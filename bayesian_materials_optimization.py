import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import optuna

warnings.filterwarnings("ignore")

BASE = "/mnt/c/Users/jayam/Downloads/IITH_ML_project"

sys.path.insert(0, BASE)

from predictions.predict_all_properties import (
    build_prediction_matrix,
    get_element_symbol,
)

# ============================================================
# CONFIGURATION
# ============================================================

N_TRIALS = 1000
SEED = 42

ELEMENTS = [
    "Al", "Co", "Cr", "Cu", "Fe", "Mn",
    "Mo", "Nb", "Ni", "Ti", "V", "W"
]

MODEL_FILES = {
    "YS": "YS_XGBoost_Model.pkl",
    "UTS": "UTS_XGBoost_Model.pkl",
    "Elongation": "Elongation_XGBoost_Model.pkl",
    "Hardness": "Hardness_ExtraTrees_Model.pkl",
}

MODEL_PATHS = {
    k: os.path.join(BASE, v)
    for k, v in MODEL_FILES.items()
}

BOOTSTRAP_DIR = os.path.join(
    BASE,
    "uncertainty",
    "models"
)

OUTPUT_DIR = os.path.join(
    BASE,
    "predictions",
    "results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD MODELS
# ============================================================

print("=" * 75)
print("AI-ASSISTED HEA/MPEA BAYESIAN MATERIALS OPTIMIZATION")
print("=" * 75)

models = {}

for name, path in MODEL_PATHS.items():
    print(f"Loading {name}: {path}")
    models[name] = joblib.load(path)

# Load bootstrap YS ensemble.
bootstrap_models = []

for i in range(1, 16):
    path = os.path.join(
        BOOTSTRAP_DIR,
        f"YS_Bootstrap_Model_{i:02d}.pkl"
    )

    if os.path.exists(path):
        bootstrap_models.append(joblib.load(path))

print(f"\nBootstrap YS models loaded: {len(bootstrap_models)}")

# ============================================================
# LOAD EXPERIMENTAL COMPOSITIONS FOR NOVELTY
# ============================================================

DATA_FILE = os.path.join(
    BASE,
    "HEA_MPEA_ML_Ready_Deduplicated.csv"
)

experimental = pd.read_csv(DATA_FILE)

exp_cols = [f"{e}_at_pct" for e in [
    "Ag","Al","B","C","Ca","Co","Cr","Cu","Fe","Ga",
    "Hf","I","Li","Mg","Mn","Mo","Nb","Nd","Ni","O",
    "Pd","Re","Ru","S","Sc","Si","Sn","T","Ta","Ti",
    "V","W","Y","Zn","Zr"
]]

exp_compositions = (
    experimental[exp_cols]
    .dropna(how="all")
    .fillna(0)
    .to_numpy(dtype=float)
)

# Normalize by composition magnitude.
exp_norm = np.linalg.norm(
    exp_compositions,
    axis=1,
    keepdims=True
)

exp_norm[exp_norm == 0] = 1

exp_unit = exp_compositions / exp_norm

print(f"Experimental compositions available: {len(exp_unit)}")

# ============================================================
# REFERENCE RANGES
# ============================================================
#
# Use observed training-property ranges rather than arbitrary
# engineering limits.

reference_ranges = {}

targets = {
    "YS": "YS_Tensile_MPa",
    "UTS": "UTS_Tensile_MPa",
    "Elongation": "Elongation_Tensile_pct",
    "Hardness": "Hardness_HV",
}

for name, column in targets.items():

    values = pd.to_numeric(
        experimental[column],
        errors="coerce"
    ).dropna()

    reference_ranges[name] = (
        float(values.quantile(0.05)),
        float(values.quantile(0.95))
    )

    print(
        f"{name:12s}: "
        f"{reference_ranges[name][0]:.3f} - "
        f"{reference_ranges[name][1]:.3f}"
    )

# ============================================================
# COMPOSITION BUILDER
# ============================================================

def build_candidate(values):

    row = {
        "candidate_id": "OPTUNA_TEMP",
        "composition": "",
        "n_elements": len(values),
    }

    for element in ELEMENTS:
        row[element] = 0.0

    for element, value in values.items():
        row[element] = value

    row["composition"] = "".join(
        f"{e}{values.get(e, 0.0):.2f}"
        for e in ELEMENTS
        if values.get(e, 0.0) > 0
    )

    return pd.DataFrame([row])

# ============================================================
# PREDICTION
# ============================================================

def predict_property(candidate_df, model):

    X = build_prediction_matrix(
        candidate_df,
        model
    )

    prediction = model.predict(X)

    return float(np.asarray(prediction)[0])

# ============================================================
# YS UNCERTAINTY
# ============================================================

def predict_ys_uncertainty(candidate_df):

    predictions = []

    for bundle in bootstrap_models:

        # Bootstrap models were saved as:
        # {"preprocessor": ..., "model": ..., "seed": ...}

        prep = bundle["preprocessor"]
        model = bundle["model"]

        # Bootstrap models store the fitted preprocessor and
        # estimator separately.
        # The preprocessor contains the authoritative 52-feature
        # input schema because the XGB estimator itself does not
        # expose feature_names_in_ after preprocessing.

        X = build_prediction_matrix(
            candidate_df,
            bundle
        )

        X_transformed = prep.transform(X)

        predictions.append(
            float(model.predict(X_transformed)[0])
        )

    if not predictions:
        return np.nan, np.nan

    predictions = np.asarray(predictions)

    return (
        float(predictions.mean()),
        float(predictions.std(ddof=1))
    )

# ============================================================
# NOVELTY
# ============================================================

def novelty_distance(candidate_df):

    vector = np.zeros(35, dtype=float)

    full_elements = [
        "Ag","Al","B","C","Ca","Co","Cr","Cu","Fe","Ga",
        "Hf","I","Li","Mg","Mn","Mo","Nb","Nd","Ni","O",
        "Pd","Re","Ru","S","Sc","Si","Sn","T","Ta","Ti",
        "V","W","Y","Zn","Zr"
    ]

    for i, e in enumerate(full_elements):

        if e in candidate_df.columns:
            vector[i] = float(
                candidate_df.iloc[0][e]
            )

    norm = np.linalg.norm(vector)

    if norm == 0:
        return np.inf

    unit = vector / norm

    cosine_similarity = exp_unit @ unit

    distance = 1.0 - cosine_similarity.max()

    return float(distance)

# ============================================================
# NORMALIZATION
# ============================================================

def normalize(value, low, high):

    if high <= low:
        return 0.0

    x = (value - low) / (high - low)

    return float(np.clip(x, 0.0, 1.0))

# ============================================================
# OPTUNA OBJECTIVE
# ============================================================

def objective(trial):

    # --------------------------------------------------------
    # Select a valid 4-6 element principal-element combination.
    #
    # The categorical space is fixed before optimization, so
    # Optuna does not encounter the dynamic categorical-space
    # error. Every selectable combination is already valid and
    # contains unique elements.
    # --------------------------------------------------------

    from itertools import combinations

    all_element_sets = [
        combo
        for n in range(4, 7)
        for combo in combinations(ELEMENTS, n)
    ]

    element_set_labels = [
        ",".join(combo)
        for combo in all_element_sets
    ]

    selected_label = trial.suggest_categorical(
        "element_set",
        element_set_labels
    )

    selected_elements = selected_label.split(",")
    n_elements = len(selected_elements)

    # --------------------------------------------------------
    # Generate positive composition fractions.
    # --------------------------------------------------------

    raw = np.array([
        trial.suggest_float(
            f"x_{e}",
            5.0,
            45.0
        )
        for e in selected_elements
    ])

    # Normalize exactly to 100 at.%.
    fractions = 100.0 * raw / raw.sum()

    values = {
        e: float(v)
        for e, v in zip(
            selected_elements,
            fractions
        )
    }

    candidate = build_candidate(values)

    # --------------------------------------------------------
    # Predict all four properties.
    # --------------------------------------------------------

    predictions = {}

    for name, model in models.items():

        predictions[name] = predict_property(
            candidate,
            model
        )

    # --------------------------------------------------------
    # YS ensemble uncertainty.
    # --------------------------------------------------------

    ys_mean, ys_uncertainty = predict_ys_uncertainty(
        candidate
    )

    if not np.isfinite(ys_mean) or not np.isfinite(ys_uncertainty):
        raise optuna.TrialPruned(
            "Invalid YS uncertainty prediction."
        )

    # --------------------------------------------------------
    # Normalize performance.
    # --------------------------------------------------------

    scores = {}

    for name in [
        "YS",
        "UTS",
        "Elongation",
        "Hardness"
    ]:

        value = predictions[name]

        if not np.isfinite(value):
            raise optuna.TrialPruned(
                f"Invalid {name} prediction."
            )

        low, high = reference_ranges[name]

        scores[name] = normalize(
            value,
            low,
            high
        )

    performance = float(
        np.mean(list(scores.values()))
    )

    # --------------------------------------------------------
    # Relative YS uncertainty.
    # --------------------------------------------------------

    relative_uncertainty = (
        ys_uncertainty /
        max(abs(ys_mean), 1.0)
    )

    # --------------------------------------------------------
    # Novelty.
    # --------------------------------------------------------

    novelty = novelty_distance(candidate)

    if not np.isfinite(novelty):
        raise optuna.TrialPruned(
            "Invalid novelty distance."
        )

    # Moderate novelty is useful, but extreme extrapolation
    # should be penalized.
    novelty_penalty = min(
        novelty / 0.5,
        1.0
    )

    # --------------------------------------------------------
    # Composite acquisition score.
    # --------------------------------------------------------

    score = float(
        performance
        - 0.25 * relative_uncertainty
        - 0.10 * novelty_penalty
    )

    # --------------------------------------------------------
    # Store trial information.
    # --------------------------------------------------------

    trial.set_user_attr(
        "composition",
        candidate.iloc[0]["composition"]
    )

    trial.set_user_attr(
        "n_elements",
        n_elements
    )

    trial.set_user_attr(
        "YS",
        predictions["YS"]
    )

    trial.set_user_attr(
        "UTS",
        predictions["UTS"]
    )

    trial.set_user_attr(
        "Elongation",
        predictions["Elongation"]
    )

    trial.set_user_attr(
        "Hardness",
        predictions["Hardness"]
    )

    trial.set_user_attr(
        "YS_Ensemble_Mean",
        ys_mean
    )

    trial.set_user_attr(
        "YS_Uncertainty",
        ys_uncertainty
    )

    trial.set_user_attr(
        "Relative_Uncertainty",
        relative_uncertainty
    )

    trial.set_user_attr(
        "Novelty_Distance",
        novelty
    )

    trial.set_user_attr(
        "Performance_Score",
        performance
    )

    trial.set_user_attr(
        "Acquisition_Score",
        score
    )

    return score

# ============================================================
# RUN OPTUNA
# ============================================================

sampler = optuna.samplers.TPESampler(
    seed=SEED
)

study = optuna.create_study(
    direction="maximize",
    sampler=sampler,
    study_name="HEA_MPEA_Bayesian_Discovery"
)

print()
print(f"Running {N_TRIALS} optimization trials...")

study.optimize(
    objective,
    n_trials=N_TRIALS,
    show_progress_bar=True
)

# ============================================================
# COLLECT RESULTS
# ============================================================

rows = []

for trial in study.trials:

    if trial.state != optuna.trial.TrialState.COMPLETE:
        continue

    row = {
        "trial": trial.number,
        "n_elements": trial.user_attrs.get("n_elements"),
        "composition": trial.user_attrs.get("composition"),
        "YS_predicted_MPa": trial.user_attrs.get("YS"),
        "UTS_predicted_MPa": trial.user_attrs.get("UTS"),
        "Elongation_predicted_pct": trial.user_attrs.get("Elongation"),
        "Hardness_predicted_HV": trial.user_attrs.get("Hardness"),
        "YS_Ensemble_Mean_MPa": trial.user_attrs.get("YS_Ensemble_Mean"),
        "YS_Uncertainty_MPa": trial.user_attrs.get("YS_Uncertainty"),
        "Relative_YS_Uncertainty": trial.user_attrs.get("Relative_Uncertainty"),
        "Novelty_Distance": trial.user_attrs.get("Novelty_Distance"),
        "Performance_Score": trial.user_attrs.get("Performance_Score"),
        "Acquisition_Score": trial.user_attrs.get("Acquisition_Score"),
    }

    for e in ELEMENTS:
        row[e] = 0.0

    # Recover elemental values from the trial parameters.
    for key, value in trial.params.items():

        if key.startswith("x_"):
            element = key[2:]

            if element in ELEMENTS:
                row[element] = value

    rows.append(row)

results = pd.DataFrame(rows)

# Remove incomplete rows defensively.
if not results.empty:
    results = results[
        results["composition"].notna()
        & results["Acquisition_Score"].notna()
    ].copy()

# ============================================================
# DEDUPLICATION
# ============================================================

results = results.drop_duplicates(
    subset=["composition"]
).reset_index(drop=True)

results = results.sort_values(
    "Acquisition_Score",
    ascending=False
).reset_index(drop=True)

results.insert(
    0,
    "optimization_rank",
    np.arange(1, len(results) + 1)
)

# ============================================================
# SAVE
# ============================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "HEA_MPEA_Bayesian_Optimization_Results.csv"
)

top_file = os.path.join(
    OUTPUT_DIR,
    "HEA_MPEA_Bayesian_Top50.csv"
)

results.to_csv(
    output_file,
    index=False
)

results.head(50).to_csv(
    top_file,
    index=False
)

# ============================================================
# REPORT
# ============================================================

print()
print("=" * 75)
print("BAYESIAN OPTIMIZATION COMPLETE")
print("=" * 75)

print("Trials completed:", len(results))
print("Best acquisition score:",
      results.iloc[0]["Acquisition_Score"])

print("\nTop 10 candidates:")
print(
    results[
        [
            "optimization_rank",
            "composition",
            "YS_predicted_MPa",
            "UTS_predicted_MPa",
            "Elongation_predicted_pct",
            "Hardness_predicted_HV",
            "YS_Uncertainty_MPa",
            "Novelty_Distance",
            "Acquisition_Score",
        ]
    ].head(10).to_string(index=False)
)

print("\nSaved:")
print(output_file)
print(top_file)
