import os
import joblib
import numpy as np
import pandas as pd

BASE = "/mnt/c/Users/jayam/Downloads/IITH_ML_project"

CANDIDATE_FILE = os.path.join(
    BASE,
    "uncertainty",
    "candidates",
    "results",
    "HEA_MPEA_Candidate_Compositions.csv"
)

UTS_FILE = os.path.join(
    BASE,
    "predictions",
    "results",
    "UTS_5000_Candidate_Validation.csv"
)

ELONG_FILE = os.path.join(
    BASE,
    "predictions",
    "results",
    "Elongation_5000_Candidate_Validation.csv"
)

YS_MODEL_FILE = os.path.join(
    BASE,
    "YS_XGBoost_Model.pkl"
)

UTS_MODEL_FILE = os.path.join(
    BASE,
    "UTS_XGBoost_Model.pkl"
)

ELONG_MODEL_FILE = os.path.join(
    BASE,
    "Elongation_XGBoost_Model.pkl"
)

HARDNESS_MODEL_FILE = os.path.join(
    BASE,
    "Hardness_ExtraTrees_Model.pkl"
)

RESULT_DIR = os.path.join(
    BASE,
    "predictions",
    "results"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

print("=" * 70)
print("FINAL HEA/MPEA CANDIDATE RANKING")
print("=" * 70)

# ============================================================
# LOAD
# ============================================================

candidates = pd.read_csv(CANDIDATE_FILE)
uts = pd.read_csv(UTS_FILE)
elong = pd.read_csv(ELONG_FILE)

print("\nCandidate records:", len(candidates))
print("UTS predictions :", len(uts))
print("Elongation      :", len(elong))

# ============================================================
# CHECK IDS
# ============================================================

if not candidates["candidate_id"].equals(
    uts["candidate_id"]
):
    raise RuntimeError(
        "Candidate ID mismatch in UTS file."
    )

if not candidates["candidate_id"].equals(
    elong["candidate_id"]
):
    raise RuntimeError(
        "Candidate ID mismatch in Elongation file."
    )

# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading models...")

ys_model = joblib.load(
    YS_MODEL_FILE
)

uts_model = joblib.load(
    UTS_MODEL_FILE
)

elong_model = joblib.load(
    ELONG_MODEL_FILE
)

hardness_model = joblib.load(
    HARDNESS_MODEL_FILE
)

print("All four models loaded.")

# ============================================================
# COMMON CANDIDATE ELEMENT MAP
# ============================================================

candidate_map = {
    "al": "Al",
    "b": "B",
    "c": "C",
    "co": "Co",
    "cr": "Cr",
    "cu": "Cu",
    "fe": "Fe",
    "mn": "Mn",
    "mo": "Mo",
    "nb": "Nb",
    "ni": "Ni",
    "si": "Si",
    "ta": "Ta",
    "ti": "Ti",
    "v": "V",
    "w": "W",
    "zr": "Zr",
    "ag": "Ag",
    "ca": "Ca",
    "ga": "Ga",
    "hf": "Hf",
    "i": "I",
    "li": "Li",
    "mg": "Mg",
    "nd": "Nd",
    "o": "O",
    "pd": "Pd",
    "re": "Re",
    "ru": "Ru",
    "s": "S",
    "sc": "Sc",
    "sn": "Sn",
    "t": "T",
    "y": "Y",
    "zn": "Zn"
}

# ============================================================
# FUNCTION: BUILD MODEL INPUT
# ============================================================

def build_input(model):

    pre = model.named_steps["preprocessor"]

    expected_features = list(
        pre.feature_names_in_
    )

    X = pd.DataFrame(
        index=candidates.index
    )

    for feature in expected_features:

        # ----------------------------------------------------
        # ELEMENT FEATURES
        # ----------------------------------------------------

        if feature.endswith("_at_pct"):

            element_key = feature[:-7].lower()

            source = candidate_map.get(
                element_key
            )

            if source is not None and source in candidates.columns:

                X[feature] = pd.to_numeric(
                    candidates[source],
                    errors="coerce"
                )

            else:

                X[feature] = 0.0

        # ----------------------------------------------------
        # TEMPERATURE
        # ----------------------------------------------------

        elif feature.lower() == "test_temperature_c":

            X[feature] = 25.0

        # ----------------------------------------------------
        # DERIVED / PHYSICAL FEATURES
        # ----------------------------------------------------

        elif feature.lower() in [
            "vec",
            "atomic_size_mismatch",
            "mixing_enthalpy",
            "mixing_entropy",
            "density_exp_g_cm3",
            "density_calc_g_cm3",
            "grain_size_um",
            "precipitate_size_nm",
            "matrix_volume_pct"
        ]:

            X[feature] = np.nan

        # ----------------------------------------------------
        # CATEGORICAL FEATURES
        # ----------------------------------------------------

        elif feature.lower() == "test_type":

            X[feature] = "T"

        elif feature.lower() == "phase":

            X[feature] = "UNKNOWN"

        elif feature.lower() == "processing_method":

            X[feature] = "CAST"

        elif feature.lower() == "alloy_class":

            X[feature] = "HEA"

        elif feature.lower() == "equilibrium_condition":

            X[feature] = "UNKNOWN"

        elif feature.lower() == "single_multiphase":

            X[feature] = "UNKNOWN"

        elif feature.lower() == "precipitate_info":

            X[feature] = "UNKNOWN"

        else:

            # Safe fallback
            X[feature] = np.nan

    return X


# ============================================================
# PREDICT YS
# ============================================================

print("\nPredicting Yield Strength...")

X_ys = build_input(
    ys_model
)

print(
    "YS input:",
    X_ys.shape
)

ys_pred = ys_model.predict(
    X_ys
)

print(
    "YS prediction range:",
    round(float(ys_pred.min()), 3),
    "-",
    round(float(ys_pred.max()), 3)
)

print(
    "YS unique:",
    len(
        np.unique(
            np.round(
                ys_pred,
                6
            )
        )
    )
)

# ============================================================
# PREDICT UTS
# ============================================================

print("\nUsing validated UTS predictions...")

uts_pred = uts[
    "Predicted_UTS_MPa"
].to_numpy()

print(
    "UTS prediction range:",
    round(float(uts_pred.min()), 3),
    "-",
    round(float(uts_pred.max()), 3)
)

print(
    "UTS unique:",
    len(
        np.unique(
            np.round(
                uts_pred,
                6
            )
        )
    )
)

# ============================================================
# PREDICT ELONGATION
# ============================================================

print("\nUsing validated Elongation predictions...")

elong_pred = elong[
    "Predicted_Elongation_pct"
].to_numpy()

print(
    "Elongation prediction range:",
    round(float(elong_pred.min()), 3),
    "-",
    round(float(elong_pred.max()), 3)
)

print(
    "Elongation unique:",
    len(
        np.unique(
            np.round(
                elong_pred,
                6
            )
        )
    )
)

# ============================================================
# PREDICT HARDNESS
# ============================================================

print("\nPredicting Hardness...")

X_hardness = build_input(
    hardness_model
)

print(
    "Hardness input:",
    X_hardness.shape
)

hardness_pred = hardness_model.predict(
    X_hardness
)

print(
    "Hardness prediction range:",
    round(float(hardness_pred.min()), 3),
    "-",
    round(float(hardness_pred.max()), 3)
)

print(
    "Hardness unique:",
    len(
        np.unique(
            np.round(
                hardness_pred,
                6
            )
        )
    )
)

# ============================================================
# FINAL TABLE
# ============================================================

result = candidates.copy()

result["Predicted_YS_MPa"] = ys_pred

result["Predicted_UTS_MPa"] = uts_pred

result["Predicted_Elongation_pct"] = elong_pred

result["Predicted_Hardness_HV"] = hardness_pred

# ============================================================
# UTS / YS
# ============================================================

result["UTS_to_YS_Ratio"] = (
    result["Predicted_UTS_MPa"]
    /
    result["Predicted_YS_MPa"]
)

result["UTS_YS_Consistency"] = (
    result["Predicted_UTS_MPa"]
    >=
    result["Predicted_YS_MPa"]
)

# ============================================================
# MIN-MAX NORMALIZATION
# ============================================================

def minmax(series):

    mn = series.min()
    mx = series.max()

    if mx == mn:
        return pd.Series(
            0.5,
            index=series.index
        )

    return (
        (series - mn)
        /
        (mx - mn)
    )


result["YS_Score"] = minmax(
    result["Predicted_YS_MPa"]
)

result["UTS_Score"] = minmax(
    result["Predicted_UTS_MPa"]
)

result["Elongation_Score"] = minmax(
    result["Predicted_Elongation_pct"]
)

result["Hardness_Score"] = minmax(
    result["Predicted_Hardness_HV"]
)

# ============================================================
# BALANCED SCORE
# ============================================================

result["Balanced_Score"] = (
    0.30 * result["YS_Score"]
    +
    0.30 * result["UTS_Score"]
    +
    0.25 * result["Elongation_Score"]
    +
    0.15 * result["Hardness_Score"]
)

# ============================================================
# PARETO FRONT
# ============================================================

objectives = [
    "Predicted_YS_MPa",
    "Predicted_UTS_MPa",
    "Predicted_Elongation_pct",
    "Predicted_Hardness_HV"
]

values = result[
    objectives
].to_numpy()

n = len(values)

pareto = np.ones(
    n,
    dtype=bool
)

for i in range(n):

    if not pareto[i]:
        continue

    dominates = (
        np.all(
            values >= values[i],
            axis=1
        )
        &
        np.any(
            values > values[i],
            axis=1
        )
    )

    dominates[i] = False

    if np.any(dominates):
        pareto[i] = False

result["Pareto_Optimal"] = pareto

# ============================================================
# RANK
# ============================================================

result = result.sort_values(
    [
        "Pareto_Optimal",
        "Balanced_Score"
    ],
    ascending=[
        False,
        False
    ]
).reset_index(
    drop=True
)

result["Overall_Rank"] = (
    np.arange(
        len(result)
    )
    + 1
)

# ============================================================
# SAVE FULL DATASET
# ============================================================

FULL_FILE = os.path.join(
    RESULT_DIR,
    "HEA_MPEA_5000_Final_Predictions.csv"
)

result.to_csv(
    FULL_FILE,
    index=False
)

# ============================================================
# TOP 100
# ============================================================

top100 = result.head(
    100
).copy()

TOP100_FILE = os.path.join(
    RESULT_DIR,
    "HEA_MPEA_Top100_Candidates.csv"
)

top100.to_csv(
    TOP100_FILE,
    index=False
)

# ============================================================
# PARETO
# ============================================================

pareto_df = result[
    result["Pareto_Optimal"]
].copy()

PARETO_FILE = os.path.join(
    RESULT_DIR,
    "HEA_MPEA_Pareto_Optimal_Candidates.csv"
)

pareto_df.to_csv(
    PARETO_FILE,
    index=False
)

# ============================================================
# TOP 20
# ============================================================

top20 = result.head(
    20
).copy()

TOP20_FILE = os.path.join(
    RESULT_DIR,
    "HEA_MPEA_Top20_Candidates.csv"
)

top20.to_csv(
    TOP20_FILE,
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL PREDICTION SUMMARY")
print("=" * 70)

for col in objectives:

    print(
        f"\n{col}"
    )

    print(
        "  Min   :",
        round(float(result[col].min()), 3)
    )

    print(
        "  Max   :",
        round(float(result[col].max()), 3)
    )

    print(
        "  Mean  :",
        round(float(result[col].mean()), 3)
    )

    print(
        "  Median:",
        round(float(result[col].median()), 3)
    )

# ============================================================
# PARETO SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PARETO ANALYSIS")
print("=" * 70)

print(
    "Pareto-optimal candidates:",
    int(result["Pareto_Optimal"].sum())
)

print(
    "Pareto percentage:",
    round(
        100
        * result["Pareto_Optimal"].sum()
        / len(result),
        2
    ),
    "%"
)

# ============================================================
# TOP 20
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 CANDIDATES")
print("=" * 70)

display_columns = [
    "Overall_Rank",
    "candidate_id",
    "composition",
    "Predicted_YS_MPa",
    "Predicted_UTS_MPa",
    "Predicted_Elongation_pct",
    "Predicted_Hardness_HV",
    "UTS_to_YS_Ratio",
    "Balanced_Score",
    "Pareto_Optimal"
]

print(
    top20[
        display_columns
    ].to_string(
        index=False
    )
)

# ============================================================
# FILES
# ============================================================

print("\n" + "=" * 70)
print("FINAL FILES")
print("=" * 70)

print(FULL_FILE)
print(TOP100_FILE)
print(PARETO_FILE)
print(TOP20_FILE)

print("\n" + "=" * 70)
print("FINAL CANDIDATE RANKING COMPLETE")
print("=" * 70)
