import os
import glob
import joblib
import pandas as pd
import numpy as np

BASE = "predictions/results"
UNC = "uncertainty/results"
OUT = "uncertainty/results"

os.makedirs(OUT, exist_ok=True)

CANDIDATE_FILE = f"{BASE}/HEA_MPEA_5000_Final_Predictions.csv"

print("=" * 70)
print("UNCERTAINTY-AWARE CANDIDATE RANKING")
print("=" * 70)

# ------------------------------------------------------------
# Load final candidate predictions
# ------------------------------------------------------------
df = pd.read_csv(CANDIDATE_FILE)

print("\nCandidates loaded:", len(df))

# ------------------------------------------------------------
# Load ensemble model files
# ------------------------------------------------------------
model_files = sorted(
    glob.glob(f"{UNC}/*.pkl")
)

print("\nPKL files found:")
for f in model_files:
    print(os.path.basename(f))

# Prefer the saved uncertainty preprocessor only if needed.
# The ensemble prediction file itself cannot be used for candidates.
ensemble_models = []

for f in model_files:
    name = os.path.basename(f).lower()

    if (
        "model" in name
        and "ys" in name
        and "ensemble" in name
    ):
        try:
            ensemble_models.append(joblib.load(f))
        except Exception as e:
            print("Could not load:", f, e)

# ------------------------------------------------------------
# Search for individual bootstrap models
# ------------------------------------------------------------
if len(ensemble_models) == 0:

    possible_dirs = [
        "uncertainty/models",
        "uncertainty/results",
        "models",
    ]

    for directory in possible_dirs:

        if not os.path.exists(directory):
            continue

        files = sorted(
            glob.glob(
                f"{directory}/*.pkl"
            )
        )

        for f in files:

            name = os.path.basename(f).lower()

            if (
                "ys" in name
                and (
                    "bootstrap" in name
                    or "ensemble" in name
                    or "model" in name
                )
            ):
                try:
                    m = joblib.load(f)

                    # Accept sklearn pipelines
                    if hasattr(m, "predict"):
                        ensemble_models.append(m)

                except Exception:
                    pass

# Remove duplicates by object identity
unique_models = []
seen = set()

for m in ensemble_models:
    if id(m) not in seen:
        unique_models.append(m)
        seen.add(id(m))

ensemble_models = unique_models

print(
    "\nLoaded YS ensemble models:",
    len(ensemble_models)
)

# ------------------------------------------------------------
# If bootstrap models are not saved individually,
# reconstruct uncertainty using the final model architecture
# from the saved uncertainty ensemble predictions is impossible
# for unseen candidates.
# ------------------------------------------------------------
if len(ensemble_models) < 2:

    print("\nWARNING:")
    print(
        "Individual YS bootstrap models were not found."
    )
    print(
        "The existing YS_ensemble_predictions.csv contains "
        "uncertainty only for validation records, not candidates."
    )
    print(
        "Candidate uncertainty therefore cannot be calculated "
        "from that file alone."
    )

    print("\nSearching for model files recursively...")

    all_pkls = sorted(
        glob.glob(
            "**/*.pkl",
            recursive=True
        )
    )

    for f in all_pkls:
        print(f)

    raise SystemExit(
        "\nSTOP: Please send the printed PKL file list. "
        "We need to locate the saved bootstrap models before "
        "performing uncertainty-aware candidate ranking."
    )

# ------------------------------------------------------------
# Determine model input columns
# ------------------------------------------------------------
first_model = ensemble_models[0]

if hasattr(first_model, "named_steps"):
    prep = first_model.named_steps["preprocessor"]
    expected = list(
        prep.feature_names_in_
    )
else:
    expected = None

if expected is None:
    raise ValueError(
        "Could not determine expected YS model features."
    )

# ------------------------------------------------------------
# Map candidate columns to model features
# ------------------------------------------------------------
candidate_map = {
    c.lower(): c
    for c in df.columns
}

# Candidate file has only 12 elemental columns.
# Other required YS features must be supplied consistently
# with the candidate-generation assumptions.
#
# Use zeros for absent elements and standard baseline
# conditions matching the candidate prediction workflow.
# ------------------------------------------------------------
required = {}

for feature in expected:

    key = feature.lower()

    if key in candidate_map:
        required[feature] = df[
            candidate_map[key]
        ]

    elif key.endswith("_at_pct"):
        required[feature] = 0.0

    elif key == "test_temperature_c":
        required[feature] = 25.0

    elif key == "vec":
        required[feature] = np.nan

    elif key == "atomic_size_mismatch":
        required[feature] = np.nan

    elif key == "mixing_enthalpy":
        required[feature] = np.nan

    elif key == "mixing_entropy":
        required[feature] = np.nan

    elif key == "density_calc_g_cm3":
        required[feature] = np.nan

    elif key == "density_exp_g_cm3":
        required[feature] = np.nan

    elif key == "grain_size_um":
        required[feature] = np.nan

    elif key == "precipitate_size_nm":
        required[feature] = np.nan

    elif key == "matrix_volume_pct":
        required[feature] = np.nan

    elif key == "test_type":
        required[feature] = "C"

    elif key == "phase":
        required[feature] = "BCC"

    elif key == "processing_method":
        required[feature] = "WROUGHT"

    elif key == "alloy_class":
        required[feature] = "HEA"

    elif key == "equilibrium_condition":
        required[feature] = np.nan

    elif key == "single_multiphase":
        required[feature] = np.nan

    elif key == "precipitate_info":
        required[feature] = "None"

    else:
        required[feature] = np.nan

Xcand = pd.DataFrame(
    required,
    index=df.index
)

# ------------------------------------------------------------
# Predict with every bootstrap model
# ------------------------------------------------------------
predictions = []

print(
    "\nPredicting 5,000 candidates with",
    len(ensemble_models),
    "YS ensemble models..."
)

for i, model in enumerate(
    ensemble_models,
    start=1
):

    try:
        p = np.asarray(
            model.predict(Xcand)
        ).reshape(-1)

        predictions.append(p)

        print(
            f"Model {i:02d}: "
            f"min={p.min():.2f}, "
            f"max={p.max():.2f}"
        )

    except Exception as e:

        print(
            f"Model {i:02d} FAILED:",
            repr(e)
        )

if len(predictions) < 2:
    raise RuntimeError(
        "Fewer than two successful ensemble predictions."
    )

P = np.vstack(predictions)

# ------------------------------------------------------------
# Candidate uncertainty
# ------------------------------------------------------------
df["YS_Ensemble_Mean_MPa"] = np.mean(
    P,
    axis=0
)

df["YS_Uncertainty_STD_MPa"] = np.std(
    P,
    axis=0,
    ddof=1
)

# ------------------------------------------------------------
# Compare ensemble mean with existing YS prediction
# ------------------------------------------------------------
df["YS_Ensemble_Difference_MPa"] = (
    df["YS_Ensemble_Mean_MPa"]
    - df["Predicted_YS_MPa"]
)

# ------------------------------------------------------------
# Normalize objectives
# ------------------------------------------------------------
def minmax(series):
    series = pd.to_numeric(
        series,
        errors="coerce"
    )

    mn = series.min()
    mx = series.max()

    if mx == mn:
        return pd.Series(
            1.0,
            index=series.index
        )

    return (series - mn) / (mx - mn)

df["YS_Normalized"] = minmax(
    df["Predicted_YS_MPa"]
)

df["UTS_Normalized"] = minmax(
    df["Predicted_UTS_MPa"]
)

df["Elongation_Normalized"] = minmax(
    df["Predicted_Elongation_pct"]
)

df["Hardness_Normalized"] = minmax(
    df["Predicted_Hardness_HV"]
)

df["Uncertainty_Normalized"] = minmax(
    df["YS_Uncertainty_STD_MPa"]
)

df["Uncertainty_Confidence"] = (
    1 -
    df["Uncertainty_Normalized"]
)

# ------------------------------------------------------------
# Performance score
# ------------------------------------------------------------
df["Performance_Score"] = (
    0.25 * df["YS_Normalized"] +
    0.30 * df["UTS_Normalized"] +
    0.30 * df["Elongation_Normalized"] +
    0.15 * df["Hardness_Normalized"]
)

# ------------------------------------------------------------
# Uncertainty-aware score
# ------------------------------------------------------------
df["Uncertainty_Aware_Score"] = (
    0.80 * df["Performance_Score"] +
    0.20 * df["Uncertainty_Confidence"]
)

# ------------------------------------------------------------
# Risk categories
# ------------------------------------------------------------
q25 = df[
    "YS_Uncertainty_STD_MPa"
].quantile(0.25)

q75 = df[
    "YS_Uncertainty_STD_MPa"
].quantile(0.75)

def risk(x):

    if x <= q25:
        return "Low"

    if x >= q75:
        return "High"

    return "Medium"

df["Uncertainty_Risk"] = (
    df["YS_Uncertainty_STD_MPa"]
    .apply(risk)
)

# ------------------------------------------------------------
# Rank
# ------------------------------------------------------------
df = df.sort_values(
    "Uncertainty_Aware_Score",
    ascending=False
).reset_index(drop=True)

df["Uncertainty_Aware_Rank"] = (
    np.arange(len(df)) + 1
)

# ------------------------------------------------------------
# High performance + low uncertainty
# ------------------------------------------------------------
performance_threshold = df[
    "Performance_Score"
].quantile(0.75)

uncertainty_threshold = df[
    "YS_Uncertainty_STD_MPa"
].quantile(0.50)

df[
    "High_Performance_Low_Uncertainty"
] = (
    (df["Performance_Score"]
     >= performance_threshold)
    &
    (df["YS_Uncertainty_STD_MPa"]
     <= uncertainty_threshold)
)

shortlist = df[
    df["High_Performance_Low_Uncertainty"]
].copy()

shortlist = shortlist.sort_values(
    "Uncertainty_Aware_Score",
    ascending=False
)

# ------------------------------------------------------------
# Pareto subset
# ------------------------------------------------------------
pareto = df[
    df["Pareto_Optimal"] == True
].copy()

pareto = pareto.sort_values(
    "Uncertainty_Aware_Score",
    ascending=False
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
full_file = (
    f"{OUT}/HEA_MPEA_5000_Uncertainty_Aware_Ranking.csv"
)

top_file = (
    f"{OUT}/HEA_MPEA_Top50_Uncertainty_Aware.csv"
)

short_file = (
    f"{OUT}/HEA_MPEA_Low_Uncertainty_High_Performance.csv"
)

pareto_file = (
    f"{OUT}/HEA_MPEA_Pareto_Uncertainty_Ranked.csv"
)

df.to_csv(
    full_file,
    index=False
)

df.head(50).to_csv(
    top_file,
    index=False
)

shortlist.to_csv(
    short_file,
    index=False
)

pareto.to_csv(
    pareto_file,
    index=False
)

# ------------------------------------------------------------
# Print
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("UNCERTAINTY RESULTS")
print("=" * 70)

print(
    "\nYS uncertainty:"
)

print(
    df[
        "YS_Uncertainty_STD_MPa"
    ].describe()
)

print(
    "\nLow uncertainty threshold:",
    round(uncertainty_threshold, 3)
)

print(
    "High uncertainty threshold:",
    round(q75, 3)
)

print(
    "\nHigh-performance + low-uncertainty:",
    len(shortlist),
    f"({100 * len(shortlist) / len(df):.2f}%)"
)

cols = [
    "candidate_id",
    "composition",
    "Predicted_YS_MPa",
    "Predicted_UTS_MPa",
    "Predicted_Elongation_pct",
    "Predicted_Hardness_HV",
    "YS_Uncertainty_STD_MPa",
    "Uncertainty_Risk",
    "Performance_Score",
    "Uncertainty_Aware_Score",
    "Uncertainty_Aware_Rank",
    "Pareto_Optimal"
]

print(
    "\nTOP 20 UNCERTAINTY-AWARE CANDIDATES"
)

print(
    df.head(20)[cols].to_string(
        index=False
    )
)

print(
    "\nFiles created:"
)

print(full_file)
print(top_file)
print(short_file)
print(pareto_file)

print(
    "\nNOTE: YS uncertainty is a relative ensemble "
    "uncertainty measure, NOT a calibrated 95% interval."
)
