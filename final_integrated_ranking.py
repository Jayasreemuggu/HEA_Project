import os
import pandas as pd
import numpy as np

BASE = "/mnt/c/Users/jayam/Downloads/IITH_ML_project"

PRED_FILE = os.path.join(
    BASE, "predictions", "results",
    "HEA_MPEA_5000_Final_Predictions.csv"
)

UNC_FILE = os.path.join(
    BASE, "uncertainty", "results",
    "HEA_MPEA_5000_Uncertainty_Aware_Ranking_CORRECTED.csv"
)

OUT = os.path.join(
    BASE, "predictions", "results"
)

os.makedirs(OUT, exist_ok=True)

print("=" * 75)
print("FINAL INTEGRATED HEA/MPEA CANDIDATE RANKING")
print("=" * 75)

# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

pred = pd.read_csv(PRED_FILE)
unc = pd.read_csv(UNC_FILE)

print("\nFour-property predictions:", len(pred))
print("Corrected uncertainty:", len(unc))

if len(pred) != 5000 or len(unc) != 5000:
    raise RuntimeError("Expected exactly 5000 candidates in both files.")

# ------------------------------------------------------------
# CHECK IDS
# ------------------------------------------------------------

if not set(pred["candidate_id"]) == set(unc["candidate_id"]):
    raise RuntimeError("Candidate IDs do not match.")

# ------------------------------------------------------------
# KEEP ORIGINAL FOUR-PROPERTY PREDICTIONS
# ------------------------------------------------------------

property_cols = [
    "Predicted_YS_MPa",
    "Predicted_UTS_MPa",
    "Predicted_Elongation_pct",
    "Predicted_Hardness_HV",
]

missing = [
    c for c in property_cols
    if c not in pred.columns
]

if missing:
    raise RuntimeError(
        f"Missing property columns: {missing}"
    )

result = pred.copy()

# ------------------------------------------------------------
# MERGE CORRECTED UNCERTAINTY
# ------------------------------------------------------------

unc_cols = [
    "candidate_id",
    "YS_Ensemble_Mean_MPa",
    "YS_Uncertainty_STD_MPa",
    "YS_Relative_Uncertainty",
    "Uncertainty_Risk",
]

unc_cols = [
    c for c in unc_cols
    if c in unc.columns
]

result = result.drop(
    columns=[
        c for c in unc_cols
        if c != "candidate_id" and c in result.columns
    ],
    errors="ignore"
)

result = result.merge(
    unc[unc_cols],
    on="candidate_id",
    how="left",
    validate="one_to_one"
)

# ------------------------------------------------------------
# VALIDATE
# ------------------------------------------------------------

if result["YS_Ensemble_Mean_MPa"].isna().any():
    raise RuntimeError("Missing ensemble YS predictions.")

if result["YS_Uncertainty_STD_MPa"].isna().any():
    raise RuntimeError("Missing YS uncertainty values.")

print("\nCorrected uncertainty successfully merged.")

print(
    "Unique ensemble YS:",
    result["YS_Ensemble_Mean_MPa"].nunique()
)

print(
    "Unique uncertainty:",
    result["YS_Uncertainty_STD_MPa"].nunique()
)

# ------------------------------------------------------------
# NORMALIZE FOUR PROPERTIES
# ------------------------------------------------------------

for col in property_cols:

    values = pd.to_numeric(
        result[col],
        errors="coerce"
    )

    if values.isna().any():
        raise RuntimeError(
            f"Missing values in {col}"
        )

    vmin = values.min()
    vmax = values.max()

    result[col + "_Normalized"] = (
        (values - vmin) /
        (vmax - vmin)
    )

# ------------------------------------------------------------
# PERFORMANCE SCORE
# ------------------------------------------------------------

result["Performance_Score"] = (
    result["Predicted_YS_MPa_Normalized"]
    + result["Predicted_UTS_MPa_Normalized"]
    + result["Predicted_Elongation_pct_Normalized"]
    + result["Predicted_Hardness_HV_Normalized"]
) / 4.0

# ------------------------------------------------------------
# UNCERTAINTY PENALTY
# ------------------------------------------------------------

result["Uncertainty_Penalty"] = (
    result["YS_Relative_Uncertainty"]
)

# Lower uncertainty = higher confidence.
#
# Score is deliberately kept interpretable:
# performance divided by (1 + relative uncertainty).
# ------------------------------------------------------------

result["Uncertainty_Aware_Score"] = (
    result["Performance_Score"] /
    (
        1.0 +
        result["Uncertainty_Penalty"]
    )
)

# ------------------------------------------------------------
# PARETO OPTIMALITY
# ------------------------------------------------------------

objectives = [
    "Predicted_YS_MPa",
    "Predicted_UTS_MPa",
    "Predicted_Elongation_pct",
    "Predicted_Hardness_HV",
]

values = result[objectives].to_numpy(dtype=float)

n = len(result)
pareto = np.ones(n, dtype=bool)

for i in range(n):

    if not pareto[i]:
        continue

    dominated = np.all(
        values >= values[i],
        axis=1
    ) & np.any(
        values > values[i],
        axis=1
    )

    dominated[i] = False

    pareto[dominated] = False

result["Pareto_Optimal"] = pareto

# ------------------------------------------------------------
# SORT
# ------------------------------------------------------------

result = result.sort_values(
    [
        "Uncertainty_Aware_Score",
        "Performance_Score"
    ],
    ascending=[False, False]
).reset_index(drop=True)

result["Final_Rank"] = (
    np.arange(len(result)) + 1
)

# ------------------------------------------------------------
# PARETO RANK
# ------------------------------------------------------------

pareto_result = result[
    result["Pareto_Optimal"]
].copy()

pareto_result = pareto_result.sort_values(
    "Uncertainty_Aware_Score",
    ascending=False
).reset_index(drop=True)

pareto_result["Pareto_Rank"] = (
    np.arange(len(pareto_result)) + 1
)

# ------------------------------------------------------------
# TOP 100
# ------------------------------------------------------------

top100 = result.head(100).copy()

top20 = result.head(20).copy()

# ------------------------------------------------------------
# LOW-UNCERTAINTY HIGH-PERFORMANCE
# ------------------------------------------------------------

unc_threshold = result[
    "YS_Uncertainty_STD_MPa"
].quantile(0.50)

perf_threshold = result[
    "Performance_Score"
].quantile(0.75)

low_unc_high_perf = result[
    (result["YS_Uncertainty_STD_MPa"] <= unc_threshold)
    &
    (result["Performance_Score"] >= perf_threshold)
].copy()

low_unc_high_perf = low_unc_high_perf.sort_values(
    "Uncertainty_Aware_Score",
    ascending=False
)

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

main_file = os.path.join(
    OUT,
    "HEA_MPEA_FINAL_INTEGRATED_RANKING.csv"
)

top100_file = os.path.join(
    OUT,
    "HEA_MPEA_FINAL_TOP100.csv"
)

top20_file = os.path.join(
    OUT,
    "HEA_MPEA_FINAL_TOP20.csv"
)

pareto_file = os.path.join(
    OUT,
    "HEA_MPEA_FINAL_PARETO.csv"
)

low_file = os.path.join(
    OUT,
    "HEA_MPEA_FINAL_LOW_UNCERTAINTY_HIGH_PERFORMANCE.csv"
)

result.to_csv(
    main_file,
    index=False
)

top100.to_csv(
    top100_file,
    index=False
)

top20.to_csv(
    top20_file,
    index=False
)

pareto_result.to_csv(
    pareto_file,
    index=False
)

low_unc_high_perf.to_csv(
    low_file,
    index=False
)

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("FINAL RESULTS")
print("=" * 75)

print(
    f"\nPareto-optimal candidates:"
    f" {len(pareto_result)} / {len(result)}"
    f" ({100 * len(pareto_result) / len(result):.2f}%)"
)

print(
    f"Low-uncertainty + high-performance:"
    f" {len(low_unc_high_perf)}"
)

print("\nTOP 20 FINAL CANDIDATES")
print("-" * 75)

display_cols = [
    "Final_Rank",
    "candidate_id",
    "composition",
    "Predicted_YS_MPa",
    "Predicted_UTS_MPa",
    "Predicted_Elongation_pct",
    "Predicted_Hardness_HV",
    "YS_Ensemble_Mean_MPa",
    "YS_Uncertainty_STD_MPa",
    "Performance_Score",
    "Uncertainty_Aware_Score",
    "Pareto_Optimal",
]

print(
    result[display_cols]
    .head(20)
    .to_string(index=False)
)

print("\nFiles created:")

for f in [
    main_file,
    top100_file,
    top20_file,
    pareto_file,
    low_file,
]:
    print(f)

print("\nInterpretation:")
print(
    "YS_Uncertainty_STD_MPa is the standard deviation "
    "of the 15-model bootstrap ensemble."
)

print(
    "It is a relative model-risk measure, not a calibrated "
    "95% prediction interval."
)

print(
    "\nCandidate predictions are model-based screening results "
    "within the generated candidate composition space and "
    "require experimental validation."
)

