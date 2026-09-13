import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

BASE = "/mnt/c/Users/jayam/Downloads/IITH_ML_project"

EXPERIMENTAL = os.path.join(
    BASE,
    "HEA_MPEA_ML_Ready_Deduplicated.csv"
)

CANDIDATES = os.path.join(
    BASE,
    "predictions",
    "results",
    "HEA_MPEA_FINAL_INTEGRATED_RANKING.csv"
)

OUT_DIR = os.path.join(
    BASE,
    "novelty",
    "results"
)

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================

exp = pd.read_csv(EXPERIMENTAL)
cand = pd.read_csv(CANDIDATES)

print("=" * 75)
print("HEA/MPEA COMPOSITION NOVELTY ANALYSIS")
print("=" * 75)

print("Experimental records:", len(exp))
print("Candidate records:", len(cand))

# Candidate-generation design space
ELEMENTS = [
    "Al", "Co", "Cr", "Cu", "Fe", "Mn",
    "Mo", "Nb", "Ni", "Ti", "V", "W"
]

# ============================================================
# 2. BUILD EXPERIMENTAL COMPOSITION MATRIX
# ============================================================

exp_cols = [f"{e}_at_pct" for e in ELEMENTS]

missing_exp = [
    c for c in exp_cols
    if c not in exp.columns
]

missing_cand = [
    e for e in ELEMENTS
    if e not in cand.columns
]

if missing_exp:
    raise RuntimeError(
        f"Missing experimental composition columns: {missing_exp}"
    )

if missing_cand:
    raise RuntimeError(
        f"Missing candidate element columns: {missing_cand}"
    )

X_exp = exp[exp_cols].apply(
    pd.to_numeric,
    errors="coerce"
).fillna(0.0).values

X_cand = cand[ELEMENTS].apply(
    pd.to_numeric,
    errors="coerce"
).fillna(0.0).values

print("\nComposition matrix:")
print("Experimental:", X_exp.shape)
print("Candidates:", X_cand.shape)

# ============================================================
# 3. REMOVE INVALID EXPERIMENTAL COMPOSITIONS
# ============================================================

exp_sum = X_exp.sum(axis=1)

valid_exp = (
    np.isfinite(X_exp).all(axis=1)
    &
    (exp_sum > 95)
    &
    (exp_sum < 105)
)

X_exp_valid = X_exp[valid_exp]

exp_valid_df = exp.loc[valid_exp].copy()

print(
    "\nValid experimental compositions:",
    len(X_exp_valid)
)

# ============================================================
# 4. NORMALIZE COMPOSITION SPACE
# ============================================================

# Standardization makes the distance comparable across
# the 12 elemental dimensions.

scaler = StandardScaler()

X_exp_scaled = scaler.fit_transform(
    X_exp_valid
)

X_cand_scaled = scaler.transform(
    X_cand
)

# ============================================================
# 5. FIND NEAREST EXPERIMENTAL ALLOY
# ============================================================

nn = NearestNeighbors(
    n_neighbors=1,
    metric="euclidean",
    algorithm="auto"
)

nn.fit(X_exp_scaled)

distances, indices = nn.kneighbors(
    X_cand_scaled
)

distances = distances[:, 0]
indices = indices[:, 0]

nearest_rows = exp_valid_df.iloc[
    indices
].reset_index(drop=True)

# ============================================================
# 6. CREATE NOVELTY METRICS
# ============================================================

result = cand.copy()

result["Nearest_Experimental_Distance"] = distances

result["Nearest_Experimental_Record_ID"] = (
    nearest_rows["Record_ID"].values
)

result["Nearest_Experimental_Alloy"] = (
    nearest_rows["Alloy_Name"].fillna("UNKNOWN").values
)

result["Nearest_Experimental_Composition"] = (
    nearest_rows["Composition_Canonical"]
    .fillna("UNKNOWN")
    .values
)

# Percentile-based novelty is easier to interpret:
# higher percentile = more novel than most generated candidates.

result["Novelty_Percentile"] = (
    result["Nearest_Experimental_Distance"]
    .rank(pct=True)
    * 100
)

# Normalized novelty score 0–1.
dmin = result["Nearest_Experimental_Distance"].min()
dmax = result["Nearest_Experimental_Distance"].max()

if dmax > dmin:
    result["Novelty_Score"] = (
        result["Nearest_Experimental_Distance"] - dmin
    ) / (dmax - dmin)
else:
    result["Novelty_Score"] = 0.0

# ============================================================
# 7. COMBINE PERFORMANCE + UNCERTAINTY + NOVELTY
# ============================================================

# Existing integrated ranking already contains:
#
# Performance_Score
# YS_Relative_Uncertainty
# Uncertainty_Aware_Score
#
# We retain those values and introduce novelty as an
# additional discovery objective.

required = [
    "Performance_Score",
    "YS_Relative_Uncertainty",
    "Uncertainty_Aware_Score"
]

missing_required = [
    c for c in required
    if c not in result.columns
]

if missing_required:
    raise RuntimeError(
        "Missing integrated-ranking columns: "
        + str(missing_required)
    )

# Normalize performance explicitly.
performance = result["Performance_Score"].astype(float)

pmin = performance.min()
pmax = performance.max()

if pmax > pmin:
    performance_norm = (
        performance - pmin
    ) / (pmax - pmin)
else:
    performance_norm = pd.Series(
        0.0,
        index=result.index
    )

# Relative uncertainty is used as risk.
uncertainty = (
    result["YS_Relative_Uncertainty"]
    .astype(float)
    .clip(lower=0)
)

# Convert uncertainty to a reward where lower uncertainty
# gives a larger value.

u_min = uncertainty.min()
u_max = uncertainty.max()

if u_max > u_min:
    uncertainty_quality = 1 - (
        (uncertainty - u_min)
        / (u_max - u_min)
    )
else:
    uncertainty_quality = pd.Series(
        1.0,
        index=result.index
    )

# ============================================================
# DISCOVERY SCORE
# ============================================================

# Balanced weights:
#
# 50% predicted performance
# 25% novelty
# 25% uncertainty quality
#
# This is intentionally different from the previous
# uncertainty-aware score.

result["Discovery_Score"] = (
    0.50 * performance_norm
    +
    0.25 * result["Novelty_Score"]
    +
    0.25 * uncertainty_quality
)

result["Discovery_Rank"] = (
    result["Discovery_Score"]
    .rank(
        ascending=False,
        method="first"
    )
    .astype(int)
)

# ============================================================
# 8. NOVELTY CATEGORIES
# ============================================================

result["Novelty_Category"] = pd.cut(
    result["Novelty_Percentile"],
    bins=[0, 25, 50, 75, 90, 100],
    labels=[
        "Low",
        "Moderate",
        "High",
        "Very High",
        "Extreme"
    ],
    include_lowest=True
)

# ============================================================
# 9. SAVE COMPLETE RESULT
# ============================================================

result = result.sort_values(
    "Discovery_Rank"
).reset_index(drop=True)

complete_file = os.path.join(
    OUT_DIR,
    "HEA_MPEA_5000_Novelty_Analysis.csv"
)

result.to_csv(
    complete_file,
    index=False
)

# ============================================================
# 10. TOP 50
# ============================================================

top50 = result.head(50).copy()

top50_file = os.path.join(
    OUT_DIR,
    "HEA_MPEA_Top50_Discovery_Novelty.csv"
)

top50.to_csv(
    top50_file,
    index=False
)

# ============================================================
# 11. TOP 20
# ============================================================

top20 = result.head(20).copy()

top20_file = os.path.join(
    OUT_DIR,
    "HEA_MPEA_Top20_Discovery_Novelty.csv"
)

top20.to_csv(
    top20_file,
    index=False
)

# ============================================================
# 12. HIGH PERFORMANCE + HIGH NOVELTY + LOW UNCERTAINTY
# ============================================================

performance_threshold = (
    performance.quantile(0.75)
)

novelty_threshold = (
    result["Novelty_Percentile"].quantile(0.75)
)

uncertainty_threshold = (
    uncertainty.median()
)

discovery_shortlist = result[
    (performance >= performance_threshold)
    &
    (result["Novelty_Percentile"] >= novelty_threshold)
    &
    (uncertainty <= uncertainty_threshold)
].copy()

discovery_shortlist = discovery_shortlist.sort_values(
    "Discovery_Score",
    ascending=False
)

shortlist_file = os.path.join(
    OUT_DIR,
    "HEA_MPEA_HighPerformance_HighNovelty_LowUncertainty.csv"
)

discovery_shortlist.to_csv(
    shortlist_file,
    index=False
)

# ============================================================
# 13. PLOT: NOVELTY VS PERFORMANCE
# ============================================================

plt.figure(figsize=(9, 7))

plt.scatter(
    result["Novelty_Score"],
    result["Performance_Score"],
    s=12,
    alpha=0.55
)

plt.xlabel("Composition Novelty Score")
plt.ylabel("Performance Score")
plt.title(
    "HEA/MPEA Candidate Novelty vs Predicted Performance"
)

plt.tight_layout()

novelty_plot = os.path.join(
    OUT_DIR,
    "Novelty_vs_Performance.png"
)

plt.savefig(
    novelty_plot,
    dpi=200
)

plt.close()

# ============================================================
# 14. PLOT: DISCOVERY SCORE DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 6))

plt.hist(
    result["Discovery_Score"],
    bins=40
)

plt.xlabel("Discovery Score")
plt.ylabel("Number of Candidates")
plt.title(
    "Distribution of AI-Assisted Materials Discovery Score"
)

plt.tight_layout()

score_plot = os.path.join(
    OUT_DIR,
    "Discovery_Score_Distribution.png"
)

plt.savefig(
    score_plot,
    dpi=200
)

plt.close()

# ============================================================
# 15. SUMMARY STATISTICS
# ============================================================

summary = pd.DataFrame({
    "Metric": [
        "Experimental records",
        "Valid experimental compositions",
        "Generated candidates",
        "Minimum novelty distance",
        "Median novelty distance",
        "Maximum novelty distance",
        "Median novelty percentile",
        "High-performance/high-novelty/low-uncertainty candidates",
        "Top candidate"
    ],
    "Value": [
        len(exp),
        len(X_exp_valid),
        len(result),
        result["Nearest_Experimental_Distance"].min(),
        result["Nearest_Experimental_Distance"].median(),
        result["Nearest_Experimental_Distance"].max(),
        result["Novelty_Percentile"].median(),
        len(discovery_shortlist),
        result.iloc[0]["candidate_id"]
    ]
})

summary_file = os.path.join(
    OUT_DIR,
    "Novelty_Analysis_Summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)

# ============================================================
# 16. PRINT RESULTS
# ============================================================

print("\n" + "=" * 75)
print("NOVELTY ANALYSIS COMPLETE")
print("=" * 75)

print(
    f"Experimental records: {len(exp)}"
)

print(
    f"Valid experimental compositions: {len(X_exp_valid)}"
)

print(
    f"Candidates analyzed: {len(result)}"
)

print(
    "\nNovelty distance:"
)

print(
    f"  Minimum : {result['Nearest_Experimental_Distance'].min():.4f}"
)

print(
    f"  Median  : {result['Nearest_Experimental_Distance'].median():.4f}"
)

print(
    f"  Maximum : {result['Nearest_Experimental_Distance'].max():.4f}"
)

print(
    "\nDiscovery shortlist:"
)

print(
    "  High performance + high novelty + low uncertainty:",
    len(discovery_shortlist)
)

print(
    "\nTOP 20 DISCOVERY CANDIDATES"
)

display_cols = [
    "Discovery_Rank",
    "candidate_id",
    "composition",
    "Predicted_YS_MPa",
    "Predicted_UTS_MPa",
    "Predicted_Elongation_pct",
    "Predicted_Hardness_HV",
    "YS_Uncertainty_STD_MPa",
    "Novelty_Percentile",
    "Performance_Score",
    "Discovery_Score"
]

print(
    result[display_cols]
    .head(20)
    .to_string(index=False)
)

print("\nFiles created:")

for path in [
    complete_file,
    top50_file,
    top20_file,
    shortlist_file,
    novelty_plot,
    score_plot,
    summary_file
]:
    print(path)

print("=" * 75)
