import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

BASE = "/mnt/c/Users/jayam/Downloads/IITH_ML_project"

EXP_FILE = os.path.join(
    BASE,
    "HEA_MPEA_ML_Ready_Deduplicated.csv"
)

CAND_FILE = os.path.join(
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
# 1. FULL 35-ELEMENT SPACE
# ============================================================

ELEMENTS = [
    "Ag", "Al", "B", "C", "Ca", "Co", "Cr", "Cu",
    "Fe", "Ga", "Hf", "I", "Li", "Mg", "Mn", "Mo",
    "Nb", "Nd", "Ni", "O", "Pd", "Re", "Ru", "S",
    "Sc", "Si", "Sn", "T", "Ta", "Ti", "V", "W",
    "Y", "Zn", "Zr"
]

EXP_COLS = [
    f"{e}_at_pct"
    for e in ELEMENTS
]

print("=" * 80)
print("FULL 35-ELEMENT HEA/MPEA NOVELTY ANALYSIS")
print("=" * 80)

# ============================================================
# 2. LOAD DATA
# ============================================================

exp = pd.read_csv(EXP_FILE)
cand = pd.read_csv(CAND_FILE)

print("Experimental records :", len(exp))
print("Candidate records    :", len(cand))
print("Composition dimensions:", len(ELEMENTS))

# ============================================================
# 3. VERIFY EXPERIMENTAL COLUMNS
# ============================================================

missing = [
    c for c in EXP_COLS
    if c not in exp.columns
]

if missing:
    raise RuntimeError(
        "Missing experimental columns:\n"
        + str(missing)
    )

# ============================================================
# 4. BUILD EXPERIMENTAL MATRIX
# ============================================================

X_exp = (
    exp[EXP_COLS]
    .apply(pd.to_numeric, errors="coerce")
    .fillna(0.0)
    .values
)

# Candidate dataframe contains only the 12-element
# generation space. Missing elements in the full
# 35-element space are therefore zero.

X_cand = np.zeros(
    (len(cand), len(ELEMENTS)),
    dtype=float
)

for j, element in enumerate(ELEMENTS):

    if element in cand.columns:
        X_cand[:, j] = pd.to_numeric(
            cand[element],
            errors="coerce"
        ).fillna(0.0)

# ============================================================
# 5. VALID EXPERIMENTAL COMPOSITIONS
# ============================================================

exp_sum = X_exp.sum(axis=1)

valid_mask = (
    np.isfinite(X_exp).all(axis=1)
    &
    (exp_sum >= 95.0)
    &
    (exp_sum <= 105.0)
)

X_exp_valid = X_exp[valid_mask]

exp_valid = exp.loc[
    valid_mask
].copy()

print(
    "\nValid experimental compositions:",
    len(X_exp_valid)
)

# ============================================================
# 6. STANDARDIZE FULL COMPOSITION SPACE
# ============================================================

scaler = StandardScaler()

X_exp_scaled = scaler.fit_transform(
    X_exp_valid
)

X_cand_scaled = scaler.transform(
    X_cand
)

print(
    "Scaled experimental matrix:",
    X_exp_scaled.shape
)

print(
    "Scaled candidate matrix:",
    X_cand_scaled.shape
)

# ============================================================
# 7. NEAREST-NEIGHBOR SEARCH
# ============================================================

nn = NearestNeighbors(
    n_neighbors=1,
    metric="euclidean"
)

nn.fit(X_exp_scaled)

distances, indices = nn.kneighbors(
    X_cand_scaled
)

distances = distances[:, 0]
indices = indices[:, 0]

nearest = exp_valid.iloc[
    indices
].reset_index(drop=True)

# ============================================================
# 8. NOVELTY METRICS
# ============================================================

result = cand.copy()

result["Novelty_35D_Distance"] = distances

result["Nearest_Experimental_Record_ID"] = (
    nearest["Record_ID"].values
)

result["Nearest_Experimental_Alloy"] = (
    nearest["Alloy_Name"]
    .fillna("UNKNOWN")
    .values
)

result["Nearest_Experimental_Composition"] = (
    nearest["Composition_Canonical"]
    .fillna("UNKNOWN")
    .values
)

# Percentile:
# 100 = more novel than almost all candidates

result["Novelty_35D_Percentile"] = (
    result["Novelty_35D_Distance"]
    .rank(pct=True)
    * 100
)

# Min-max novelty score

dmin = result["Novelty_35D_Distance"].min()
dmax = result["Novelty_35D_Distance"].max()

if dmax > dmin:
    result["Novelty_35D_Score"] = (
        result["Novelty_35D_Distance"] - dmin
    ) / (dmax - dmin)
else:
    result["Novelty_35D_Score"] = 0.0

# ============================================================
# 9. DISCOVERY SCORE
# ============================================================

performance = (
    result["Performance_Score"]
    .astype(float)
)

uncertainty = (
    result["YS_Relative_Uncertainty"]
    .astype(float)
    .clip(lower=0)
)

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

umin = uncertainty.min()
umax = uncertainty.max()

if umax > umin:
    uncertainty_quality = 1 - (
        (uncertainty - umin)
        / (umax - umin)
    )
else:
    uncertainty_quality = pd.Series(
        1.0,
        index=result.index
    )

# Same conceptual weighting as previous analysis:
#
# 50% performance
# 25% novelty
# 25% uncertainty quality

result["Discovery_35D_Score"] = (
    0.50 * performance_norm
    +
    0.25 * result["Novelty_35D_Score"]
    +
    0.25 * uncertainty_quality
)

result["Discovery_35D_Rank"] = (
    result["Discovery_35D_Score"]
    .rank(
        ascending=False,
        method="first"
    )
    .astype(int)
)

# ============================================================
# 10. NOVELTY CATEGORY
# ============================================================

result["Novelty_35D_Category"] = pd.cut(
    result["Novelty_35D_Percentile"],
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
# 11. SORT
# ============================================================

result = result.sort_values(
    "Discovery_35D_Rank"
).reset_index(drop=True)

# ============================================================
# 12. SAVE COMPLETE DATASET
# ============================================================

complete_file = os.path.join(
    OUT_DIR,
    "HEA_MPEA_5000_Novelty_35Element.csv"
)

result.to_csv(
    complete_file,
    index=False
)

# ============================================================
# 13. TOP 50
# ============================================================

top50 = result.head(50)

top50_file = os.path.join(
    OUT_DIR,
    "HEA_MPEA_Top50_Discovery_35Element.csv"
)

top50.to_csv(
    top50_file,
    index=False
)

# ============================================================
# 14. TOP 20
# ============================================================

top20 = result.head(20)

top20_file = os.path.join(
    OUT_DIR,
    "HEA_MPEA_Top20_Discovery_35Element.csv"
)

top20.to_csv(
    top20_file,
    index=False
)

# ============================================================
# 15. HIGH PERFORMANCE + HIGH NOVELTY + LOW UNCERTAINTY
# ============================================================

performance_threshold = performance.quantile(0.75)
novelty_threshold = result[
    "Novelty_35D_Percentile"
].quantile(0.75)
uncertainty_threshold = uncertainty.median()

shortlist = result[
    (performance >= performance_threshold)
    &
    (
        result["Novelty_35D_Percentile"]
        >= novelty_threshold
    )
    &
    (uncertainty <= uncertainty_threshold)
].copy()

shortlist = shortlist.sort_values(
    "Discovery_35D_Score",
    ascending=False
)

shortlist_file = os.path.join(
    OUT_DIR,
    "HEA_MPEA_35D_HighPerformance_HighNovelty_LowUncertainty.csv"
)

shortlist.to_csv(
    shortlist_file,
    index=False
)

# ============================================================
# 16. COMPARE 12D VS 35D NOVELTY
# ============================================================

old_file = os.path.join(
    OUT_DIR,
    "HEA_MPEA_5000_Novelty_Analysis.csv"
)

if os.path.exists(old_file):

    old = pd.read_csv(old_file)

    compare_cols = [
        "candidate_id",
        "Novelty_Percentile",
        "Novelty_Score",
        "Discovery_Rank",
        "Discovery_Score"
    ]

    old_small = old[
        [
            c for c in compare_cols
            if c in old.columns
        ]
    ].copy()

    comparison = result[
        [
            "candidate_id",
            "Novelty_35D_Percentile",
            "Novelty_35D_Score",
            "Discovery_35D_Rank",
            "Discovery_35D_Score"
        ]
    ].copy()

    comparison_file = os.path.join(
        OUT_DIR,
        "Novelty_12D_vs_35D_Comparison.csv"
    )

    comparison.to_csv(
        comparison_file,
        index=False
    )

# ============================================================
# 17. NOVELTY VS PERFORMANCE PLOT
# ============================================================

plt.figure(figsize=(9, 7))

plt.scatter(
    result["Novelty_35D_Score"],
    result["Performance_Score"],
    s=12,
    alpha=0.55
)

plt.xlabel(
    "35-Dimensional Composition Novelty Score"
)

plt.ylabel(
    "Performance Score"
)

plt.title(
    "HEA/MPEA Composition Novelty vs Predicted Performance"
)

plt.tight_layout()

plot1 = os.path.join(
    OUT_DIR,
    "Novelty_35D_vs_Performance.png"
)

plt.savefig(
    plot1,
    dpi=200
)

plt.close()

# ============================================================
# 18. NOVELTY DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 6))

plt.hist(
    result["Novelty_35D_Distance"],
    bins=40
)

plt.xlabel(
    "Nearest Experimental Alloy Distance"
)

plt.ylabel(
    "Number of Candidates"
)

plt.title(
    "35-Dimensional Composition Novelty Distribution"
)

plt.tight_layout()

plot2 = os.path.join(
    OUT_DIR,
    "Novelty_35D_Distribution.png"
)

plt.savefig(
    plot2,
    dpi=200
)

plt.close()

# ============================================================
# 19. SUMMARY
# ============================================================

summary = pd.DataFrame({
    "Metric": [
        "Experimental records",
        "Valid experimental compositions",
        "Composition dimensions",
        "Generated candidates",
        "Minimum 35D novelty distance",
        "Median 35D novelty distance",
        "Maximum 35D novelty distance",
        "Median novelty percentile",
        "High-performance/high-novelty/low-uncertainty candidates",
        "Top discovery candidate"
    ],
    "Value": [
        len(exp),
        len(X_exp_valid),
        len(ELEMENTS),
        len(result),
        result["Novelty_35D_Distance"].min(),
        result["Novelty_35D_Distance"].median(),
        result["Novelty_35D_Distance"].max(),
        result["Novelty_35D_Percentile"].median(),
        len(shortlist),
        result.iloc[0]["candidate_id"]
    ]
})

summary_file = os.path.join(
    OUT_DIR,
    "Novelty_35D_Summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)

# ============================================================
# 20. PRINT RESULTS
# ============================================================

print("\n" + "=" * 80)
print("35-ELEMENT NOVELTY ANALYSIS COMPLETE")
print("=" * 80)

print(
    f"Experimental records       : {len(exp)}"
)

print(
    f"Valid experimental records : {len(X_exp_valid)}"
)

print(
    f"Composition dimensions     : {len(ELEMENTS)}"
)

print(
    f"Candidates analyzed        : {len(result)}"
)

print("\n35D Novelty distance:")

print(
    f"  Minimum : "
    f"{result['Novelty_35D_Distance'].min():.4f}"
)

print(
    f"  Median  : "
    f"{result['Novelty_35D_Distance'].median():.4f}"
)

print(
    f"  Maximum : "
    f"{result['Novelty_35D_Distance'].max():.4f}"
)

print(
    "\nHigh-performance + high-novelty + low-uncertainty:"
)

print(
    f"  {len(shortlist)} candidates"
)

print("\nTOP 20 DISCOVERY CANDIDATES")

display_cols = [
    "Discovery_35D_Rank",
    "candidate_id",
    "composition",
    "Predicted_YS_MPa",
    "Predicted_UTS_MPa",
    "Predicted_Elongation_pct",
    "Predicted_Hardness_HV",
    "YS_Uncertainty_STD_MPa",
    "Novelty_35D_Percentile",
    "Performance_Score",
    "Discovery_35D_Score"
]

print(
    result[
        display_cols
    ].head(20).to_string(index=False)
)

print("\nFiles created:")

for f in [
    complete_file,
    top50_file,
    top20_file,
    shortlist_file,
    plot1,
    plot2,
    summary_file
]:
    print(f)

print("=" * 80)
