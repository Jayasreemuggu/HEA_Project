import os
import sys
from pathlib import Path
import pandas as pd

BASE = Path("/mnt/c/Users/jayam/Downloads/IITH_ML_project")

print("=" * 90)
print("AI-ASSISTED HEA/MPEA MATERIALS DISCOVERY")
print("FINAL REPRODUCIBLE PIPELINE CHECK")
print("=" * 90)

# ============================================================
# PROJECT COMPONENTS
# ============================================================

components = {
    "Dataset":
        BASE / "HEA_MPEA_ML_Ready_Deduplicated.csv",

    "YS Model":
        BASE / "YS_XGBoost_Model.pkl",

    "UTS Model":
        BASE / "UTS_XGBoost_Model.pkl",

    "Elongation Model":
        BASE / "Elongation_XGBoost_Model.pkl",

    "Hardness Model":
        BASE / "Hardness_ExtraTrees_Model.pkl",

    "GNN Benchmark":
        BASE / "GNN_YieldStrength_EXACT_Metrics.csv",

    "Benchmark":
        BASE / "evaluation/results/Final_Model_Benchmark.csv",

    "Candidate Predictions":
        BASE / "predictions/results/HEA_MPEA_5000_Final_Predictions.csv",

    "Integrated Ranking":
        BASE / "predictions/results/HEA_MPEA_FINAL_INTEGRATED_RANKING.csv",

    "Top 100":
        BASE / "predictions/results/HEA_MPEA_FINAL_TOP100.csv",

    "Pareto Candidates":
        BASE / "predictions/results/HEA_MPEA_FINAL_PARETO.csv",

    "Low Uncertainty High Performance":
        BASE / "predictions/results/HEA_MPEA_FINAL_LOW_UNCERTAINTY_HIGH_PERFORMANCE.csv",

    "35D Novelty":
        BASE / "novelty/results/HEA_MPEA_5000_Novelty_35Element.csv",

    "Novelty Discovery":
        BASE / "novelty/results/HEA_MPEA_35D_HighPerformance_HighNovelty_LowUncertainty.csv",

    "Active Learning":
        BASE / "active_learning/results/YS_Active_Learning_Controlled_Final.csv",

    "Active Learning Summary":
        BASE / "active_learning/results/YS_Active_Learning_Controlled_Summary.csv",

    "README":
        BASE / "README.md",
}

# ============================================================
# CHECK FILES
# ============================================================

print("\n[1/7] Checking project components...")

missing = []

for name, path in components.items():

    if path.exists():
        print(f"[OK]      {name:<38} {path}")
    else:
        print(f"[MISSING] {name:<38} {path}")
        missing.append(name)

if missing:
    print("\nERROR: Missing required components:")
    for x in missing:
        print(" -", x)

    sys.exit(1)

# ============================================================
# DATASET VALIDATION
# ============================================================

print("\n[2/7] Validating master dataset...")

dataset = pd.read_csv(
    components["Dataset"]
)

print(
    f"Records           : {len(dataset)}"
)

print(
    f"Columns           : {len(dataset.columns)}"
)

print(
    f"YS records        : "
    f"{dataset['YS_Tensile_MPa'].notna().sum()}"
)

print(
    f"UTS records       : "
    f"{dataset['UTS_Tensile_MPa'].notna().sum()}"
)

print(
    f"Elongation records: "
    f"{dataset['Elongation_Tensile_pct'].notna().sum()}"
)

# ============================================================
# BENCHMARK
# ============================================================

print("\n[3/7] Validating model benchmark...")

benchmark = pd.read_csv(
    components["Benchmark"]
)

print(
    benchmark.to_string(
        index=False
    )
)

# ============================================================
# CANDIDATE SCREENING
# ============================================================

print("\n[4/7] Validating candidate discovery...")

ranking = pd.read_csv(
    components["Integrated Ranking"]
)

pareto = pd.read_csv(
    components["Pareto Candidates"]
)

low_unc = pd.read_csv(
    components[
        "Low Uncertainty High Performance"
    ]
)

novelty = pd.read_csv(
    components["35D Novelty"]
)

novelty_discovery = pd.read_csv(
    components["Novelty Discovery"]
)

print(
    f"Hypothetical candidates       : "
    f"{len(ranking)}"
)

print(
    f"Pareto candidates              : "
    f"{len(pareto)} "
    f"({100*len(pareto)/len(ranking):.2f}%)"
)

print(
    f"Low uncertainty/high performance: "
    f"{len(low_unc)}"
)

print(
    f"35D novelty candidates         : "
    f"{len(novelty)}"
)

print(
    f"Novel + performance + low risk : "
    f"{len(novelty_discovery)}"
)

# ============================================================
# ACTIVE LEARNING
# ============================================================

print("\n[5/7] Validating active learning...")

al = pd.read_csv(
    components["Active Learning"]
)

print(
    al.to_string(
        index=False
    )
)

unc = al[
    al["Strategy"] == "Uncertainty"
].iloc[0]

rnd = al[
    al["Strategy"] == "Random"
].iloc[0]

r2_gain = (
    unc["Mean_R2"]
    - rnd["Mean_R2"]
)

mae_gain = (
    rnd["Mean_MAE_MPa"]
    - unc["Mean_MAE_MPa"]
)

rmse_gain = (
    rnd["Mean_RMSE_MPa"]
    - unc["Mean_RMSE_MPa"]
)

print(
    f"\nUncertainty R² gain : "
    f"{r2_gain:+.4f}"
)

print(
    f"MAE improvement     : "
    f"{mae_gain:+.2f} MPa"
)

print(
    f"RMSE improvement    : "
    f"{rmse_gain:+.2f} MPa"
)

# ============================================================
# FINAL TOP CANDIDATES
# ============================================================

print("\n[6/7] Final candidate shortlist...")

top = ranking.head(10)

candidate_columns = [
    "candidate_id",
    "composition",
    "Predicted_YS_MPa",
    "Predicted_UTS_MPa",
    "Predicted_Elongation_pct",
    "Predicted_Hardness_HV",
    "YS_Uncertainty_STD_MPa",
    "Performance_Score",
    "Uncertainty_Aware_Score",
]

candidate_columns = [
    c for c in candidate_columns
    if c in top.columns
]

print(
    top[
        candidate_columns
    ].to_string(
        index=False
    )
)

# ============================================================
# FINAL FIGURES
# ============================================================

print("\n[7/7] Checking final figures...")

figure_dir = BASE / "final_figures"

expected_figures = [
    "01_Model_R2_Comparison.png",
    "02_Four_Property_R2.png",
    "03_YS_SHAP_Top15.png",
    "04_Pareto_Screening.png",
    "05_YS_Uncertainty_Distribution.png",
    "06_Top10_Property_Comparison.png",
]

for fig in expected_figures:

    path = figure_dir / fig

    if path.exists():
        print(
            f"[OK] {fig}"
        )
    else:
        print(
            f"[MISSING] {fig}"
        )

# ============================================================
# FINAL SUMMARY
# ============================================================

summary_file = (
    BASE /
    "FINAL_PROJECT_SUMMARY.txt"
)

summary = f"""
AI-ASSISTED HEA/MPEA MATERIALS DISCOVERY
========================================

DATA
----
Experimental records: {len(dataset)}
YS records: {dataset['YS_Tensile_MPa'].notna().sum()}
UTS records: {dataset['UTS_Tensile_MPa'].notna().sum()}
Elongation records: {dataset['Elongation_Tensile_pct'].notna().sum()}

MODELS
------
YS: XGBoost
UTS: XGBoost
Elongation: XGBoost
Hardness: Extra Trees
Deep-learning benchmark: GCN

DISCOVERY
---------
Hypothetical candidates: {len(ranking)}
Pareto candidates: {len(pareto)}
Low-uncertainty/high-performance candidates: {len(low_unc)}
35D novelty candidates: {len(novelty)}
Novelty + performance + low uncertainty candidates:
{len(novelty_discovery)}

ACTIVE LEARNING
---------------
Random final R2:
{rnd['Mean_R2']:.4f} +/- {rnd['Std_R2']:.4f}

Uncertainty final R2:
{unc['Mean_R2']:.4f} +/- {unc['Std_R2']:.4f}

R2 improvement:
{r2_gain:+.4f}

MAE improvement:
{mae_gain:+.2f} MPa

RMSE improvement:
{rmse_gain:+.2f} MPa

INTERPRETATION
--------------
The active-learning experiment shows a modest improvement
for uncertainty-guided acquisition at a matched composition
budget across five independent seeds.

Candidate properties are machine-learning predictions within
the explored composition space and require experimental
validation.

FINAL TOP CANDIDATE
-------------------
{top.iloc[0]['candidate_id']}
{top.iloc[0]['composition']}

Predicted YS:
{top.iloc[0]['Predicted_YS_MPa']:.2f} MPa

Predicted UTS:
{top.iloc[0]['Predicted_UTS_MPa']:.2f} MPa

Predicted elongation:
{top.iloc[0]['Predicted_Elongation_pct']:.2f} %

Predicted hardness:
{top.iloc[0]['Predicted_Hardness_HV']:.2f} HV

YS uncertainty:
{top.iloc[0]['YS_Uncertainty_STD_MPa']:.2f} MPa

Performance score:
{top.iloc[0]['Performance_Score']:.4f}

Uncertainty-aware score:
{top.iloc[0]['Uncertainty_Aware_Score']:.4f}
"""

summary_file.write_text(
    summary.strip() + "\n",
    encoding="utf-8"
)

print("\n" + "=" * 90)
print("FINAL PIPELINE VALIDATION COMPLETE")
print("=" * 90)

print(
    f"\nSummary saved to:\n{summary_file}"
)

print(
    "\nProject status: READY FOR FINAL DOCUMENTATION / GITHUB PACKAGING"
)

print("=" * 90)
