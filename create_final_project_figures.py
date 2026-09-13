import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE = "/mnt/c/Users/jayam/Downloads/IITH_ML_project"
OUT = os.path.join(BASE, "final_figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams["figure.dpi"] = 150

# ============================================================
# 1. MODEL R2 COMPARISON
# ============================================================

benchmark = pd.DataFrame({
    "Model": [
        "YS Random Forest",
        "YS Extra Trees",
        "YS XGBoost",
        "YS GNN-GCN",
        "YS GNN-GAT",
        "UTS XGBoost",
        "Elongation XGBoost",
        "Hardness Extra Trees"
    ],
    "R2": [
        0.6369,
        0.6818,
        0.7128,
        0.5767,
        -0.0002,
        0.7260,
        0.3445,
        0.6930
    ]
})

plt.figure(figsize=(10, 6))
plt.bar(benchmark["Model"], benchmark["R2"])
plt.ylabel("R²")
plt.xlabel("Model")
plt.title("Model Performance Comparison")
plt.xticks(rotation=45, ha="right")
plt.axhline(0, linewidth=0.8)
plt.tight_layout()
plt.savefig(
    os.path.join(OUT, "01_Model_R2_Comparison.png")
)
plt.close()

# ============================================================
# 2. FOUR PROPERTY PERFORMANCE
# ============================================================

properties = [
    "Yield Strength",
    "UTS",
    "Elongation",
    "Hardness"
]

r2 = [
    0.7128,
    0.7260,
    0.3445,
    0.6930
]

plt.figure(figsize=(8, 6))
plt.bar(properties, r2)
plt.ylabel("R²")
plt.title("Final Four-Property ML Model Performance")
plt.ylim(0, 0.8)

for i, value in enumerate(r2):
    plt.text(
        i,
        value + 0.015,
        f"{value:.4f}",
        ha="center"
    )

plt.tight_layout()
plt.savefig(
    os.path.join(OUT, "02_Four_Property_R2.png")
)
plt.close()

# ============================================================
# 3. YS SHAP
# ============================================================

shap_file = os.path.join(
    BASE,
    "explainability",
    "results",
    "YS_SHAP_Feature_Importance.csv"
)

shap_df = pd.read_csv(shap_file)

# Detect columns automatically
feature_col = None
value_col = None

for c in shap_df.columns:

    cl = c.lower()

    if feature_col is None and (
        "feature" in cl or
        "name" in cl
    ):
        feature_col = c

    if value_col is None and (
        "mean" in cl and "shap" in cl
    ):
        value_col = c

if feature_col is None:
    feature_col = shap_df.columns[0]

if value_col is None:
    numeric_cols = shap_df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if not numeric_cols:
        raise RuntimeError(
            "Could not identify SHAP importance column."
        )

    value_col = numeric_cols[-1]

shap_top = shap_df.sort_values(
    value_col,
    ascending=False
).head(15)

shap_top = shap_top.iloc[::-1]

plt.figure(figsize=(9, 7))
plt.barh(
    shap_top[feature_col].astype(str),
    shap_top[value_col]
)
plt.xlabel("Mean |SHAP Value|")
plt.ylabel("Feature")
plt.title("Top 15 Features Influencing Yield Strength")
plt.tight_layout()
plt.savefig(
    os.path.join(OUT, "03_YS_SHAP_Top15.png")
)
plt.close()

# ============================================================
# 4. PARETO SCREENING
# ============================================================

final_file = os.path.join(
    BASE,
    "predictions",
    "results",
    "HEA_MPEA_FINAL_INTEGRATED_RANKING.csv"
)

df = pd.read_csv(final_file)

pareto_count = int(
    df["Pareto_Optimal"].sum()
)

non_pareto_count = len(df) - pareto_count

plt.figure(figsize=(7, 6))
plt.bar(
    ["Pareto Optimal", "Non-Pareto"],
    [pareto_count, non_pareto_count]
)
plt.ylabel("Number of Candidates")
plt.title("Pareto Screening of 5,000 HEA/MPEA Candidates")

for i, value in enumerate(
    [pareto_count, non_pareto_count]
):
    plt.text(
        i,
        value + 50,
        str(value),
        ha="center"
    )

plt.tight_layout()
plt.savefig(
    os.path.join(OUT, "04_Pareto_Screening.png")
)
plt.close()

# ============================================================
# 5. YS UNCERTAINTY DISTRIBUTION
# ============================================================

unc = pd.to_numeric(
    df["YS_Uncertainty_STD_MPa"],
    errors="coerce"
)

plt.figure(figsize=(8, 6))
plt.hist(
    unc,
    bins=40
)
plt.xlabel("YS Ensemble Uncertainty (MPa)")
plt.ylabel("Number of Candidates")
plt.title("Bootstrap Ensemble Uncertainty Across 5,000 Candidates")

plt.axvline(
    unc.median(),
    linestyle="--",
    linewidth=1.5,
    label=f"Median = {unc.median():.2f} MPa"
)

plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(OUT, "05_YS_Uncertainty_Distribution.png")
)
plt.close()

# ============================================================
# 6. TOP 10 CANDIDATE PROPERTY COMPARISON
# ============================================================

top10 = df.head(10).copy()

labels = top10["candidate_id"].astype(str)

property_data = {
    "YS": top10["Predicted_YS_MPa"].values,
    "UTS": top10["Predicted_UTS_MPa"].values,
    "Elongation": top10["Predicted_Elongation_pct"].values,
    "Hardness": top10["Predicted_Hardness_HV"].values
}

# Separate normalized comparison so different units
# do not distort the plot.

normalized = pd.DataFrame(index=top10.index)

for name, values in property_data.items():

    values = np.asarray(values, dtype=float)

    normalized[name] = (
        (values - values.min()) /
        (values.max() - values.min())
        if values.max() > values.min()
        else 0
    )

x = np.arange(len(top10))
width = 0.2

plt.figure(figsize=(12, 6))

for i, (name, values) in enumerate(
    normalized.items()
):

    plt.bar(
        x + (i - 1.5) * width,
        values,
        width,
        label=name
    )

plt.xticks(
    x,
    labels,
    rotation=45,
    ha="right"
)

plt.ylabel("Normalized Predicted Property")
plt.xlabel("Candidate")
plt.title("Normalized Property Comparison of Top 10 Candidates")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(OUT, "06_Top10_Property_Comparison.png")
)

plt.close()

# ============================================================
# SAVE FIGURE SUMMARY
# ============================================================

summary = pd.DataFrame({
    "Figure": [
        "01_Model_R2_Comparison.png",
        "02_Four_Property_R2.png",
        "03_YS_SHAP_Top15.png",
        "04_Pareto_Screening.png",
        "05_YS_Uncertainty_Distribution.png",
        "06_Top10_Property_Comparison.png"
    ],
    "Purpose": [
        "Comparison of ML and GNN model R²",
        "Performance of final four property models",
        "Explainable AI feature importance for YS",
        "Multi-objective candidate screening",
        "Bootstrap model uncertainty",
        "Comparison of top candidate properties"
    ]
})

summary.to_csv(
    os.path.join(
        OUT,
        "Final_Figure_Index.csv"
    ),
    index=False
)

print("=" * 70)
print("FINAL PROJECT FIGURES CREATED")
print("=" * 70)

for filename in summary["Figure"]:
    print(
        os.path.join(
            OUT,
            filename
        )
    )

print(
    "\nPareto candidates:",
    pareto_count,
    "/",
    len(df)
)

print(
    "YS uncertainty median:",
    f"{unc.median():.2f} MPa"
)

print(
    "\nFigure index:",
    os.path.join(
        OUT,
        "Final_Figure_Index.csv"
    )
)
