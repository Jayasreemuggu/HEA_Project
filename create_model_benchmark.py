import os
import pandas as pd
import matplotlib.pyplot as plt

OUT = "evaluation/results"
os.makedirs(OUT, exist_ok=True)

# Final validated results
data = [
    ["YS", "Random Forest", 354.81, 474.60, 0.2095],
    ["YS", "Random Forest + Conditions", 251.74, 330.83, 0.6159],
    ["YS", "Random Forest + Physical", 240.73, 321.68, 0.6369],
    ["YS", "Extra Trees", 207.18, 301.10, 0.6818],
    ["YS", "XGBoost", 198.56, 286.06, 0.7128],
    ["YS", "GNN-GCN", 257.79, 347.32, 0.5767],
    ["YS", "GNN-GAT", 435.87, 533.88, -0.0002],

    ["UTS", "XGBoost", 289.48, 423.54, 0.7260],
    ["Elongation", "XGBoost", 12.73, 17.49, 0.3445],
    ["Hardness", "Extra Trees", 78.45, 108.20, 0.6930],
]

df = pd.DataFrame(
    data,
    columns=["Property", "Model", "MAE", "RMSE", "R2"]
)

df.to_csv(
    f"{OUT}/Final_Model_Benchmark.csv",
    index=False
)

# ------------------------------------------------------------
# YS comparison
# ------------------------------------------------------------
ys = df[df["Property"] == "YS"].copy()

plt.figure(figsize=(10, 6))
plt.bar(ys["Model"], ys["R2"])
plt.ylabel("R²")
plt.xlabel("Model")
plt.title("YS Prediction: Model Comparison")
plt.xticks(rotation=25, ha="right")
plt.axhline(0, linewidth=1)
plt.tight_layout()
plt.savefig(
    f"{OUT}/YS_Model_R2_Comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ------------------------------------------------------------
# YS MAE comparison
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.bar(ys["Model"], ys["MAE"])
plt.ylabel("MAE (MPa)")
plt.xlabel("Model")
plt.title("YS Prediction: MAE Comparison")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(
    f"{OUT}/YS_Model_MAE_Comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ------------------------------------------------------------
# Final property model comparison
# ------------------------------------------------------------
final_models = df[
    df["Model"].isin(["XGBoost", "Extra Trees"])
].copy()

plt.figure(figsize=(9, 6))
plt.bar(
    final_models["Property"],
    final_models["R2"]
)
plt.ylabel("R²")
plt.xlabel("Property")
plt.title("Final Model Performance Across Properties")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(
    f"{OUT}/Final_Property_R2_Comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("\n" + "=" * 70)
print("FINAL MODEL BENCHMARK")
print("=" * 70)
print(df.to_string(index=False))

print("\nBest YS model:")
best_ys = ys.loc[ys["R2"].idxmax()]
print(best_ys.to_string())

print("\nFiles created:")
print(f"{OUT}/Final_Model_Benchmark.csv")
print(f"{OUT}/YS_Model_R2_Comparison.png")
print(f"{OUT}/YS_Model_MAE_Comparison.png")
print(f"{OUT}/Final_Property_R2_Comparison.png")
