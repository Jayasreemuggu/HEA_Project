import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("=" * 70)
print("XGBOOST - YIELD STRENGTH EVALUATION")
print("=" * 70)

# =========================================================
# 1. LOAD PREDICTIONS
# =========================================================

df = pd.read_csv("YS_XGBoost_Predictions.csv")

df["Actual_YS_MPa"] = pd.to_numeric(
    df["Actual_YS_MPa"],
    errors="coerce"
)

df["Predicted_YS_MPa"] = pd.to_numeric(
    df["Predicted_YS_MPa"],
    errors="coerce"
)

df = df.dropna(
    subset=["Actual_YS_MPa", "Predicted_YS_MPa"]
).reset_index(drop=True)

actual = df["Actual_YS_MPa"]
predicted = df["Predicted_YS_MPa"]

print("\nLoaded predictions:", df.shape)

# =========================================================
# 2. METRICS
# =========================================================

mae = mean_absolute_error(actual, predicted)

rmse = np.sqrt(
    mean_squared_error(actual, predicted)
)

r2 = r2_score(actual, predicted)

print("\n" + "=" * 70)
print("XGBOOST METRICS")
print("=" * 70)

print(f"MAE  : {mae:.2f} MPa")
print(f"RMSE : {rmse:.2f} MPa")
print(f"R²   : {r2:.4f}")

# =========================================================
# 3. ERROR STATISTICS
# =========================================================

error = actual - predicted
absolute_error = np.abs(error)

print("\n" + "=" * 70)
print("ERROR STATISTICS")
print("=" * 70)

print(error.describe())

print(f"\nMean Absolute Error : {absolute_error.mean():.2f} MPa")
print(f"Maximum Absolute Error : {absolute_error.max():.2f} MPa")
print(f"Mean Error (Actual - Predicted) : {error.mean():.2f} MPa")

# =========================================================
# 4. ACTUAL VS PREDICTED
# =========================================================

plt.figure(figsize=(8, 7))

plt.scatter(
    actual,
    predicted,
    alpha=0.7
)

min_value = min(actual.min(), predicted.min())
max_value = max(actual.max(), predicted.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Yield Strength (MPa)")
plt.ylabel("Predicted Yield Strength (MPa)")
plt.title("XGBoost - Actual vs Predicted Yield Strength")

plt.tight_layout()

plt.savefig(
    "YS_XGBoost_Actual_vs_Predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =========================================================
# 5. ERROR DISTRIBUTION
# =========================================================

plt.figure(figsize=(8, 6))

plt.hist(
    error,
    bins=30
)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel("Prediction Error (Actual - Predicted) [MPa]")
plt.ylabel("Number of Records")
plt.title("XGBoost - Yield Strength Error Distribution")

plt.tight_layout()

plt.savefig(
    "YS_XGBoost_Error_Distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =========================================================
# 6. ERROR BY YIELD STRENGTH RANGE
# =========================================================

def classify_ys(value):

    if value < 500:
        return "<500"

    elif value < 1000:
        return "500-1000"

    elif value < 1500:
        return "1000-1500"

    elif value < 2000:
        return "1500-2000"

    else:
        return ">2000"


df["YS_Range"] = actual.apply(classify_ys)
df["Error_MPa"] = error
df["Absolute_Error_MPa"] = absolute_error

range_order = [
    "<500",
    "500-1000",
    "1000-1500",
    "1500-2000",
    ">2000"
]

range_results = []

for ys_range in range_order:

    subset = df[df["YS_Range"] == ys_range]

    if len(subset) == 0:
        continue

    range_mae = mean_absolute_error(
        subset["Actual_YS_MPa"],
        subset["Predicted_YS_MPa"]
    )

    range_rmse = np.sqrt(
        mean_squared_error(
            subset["Actual_YS_MPa"],
            subset["Predicted_YS_MPa"]
        )
    )

    range_results.append({
        "YS_Range": ys_range,
        "Records": len(subset),
        "Mean_Actual_YS": subset["Actual_YS_MPa"].mean(),
        "Mean_Predicted_YS": subset["Predicted_YS_MPa"].mean(),
        "MAE": range_mae,
        "RMSE": range_rmse
    })

range_df = pd.DataFrame(range_results)

print("\n" + "=" * 70)
print("ERROR BY YIELD STRENGTH RANGE")
print("=" * 70)

print(
    range_df.to_string(index=False)
)

range_df.to_csv(
    "YS_XGBoost_Error_By_Range.csv",
    index=False
)

# =========================================================
# 7. WORST 20 PREDICTIONS
# =========================================================

worst_20 = df.sort_values(
    by="Absolute_Error_MPa",
    ascending=False
).head(20)

print("\n" + "=" * 70)
print("TOP 20 WORST PREDICTIONS")
print("=" * 70)

print(
    worst_20[
        [
            "Actual_YS_MPa",
            "Predicted_YS_MPa",
            "Error_MPa",
            "Absolute_Error_MPa"
        ]
    ].to_string(index=False)
)

worst_20[
    [
        "Actual_YS_MPa",
        "Predicted_YS_MPa",
        "Error_MPa",
        "Absolute_Error_MPa"
    ]
].to_csv(
    "YS_XGBoost_Worst_20_Predictions.csv",
    index=False
)

# =========================================================
# 8. COMPLETE
# =========================================================

print("\n" + "=" * 70)
print("XGBOOST EVALUATION COMPLETE")
print("=" * 70)

print("\nGenerated:")
print("1. YS_XGBoost_Actual_vs_Predicted.png")
print("2. YS_XGBoost_Error_Distribution.png")
print("3. YS_XGBoost_Error_By_Range.csv")
print("4. YS_XGBoost_Worst_20_Predictions.csv")