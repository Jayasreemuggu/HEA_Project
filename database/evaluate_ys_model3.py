import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# DAY 3 - MODEL 3 EVALUATION
# ============================================================

print("=" * 70)
print("DAY 3 - YIELD STRENGTH MODEL 3 EVALUATION")
print("=" * 70)


# ============================================================
# 1. LOAD PREDICTIONS
# ============================================================

file_path = "YS_Model3_Predictions.csv"

df = pd.read_csv(file_path)

print("\nLoaded predictions:")
print(df.shape)


# ============================================================
# 2. ACTUAL / PREDICTED
# ============================================================

actual = df["Actual_YS_MPa"]

predicted = df["Predicted_YS_MPa"]

error = df["Error_MPa"]

absolute_error = np.abs(error)


# ============================================================
# 3. METRICS
# ============================================================

mae = mean_absolute_error(
    actual,
    predicted
)

rmse = np.sqrt(
    mean_squared_error(
        actual,
        predicted
    )
)

r2 = r2_score(
    actual,
    predicted
)


print("\n" + "=" * 70)
print("MODEL 3 METRICS")
print("=" * 70)

print(f"MAE  : {mae:.2f} MPa")
print(f"RMSE : {rmse:.2f} MPa")
print(f"R²   : {r2:.4f}")


# ============================================================
# 4. ERROR STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("ERROR STATISTICS")
print("=" * 70)

print(
    pd.Series(error).describe()
)

print(
    f"\nMean Absolute Error : "
    f"{absolute_error.mean():.2f} MPa"
)

print(
    f"Maximum Absolute Error : "
    f"{absolute_error.max():.2f} MPa"
)

print(
    f"Mean Error (Actual - Predicted) : "
    f"{error.mean():.2f} MPa"
)


# ============================================================
# 5. ACTUAL VS PREDICTED
# ============================================================

plt.figure(
    figsize=(8, 7)
)

plt.scatter(
    actual,
    predicted,
    alpha=0.65
)

# Perfect prediction line
min_value = min(
    actual.min(),
    predicted.min()
)

max_value = max(
    actual.max(),
    predicted.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel(
    "Actual Yield Strength (MPa)"
)

plt.ylabel(
    "Predicted Yield Strength (MPa)"
)

plt.title(
    "Model 3: Actual vs Predicted Yield Strength"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "YS_Model3_Actual_vs_Predicted.png",
    dpi=300
)

plt.show()


# ============================================================
# 6. ERROR DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.hist(
    error,
    bins=40,
    alpha=0.75
)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Prediction Error (Actual - Predicted) [MPa]"
)

plt.ylabel(
    "Number of Records"
)

plt.title(
    "Model 3: Yield Strength Prediction Error Distribution"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "YS_Model3_Error_Distribution.png",
    dpi=300
)

plt.show()


# ============================================================
# 7. ABSOLUTE ERROR DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.hist(
    absolute_error,
    bins=40,
    alpha=0.75
)

plt.xlabel(
    "Absolute Prediction Error (MPa)"
)

plt.ylabel(
    "Number of Records"
)

plt.title(
    "Model 3: Absolute Prediction Error"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "YS_Model3_Absolute_Error.png",
    dpi=300
)

plt.show()


# ============================================================
# 8. ERROR BY YIELD STRENGTH RANGE
# ============================================================

df["YS_Range"] = pd.cut(
    actual,
    bins=[
        -np.inf,
        500,
        1000,
        1500,
        2000,
        np.inf
    ],
    labels=[
        "<500",
        "500-1000",
        "1000-1500",
        "1500-2000",
        ">2000"
    ]
)


range_results = []


for group_name, group in df.groupby(
    "YS_Range",
    observed=False
):

    if len(group) == 0:
        continue

    group_actual = group[
        "Actual_YS_MPa"
    ]

    group_predicted = group[
        "Predicted_YS_MPa"
    ]

    group_mae = mean_absolute_error(
        group_actual,
        group_predicted
    )

    group_rmse = np.sqrt(
        mean_squared_error(
            group_actual,
            group_predicted
        )
    )

    range_results.append({

        "YS_Range": str(group_name),

        "Records": len(group),

        "Mean_Actual_YS": group_actual.mean(),

        "Mean_Predicted_YS": group_predicted.mean(),

        "MAE": group_mae,

        "RMSE": group_rmse
    })


range_df = pd.DataFrame(
    range_results
)


print("\n" + "=" * 70)
print("ERROR BY YIELD STRENGTH RANGE")
print("=" * 70)

print(
    range_df.to_string(
        index=False
    )
)


# ============================================================
# 9. SAVE RANGE ANALYSIS
# ============================================================

range_df.to_csv(
    "YS_Model3_Error_By_Range.csv",
    index=False
)


print(
    "\nSaved:"
)

print(
    "YS_Model3_Error_By_Range.csv"
)


# ============================================================
# 10. IDENTIFY WORST PREDICTIONS
# ============================================================

worst = df.sort_values(
    "Absolute_Error"
    if "Absolute_Error" in df.columns
    else "Error_MPa",
    key=lambda x: np.abs(x),
    ascending=False
).head(20)


# Add absolute error explicitly
df["Absolute_Error_MPa"] = (
    np.abs(
        df["Error_MPa"]
    )
)


worst = df.sort_values(
    "Absolute_Error_MPa",
    ascending=False
).head(20)


print("\n" + "=" * 70)
print("TOP 20 WORST PREDICTIONS")
print("=" * 70)

print(
    worst[
        [
            "Actual_YS_MPa",
            "Predicted_YS_MPa",
            "Error_MPa",
            "Absolute_Error_MPa"
        ]
    ].to_string(index=False)
)


# ============================================================
# 11. SAVE WORST CASES
# ============================================================

worst.to_csv(
    "YS_Model3_Worst_20_Predictions.csv",
    index=False
)


print(
    "\nSaved:"
)

print(
    "YS_Model3_Worst_20_Predictions.csv"
)


# ============================================================
# 12. FINAL
# ============================================================

print("\n" + "=" * 70)
print("MODEL 3 EVALUATION COMPLETE")
print("=" * 70)

print("\nGenerated files:")

print(
    "1. YS_Model3_Actual_vs_Predicted.png"
)

print(
    "2. YS_Model3_Error_Distribution.png"
)

print(
    "3. YS_Model3_Absolute_Error.png"
)

print(
    "4. YS_Model3_Error_By_Range.csv"
)

print(
    "5. YS_Model3_Worst_20_Predictions.csv"
)

print("=" * 70)