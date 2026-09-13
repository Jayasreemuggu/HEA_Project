import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


print("=" * 70)
print("UTS MODEL 3 - ERROR ANALYSIS")
print("=" * 70)


# =========================================================
# 1. LOAD PREDICTIONS
# =========================================================

predictions = pd.read_csv(
    "UTS_Model3_Predictions.csv"
)

print("\nPredictions shape:", predictions.shape)


# =========================================================
# 2. EXTRACT VALUES
# =========================================================

y_actual = predictions["Actual_UTS_MPa"]
y_pred = predictions["Predicted_UTS_MPa"]

error = y_actual - y_pred
absolute_error = np.abs(error)


# =========================================================
# 3. OVERALL METRICS
# =========================================================

mae = mean_absolute_error(
    y_actual,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_actual,
        y_pred
    )
)

r2 = r2_score(
    y_actual,
    y_pred
)

print("\n" + "=" * 70)
print("OVERALL MODEL PERFORMANCE")
print("=" * 70)

print(f"MAE  : {mae:.2f} MPa")
print(f"RMSE : {rmse:.2f} MPa")
print(f"R²   : {r2:.4f}")


# =========================================================
# 4. ERROR STATISTICS
# =========================================================

print("\n" + "=" * 70)
print("ERROR STATISTICS")
print("=" * 70)

print(f"Mean Error       : {error.mean():.2f} MPa")
print(f"Error Std        : {error.std():.2f} MPa")
print(f"Minimum Error    : {error.min():.2f} MPa")
print(f"Maximum Error    : {error.max():.2f} MPa")
print(f"Maximum Abs Error: {absolute_error.max():.2f} MPa")


# =========================================================
# 5. ACTUAL VS PREDICTED PLOT
# =========================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_actual,
    y_pred,
    alpha=0.7
)

# Ideal prediction line
min_value = min(
    y_actual.min(),
    y_pred.min()
)

max_value = max(
    y_actual.max(),
    y_pred.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual UTS (MPa)")
plt.ylabel("Predicted UTS (MPa)")
plt.title("UTS Model 3 - Actual vs Predicted")

plt.tight_layout()

plt.savefig(
    "UTS_Model3_Actual_vs_Predicted.png",
    dpi=300
)

plt.close()


# =========================================================
# 6. ERROR DISTRIBUTION
# =========================================================

plt.figure(figsize=(8, 6))

plt.hist(
    error,
    bins=30,
    edgecolor="black"
)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel("Prediction Error (Actual - Predicted) [MPa]")
plt.ylabel("Frequency")
plt.title("UTS Model 3 - Error Distribution")

plt.tight_layout()

plt.savefig(
    "UTS_Model3_Error_Distribution.png",
    dpi=300
)

plt.close()


# =========================================================
# 7. CREATE UTS RANGE
# =========================================================

def uts_range(value):

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


predictions["UTS_Range"] = (
    predictions["Actual_UTS_MPa"]
    .apply(uts_range)
)


# =========================================================
# 8. ERROR BY UTS RANGE
# =========================================================

range_results = []

range_order = [
    "<500",
    "500-1000",
    "1000-1500",
    "1500-2000",
    ">2000"
]

for category in range_order:

    subset = predictions[
        predictions["UTS_Range"] == category
    ]

    if len(subset) == 0:
        continue

    actual = subset["Actual_UTS_MPa"]
    predicted = subset["Predicted_UTS_MPa"]

    range_mae = mean_absolute_error(
        actual,
        predicted
    )

    range_rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    range_results.append({
        "UTS_Range": category,
        "Records": len(subset),
        "Mean_Actual_UTS": actual.mean(),
        "Mean_Predicted_UTS": predicted.mean(),
        "MAE": range_mae,
        "RMSE": range_rmse
    })


range_df = pd.DataFrame(
    range_results
)


print("\n" + "=" * 70)
print("ERROR BY UTS RANGE")
print("=" * 70)

print(
    range_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)


# =========================================================
# 9. SAVE RANGE ANALYSIS
# =========================================================

range_df.to_csv(
    "UTS_Model3_Error_By_Range.csv",
    index=False
)


# =========================================================
# 10. WORST PREDICTIONS
# =========================================================

worst = predictions.sort_values(
    "Absolute_Error_MPa",
    ascending=False
).head(20)


print("\n" + "=" * 70)
print("TOP 20 WORST PREDICTIONS")
print("=" * 70)

print(
    worst[
        [
            "Actual_UTS_MPa",
            "Predicted_UTS_MPa",
            "Error_MPa",
            "Absolute_Error_MPa"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)


# =========================================================
# 11. SAVE WORST PREDICTIONS
# =========================================================

worst[
    [
        "Actual_UTS_MPa",
        "Predicted_UTS_MPa",
        "Error_MPa",
        "Absolute_Error_MPa"
    ]
].to_csv(
    "UTS_Model3_Worst_Predictions.csv",
    index=False
)


# =========================================================
# 12. BIAS
# =========================================================

mean_error = error.mean()

print("\n" + "=" * 70)
print("BIAS INTERPRETATION")
print("=" * 70)

if mean_error > 0:
    print(
        f"Mean error = {mean_error:.2f} MPa."
    )
    print(
        "The model slightly UNDERPREDICTS UTS overall."
    )

elif mean_error < 0:
    print(
        f"Mean error = {mean_error:.2f} MPa."
    )
    print(
        "The model slightly OVERPREDICTS UTS overall."
    )

else:
    print("Mean error is approximately zero.")


# =========================================================
# 13. SAVE COMPLETE EVALUATION
# =========================================================

evaluation_summary = pd.DataFrame({
    "Metric": [
        "MAE_MPa",
        "RMSE_MPa",
        "R2",
        "Mean_Error_MPa",
        "Error_Std_MPa",
        "Min_Error_MPa",
        "Max_Error_MPa",
        "Max_Absolute_Error_MPa"
    ],
    "Value": [
        mae,
        rmse,
        r2,
        error.mean(),
        error.std(),
        error.min(),
        error.max(),
        absolute_error.max()
    ]
})

evaluation_summary.to_csv(
    "UTS_Model3_Evaluation_Summary.csv",
    index=False
)


# =========================================================
# 14. FINAL
# =========================================================

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print("1. UTS_Model3_Actual_vs_Predicted.png")
print("2. UTS_Model3_Error_Distribution.png")
print("3. UTS_Model3_Error_By_Range.csv")
print("4. UTS_Model3_Worst_Predictions.csv")
print("5. UTS_Model3_Evaluation_Summary.csv")

print("\n" + "=" * 70)
print("UTS MODEL 3 ERROR ANALYSIS COMPLETE")
print("=" * 70)