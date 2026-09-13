import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# EXTRA TREES YS EVALUATION
# ============================================================

print("=" * 70)
print("EXTRATREES - YIELD STRENGTH EVALUATION")
print("=" * 70)


# ============================================================
# 1. LOAD PREDICTIONS
# ============================================================

df = pd.read_csv(
    "YS_ExtraTrees_Predictions.csv"
)

print(
    "\nLoaded predictions:",
    df.shape
)


# ============================================================
# 2. ACTUAL / PREDICTED / ERROR
# ============================================================

actual = df[
    "Actual_YS_MPa"
]

predicted = df[
    "Predicted_YS_MPa"
]

error = (
    actual
    -
    predicted
)

absolute_error = np.abs(
    error
)


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


print(
    "\n" + "=" * 70
)

print(
    "EXTRATREES METRICS"
)

print(
    "=" * 70
)

print(
    f"MAE  : {mae:.2f} MPa"
)

print(
    f"RMSE : {rmse:.2f} MPa"
)

print(
    f"R²   : {r2:.4f}"
)


# ============================================================
# 4. ERROR STATISTICS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "ERROR STATISTICS"
)

print(
    "=" * 70
)

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
    "ExtraTrees: Actual vs Predicted Yield Strength"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "YS_ExtraTrees_Actual_vs_Predicted.png",
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
    "ExtraTrees: Yield Strength Prediction Error"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "YS_ExtraTrees_Error_Distribution.png",
    dpi=300
)

plt.show()


# ============================================================
# 7. ERROR BY YS RANGE
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


print(
    "\n" + "=" * 70
)

print(
    "ERROR BY YIELD STRENGTH RANGE"
)

print(
    "=" * 70
)

print(
    range_df.to_string(
        index=False
    )
)


# ============================================================
# 8. SAVE RANGE ANALYSIS
# ============================================================

range_df.to_csv(
    "YS_ExtraTrees_Error_By_Range.csv",
    index=False
)


# ============================================================
# 9. WORST PREDICTIONS
# ============================================================

df["Absolute_Error_MPa"] = (
    np.abs(
        df["Error_MPa"]
    )
)


worst = (
    df
    .sort_values(
        "Absolute_Error_MPa",
        ascending=False
    )
    .head(20)
)


print(
    "\n" + "=" * 70
)

print(
    "TOP 20 WORST PREDICTIONS"
)

print(
    "=" * 70
)

print(
    worst[
        [
            "Actual_YS_MPa",
            "Predicted_YS_MPa",
            "Error_MPa",
            "Absolute_Error_MPa"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 10. SAVE WORST CASES
# ============================================================

worst.to_csv(
    "YS_ExtraTrees_Worst_20_Predictions.csv",
    index=False
)


# ============================================================
# 11. FINAL
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "EXTRATREES EVALUATION COMPLETE"
)

print(
    "=" * 70
)

print(
    "\nGenerated:"
)

print(
    "1. YS_ExtraTrees_Actual_vs_Predicted.png"
)

print(
    "2. YS_ExtraTrees_Error_Distribution.png"
)

print(
    "3. YS_ExtraTrees_Error_By_Range.csv"
)

print(
    "4. YS_ExtraTrees_Worst_20_Predictions.csv"
)

print(
    "=" * 70
)