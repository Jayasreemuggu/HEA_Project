import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load prediction results
df = pd.read_csv("YS_Model2_Predictions.csv")

actual = df["actual_ys_mpa"]
predicted = df["predicted_ys_mpa"]


# ============================================================
# Actual vs Predicted
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    actual,
    predicted,
    alpha=0.6
)

# Perfect prediction line
minimum = min(actual.min(), predicted.min())
maximum = max(actual.max(), predicted.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Yield Strength (MPa)")
plt.ylabel("Predicted Yield Strength (MPa)")
plt.title("Random Forest: Actual vs Predicted Yield Strength")

plt.tight_layout()

plt.savefig(
    "YS_Model2_Actual_vs_Predicted.png",
    dpi=300
)

plt.show()


# ============================================================
# Error distribution
# ============================================================

errors = actual - predicted

plt.figure(figsize=(8, 6))

plt.hist(
    errors,
    bins=30
)

plt.xlabel("Prediction Error (MPa)")
plt.ylabel("Frequency")
plt.title("Yield Strength Prediction Error Distribution")

plt.tight_layout()

plt.savefig(
    "YS_Model2_Error_Distribution.png",
    dpi=300
)

plt.show()


# ============================================================
# Error statistics
# ============================================================

print("\nPrediction Error Statistics")
print("---------------------------")

print(
    errors.describe()
)

print("\nMean Error (Bias):")
print(
    f"{errors.mean():.2f} MPa"
)

print("\nMean Absolute Error:")
print(
    f"{np.abs(errors).mean():.2f} MPa"
)

print("\nMaximum Absolute Error:")
print(
    f"{np.abs(errors).max():.2f} MPa"
)


print("\nPlots saved:")
print("YS_Model2_Actual_vs_Predicted.png")
print("YS_Model2_Error_Distribution.png")