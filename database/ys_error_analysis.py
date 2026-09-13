import pandas as pd
import numpy as np


# Load predictions
df = pd.read_csv("YS_Model2_Predictions.csv")

actual = df["actual_ys_mpa"]
predicted = df["predicted_ys_mpa"]

# Absolute error
df["absolute_error"] = np.abs(
    actual - predicted
)


# --------------------------------------------------
# Define YS ranges
# --------------------------------------------------

bins = [0, 500, 1000, 1500, 2000, np.inf]

labels = [
    "<500 MPa",
    "500–1000 MPa",
    "1000–1500 MPa",
    "1500–2000 MPa",
    ">2000 MPa"
]

df["YS_Range"] = pd.cut(
    actual,
    bins=bins,
    labels=labels,
    right=False
)


# --------------------------------------------------
# Calculate error statistics
# --------------------------------------------------

analysis = df.groupby(
    "YS_Range",
    observed=True
).agg(
    Records=("actual_ys_mpa", "count"),
    Mean_Actual_YS=("actual_ys_mpa", "mean"),
    Mean_Predicted_YS=("predicted_ys_mpa", "mean"),
    MAE=("absolute_error", "mean")
).reset_index()


# RMSE for each range
rmse_values = []

for group in labels:

    subset = df[
        df["YS_Range"] == group
    ]

    if len(subset) > 0:

        rmse = np.sqrt(
            np.mean(
                (
                    subset["actual_ys_mpa"]
                    - subset["predicted_ys_mpa"]
                ) ** 2
            )
        )

    else:
        rmse = np.nan

    rmse_values.append(rmse)


analysis["RMSE"] = rmse_values


# --------------------------------------------------
# Display
# --------------------------------------------------

print("\nYS Error Analysis by Strength Range")
print("===================================")

print(
    analysis.to_string(index=False)
)


# --------------------------------------------------
# Save
# --------------------------------------------------

analysis.to_csv(
    "YS_Error_Analysis_By_Range.csv",
    index=False
)

print("\nSaved:")
print("YS_Error_Analysis_By_Range.csv")