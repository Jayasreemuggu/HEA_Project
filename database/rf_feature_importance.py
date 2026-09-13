import pandas as pd
import joblib


ELEMENTS = [
    "ag", "al", "b", "c", "ca", "co", "cr", "cu",
    "fe", "ga", "hf", "i", "li", "mg", "mn", "mo",
    "nb", "nd", "ni", "o", "pd", "re", "ru", "s",
    "sc", "si", "sn", "t", "ta", "ti", "v", "w", "y",
    "zn", "zr"
]


# Load model
model = joblib.load("YS_RandomForest_Model.pkl")

# Feature importance
importance = pd.DataFrame({
    "Element": ELEMENTS,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\nRandom Forest Feature Importance")
print("---------------------------------")
print(importance.to_string(index=False))

# Save
importance.to_csv(
    "YS_RF_Feature_Importance.csv",
    index=False
)

print("\nSaved:")
print("YS_RF_Feature_Importance.csv")