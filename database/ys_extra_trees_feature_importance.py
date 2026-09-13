import pandas as pd
import matplotlib.pyplot as plt
import joblib

print("=" * 70)
print("EXTRATREES - YIELD STRENGTH FEATURE IMPORTANCE")
print("=" * 70)

# ---------------------------------------------------------
# Load trained ExtraTrees model
# ---------------------------------------------------------

MODEL_PATH = "YS_ExtraTrees_Model.pkl"

model = joblib.load(MODEL_PATH)

print("\nModel loaded successfully.")

# ---------------------------------------------------------
# Inspect pipeline steps
# ---------------------------------------------------------

print("\nPipeline steps:")
print(model.named_steps)

# ---------------------------------------------------------
# Automatically find preprocessor
# ---------------------------------------------------------

preprocessor = None
estimator = None

for name, step in model.named_steps.items():

    if hasattr(step, "get_feature_names_out"):
        preprocessor = step

    if hasattr(step, "feature_importances_"):
        estimator = step

print("\nDetected preprocessor:", type(preprocessor).__name__)
print("Detected estimator:", type(estimator).__name__)

# ---------------------------------------------------------
# Safety check
# ---------------------------------------------------------

if preprocessor is None:
    raise ValueError("Could not find preprocessing step.")

if estimator is None:
    raise ValueError("Could not find ExtraTrees estimator.")

# ---------------------------------------------------------
# Get transformed feature names
# ---------------------------------------------------------

feature_names = preprocessor.get_feature_names_out()

importances = estimator.feature_importances_

print("\nNumber of transformed features:", len(feature_names))
print("Number of importance values:", len(importances))

# ---------------------------------------------------------
# Feature importance dataframe
# ---------------------------------------------------------

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)

# ---------------------------------------------------------
# Top 30 features
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TOP 30 FEATURES")
print("=" * 70)

print(
    importance_df.head(30).to_string(index=False)
)

# ---------------------------------------------------------
# Save complete feature importance
# ---------------------------------------------------------

importance_df.to_csv(
    "YS_ExtraTrees_Feature_Importance.csv",
    index=False
)

print("\nSaved:")
print("YS_ExtraTrees_Feature_Importance.csv")

# ---------------------------------------------------------
# Plot Top 20
# ---------------------------------------------------------

top20 = importance_df.head(20).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(10, 8))

plt.barh(
    top20["Feature"],
    top20["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("ExtraTrees Feature Importance - Yield Strength")

plt.tight_layout()

plt.savefig(
    "YS_ExtraTrees_Feature_Importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("YS_ExtraTrees_Feature_Importance.png")

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE ANALYSIS COMPLETE")
print("=" * 70)