import pandas as pd
import matplotlib.pyplot as plt
import joblib

print("=" * 70)
print("XGBOOST - YIELD STRENGTH FEATURE IMPORTANCE")
print("=" * 70)

# ---------------------------------------------------------
# 1. LOAD MODEL
# ---------------------------------------------------------

model = joblib.load("YS_XGBoost_Model.pkl")

print("\nModel loaded successfully.")

# ---------------------------------------------------------
# 2. GET PIPELINE COMPONENTS
# ---------------------------------------------------------

preprocessor = model.named_steps["preprocessor"]
xgb_model = model.named_steps["model"]

print("Detected estimator:", type(xgb_model).__name__)

# ---------------------------------------------------------
# 3. GET TRANSFORMED FEATURE NAMES
# ---------------------------------------------------------

feature_names = preprocessor.get_feature_names_out()
importances = xgb_model.feature_importances_

print("Number of transformed features:", len(feature_names))
print("Number of importance values:", len(importances))

# ---------------------------------------------------------
# 4. CREATE FEATURE IMPORTANCE TABLE
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
# 5. DISPLAY TOP 30
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TOP 30 XGBOOST FEATURES")
print("=" * 70)

print(
    importance_df.head(30).to_string(index=False)
)

# ---------------------------------------------------------
# 6. SAVE CSV
# ---------------------------------------------------------

importance_df.to_csv(
    "YS_XGBoost_Feature_Importance.csv",
    index=False
)

print("\nSaved:")
print("YS_XGBoost_Feature_Importance.csv")

# ---------------------------------------------------------
# 7. PLOT TOP 20
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
plt.title("XGBoost Feature Importance - Yield Strength")

plt.tight_layout()

plt.savefig(
    "YS_XGBoost_Feature_Importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("YS_XGBoost_Feature_Importance.png")

print("\n" + "=" * 70)
print("XGBOOST FEATURE IMPORTANCE ANALYSIS COMPLETE")
print("=" * 70)