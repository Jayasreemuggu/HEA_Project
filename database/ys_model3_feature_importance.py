import pandas as pd
import numpy as np
import joblib


# ============================================================
# MODEL 3 FEATURE IMPORTANCE
# ============================================================

print("=" * 70)
print("MODEL 3 - FEATURE IMPORTANCE")
print("=" * 70)


# ============================================================
# 1. LOAD MODEL
# ============================================================

model_path = "YS_RandomForest_Model3.pkl"

pipeline = joblib.load(model_path)

print("\nModel loaded successfully.")


# ============================================================
# 2. GET PREPROCESSOR AND RANDOM FOREST
# ============================================================

preprocessor = pipeline.named_steps[
    "preprocessor"
]

model = pipeline.named_steps[
    "model"
]


# ============================================================
# 3. GET TRANSFORMED FEATURE NAMES
# ============================================================

feature_names = (
    preprocessor
    .get_feature_names_out()
)


importance = model.feature_importances_


print(
    "\nNumber of transformed features:",
    len(feature_names)
)

print(
    "Number of importance values:",
    len(importance)
)


# ============================================================
# 4. CREATE FEATURE IMPORTANCE TABLE
# ============================================================

importance_df = pd.DataFrame({

    "Feature": feature_names,

    "Importance": importance
})


importance_df = (
    importance_df
    .sort_values(
        "Importance",
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# 5. PRINT TOP 30
# ============================================================

print("\n" + "=" * 70)
print("TOP 30 FEATURES")
print("=" * 70)

print(
    importance_df.head(30).to_string(
        index=False
    )
)


# ============================================================
# 6. SAVE FULL FEATURE IMPORTANCE
# ============================================================

importance_df.to_csv(
    "YS_Model3_Feature_Importance.csv",
    index=False
)


print(
    "\nSaved:"
)

print(
    "YS_Model3_Feature_Importance.csv"
)


# ============================================================
# 7. PLOT TOP 20
# ============================================================

import matplotlib.pyplot as plt


top20 = (
    importance_df
    .head(20)
    .sort_values(
        "Importance",
        ascending=True
    )
)


plt.figure(
    figsize=(10, 8)
)


plt.barh(
    top20["Feature"],
    top20["Importance"]
)


plt.xlabel(
    "Random Forest Feature Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Model 3 - Top 20 Feature Importance"
)


plt.tight_layout()


plt.savefig(
    "YS_Model3_Feature_Importance.png",
    dpi=300
)


plt.show()


# ============================================================
# 8. FINAL
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE ANALYSIS COMPLETE")
print("=" * 70)

print(
    "\nGenerated:"
)

print(
    "1. YS_Model3_Feature_Importance.csv"
)

print(
    "2. YS_Model3_Feature_Importance.png"
)

print("=" * 70)