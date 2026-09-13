import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD MODEL AND TEST DATA
# ============================================================

model_pipeline = joblib.load(
    "YS_RandomForest_Model2.pkl"
)

predictions = pd.read_csv(
    "YS_Model2_Predictions.csv"
)


# ============================================================
# 2. DEFINE FEATURES
# ============================================================

ELEMENTS = [
    "ag", "al", "b", "c", "ca", "co", "cr", "cu",
    "fe", "ga", "hf", "i", "li", "mg", "mn", "mo",
    "nb", "nd", "ni", "o", "pd", "re", "ru", "s",
    "sc", "si", "sn", "t", "ta", "ti", "v", "w", "y",
    "zn", "zr"
]

NUMERIC_FEATURES = [
    "test_temperature_c"
]

CATEGORICAL_FEATURES = [
    "test_type",
    "phase",
    "processing_method",
    "alloy_class"
]

FEATURES = (
    ELEMENTS
    + NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


# ============================================================
# 3. EXTRACT PREPROCESSOR AND RANDOM FOREST
# ============================================================

preprocessor = model_pipeline.named_steps[
    "preprocessor"
]

rf_model = model_pipeline.named_steps[
    "model"
]


# ============================================================
# 4. TRANSFORM TEST DATA
# ============================================================

X_test = predictions[FEATURES].copy()

X_transformed = preprocessor.transform(
    X_test
)


# ============================================================
# 5. GET TRANSFORMED FEATURE NAMES
# ============================================================

feature_names = (
    preprocessor.get_feature_names_out()
)


print("Original features:", len(FEATURES))

print(
    "Transformed features:",
    len(feature_names)
)


# ============================================================
# 6. CREATE SHAP EXPLAINER
# ============================================================

print("\nCreating SHAP explainer...")

explainer = shap.TreeExplainer(
    rf_model
)


# ============================================================
# 7. CALCULATE SHAP VALUES
# ============================================================

print("Calculating SHAP values...")

shap_values = explainer.shap_values(
    X_transformed
)

print("SHAP calculation completed.")


# ============================================================
# 8. GLOBAL SHAP SUMMARY PLOT
# ============================================================

print("\nCreating SHAP summary plot...")

plt.figure()

shap.summary_plot(
    shap_values,
    X_transformed,
    feature_names=feature_names,
    show=False
)

plt.tight_layout()

plt.savefig(
    "YS_Model2_SHAP_Summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 9. SHAP BAR PLOT
# ============================================================

plt.figure()

shap.summary_plot(
    shap_values,
    X_transformed,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)

plt.tight_layout()

plt.savefig(
    "YS_Model2_SHAP_Bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 10. CALCULATE MEAN ABSOLUTE SHAP IMPORTANCE
# ============================================================

mean_abs_shap = np.abs(
    shap_values
).mean(axis=0)


shap_importance = pd.DataFrame({
    "Feature": feature_names,
    "Mean_Absolute_SHAP": mean_abs_shap
})


shap_importance = shap_importance.sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
)


# ============================================================
# 11. DISPLAY TOP FEATURES
# ============================================================

print("\nTop 20 SHAP Features")
print("====================")

print(
    shap_importance.head(20).to_string(
        index=False
    )
)


# ============================================================
# 12. SAVE SHAP IMPORTANCE
# ============================================================

shap_importance.to_csv(
    "YS_Model2_SHAP_Importance.csv",
    index=False
)


# ============================================================
# 13. INDIVIDUAL PREDICTION
# ============================================================

sample_index = 0

sample_shap = shap_values[
    sample_index
]

sample_data = X_test.iloc[
    sample_index
]


individual_explanation = pd.DataFrame({
    "Feature": feature_names,
    "SHAP_Value": sample_shap
})


individual_explanation[
    "Absolute_SHAP"
] = np.abs(
    individual_explanation["SHAP_Value"]
)


individual_explanation = (
    individual_explanation
    .sort_values(
        "Absolute_SHAP",
        ascending=False
    )
)


print("\nIndividual Prediction Explanation")
print("=================================")

print(
    individual_explanation.head(15).to_string(
        index=False
    )
)


# ============================================================
# 14. SAVE INDIVIDUAL EXPLANATION
# ============================================================

individual_explanation.to_csv(
    "YS_Model2_Individual_SHAP.csv",
    index=False
)


# ============================================================
# 15. FINAL OUTPUT
# ============================================================

print("\nSHAP analysis completed.")

print("\nFiles saved:")
print("-------------------------------")
print("YS_Model2_SHAP_Summary.png")
print("YS_Model2_SHAP_Bar.png")
print("YS_Model2_SHAP_Importance.csv")
print("YS_Model2_Individual_SHAP.csv")