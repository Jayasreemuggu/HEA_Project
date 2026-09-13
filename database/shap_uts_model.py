import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import re

print("=" * 70)
print("UTS XGBOOST - SHAP EXPLAINABILITY")
print("=" * 70)

# =========================================================
# 1. LOAD MODEL
# =========================================================

print("\nLoading XGBoost model...")

model = joblib.load(
    "UTS_XGBoost_Model.pkl"
)

print("Model loaded successfully.")


# =========================================================
# 2. LOAD TEST DATA
# =========================================================

from database.connection import engine

query = """
SELECT *
FROM hea_mpea.dataset_v1_raw
WHERE uts_tensile_mpa IS NOT NULL
"""

print("\nLoading UTS data...")

df = pd.read_sql(query, engine)

print("Dataset shape:", df.shape)


# =========================================================
# 3. FEATURES
# =========================================================

ELEMENTS = [
    "al_at_pct", "b_at_pct", "c_at_pct", "co_at_pct",
    "cr_at_pct", "cu_at_pct", "fe_at_pct", "mn_at_pct",
    "mo_at_pct", "nb_at_pct", "ni_at_pct", "si_at_pct",
    "ta_at_pct", "ti_at_pct", "v_at_pct", "w_at_pct",
    "zr_at_pct", "ag_at_pct", "ca_at_pct", "ga_at_pct",
    "hf_at_pct", "i_at_pct", "li_at_pct", "mg_at_pct",
    "nd_at_pct", "o_at_pct", "pd_at_pct", "re_at_pct",
    "ru_at_pct", "s_at_pct", "sc_at_pct", "sn_at_pct",
    "t_at_pct", "y_at_pct", "zn_at_pct"
]

NUMERIC_FEATURES = [
    "test_temperature_c",
    "density_exp_g_cm3",
    "density_calc_g_cm3",
    "grain_size_um",
    "precipitate_size_nm",
    "matrix_volume_pct"
]

CATEGORICAL_FEATURES = [
    "test_type",
    "phase",
    "processing_method",
    "alloy_class",
    "equilibrium_condition",
    "single_multiphase",
    "precipitate_info"
]

FEATURES = (
    ELEMENTS +
    NUMERIC_FEATURES +
    CATEGORICAL_FEATURES
)


# =========================================================
# 4. RECREATE EXACT TEST SPLIT
# =========================================================

from sklearn.model_selection import GroupShuffleSplit

for col in ELEMENTS:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df[ELEMENTS] = df[ELEMENTS].fillna(0)

for col in NUMERIC_FEATURES:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df["uts_tensile_mpa"] = pd.to_numeric(
    df["uts_tensile_mpa"],
    errors="coerce"
)

df = df[
    df["uts_tensile_mpa"].notna()
].copy()

X = df[FEATURES].copy()
y = df["uts_tensile_mpa"].copy()

groups = df["composition_canonical"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(
        X,
        y,
        groups=groups
    )
)

X_test = X.iloc[test_idx].copy()

print("\nTest records:", len(X_test))


# =========================================================
# 5. TRANSFORM TEST DATA
# =========================================================

print("\nTransforming test data...")

preprocessor = model.named_steps["preprocessor"]
xgb_model = model.named_steps["model"]

X_test_transformed = preprocessor.transform(
    X_test
)

print(
    "Transformed shape:",
    X_test_transformed.shape
)


# =========================================================
# 6. GET FEATURE NAMES
# =========================================================

feature_names = (
    preprocessor
    .get_feature_names_out()
)

# Clean names for XGBoost / SHAP compatibility
clean_feature_names = []

for name in feature_names:

    name = str(name)

    name = re.sub(
        r"[\[\]<>]",
        "_",
        name
    )

    clean_feature_names.append(name)


# =========================================================
# 7. CONVERT TO NUMPY
# =========================================================

X_test_array = np.asarray(
    X_test_transformed,
    dtype=np.float64
)

print(
    "NumPy test matrix:",
    X_test_array.shape
)


# =========================================================
# 8. SHAP EXPLAINER
# =========================================================

print("\nCreating SHAP TreeExplainer...")

explainer = shap.TreeExplainer(
    xgb_model
)

print("Calculating SHAP values...")

shap_values = explainer.shap_values(
    X_test_array
)

print(
    "SHAP matrix shape:",
    np.asarray(shap_values).shape
)


# =========================================================
# 9. GLOBAL SHAP IMPORTANCE
# =========================================================

mean_abs_shap = np.mean(
    np.abs(shap_values),
    axis=0
)

importance = pd.DataFrame({
    "Feature": clean_feature_names,
    "Mean_Absolute_SHAP": mean_abs_shap
})

importance = importance.sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
).reset_index(drop=True)

importance.to_csv(
    "UTS_XGBoost_SHAP_Importance.csv",
    index=False
)


# =========================================================
# 10. PRINT TOP FEATURES
# =========================================================

print("\n" + "=" * 70)
print("TOP 30 SHAP FEATURES")
print("=" * 70)

print(
    importance.head(30).to_string(
        index=False
    )
)


# =========================================================
# 11. SHAP SUMMARY PLOT
# =========================================================

print("\nCreating SHAP summary plot...")

plt.figure(figsize=(10, 8))

shap.summary_plot(
    shap_values,
    X_test_array,
    feature_names=clean_feature_names,
    show=False
)

plt.tight_layout()

plt.savefig(
    "UTS_XGBoost_SHAP_Summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 12. SHAP BAR PLOT
# =========================================================

print("Creating SHAP bar plot...")

plt.figure(figsize=(10, 8))

shap.summary_plot(
    shap_values,
    X_test_array,
    feature_names=clean_feature_names,
    plot_type="bar",
    show=False
)

plt.tight_layout()

plt.savefig(
    "UTS_XGBoost_SHAP_Bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 13. INDIVIDUAL SHAP VALUES
# =========================================================

print("Saving individual SHAP values...")

individual = pd.DataFrame(
    shap_values,
    columns=clean_feature_names
)

individual.to_csv(
    "UTS_XGBoost_Individual_SHAP.csv",
    index=False
)


# =========================================================
# 14. FIRST TEST ALLOY
# =========================================================

first_shap = pd.DataFrame({
    "Feature": clean_feature_names,
    "SHAP_Value": shap_values[0]
})

first_shap["Absolute_SHAP"] = np.abs(
    first_shap["SHAP_Value"]
)

first_shap = first_shap.sort_values(
    "Absolute_SHAP",
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 70)
print("FIRST TEST ALLOY - TOP SHAP CONTRIBUTIONS")
print("=" * 70)

print(
    first_shap.head(20).to_string(
        index=False
    )
)


# =========================================================
# 15. SAVE INDIVIDUAL TOP SHAP
# =========================================================

first_shap.to_csv(
    "UTS_XGBoost_First_Alloy_SHAP.csv",
    index=False
)


# =========================================================
# 16. FINAL
# =========================================================

print("\n" + "=" * 70)
print("UTS SHAP ANALYSIS COMPLETE")
print("=" * 70)

print("\nSaved:")
print("1. UTS_XGBoost_SHAP_Importance.csv")
print("2. UTS_XGBoost_SHAP_Summary.png")
print("3. UTS_XGBoost_SHAP_Bar.png")
print("4. UTS_XGBoost_Individual_SHAP.csv")
print("5. UTS_XGBoost_First_Alloy_SHAP.csv")