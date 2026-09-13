import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap

from sqlalchemy import text
from database.connection import engine


print("=" * 70)
print("SHAP - XGBOOST YIELD STRENGTH EXPLAINABILITY")
print("=" * 70)

# =========================================================
# 1. LOAD MODEL
# =========================================================

MODEL_PATH = "YS_XGBoost_Model.pkl"

pipeline = joblib.load(MODEL_PATH)

print("\nXGBoost pipeline loaded successfully.")

preprocessor = pipeline.named_steps["preprocessor"]
xgb_model = pipeline.named_steps["model"]

# =========================================================
# 2. LOAD SAME YS DATA
# =========================================================

query = """
SELECT *
FROM hea_mpea.dataset_v1_raw
WHERE ys_tensile_mpa IS NOT NULL
"""

df = pd.read_sql(text(query), engine)

print("Loaded data:", df.shape)

# =========================================================
# 3. FEATURE DEFINITIONS
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
    "vec",
    "atomic_size_mismatch",
    "mixing_enthalpy",
    "mixing_entropy",
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
    ELEMENTS
    + NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)

# =========================================================
# 4. PREPARE FEATURES
# =========================================================

df = df[
    FEATURES
    + ["ys_tensile_mpa", "composition_canonical"]
].copy()

df = df.dropna(
    subset=["composition_canonical"]
).reset_index(drop=True)

# Convert elemental values to numeric
for element in ELEMENTS:
    df[element] = pd.to_numeric(
        df[element],
        errors="coerce"
    )

X = df[FEATURES].copy()

for element in ELEMENTS:
    X[element] = X[element].fillna(0)

y = pd.to_numeric(
    df["ys_tensile_mpa"],
    errors="coerce"
)

# =========================================================
# 5. USE THE SAME GROUPED TEST SPLIT
# =========================================================

from sklearn.model_selection import GroupShuffleSplit

groups = df["composition_canonical"]

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    gss.split(X, y, groups=groups)
)

X_test = X.iloc[test_idx].copy()

print("Test records:", len(X_test))

# =========================================================
# 6. TRANSFORM FEATURES
# =========================================================

X_test_transformed = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()

# Clean feature names for XGBoost/SHAP compatibility
clean_feature_names = [
    name.replace("[", "(")
        .replace("]", ")")
        .replace("<", "less_than_")
        .replace(">", "greater_than_")
    for name in feature_names
]

X_test_transformed = pd.DataFrame(
    X_test_transformed,
    columns=clean_feature_names
)

feature_names = clean_feature_names

print(
    "Transformed test shape:",
    X_test_transformed.shape
)

# =========================================================
# 7. SHAP EXPLAINER
# =========================================================

print("\nCreating SHAP TreeExplainer...")

explainer = shap.TreeExplainer(
    xgb_model,
    feature_perturbation="tree_path_dependent",
    model_output="raw"
)

# Convert to NumPy array before passing to SHAP.
# This avoids XGBoost/pandas feature-name compatibility issues.
X_test_shap = X_test_transformed.to_numpy(dtype=float)

shap_values = explainer.shap_values(
    X_test_shap
)

print("SHAP calculation complete.")

# =========================================================
# 8. GLOBAL SHAP IMPORTANCE
# =========================================================

mean_abs_shap = np.abs(
    shap_values
).mean(axis=0)

shap_importance = pd.DataFrame({
    "Feature": feature_names,
    "Mean_Absolute_SHAP": mean_abs_shap
})

shap_importance = shap_importance.sort_values(
    by="Mean_Absolute_SHAP",
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 70)
print("TOP 30 SHAP FEATURES")
print("=" * 70)

print(
    shap_importance.head(30).to_string(index=False)
)

# =========================================================
# 9. SAVE SHAP IMPORTANCE
# =========================================================

shap_importance.to_csv(
    "YS_XGBoost_SHAP_Importance.csv",
    index=False
)

print(
    "\nSaved: YS_XGBoost_SHAP_Importance.csv"
)

# =========================================================
# 10. SHAP SUMMARY PLOT
# =========================================================

plt.figure()

shap.summary_plot(
    shap_values,
    X_test_shap,
    feature_names=feature_names,
    show=False,
    max_display=20
)

plt.tight_layout()

plt.savefig(
    "YS_XGBoost_SHAP_Summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved: YS_XGBoost_SHAP_Summary.png"
)

# =========================================================
# 11. SHAP BAR PLOT
# =========================================================

plt.figure()

shap.summary_plot(
    shap_values,
    X_test_shap,
    feature_names=feature_names,
    plot_type="bar",
    show=False,
    max_display=20
)

plt.tight_layout()

plt.savefig(
    "YS_XGBoost_SHAP_Bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved: YS_XGBoost_SHAP_Bar.png"
)

# =========================================================
# 12. INDIVIDUAL ALLOY EXPLANATION
# =========================================================

sample_index = 0

individual_shap = pd.DataFrame({
    "Feature": feature_names,
    "SHAP_Value": shap_values[sample_index],
    "Absolute_SHAP": np.abs(
        shap_values[sample_index]
    )
})

individual_shap = individual_shap.sort_values(
    by="Absolute_SHAP",
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 70)
print("TOP FEATURES FOR FIRST TEST ALLOY")
print("=" * 70)

print(
    individual_shap.head(20).to_string(index=False)
)

individual_shap.to_csv(
    "YS_XGBoost_Individual_SHAP.csv",
    index=False
)

print(
    "\nSaved: YS_XGBoost_Individual_SHAP.csv"
)

# =========================================================
# 13. COMPLETE
# =========================================================

print("\n" + "=" * 70)
print("SHAP ANALYSIS COMPLETE")
print("=" * 70)

print("\nGenerated:")
print("1. YS_XGBoost_SHAP_Importance.csv")
print("2. YS_XGBoost_SHAP_Summary.png")
print("3. YS_XGBoost_SHAP_Bar.png")
print("4. YS_XGBoost_Individual_SHAP.csv")