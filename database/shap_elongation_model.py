import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

print("=" * 70)
print("SHAP ANALYSIS - ELONGATION XGBOOST MODEL")
print("=" * 70)


# ============================================================
# 1. LOAD MODEL
# ============================================================

model = joblib.load(
    "Elongation_XGBoost_Model.pkl"
)

print("\nModel loaded successfully.")


# ============================================================
# 2. LOAD DATA
# ============================================================

clean_df = pd.read_csv(
    "Elongation_ML_Dataset.csv"
)

print(
    "Cleaned dataset:",
    clean_df.shape
)


# ============================================================
# 3. LOAD POSTGRESQL DATA
# ============================================================

from sqlalchemy import text
from database.connection import engine


query = """
SELECT
    d.*,

    c.al, c.b, c.c, c.co, c.cr, c.cu, c.fe, c.mn,
    c.mo, c.nb, c.ni, c.si, c.ta, c.ti, c.v, c.w,
    c.zr, c.ag, c.ca, c.ga, c.hf, c.i, c.li, c.mg,
    c.nd, c.o, c.pd, c.re, c.ru, c.s, c.sc, c.sn,
    c.t, c.y, c.zn

FROM hea_mpea.dataset_v1_raw d

LEFT JOIN hea_mpea.alloys a
    ON d.composition_canonical = a.composition_canonical

LEFT JOIN hea_mpea.compositions c
    ON a.alloy_id = c.alloy_id

WHERE d.elongation_tensile_pct IS NOT NULL
"""

with engine.connect() as conn:
    pg_df = pd.read_sql(
        text(query),
        conn
    )


# ============================================================
# 4. MATCH EXACT CLEANED DATA
# ============================================================

match_columns = [
    "composition_canonical",
    "test_temperature_c",
    "test_type",
    "phase",
    "processing_method",
    "elongation_tensile_pct"
]


def make_key(df):

    temp = df[match_columns].copy()

    for col in match_columns:

        if pd.api.types.is_numeric_dtype(
            temp[col]
        ):
            temp[col] = pd.to_numeric(
                temp[col],
                errors="coerce"
            ).round(8)

        temp[col] = (
            temp[col]
            .fillna("<NA>")
            .astype(str)
        )

    return temp.astype(str).agg(
        "|".join,
        axis=1
    )


clean_df["_match_key"] = make_key(
    clean_df
)

pg_df["_match_key"] = make_key(
    pg_df
)


pg_unique = pg_df.drop_duplicates(
    subset="_match_key",
    keep="first"
)


df = pg_unique[
    pg_unique["_match_key"].isin(
        set(clean_df["_match_key"])
    )
].copy()


print(
    "Matched records:",
    len(df)
)


# ============================================================
# 5. FEATURES
# ============================================================

element_features = [
    "al", "b", "c", "co", "cr", "cu", "fe", "mn",
    "mo", "nb", "ni", "si", "ta", "ti", "v", "w",
    "zr", "ag", "ca", "ga", "hf", "i", "li", "mg",
    "nd", "o", "pd", "re", "ru", "s", "sc", "sn",
    "t", "y", "zn"
]


numeric_features = [
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


categorical_features = [
    "test_type",
    "phase",
    "processing_method",
    "alloy_class",
    "equilibrium_condition",
    "single_multiphase",
    "precipitate_info"
]


numeric_features = [
    c for c in numeric_features
    if c in df.columns
]

categorical_features = [
    c for c in categorical_features
    if c in df.columns
]


features = (
    element_features
    + numeric_features
    + categorical_features
)


X = df[features].copy()


# ============================================================
# 6. NUMERIC CONVERSION
# ============================================================

for col in (
    element_features
    + numeric_features
):

    X[col] = pd.to_numeric(
        X[col],
        errors="coerce"
    )


X[element_features] = (
    X[element_features]
    .fillna(0)
)


# ============================================================
# 7. APPLY SAME PREPROCESSOR FROM MODEL
# ============================================================

preprocessor = (
    model.named_steps["preprocessor"]
)

X_transformed = preprocessor.transform(X)


print(
    "\nTransformed feature matrix:",
    X_transformed.shape
)


# Convert sparse matrix if necessary
if hasattr(
    X_transformed,
    "toarray"
):

    X_transformed = (
        X_transformed.toarray()
    )


# ============================================================
# 8. FEATURE NAMES
# ============================================================

feature_names = (
    preprocessor
    .get_feature_names_out()
)


feature_names = [
    str(x).replace(
        " ",
        "_"
    )
    for x in feature_names
]


# ============================================================
# 9. XGBOOST MODEL
# ============================================================

xgb_model = (
    model.named_steps["model"]
)


# ============================================================
# 10. SHAP EXPLAINER
# ============================================================

print(
    "\nCalculating SHAP values..."
)

explainer = shap.TreeExplainer(
    xgb_model
)

shap_values = explainer.shap_values(
    X_transformed
)


print(
    "SHAP matrix:",
    np.array(shap_values).shape
)


# ============================================================
# 11. GLOBAL SHAP IMPORTANCE
# ============================================================

mean_abs_shap = np.mean(
    np.abs(shap_values),
    axis=0
)


importance = pd.DataFrame({
    "Feature": feature_names,
    "Mean_Absolute_SHAP":
        mean_abs_shap
})


importance = importance.sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
)


print("\n" + "=" * 70)
print("TOP SHAP FEATURES")
print("=" * 70)

print(
    importance.head(20).to_string(
        index=False
    )
)


# ============================================================
# 12. SAVE IMPORTANCE
# ============================================================

importance.to_csv(
    "Elongation_XGBoost_SHAP_Importance.csv",
    index=False
)


# ============================================================
# 13. SHAP SUMMARY PLOT
# ============================================================

plt.figure()

shap.summary_plot(
    shap_values,
    X_transformed,
    feature_names=feature_names,
    show=False
)

plt.tight_layout()

plt.savefig(
    "Elongation_XGBoost_SHAP_Summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 14. SHAP BAR PLOT
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
    "Elongation_XGBoost_SHAP_Bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 15. INDIVIDUAL TEST RECORD
# ============================================================

first_shap = pd.DataFrame({

    "Feature": feature_names,

    "SHAP_Value":
        shap_values[0],

    "Absolute_SHAP":
        np.abs(shap_values[0])

})


first_shap = first_shap.sort_values(
    "Absolute_SHAP",
    ascending=False
)


first_shap.to_csv(
    "Elongation_XGBoost_First_Record_SHAP.csv",
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\nSaved:")

print(
    "Elongation_XGBoost_SHAP_Importance.csv"
)

print(
    "Elongation_XGBoost_SHAP_Summary.png"
)

print(
    "Elongation_XGBoost_SHAP_Bar.png"
)

print(
    "Elongation_XGBoost_First_Record_SHAP.csv"
)

print("\n" + "=" * 70)
print("ELONGATION SHAP ANALYSIS COMPLETE")
print("=" * 70)