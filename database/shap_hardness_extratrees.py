# ================================================================
# SHAP ANALYSIS - FINAL HARDNESS EXTRATREES MODEL
# ================================================================

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from database.connection import engine


# ================================================================
# CONFIGURATION
# ================================================================

DATASET_PATH = "Hardness_ML_Dataset.csv"
MODEL_PATH = "Hardness_ExtraTrees_Model.pkl"

SHAP_IMPORTANCE_OUTPUT = (
    "Hardness_ExtraTrees_SHAP_Importance.csv"
)

SHAP_SUMMARY_OUTPUT = (
    "Hardness_ExtraTrees_SHAP_Summary.png"
)

SHAP_BAR_OUTPUT = (
    "Hardness_ExtraTrees_SHAP_Bar.png"
)

INDIVIDUAL_OUTPUT = (
    "Hardness_ExtraTrees_Individual_SHAP.csv"
)


# ================================================================
# FEATURES
# ================================================================

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

features = (
    element_features
    + numeric_features
    + categorical_features
)


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("SHAP ANALYSIS - FINAL HARDNESS EXTRATREES MODEL")
print("=" * 70)


# ================================================================
# LOAD MODEL
# ================================================================

print("\nLoading model...")

pipeline = joblib.load(
    MODEL_PATH
)

print(
    "Model loaded successfully."
)


# ================================================================
# LOAD CLEANED DATASET
# ================================================================

clean_df = pd.read_csv(
    DATASET_PATH
)

print(
    "\nCleaned dataset:",
    clean_df.shape
)


# ================================================================
# LOAD POSTGRESQL DATA
# ================================================================

query = """
SELECT *
FROM hea_mpea.dataset_v1_raw
WHERE hardness_hv IS NOT NULL
"""

pg_df = pd.read_sql(
    query,
    engine
)

print(
    "PostgreSQL dataset:",
    pg_df.shape
)


# ================================================================
# MATCH EXACT 717 RECORDS
# ================================================================

df = pg_df[
    pg_df["record_id"].isin(
        clean_df["record_id"]
    )
].copy()

print(
    "\nMatched records:",
    len(df)
)

if len(df) != len(clean_df):

    raise ValueError(
        "ERROR: Dataset matching failed."
    )


# ================================================================
# ADD COMPOSITION FEATURES
# ================================================================

composition_query = """
SELECT
    a.composition_canonical,

    c.al, c.b, c.c, c.co, c.cr, c.cu, c.fe, c.mn,
    c.mo, c.nb, c.ni, c.si, c.ta, c.ti, c.v, c.w,
    c.zr, c.ag, c.ca, c.ga, c.hf, c.i, c.li, c.mg,
    c.nd, c.o, c.pd, c.re, c.ru, c.s, c.sc, c.sn,
    c.t, c.y, c.zn

FROM hea_mpea.alloys a

LEFT JOIN hea_mpea.compositions c
    ON a.alloy_id = c.alloy_id
"""

composition_df = pd.read_sql(
    composition_query,
    engine
)

composition_df = (
    composition_df
    .drop_duplicates(
        subset=["composition_canonical"],
        keep="first"
    )
)

df = df.merge(
    composition_df,
    on="composition_canonical",
    how="left"
)

print(
    "After composition merge:",
    df.shape
)


# ================================================================
# PREPARE FEATURES
# ================================================================

df[element_features] = (
    df[element_features]
    .apply(
        pd.to_numeric,
        errors="coerce"
    )
    .fillna(0)
)

X = df[features].copy()


# ================================================================
# GET TRAINED PREPROCESSOR AND MODEL
# ================================================================

preprocessor = (
    pipeline.named_steps[
        "preprocessor"
    ]
)

model = (
    pipeline.named_steps[
        "model"
    ]
)


# ================================================================
# TRANSFORM DATA
# ================================================================

print(
    "\nTransforming features..."
)

X_transformed = (
    preprocessor.transform(X)
)

feature_names = (
    preprocessor
    .get_feature_names_out()
)

print(
    "Transformed shape:",
    X_transformed.shape
)

print(
    "Number of features:",
    len(feature_names)
)


# ================================================================
# CONVERT TO NUMPY
# ================================================================

X_transformed = np.asarray(
    X_transformed,
    dtype=float
)


# ================================================================
# SHAP EXPLAINER
# ================================================================

print(
    "\nCreating SHAP TreeExplainer..."
)

explainer = shap.TreeExplainer(
    model
)

shap_values = explainer.shap_values(
    X_transformed
)

shap_values = np.asarray(
    shap_values
)

print(
    "SHAP matrix shape:",
    shap_values.shape
)


# ================================================================
# GLOBAL SHAP IMPORTANCE
# ================================================================

mean_abs_shap = np.mean(
    np.abs(shap_values),
    axis=0
)

shap_importance = pd.DataFrame(
    {
        "Feature": feature_names,
        "Mean_Absolute_SHAP": mean_abs_shap
    }
)

shap_importance = (
    shap_importance
    .sort_values(
        "Mean_Absolute_SHAP",
        ascending=False
    )
    .reset_index(drop=True)
)


# ================================================================
# SAVE SHAP IMPORTANCE
# ================================================================

shap_importance.to_csv(
    SHAP_IMPORTANCE_OUTPUT,
    index=False
)


# ================================================================
# PRINT TOP FEATURES
# ================================================================

print(
    "\n" + "=" * 70
)

print(
    "TOP 20 SHAP FEATURES"
)

print(
    "=" * 70
)

print(
    shap_importance
    .head(20)
    .to_string(index=False)
)


# ================================================================
# SHAP SUMMARY PLOT
# ================================================================

print(
    "\nGenerating SHAP summary plot..."
)

plt.figure()

shap.summary_plot(
    shap_values,
    X_transformed,
    feature_names=feature_names,
    show=False
)

plt.tight_layout()

plt.savefig(
    SHAP_SUMMARY_OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ================================================================
# SHAP BAR PLOT
# ================================================================

print(
    "Generating SHAP bar plot..."
)

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
    SHAP_BAR_OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ================================================================
# INDIVIDUAL RECORD
# ================================================================

print(
    "Generating individual SHAP explanation..."
)

record_index = 0

individual_shap = pd.DataFrame(
    {
        "Feature": feature_names,
        "SHAP_Value":
            shap_values[record_index]
    }
)

individual_shap["Absolute_SHAP"] = (
    np.abs(
        individual_shap["SHAP_Value"]
    )
)

individual_shap = (
    individual_shap
    .sort_values(
        "Absolute_SHAP",
        ascending=False
    )
    .reset_index(drop=True)
)

individual_shap.to_csv(
    INDIVIDUAL_OUTPUT,
    index=False
)


# ================================================================
# FINAL OUTPUT
# ================================================================

print(
    "\n" + "=" * 70
)

print(
    "FILES SAVED"
)

print(
    "=" * 70
)

print(
    SHAP_IMPORTANCE_OUTPUT
)

print(
    SHAP_SUMMARY_OUTPUT
)

print(
    SHAP_BAR_OUTPUT
)

print(
    INDIVIDUAL_OUTPUT
)

print(
    "\nHardness SHAP analysis completed successfully."
)