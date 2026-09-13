import pandas as pd
import numpy as np
import joblib

from sqlalchemy import text
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from database.connection import engine


print("=" * 70)
print("XGBOOST - YIELD STRENGTH MODEL")
print("=" * 70)


# =========================================================
# 1. FEATURES
# =========================================================

ELEMENTS = [
    "al_at_pct",
    "b_at_pct",
    "c_at_pct",
    "co_at_pct",
    "cr_at_pct",
    "cu_at_pct",
    "fe_at_pct",
    "mn_at_pct",
    "mo_at_pct",
    "nb_at_pct",
    "ni_at_pct",
    "si_at_pct",
    "ta_at_pct",
    "ti_at_pct",
    "v_at_pct",
    "w_at_pct",
    "zr_at_pct",
    "ag_at_pct",
    "ca_at_pct",
    "ga_at_pct",
    "hf_at_pct",
    "i_at_pct",
    "li_at_pct",
    "mg_at_pct",
    "nd_at_pct",
    "o_at_pct",
    "pd_at_pct",
    "re_at_pct",
    "ru_at_pct",
    "s_at_pct",
    "sc_at_pct",
    "sn_at_pct",
    "t_at_pct",
    "y_at_pct",
    "zn_at_pct"
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

FEATURES = ELEMENTS + NUMERIC_FEATURES + CATEGORICAL_FEATURES


# =========================================================
# 2. LOAD DATA FROM POSTGRESQL
# =========================================================

query = """
SELECT *
FROM hea_mpea.dataset_v1_raw
WHERE ys_tensile_mpa IS NOT NULL
"""

print("\nLoading data from PostgreSQL...")

df = pd.read_sql(text(query), engine)

print("Loaded shape:", df.shape)


# =========================================================
# 3. PREPARE DATA
# =========================================================

df = df.copy()

# Keep only required columns
required_columns = FEATURES + [
    "ys_tensile_mpa",
    "composition_canonical"
]

df = df[required_columns]

# Remove rows without composition identifier
df = df.dropna(
    subset=["composition_canonical"]
).reset_index(drop=True)

# Target
X = df[FEATURES]
y = pd.to_numeric(
    df["ys_tensile_mpa"],
    errors="coerce"
)

# Remove rows where yield strength could not be converted to numeric
valid_target = y.notna()

df = df.loc[valid_target].reset_index(drop=True)
y = y.loc[valid_target].reset_index(drop=True)

X = df[FEATURES].copy()
groups = df["composition_canonical"].reset_index(drop=True)

groups = df["composition_canonical"]


# =========================================================
# 4. CLEAN ELEMENTAL COMPOSITION DATA
# =========================================================

# Convert all elemental composition columns to numeric.
# Non-numeric values are converted to NaN.
for element in ELEMENTS:
    df[element] = pd.to_numeric(
        df[element],
        errors="coerce"
    )

# Check composition sum
composition_sum = df[ELEMENTS].fillna(0).sum(axis=1)

print("\nComposition sum statistics:")
print(composition_sum.describe())

# Fill missing elemental concentrations with 0.
# 0 represents absence of the element.
X = df[FEATURES].copy()

for element in ELEMENTS:
    X[element] = X[element].fillna(0)

# =========================================================
# 5. GROUPED TRAIN / TEST SPLIT
# =========================================================

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    gss.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()

y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()

groups_train = groups.iloc[train_idx]
groups_test = groups.iloc[test_idx]

print("\nTrain records:", len(X_train))
print("Test records :", len(X_test))

print(
    "Unique train compositions:",
    groups_train.nunique()
)

print(
    "Unique test compositions:",
    groups_test.nunique()
)

overlap = set(groups_train).intersection(
    set(groups_test)
)

print(
    "Composition overlap:",
    len(overlap)
)


# =========================================================
# 6. PREPROCESSING
# =========================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            ELEMENTS + NUMERIC_FEATURES
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        )
    ]
)


# =========================================================
# 7. XGBOOST MODEL
# =========================================================

xgb_model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)


# =========================================================
# 8. COMPLETE PIPELINE
# =========================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", xgb_model)
    ]
)


# =========================================================
# 9. TRAIN
# =========================================================

print("\nTraining XGBoost...")

pipeline.fit(
    X_train,
    y_train
)

print("Training complete.")


# =========================================================
# 10. PREDICTION
# =========================================================

y_pred = pipeline.predict(X_test)


# =========================================================
# 11. METRICS
# =========================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n" + "=" * 70)
print("XGBOOST RESULTS")
print("=" * 70)

print(f"MAE  : {mae:.2f} MPa")
print(f"RMSE : {rmse:.2f} MPa")
print(f"R²   : {r2:.4f}")


# =========================================================
# 12. COMPARE WITH EXTRATREES
# =========================================================

extra_r2 = 0.6818
extra_mae = 207.18
extra_rmse = 301.10

print("\n" + "=" * 70)
print("COMPARISON WITH EXTRATREES")
print("=" * 70)

print(
    f"ExtraTrees R² : {extra_r2:.4f}"
)

print(
    f"XGBoost R²    : {r2:.4f}"
)

print(
    f"R² difference : {r2 - extra_r2:+.4f}"
)

print(
    f"\nExtraTrees MAE : {extra_mae:.2f} MPa"
)

print(
    f"XGBoost MAE    : {mae:.2f} MPa"
)

print(
    f"MAE difference : {mae - extra_mae:+.2f} MPa"
)

print(
    f"\nExtraTrees RMSE : {extra_rmse:.2f} MPa"
)

print(
    f"XGBoost RMSE    : {rmse:.2f} MPa"
)

print(
    f"RMSE difference : {rmse - extra_rmse:+.2f} MPa"
)


# =========================================================
# 13. SAVE MODEL
# =========================================================

joblib.dump(
    pipeline,
    "YS_XGBoost_Model_SHAP.pkl"
)

print("\nSaved:")
print("YS_XGBoost_Model.pkl")


# =========================================================
# 14. SAVE PREDICTIONS
# =========================================================
# Convert test target to numeric before calculating errors
y_test_numeric = pd.to_numeric(
    y_test,
    errors="coerce"
).to_numpy(dtype=float)

y_pred_numeric = np.asarray(
    y_pred,
    dtype=float
)

predictions = pd.DataFrame({
    "Actual_YS_MPa": y_test_numeric,
    "Predicted_YS_MPa": y_pred_numeric,
    "Error_MPa": y_test_numeric - y_pred_numeric,
    "Absolute_Error_MPa":
        np.abs(y_test_numeric - y_pred_numeric)
})

predictions.to_csv(
    "YS_XGBoost_Predictions.csv",
    index=False
)

print(
    "YS_XGBoost_Predictions.csv"
)


# =========================================================
# 15. SAVE METRICS
# =========================================================

metrics = pd.DataFrame({
    "Model": ["XGBoost"],
    "MAE_MPa": [mae],
    "RMSE_MPa": [rmse],
    "R2": [r2]
})

metrics.to_csv(
    "YS_XGBoost_Metrics.csv",
    index=False
)

print(
    "YS_XGBoost_Metrics.csv"
)

print("\n" + "=" * 70)
print("XGBOOST TRAINING COMPLETE")
print("=" * 70)