import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

from database.connection import engine


print("=" * 70)
print("XGBOOST - UTS MODEL")
print("Composition + Conditions + Physical/Microstructural Features")
print("=" * 70)


# =========================================================
# 1. LOAD DATA
# =========================================================

query = """
SELECT *
FROM hea_mpea.dataset_v1_raw
WHERE uts_tensile_mpa IS NOT NULL
"""

print("\nLoading UTS data from PostgreSQL...")

df = pd.read_sql(query, engine)

print("Original shape:", df.shape)


# =========================================================
# 2. ELEMENT FEATURES
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


# =========================================================
# 3. AVAILABLE PHYSICAL / MICROSTRUCTURAL FEATURES
# =========================================================

NUMERIC_FEATURES = [
    "test_temperature_c",
    "density_exp_g_cm3",
    "density_calc_g_cm3",
    "grain_size_um",
    "precipitate_size_nm",
    "matrix_volume_pct"
]


# =========================================================
# 4. CATEGORICAL FEATURES
# =========================================================

CATEGORICAL_FEATURES = [
    "test_type",
    "phase",
    "processing_method",
    "alloy_class",
    "equilibrium_condition",
    "single_multiphase",
    "precipitate_info"
]


TARGET = "uts_tensile_mpa"


# =========================================================
# 5. DATA TYPE CONVERSION
# =========================================================

for col in ELEMENTS:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# Missing elemental concentration = 0
df[ELEMENTS] = df[ELEMENTS].fillna(0)

for col in NUMERIC_FEATURES:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

df = df[df[TARGET].notna()].copy()

print("\nFinal records:", len(df))


# =========================================================
# 6. COMPOSITION CHECK
# =========================================================

composition_sum = df[ELEMENTS].sum(axis=1)

print("\nComposition sum statistics:")
print(composition_sum.describe())

invalid = (
    (composition_sum < 99) |
    (composition_sum > 101)
).sum()

print(
    "\nRecords with composition sum outside 99-101:",
    invalid
)


# =========================================================
# 7. FEATURES / TARGET
# =========================================================

FEATURES = (
    ELEMENTS +
    NUMERIC_FEATURES +
    CATEGORICAL_FEATURES
)

X = df[FEATURES].copy()
y = df[TARGET].copy()


# =========================================================
# 8. GROUPED TRAIN / TEST SPLIT
# =========================================================

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

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()

y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()


train_compositions = set(
    df.iloc[train_idx]["composition_canonical"]
)

test_compositions = set(
    df.iloc[test_idx]["composition_canonical"]
)

overlap = train_compositions.intersection(
    test_compositions
)


print("\n" + "=" * 70)
print("GROUPED SPLIT")
print("=" * 70)

print("Training records:", len(X_train))
print("Testing records :", len(X_test))

print(
    "Unique train compositions:",
    len(train_compositions)
)

print(
    "Unique test compositions:",
    len(test_compositions)
)

print(
    "Composition overlap:",
    len(overlap)
)


# =========================================================
# 9. PREPROCESSING
# =========================================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])

categorical_pipeline = Pipeline([
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
])

preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        NUMERIC_FEATURES
    ),
    (
        "categorical",
        categorical_pipeline,
        CATEGORICAL_FEATURES
    )
])


# =========================================================
# 10. XGBOOST
# =========================================================

xgb_model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    reg_alpha=0,
    reg_lambda=1,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)


model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        xgb_model
    )
])


# =========================================================
# 11. TRAIN
# =========================================================

print("\nTraining XGBoost...")

model.fit(
    X_train,
    y_train
)

print("Training complete.")


# =========================================================
# 12. PREDICTION
# =========================================================

y_pred = model.predict(
    X_test
)


# =========================================================
# 13. METRICS
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
print("UTS XGBOOST RESULTS")
print("=" * 70)

print(f"MAE  : {mae:.2f} MPa")
print(f"RMSE : {rmse:.2f} MPa")
print(f"R²   : {r2:.4f}")


# =========================================================
# 14. COMPARE WITH PREVIOUS MODELS
# =========================================================

comparison = pd.read_csv(
    "UTS_Model_Comparison.csv"
)

print("\n" + "=" * 70)
print("UTS MODEL COMPARISON")
print("=" * 70)

print(
    comparison.to_string(
        index=False
    )
)

print("\nXGBoost vs ExtraTrees:")

extra_r2 = comparison["R2"].iloc[3]
extra_mae = comparison["MAE_MPa"].iloc[3]
extra_rmse = comparison["RMSE_MPa"].iloc[3]

print(
    f"R² improvement  : {r2 - extra_r2:+.4f}"
)

print(
    f"MAE improvement : {extra_mae - mae:+.2f} MPa"
)

print(
    f"RMSE improvement: {extra_rmse - rmse:+.2f} MPa"
)


# =========================================================
# 15. ADD XGBOOST TO COMPARISON
# =========================================================

comparison = pd.concat(
    [
        comparison,
        pd.DataFrame({
            "Model": [
                "XGBoost - Physical + Microstructural"
            ],
            "MAE_MPa": [mae],
            "RMSE_MPa": [rmse],
            "R2": [r2]
        })
    ],
    ignore_index=True
)


# =========================================================
# 16. SAVE MODEL
# =========================================================

joblib.dump(
    model,
    "UTS_XGBoost_Model.pkl"
)


# =========================================================
# 17. SAVE PREDICTIONS
# =========================================================

predictions = pd.DataFrame({
    "Actual_UTS_MPa": y_test.values,
    "Predicted_UTS_MPa": y_pred,
    "Error_MPa": y_test.values - y_pred,
    "Absolute_Error_MPa":
        np.abs(y_test.values - y_pred)
})

predictions.to_csv(
    "UTS_XGBoost_Predictions.csv",
    index=False
)


# =========================================================
# 18. SAVE COMPARISON
# =========================================================

comparison.to_csv(
    "UTS_Model_Comparison.csv",
    index=False
)


# =========================================================
# 19. FINAL
# =========================================================

print("\nSaved:")
print("UTS_XGBoost_Model.pkl")
print("UTS_XGBoost_Predictions.csv")
print("UTS_Model_Comparison.csv")

print("\n" + "=" * 70)
print("UTS XGBOOST COMPLETE")
print("=" * 70)