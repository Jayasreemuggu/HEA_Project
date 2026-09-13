import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


print("=" * 70)
print("RANDOM FOREST - UTS MODEL 3")
print("Composition + Conditions + Physical/Microstructural Features")
print("=" * 70)


# =========================================================
# 1. LOAD DATA FROM POSTGRESQL
# =========================================================

from database.connection import engine

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
# 3. PHYSICAL / MICROSTRUCTURAL FEATURES
# =========================================================

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
# 5. CONVERT ELEMENTS TO NUMERIC
# =========================================================

for col in ELEMENTS:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# Missing elemental concentration = 0
df[ELEMENTS] = df[ELEMENTS].fillna(0)


# =========================================================
# 6. CONVERT NUMERIC FEATURES
# =========================================================

for col in NUMERIC_FEATURES:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# =========================================================
# 7. TARGET
# =========================================================

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

df = df[df[TARGET].notna()].copy()

print("\nFinal records:", len(df))


# =========================================================
# 8. COMPOSITION CHECK
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
# 9. FEATURES
# =========================================================

FEATURES = (
    ELEMENTS +
    NUMERIC_FEATURES +
    CATEGORICAL_FEATURES
)

X = df[FEATURES].copy()
y = df[TARGET].copy()


# =========================================================
# 10. GROUPED TRAIN / TEST SPLIT
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
    "Unique test compositions :",
    len(test_compositions)
)

print(
    "Composition overlap:",
    len(overlap)
)


# =========================================================
# 11. PREPROCESSING
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
# 12. RANDOM FOREST
# =========================================================

rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
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
        rf_model
    )
])


# =========================================================
# 13. TRAIN
# =========================================================

print("\nTraining UTS Model 3...")

model.fit(
    X_train,
    y_train
)

print("Training complete.")


# =========================================================
# 14. PREDICTION
# =========================================================

y_pred = model.predict(
    X_test
)


# =========================================================
# 15. METRICS
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
print("UTS MODEL 3 RESULTS")
print("=" * 70)

print(f"MAE  : {mae:.2f} MPa")
print(f"RMSE : {rmse:.2f} MPa")
print(f"R²   : {r2:.4f}")


# =========================================================
# 16. MODEL 1 / MODEL 2 COMPARISON
# =========================================================

model1_metrics = pd.read_csv(
    "UTS_RandomForest_Metrics.csv"
)

model2_metrics = pd.read_csv(
    "UTS_Model_Comparison.csv"
)

model1_r2 = model1_metrics["R2"].iloc[0]
model1_mae = model1_metrics["MAE_MPa"].iloc[0]
model1_rmse = model1_metrics["RMSE_MPa"].iloc[0]

model2_r2 = model2_metrics["R2"].iloc[1]
model2_mae = model2_metrics["MAE_MPa"].iloc[1]
model2_rmse = model2_metrics["RMSE_MPa"].iloc[1]


print("\n" + "=" * 70)
print("UTS MODEL COMPARISON")
print("=" * 70)

print(
    f"{'Model':<40}"
    f"{'R²':>10}"
    f"{'MAE':>12}"
    f"{'RMSE':>12}"
)

print(
    f"{'Model 1 - Composition Only':<40}"
    f"{model1_r2:>10.4f}"
    f"{model1_mae:>12.2f}"
    f"{model1_rmse:>12.2f}"
)

print(
    f"{'Model 2 - Composition + Conditions':<40}"
    f"{model2_r2:>10.4f}"
    f"{model2_mae:>12.2f}"
    f"{model2_rmse:>12.2f}"
)

print(
    f"{'Model 3 - Physical + Microstructural':<40}"
    f"{r2:>10.4f}"
    f"{mae:>12.2f}"
    f"{rmse:>12.2f}"
)


print("\nModel 3 vs Model 2:")

print(
    f"R² improvement  : {r2 - model2_r2:+.4f}"
)

print(
    f"MAE improvement : {model2_mae - mae:+.2f} MPa"
)

print(
    f"RMSE improvement: {model2_rmse - rmse:+.2f} MPa"
)


# =========================================================
# 17. SAVE MODEL
# =========================================================

joblib.dump(
    model,
    "UTS_RandomForest_Model3.pkl"
)


# =========================================================
# 18. SAVE PREDICTIONS
# =========================================================

predictions = pd.DataFrame({
    "Actual_UTS_MPa": y_test.values,
    "Predicted_UTS_MPa": y_pred,
    "Error_MPa": y_test.values - y_pred,
    "Absolute_Error_MPa":
        np.abs(y_test.values - y_pred)
})

predictions.to_csv(
    "UTS_Model3_Predictions.csv",
    index=False
)


# =========================================================
# 19. SAVE METRICS
# =========================================================

metrics = pd.DataFrame({
    "Model": [
        "RF - Composition Only",
        "RF - Composition + Conditions",
        "RF - Physical + Microstructural"
    ],
    "MAE_MPa": [
        model1_mae,
        model2_mae,
        mae
    ],
    "RMSE_MPa": [
        model1_rmse,
        model2_rmse,
        rmse
    ],
    "R2": [
        model1_r2,
        model2_r2,
        r2
    ]
})

metrics.to_csv(
    "UTS_Model_Comparison.csv",
    index=False
)


# =========================================================
# 20. FINAL
# =========================================================

print("\nSaved:")
print("UTS_RandomForest_Model3.pkl")
print("UTS_Model3_Predictions.csv")
print("UTS_Model_Comparison.csv")

print("\n" + "=" * 70)
print("UTS MODEL 3 COMPLETE")
print("=" * 70)