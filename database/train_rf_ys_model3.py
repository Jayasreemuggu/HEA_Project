import pandas as pd
import numpy as np
import joblib

from sqlalchemy import text
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from database.connection import engine


# ============================================================
# DAY 3 - YIELD STRENGTH MODEL 3
# ============================================================

print("=" * 70)
print("DAY 3 - YIELD STRENGTH MODEL 3")
print("=" * 70)


# ============================================================
# 1. ELEMENTAL FEATURES
# ============================================================
#
# PostgreSQL stores these as:
#
# al_at_pct
# fe_at_pct
# ni_at_pct
# etc.
#
# We will rename them to simple ML feature names later.
# ============================================================

ELEMENT_COLUMNS = [
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


# ============================================================
# 2. ADDITIONAL NUMERIC FEATURES
# ============================================================

NUMERIC_FEATURES = [
    "test_temperature_c",
    "vec",
    "atomic_size_mismatch",
    "mixing_enthalpy",
    "mixing_entropy",
    "density_exp_g_cm3",
    "density_calc_g_cm3",
    "grain_size_um",
    "predicted_solidus_c",
    "precipitate_size_nm",
    "precipitate_volume_pct",
    "matrix_volume_pct"
]


# ============================================================
# 3. CATEGORICAL FEATURES
# ============================================================

CATEGORICAL_FEATURES = [
    "test_type",
    "phase",
    "processing_method",
    "alloy_class",
    "equilibrium_condition",
    "single_multiphase",
    "precipitate_info"
]


# ============================================================
# 4. SQL QUERY
# ============================================================
#
# We use dataset_v1_raw directly.
#
# This avoids the record_id type mismatch between:
#
# dataset_v1_raw.record_id -> TEXT
# experimental_records.record_id -> INTEGER
#
# All Model 3 features are already present in dataset_v1_raw.
# ============================================================

query = """
SELECT *
FROM hea_mpea.dataset_v1_raw
WHERE ys_tensile_mpa IS NOT NULL
"""


# ============================================================
# 5. LOAD DATA
# ============================================================

print("\nLoading data from PostgreSQL...")

with engine.connect() as connection:

    df = pd.read_sql(
        text(query),
        connection
    )


print(
    "Loaded shape:",
    df.shape
)


# ============================================================
# 6. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = (
    ELEMENT_COLUMNS
    + NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
    + [
        "ys_tensile_mpa",
        "composition_canonical"
    ]
)


missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing_columns:

    print("\nERROR: Missing columns")
    print("=" * 60)

    for col in missing_columns:
        print(col)

    print("\nAvailable columns:")
    print("=" * 60)

    for col in df.columns:
        print(col)

    raise ValueError(
        "Required columns are missing."
    )


print(
    "\nAll required columns found."
)


# ============================================================
# 7. RENAME ELEMENT COLUMNS
# ============================================================
#
# Example:
#
# al_at_pct -> al
# fe_at_pct -> fe
# ni_at_pct -> ni
#
# ============================================================

ELEMENT_RENAME = {
    col: col.replace(
        "_at_pct",
        ""
    )
    for col in ELEMENT_COLUMNS
}


df = df.rename(
    columns=ELEMENT_RENAME
)


ELEMENTS = [
    col.replace(
        "_at_pct",
        ""
    )
    for col in ELEMENT_COLUMNS
]


# ============================================================
# 8. FEATURE LIST
# ============================================================

feature_columns = (
    ELEMENTS
    + NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


target_column = (
    "ys_tensile_mpa"
)


# ============================================================
# 9. SELECT DATA
# ============================================================

df_model = df[
    feature_columns
    + [
        target_column,
        "composition_canonical"
    ]
].copy()


print(
    "\nInitial Model 3 records:",
    len(df_model)
)


# ============================================================
# 10. PROCESS ELEMENTAL FEATURES
# ============================================================

print(
    "\nProcessing elemental composition..."
)


for col in ELEMENTS:

    df_model[col] = pd.to_numeric(
        df_model[col],
        errors="coerce"
    )


# Missing elemental values mean
# the element is absent.

df_model[ELEMENTS] = (
    df_model[ELEMENTS]
    .fillna(0)
)


# ============================================================
# 11. PROCESS NUMERIC FEATURES
# ============================================================

print(
    "Processing numeric features..."
)


for col in NUMERIC_FEATURES:

    df_model[col] = pd.to_numeric(
        df_model[col],
        errors="coerce"
    )


# ============================================================
# 12. PROCESS CATEGORICAL FEATURES
# ============================================================

print(
    "Processing categorical features..."
)


for col in CATEGORICAL_FEATURES:

    df_model[col] = (
        df_model[col]
        .astype("object")
    )


# ============================================================
# 13. PROCESS TARGET
# ============================================================

df_model[target_column] = pd.to_numeric(
    df_model[target_column],
    errors="coerce"
)


# ============================================================
# 14. REMOVE INVALID TARGET/GROUP ROWS
# ============================================================

df_model = df_model.dropna(
    subset=[
        target_column,
        "composition_canonical"
    ]
).reset_index(drop=True)


print(
    "\nFinal YS records:",
    len(df_model)
)


# ============================================================
# 15. COMPOSITION QUALITY CHECK
# ============================================================

composition_sum = (
    df_model[ELEMENTS]
    .sum(axis=1)
)


print(
    "\nComposition sum statistics:"
)

print(
    composition_sum.describe()
)


invalid_composition = (
    (composition_sum < 99)
    |
    (composition_sum > 101)
)


print(
    "\nComposition sums outside 99-101 at.%:",
    invalid_composition.sum()
)


# ============================================================
# 16. TARGET STATISTICS
# ============================================================

print(
    "\nYield Strength Statistics"
)

print(
    "=" * 60
)

print(
    df_model[target_column].describe()
)


# ============================================================
# 17. CREATE X, y AND GROUPS
# ============================================================

X = df_model[
    feature_columns
].copy()


y = df_model[
    target_column
].copy()


groups = df_model[
    "composition_canonical"
].copy()


# ============================================================
# 18. GROUPED TRAIN/TEST SPLIT
# ============================================================

print(
    "\nCreating grouped train/test split..."
)


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


X_train = X.iloc[
    train_idx
].copy()


X_test = X.iloc[
    test_idx
].copy()


y_train = y.iloc[
    train_idx
].copy()


y_test = y.iloc[
    test_idx
].copy()


groups_train = groups.iloc[
    train_idx
]


groups_test = groups.iloc[
    test_idx
]


# ============================================================
# 19. SPLIT INFORMATION
# ============================================================

print(
    "\nTraining records:",
    len(X_train)
)


print(
    "Testing records:",
    len(X_test)
)


print(
    "Unique train compositions:",
    groups_train.nunique()
)


print(
    "Unique test compositions:",
    groups_test.nunique()
)


# ============================================================
# 20. CHECK COMPOSITION LEAKAGE
# ============================================================

overlap = (
    set(groups_train)
    .intersection(
        set(groups_test)
    )
)


print(
    "Composition overlap:",
    len(overlap)
)


if len(overlap) == 0:

    print(
        "No composition leakage detected."
    )

else:

    print(
        "WARNING: Composition leakage detected!"
    )


# ============================================================
# 21. NUMERIC PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


# ============================================================
# 22. CATEGORICAL PREPROCESSING
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
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


# ============================================================
# 23. COLUMN TRANSFORMER
# ============================================================

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


# ============================================================
# 24. RANDOM FOREST
# ============================================================

model = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 25. COMPLETE PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            model
        )
    ]
)


# ============================================================
# 26. TRAIN
# ============================================================

print(
    "\nTraining Model 3..."
)


pipeline.fit(
    X_train,
    y_train
)


print(
    "Training completed."
)


# ============================================================
# 27. PREDICTION
# ============================================================

print(
    "\nGenerating predictions..."
)


y_pred = pipeline.predict(
    X_test
)


# ============================================================
# 28. CALCULATE METRICS
# ============================================================

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


# ============================================================
# 29. MODEL 3 RESULTS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "MODEL 3 RESULTS"
)

print(
    "=" * 70
)


print(
    f"MAE  : {mae:.2f} MPa"
)


print(
    f"RMSE : {rmse:.2f} MPa"
)


print(
    f"R²   : {r2:.4f}"
)


# ============================================================
# 30. MODEL 2 BASELINE
# ============================================================

MODEL2_R2 = 0.6159
MODEL2_MAE = 251.74
MODEL2_RMSE = 330.83


# ============================================================
# 31. MODEL COMPARISON
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "MODEL 2 vs MODEL 3"
)

print(
    "=" * 70
)


print(
    f"\nModel 2 R² : {MODEL2_R2:.4f}"
)


print(
    f"Model 3 R² : {r2:.4f}"
)


print(
    f"R² change  : {r2 - MODEL2_R2:+.4f}"
)


print(
    f"\nModel 2 MAE : {MODEL2_MAE:.2f} MPa"
)


print(
    f"Model 3 MAE : {mae:.2f} MPa"
)


print(
    f"MAE change  : {mae - MODEL2_MAE:+.2f} MPa"
)


print(
    f"\nModel 2 RMSE : {MODEL2_RMSE:.2f} MPa"
)


print(
    f"Model 3 RMSE : {rmse:.2f} MPa"
)


print(
    f"RMSE change  : {rmse - MODEL2_RMSE:+.2f} MPa"
)


# ============================================================
# 32. IMPROVEMENT PERCENTAGES
# ============================================================

r2_relative_change = (
    (r2 - MODEL2_R2)
    /
    abs(MODEL2_R2)
) * 100


mae_improvement = (
    (MODEL2_MAE - mae)
    /
    MODEL2_MAE
) * 100


rmse_improvement = (
    (MODEL2_RMSE - rmse)
    /
    MODEL2_RMSE
) * 100


print(
    "\n" + "=" * 70
)

print(
    "IMPROVEMENT"
)

print(
    "=" * 70
)


print(
    f"R² relative change : "
    f"{r2_relative_change:+.2f}%"
)


print(
    f"MAE improvement    : "
    f"{mae_improvement:+.2f}%"
)


print(
    f"RMSE improvement   : "
    f"{rmse_improvement:+.2f}%"
)


# ============================================================
# 33. SAVE MODEL
# ============================================================

model_path = (
    "YS_RandomForest_Model3.pkl"
)


joblib.dump(
    pipeline,
    model_path
)


print(
    f"\nSaved model: {model_path}"
)


# ============================================================
# 34. SAVE PREDICTIONS
# ============================================================

predictions = X_test.copy()


predictions[
    "Actual_YS_MPa"
] = y_test.values


predictions[
    "Predicted_YS_MPa"
] = y_pred


predictions[
    "Error_MPa"
] = (
    predictions[
        "Actual_YS_MPa"
    ]
    -
    predictions[
        "Predicted_YS_MPa"
    ]
)


predictions.to_csv(
    "YS_Model3_Predictions.csv",
    index=False
)


print(
    "Saved predictions: "
    "YS_Model3_Predictions.csv"
)


# ============================================================
# 35. SAVE METRICS
# ============================================================

metrics = pd.DataFrame({

    "Model": [
        "Random Forest Model 3"
    ],

    "MAE_MPa": [
        mae
    ],

    "RMSE_MPa": [
        rmse
    ],

    "R2": [
        r2
    ],

    "Train_Records": [
        len(X_train)
    ],

    "Test_Records": [
        len(X_test)
    ],

    "Unique_Train_Compositions": [
        groups_train.nunique()
    ],

    "Unique_Test_Compositions": [
        groups_test.nunique()
    ],

    "Composition_Overlap": [
        len(overlap)
    ]
})


metrics.to_csv(
    "YS_Model3_Metrics.csv",
    index=False
)


print(
    "Saved metrics: "
    "YS_Model3_Metrics.csv"
)


# ============================================================
# 36. FINAL STATUS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "DAY 3 MODEL 3 COMPLETE"
)

print(
    "=" * 70
)

print(
    "\nNext step:"
)

print(
    "Feature importance + "
    "Actual vs Predicted + "
    "Error analysis"
)

print(
    "=" * 70
)