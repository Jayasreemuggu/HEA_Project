import pandas as pd
import numpy as np
import joblib

from sqlalchemy import text

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from database.connection import engine


# ============================================================
# DAY 3 - MODEL 5
# EXTRA TREES REGRESSOR
# ============================================================

print("=" * 70)
print("DAY 3 - YIELD STRENGTH MODEL 5 - EXTRA TREES")
print("=" * 70)


# ============================================================
# 1. ELEMENTAL FEATURES
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
# 2. NUMERIC FEATURES
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
    "precipitate_size_nm",
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
# 4. LOAD DATA
# ============================================================

query = """
SELECT *
FROM hea_mpea.dataset_v1_raw
WHERE ys_tensile_mpa IS NOT NULL
"""


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
# 5. CHECK REQUIRED COLUMNS
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
# 6. RENAME ELEMENT COLUMNS
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
# 7. FEATURE COLUMNS
# ============================================================

feature_columns = (
    ELEMENTS
    + NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


target_column = "ys_tensile_mpa"


# ============================================================
# 8. CREATE MODEL DATASET
# ============================================================

df_model = df[
    feature_columns
    + [
        target_column,
        "composition_canonical"
    ]
].copy()


print(
    "\nInitial Model 5 records:",
    len(df_model)
)


# ============================================================
# 9. PROCESS ELEMENTAL FEATURES
# ============================================================

print(
    "\nProcessing elemental composition..."
)


for col in ELEMENTS:

    df_model[col] = pd.to_numeric(
        df_model[col],
        errors="coerce"
    )


# Missing elemental value = element absent

df_model[ELEMENTS] = (
    df_model[ELEMENTS]
    .fillna(0)
)


# ============================================================
# 10. PROCESS NUMERIC FEATURES
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
# 11. PROCESS CATEGORICAL FEATURES
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
# 12. PROCESS TARGET
# ============================================================

df_model[target_column] = pd.to_numeric(
    df_model[target_column],
    errors="coerce"
)


# ============================================================
# 13. REMOVE INVALID TARGET/GROUP
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
# 14. COMPOSITION CHECK
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
# 15. TARGET STATISTICS
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
# 16. X / y / GROUPS
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
# 17. SAME GROUPED SPLIT AS PREVIOUS MODELS
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
# 18. SPLIT INFORMATION
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
# 19. LEAKAGE CHECK
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
# 20. NUMERIC PIPELINE
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
# 21. CATEGORICAL PIPELINE
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
# 22. PREPROCESSOR
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
# 23. EXTRA TREES MODEL
# ============================================================

model = ExtraTreesRegressor(

    n_estimators=500,

    max_depth=None,

    min_samples_split=2,

    min_samples_leaf=1,

    max_features=1.0,

    random_state=42,

    n_jobs=-1
)


# ============================================================
# 24. PIPELINE
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
# 25. TRAIN
# ============================================================

print(
    "\nTraining ExtraTrees Model..."
)


pipeline.fit(
    X_train,
    y_train
)


print(
    "Training completed."
)


# ============================================================
# 26. PREDICTION
# ============================================================

print(
    "\nGenerating predictions..."
)


y_pred = pipeline.predict(
    X_test
)


# ============================================================
# 27. METRICS
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
# 28. RESULTS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "EXTRATREES RESULTS"
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
# 29. COMPARE AGAINST CURRENT BEST RF
# ============================================================

RF_R2 = 0.636881
RF_MAE = 240.729998
RF_RMSE = 321.675549


print(
    "\n" + "=" * 70
)

print(
    "EXTRATREES vs RANDOM FOREST MODEL 4"
)

print(
    "=" * 70
)


print(
    f"\nRandom Forest R² : {RF_R2:.4f}"
)


print(
    f"ExtraTrees R²    : {r2:.4f}"
)


print(
    f"R² change        : {r2 - RF_R2:+.4f}"
)


print(
    f"\nRandom Forest MAE : {RF_MAE:.2f} MPa"
)


print(
    f"ExtraTrees MAE    : {mae:.2f} MPa"
)


print(
    f"MAE change        : {mae - RF_MAE:+.2f} MPa"
)


print(
    f"\nRandom Forest RMSE : {RF_RMSE:.2f} MPa"
)


print(
    f"ExtraTrees RMSE    : {rmse:.2f} MPa"
)


print(
    f"RMSE change        : {rmse - RF_RMSE:+.2f} MPa"
)


# ============================================================
# 30. SAVE MODEL
# ============================================================

model_path = (
    "YS_ExtraTrees_Model.pkl"
)


joblib.dump(
    pipeline,
    model_path
)


print(
    f"\nSaved model: {model_path}"
)


# ============================================================
# 31. SAVE PREDICTIONS
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


predictions[
    "Absolute_Error_MPa"
] = np.abs(
    predictions[
        "Error_MPa"
    ]
)


predictions.to_csv(
    "YS_ExtraTrees_Predictions.csv",
    index=False
)


print(
    "Saved predictions:"
)

print(
    "YS_ExtraTrees_Predictions.csv"
)


# ============================================================
# 32. SAVE METRICS
# ============================================================

metrics = pd.DataFrame({

    "Model": [
        "ExtraTrees"
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
    "YS_ExtraTrees_Metrics.csv",
    index=False
)


print(
    "Saved metrics:"
)

print(
    "YS_ExtraTrees_Metrics.csv"
)


# ============================================================
# 33. FINAL
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "DAY 3 - EXTRATREES COMPLETE"
)

print(
    "=" * 70
)

print(
    "\nNext:"
)

print(
    "Compare ExtraTrees with Random Forest."
)

print(
    "If ExtraTrees is better, evaluate its errors."
)

print(
    "Then test XGBoost."
)

print(
    "=" * 70
)