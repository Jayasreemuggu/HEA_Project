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
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from database.connection import engine


# ============================================================
# DAY 3 - YIELD STRENGTH MODEL 4
# CLEAN RANDOM FOREST MODEL
# ============================================================

print("=" * 70)
print("DAY 3 - YIELD STRENGTH MODEL 4")
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
#
# Removed:
#   predicted_solidus_c
#   precipitate_volume_pct
#
# Both contain no observed values.
# ============================================================

NUMERIC_FEATURES = [
    "test_temperature_c",

    # Composition-derived descriptors
    "vec",
    "atomic_size_mismatch",
    "mixing_enthalpy",
    "mixing_entropy",

    # Density
    "density_exp_g_cm3",
    "density_calc_g_cm3",

    # Microstructure
    "grain_size_um",

    # Precipitate information
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
# 4. LOAD DATA FROM POSTGRESQL
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
    "\nInitial Model 4 records:",
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


# Missing element = element absent
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
# 13. REMOVE INVALID TARGET / GROUP
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
# 14. COMPOSITION QUALITY CHECK
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

print("=" * 60)

print(
    df_model[target_column].describe()
)


# ============================================================
# 16. CREATE X / y / GROUPS
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
# 17. GROUPED TRAIN / TEST SPLIT
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
# 22. COLUMN TRANSFORMER
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
# 23. RANDOM FOREST
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
    "\nTraining Model 4..."
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
# 28. MODEL 4 RESULTS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "MODEL 4 RESULTS"
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
# 29. COMPARE ALL MODELS
# ============================================================

MODEL1_R2 = 0.2095
MODEL1_MAE = 354.81
MODEL1_RMSE = 474.60

MODEL2_R2 = 0.6159
MODEL2_MAE = 251.74
MODEL2_RMSE = 330.83

MODEL3_R2 = 0.6369
MODEL3_MAE = 240.73
MODEL3_RMSE = 321.68


comparison = pd.DataFrame({

    "Model": [
        "Model 1 - Composition",
        "Model 2 - Conditions",
        "Model 3 - Physical Features",
        "Model 4 - Clean Features"
    ],

    "R2": [
        MODEL1_R2,
        MODEL2_R2,
        MODEL3_R2,
        r2
    ],

    "MAE_MPa": [
        MODEL1_MAE,
        MODEL2_MAE,
        MODEL3_MAE,
        mae
    ],

    "RMSE_MPa": [
        MODEL1_RMSE,
        MODEL2_RMSE,
        MODEL3_RMSE,
        rmse
    ]
})


print(
    "\n" + "=" * 70
)

print(
    "MODEL COMPARISON"
)

print(
    "=" * 70
)


print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# 30. MODEL 4 VS MODEL 3
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "MODEL 4 vs MODEL 3"
)

print(
    "=" * 70
)


print(
    f"\nR² change  : "
    f"{r2 - MODEL3_R2:+.4f}"
)


print(
    f"MAE change : "
    f"{mae - MODEL3_MAE:+.2f} MPa"
)


print(
    f"RMSE change: "
    f"{rmse - MODEL3_RMSE:+.2f} MPa"
)


# ============================================================
# 31. SAVE MODEL
# ============================================================

joblib.dump(
    pipeline,
    "YS_RandomForest_Model4.pkl"
)


print(
    "\nSaved model:"
)

print(
    "YS_RandomForest_Model4.pkl"
)


# ============================================================
# 32. SAVE PREDICTIONS
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
    "YS_Model4_Predictions.csv",
    index=False
)


print(
    "Saved predictions:"
)

print(
    "YS_Model4_Predictions.csv"
)


# ============================================================
# 33. SAVE METRICS
# ============================================================

metrics = pd.DataFrame({

    "Model": [
        "Random Forest Model 4"
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
    "YS_Model4_Metrics.csv",
    index=False
)


comparison.to_csv(
    "YS_Model_Comparison.csv",
    index=False
)


print(
    "Saved metrics:"
)

print(
    "YS_Model4_Metrics.csv"
)

print(
    "YS_Model_Comparison.csv"
)


# ============================================================
# 34. FINAL STATUS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "MODEL 4 COMPLETE"
)

print(
    "=" * 70
)

print(
    "\nNext:"
)

print(
    "Compare Model 4 with Model 3."
)

print(
    "Then test ExtraTrees/XGBoost."
)

print(
    "=" * 70
)