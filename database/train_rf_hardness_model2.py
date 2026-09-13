# ================================================================
# HARDNESS MODEL 2
# Random Forest - Composition + Experimental Conditions
# ================================================================

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

from database.connection import engine


# ================================================================
# CONFIGURATION
# ================================================================

DATASET_PATH = "Hardness_ML_Dataset.csv"

MODEL_OUTPUT = "Hardness_RandomForest_Model2.pkl"
PREDICTION_OUTPUT = "Hardness_Model2_Predictions.csv"
METRICS_OUTPUT = "Hardness_Model2_Metrics.csv"
IMPORTANCE_OUTPUT = "Hardness_Model2_Feature_Importance.csv"

TARGET = "hardness_hv"

element_features = [
    "al", "b", "c", "co", "cr", "cu", "fe", "mn",
    "mo", "nb", "ni", "si", "ta", "ti", "v", "w",
    "zr", "ag", "ca", "ga", "hf", "i", "li", "mg",
    "nd", "o", "pd", "re", "ru", "s", "sc", "sn",
    "t", "y", "zn"
]

numeric_features = [
    "test_temperature_c"
]

categorical_features = [
    "test_type",
    "phase",
    "processing_method",
    "alloy_class"
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
print("RANDOM FOREST - HARDNESS MODEL 2")
print("Composition + Experimental Conditions")
print("=" * 70)


# ================================================================
# LOAD CLEANED HARDNESS DATASET
# ================================================================

clean_df = pd.read_csv(DATASET_PATH)

print(
    "\nCleaned Hardness dataset:",
    clean_df.shape
)


# ================================================================
# LOAD HARDNESS RECORDS FROM POSTGRESQL
# ================================================================

pg_query = """
SELECT *
FROM hea_mpea.dataset_v1_raw
WHERE hardness_hv IS NOT NULL
"""

pg_df = pd.read_sql(
    pg_query,
    engine
)

print(
    "PostgreSQL Hardness data:",
    pg_df.shape
)


# ================================================================
# MATCH THE EXACT 717 RECORDS
# ================================================================

print("\nChecking record_id availability...")

if (
    "record_id" in clean_df.columns
    and "record_id" in pg_df.columns
):

    print(
        "record_id available in both datasets."
    )

    df = pg_df[
        pg_df["record_id"].isin(
            clean_df["record_id"]
        )
    ].copy()

else:

    print(
        "record_id not available in both datasets."
    )

    match_columns = [
        "composition_canonical",
        "hardness_hv"
    ]

    def make_match_key(data):

        temp = data[
            match_columns
        ].copy()

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

    clean_df["_match_key"] = (
        make_match_key(clean_df)
    )

    pg_df["_match_key"] = (
        make_match_key(pg_df)
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
    "\nMatched PostgreSQL records:",
    len(df)
)

print(
    "Expected cleaned records :",
    len(clean_df)
)

if len(df) != len(clean_df):

    raise ValueError(
        "ERROR: Exact 717-record matching failed."
    )

print(
    "\nFinal Model 2 records:",
    len(df)
)


# ================================================================
# ADD COMPOSITION FEATURES
# ================================================================

print("\nAdding composition features...")

composition_query = """
SELECT
    a.composition_canonical,

    c.al,
    c.b,
    c.c,
    c.co,
    c.cr,
    c.cu,
    c.fe,
    c.mn,
    c.mo,
    c.nb,
    c.ni,
    c.si,
    c.ta,
    c.ti,
    c.v,
    c.w,
    c.zr,
    c.ag,
    c.ca,
    c.ga,
    c.hf,
    c.i,
    c.li,
    c.mg,
    c.nd,
    c.o,
    c.pd,
    c.re,
    c.ru,
    c.s,
    c.sc,
    c.sn,
    c.t,
    c.y,
    c.zn

FROM hea_mpea.alloys a

LEFT JOIN hea_mpea.compositions c
    ON a.alloy_id = c.alloy_id
"""

composition_df = pd.read_sql(
    composition_query,
    engine
)

print(
    "Composition table shape:",
    composition_df.shape
)


# ================================================================
# REMOVE DUPLICATE COMPOSITION ENTRIES
# ================================================================

composition_df = composition_df.drop_duplicates(
    subset=["composition_canonical"],
    keep="first"
)

print(
    "Unique composition table shape:",
    composition_df.shape
)


# ================================================================
# MERGE COMPOSITION WITH EXPERIMENTAL DATA
# ================================================================

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
# CHECK MERGE SIZE
# ================================================================

if len(df) != 717:

    raise ValueError(
        f"ERROR: Composition merge changed record count. "
        f"Expected 717, got {len(df)}"
    )


print(
    "Composition features added successfully."
)


# ================================================================
# CONVERT COMPOSITION TO NUMERIC
# ================================================================

df[element_features] = (
    df[element_features]
    .apply(
        pd.to_numeric,
        errors="coerce"
    )
)


# Missing elemental values = 0
df[element_features] = (
    df[element_features]
    .fillna(0)
)


# ================================================================
# COMPOSITION VALIDATION
# ================================================================

composition_sum = (
    df[element_features]
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
).sum()

print(
    "\nRecords with composition sum outside 99-101:",
    invalid_composition
)

if invalid_composition > 0:

    raise ValueError(
        "ERROR: Invalid composition detected."
    )


# ================================================================
# PREPARE TARGET
# ================================================================

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

df = df.dropna(
    subset=[TARGET]
).copy()


# ================================================================
# CREATE FEATURES AND TARGET
# ================================================================

X = df[features].copy()

y = df[TARGET].copy()


print(
    "\nFeature matrix:",
    X.shape
)

print(
    "Target:",
    y.shape
)


# ================================================================
# GROUPED TRAIN / TEST SPLIT
# ================================================================

print("\n" + "=" * 70)
print("GROUPED SPLIT")
print("=" * 70)

groups = df[
    "composition_canonical"
].astype(str)


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


print(
    "Training records:",
    len(X_train)
)

print(
    "Testing records :",
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

overlap = (
    set(groups_train)
    &
    set(groups_test)
)

print(
    "Composition overlap:",
    len(overlap)
)

if len(overlap) != 0:

    raise ValueError(
        "ERROR: Composition leakage detected!"
    )


# ================================================================
# PREPROCESSING
# ================================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


categorical_transformer = Pipeline(
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


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            element_features
            + numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ================================================================
# RANDOM FOREST MODEL
# ================================================================

model = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    max_features="sqrt",
    n_jobs=-1
)


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


# ================================================================
# TRAIN
# ================================================================

print(
    "\nTraining Random Forest Model 2..."
)

pipeline.fit(
    X_train,
    y_train
)

print(
    "Training complete."
)


# ================================================================
# PREDICTION
# ================================================================

y_pred = pipeline.predict(
    X_test
)


# ================================================================
# METRICS
# ================================================================

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
print("HARDNESS MODEL 2 RESULTS")
print("=" * 70)

print(
    f"MAE  : {mae:.4f} HV"
)

print(
    f"RMSE : {rmse:.4f} HV"
)

print(
    f"R²   : {r2:.4f}"
)


# ================================================================
# LOAD MODEL 1 METRICS
# ================================================================

model1_metrics_path = (
    "Hardness_RandomForest_Metrics.csv"
)

try:

    model1_df = pd.read_csv(
        model1_metrics_path
    )

    model1_r2 = float(
        model1_df[
            model1_df["Metric"] == "R2"
        ]["Value"].iloc[0]
    )

    model1_mae = float(
        model1_df[
            model1_df["Metric"] == "MAE"
        ]["Value"].iloc[0]
    )

    model1_rmse = float(
        model1_df[
            model1_df["Metric"] == "RMSE"
        ]["Value"].iloc[0]
    )

    print("\n" + "=" * 70)
    print("MODEL 2 VS MODEL 1")
    print("=" * 70)

    print(
        f"R² improvement   : {r2 - model1_r2:+.4f}"
    )

    print(
        f"MAE improvement  : {model1_mae - mae:+.4f} HV"
    )

    print(
        f"RMSE improvement : {model1_rmse - rmse:+.4f} HV"
    )

except Exception as e:

    print(
        "\nModel 1 comparison unavailable:",
        e
    )


# ================================================================
# ERROR ANALYSIS
# ================================================================

y_test_numeric = pd.to_numeric(
    y_test,
    errors="coerce"
)

y_pred_numeric = pd.to_numeric(
    y_pred,
    errors="coerce"
)

errors = (
    y_test_numeric.values
    -
    y_pred_numeric
)

print("\n" + "=" * 70)
print("ERROR ANALYSIS")
print("=" * 70)

print(
    f"Mean Error : {np.mean(errors):.4f} HV"
)

print(
    f"Std Error  : {np.std(errors):.4f} HV"
)

print(
    f"Min Error  : {np.min(errors):.4f} HV"
)

print(
    f"Max Error  : {np.max(errors):.4f} HV"
)


# ================================================================
# SAVE PREDICTIONS
# ================================================================

prediction_df = pd.DataFrame(
    {
        "record_id":
            df.iloc[test_idx]["record_id"].values,

        "composition_canonical":
            groups_test.values,

        "actual_hardness_hv":
            y_test_numeric.values,

        "predicted_hardness_hv":
            y_pred_numeric,

        "error_hv":
            errors
    }
)

prediction_df.to_csv(
    PREDICTION_OUTPUT,
    index=False
)


# ================================================================
# SAVE METRICS
# ================================================================

metrics_df = pd.DataFrame(
    {
        "Metric": [
            "MAE",
            "RMSE",
            "R2",
            "Training_Records",
            "Testing_Records",
            "Unique_Train_Compositions",
            "Unique_Test_Compositions",
            "Composition_Overlap"
        ],

        "Value": [
            mae,
            rmse,
            r2,
            len(X_train),
            len(X_test),
            groups_train.nunique(),
            groups_test.nunique(),
            len(overlap)
        ]
    }
)

metrics_df.to_csv(
    METRICS_OUTPUT,
    index=False
)


# ================================================================
# FEATURE IMPORTANCE
# ================================================================

try:

    fitted_preprocessor = (
        pipeline.named_steps[
            "preprocessor"
        ]
    )

    fitted_model = (
        pipeline.named_steps[
            "model"
        ]
    )

    feature_names = (
        fitted_preprocessor
        .get_feature_names_out()
    )

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance":
                fitted_model.feature_importances_
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    importance_df.to_csv(
        IMPORTANCE_OUTPUT,
        index=False
    )

    print(
        "\nTop 20 Feature Importances:"
    )

    print(
        importance_df.head(20).to_string(
            index=False
        )
    )

except Exception as e:

    print(
        "\nFeature importance could not be saved:",
        e
    )


# ================================================================
# SAVE MODEL
# ================================================================

joblib.dump(
    pipeline,
    MODEL_OUTPUT
)


# ================================================================
# FINAL OUTPUT
# ================================================================

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    MODEL_OUTPUT
)

print(
    PREDICTION_OUTPUT
)

print(
    METRICS_OUTPUT
)

print(
    IMPORTANCE_OUTPUT
)

print("\nHardness Model 2 completed successfully.")