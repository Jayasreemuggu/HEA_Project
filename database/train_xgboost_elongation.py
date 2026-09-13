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
print("XGBOOST - ELONGATION MODEL 5")
print("Composition + Conditions + Physical/Microstructural Features")
print("=" * 70)


# ============================================================
# 1. LOAD CLEANED DATASET
# ============================================================

clean_df = pd.read_csv(
    "Elongation_ML_Dataset.csv"
)

print("\nCleaned Elongation dataset:", clean_df.shape)


# ============================================================
# 2. LOAD DATA FROM POSTGRESQL
# ============================================================

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

print(
    "PostgreSQL joined data:",
    pg_df.shape
)


# ============================================================
# 3. MATCH EXACT 1162 CLEANED RECORDS
# ============================================================

match_columns = [
    "composition_canonical",
    "test_temperature_c",
    "test_type",
    "phase",
    "processing_method",
    "elongation_tensile_pct"
]

print("\nMatching columns:")
print(match_columns)


def make_match_key(df):

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


clean_df["_match_key"] = make_match_key(
    clean_df
)

pg_df["_match_key"] = make_match_key(
    pg_df
)


pg_unique = pg_df.drop_duplicates(
    subset="_match_key",
    keep="first"
).copy()


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
        "ERROR: Exact 1162-record matching failed."
    )


print(
    "\nFinal Model 5 records:",
    len(df)
)


# ============================================================
# 4. FEATURES
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

y = pd.to_numeric(
    df["elongation_tensile_pct"],
    errors="coerce"
)


# ============================================================
# 5. NUMERIC CONVERSION
# ============================================================

for col in (
    element_features
    + numeric_features
):

    X[col] = pd.to_numeric(
        X[col],
        errors="coerce"
    )


# ============================================================
# 6. COMPOSITION CHECK
# ============================================================

composition_sum = X[
    element_features
].sum(axis=1)

print("\nComposition sum statistics:")
print(composition_sum.describe())

invalid = (
    (composition_sum < 99)
    |
    (composition_sum > 101)
).sum()

print(
    "\nRecords with composition sum outside 99-101:",
    invalid
)


X[element_features] = X[
    element_features
].fillna(0)


# ============================================================
# 7. GROUPED TRAIN / TEST SPLIT
# ============================================================

groups = df[
    "composition_canonical"
].fillna("UNKNOWN")


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


X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

groups_train = groups.iloc[train_idx]
groups_test = groups.iloc[test_idx]


print("\n" + "=" * 70)
print("GROUPED SPLIT")
print("=" * 70)

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
        "Composition leakage detected!"
    )


# ============================================================
# 8. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    )
])


categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),
    (
        "onehot",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])


preprocessor = ColumnTransformer([
    (
        "num",
        numeric_pipeline,
        numeric_features
    ),
    (
        "cat",
        categorical_pipeline,
        categorical_features
    )
])


# ============================================================
# 9. XGBOOST MODEL
# ============================================================

model = XGBRegressor(
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


pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        model
    )
])


# ============================================================
# 10. TRAIN
# ============================================================

print("\nTraining XGBoost Model 5...")

pipeline.fit(
    X_train,
    y_train
)

print("Training complete.")


# ============================================================
# 11. PREDICTION
# ============================================================

y_pred = pipeline.predict(
    X_test
)


# ============================================================
# 12. METRICS
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


print("\n" + "=" * 70)
print("ELONGATION MODEL 5 RESULTS")
print("=" * 70)

print(
    f"MAE  : {mae:.4f} %"
)

print(
    f"RMSE : {rmse:.4f} %"
)

print(
    f"R²   : {r2:.4f}"
)


# ============================================================
# 13. SAVE PREDICTIONS
# ============================================================

predictions = pd.DataFrame({

    "Actual_Elongation_pct":
        y_test.values,

    "Predicted_Elongation_pct":
        y_pred,

    "Error":
        y_test.values - y_pred
})


predictions.to_csv(
    "Elongation_XGBoost_Predictions.csv",
    index=False
)


# ============================================================
# 14. SAVE MODEL
# ============================================================

joblib.dump(
    pipeline,
    "Elongation_XGBoost_Model.pkl"
)


# ============================================================
# 15. MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame({

    "Model": [
        "RF - Composition Only",
        "RF - Composition + Conditions",
        "RF - Physical + Microstructural",
        "ExtraTrees - Physical + Microstructural",
        "XGBoost - Physical + Microstructural"
    ],

    "R2": [
        0.2502,
        0.2289,
        0.3185,
        0.3486,
        r2
    ],

    "MAE": [
        14.2432,
        14.7506,
        13.9365,
        12.7348,
        mae
    ],

    "RMSE": [
        18.7860,
        19.0505,
        17.9104,
        17.5104,
        rmse
    ]
})


comparison.to_csv(
    "Elongation_Model_Comparison.csv",
    index=False
)


print("\n" + "=" * 70)
print("ELONGATION MODEL COMPARISON")
print("=" * 70)

print(
    comparison.to_string(
        index=False
    )
)


print("\nModel 5 vs Model 4:")

print(
    f"R² improvement : "
    f"{r2 - 0.3486:+.4f}"
)

print(
    f"MAE improvement: "
    f"{12.7348 - mae:+.4f} %"
)

print(
    f"RMSE improvement: "
    f"{17.5104 - rmse:+.4f} %"
)


print("\nSaved:")
print("Elongation_XGBoost_Model.pkl")
print("Elongation_XGBoost_Predictions.csv")
print("Elongation_Model_Comparison.csv")


print("\n" + "=" * 70)
print("ELONGATION MODEL 5 COMPLETE")
print("=" * 70)