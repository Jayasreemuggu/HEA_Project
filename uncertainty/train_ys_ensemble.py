import os
import sys
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import psycopg2


# ============================================================
# SETTINGS
# ============================================================

N_MODELS = 15
RANDOM_STATE = 42

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("YS UNCERTAINTY QUANTIFICATION")
print("=" * 70)

query = """
SELECT *
FROM hea_mpea.dataset_v1_raw
WHERE ys_tensile_mpa IS NOT NULL
"""

# Create a fresh PostgreSQL connection using the
# same credentials as database/connection.py.
import database.connection as db
import psycopg2
import psycopg2

db_connection = psycopg2.connect(
    host=db.DB_HOST,
    port=db.DB_PORT,
    database=db.DB_NAME,
    user=db.DB_USER,
    password=db.DB_PASSWORD
)

print("Fresh PostgreSQL connection established!")

df = pd.read_sql(query, db_connection)

db_connection.close()
print("Database connection closed after data loading.")

print(f"Dataset: {df.shape}")


# ============================================================
# FEATURES
# ============================================================

ELEMENTS = [
    "ag_at_pct", "al_at_pct", "b_at_pct", "c_at_pct",
    "ca_at_pct", "co_at_pct", "cr_at_pct", "cu_at_pct",
    "fe_at_pct", "ga_at_pct", "hf_at_pct", "i_at_pct",
    "li_at_pct", "mg_at_pct", "mn_at_pct", "mo_at_pct",
    "nb_at_pct", "nd_at_pct", "ni_at_pct", "o_at_pct",
    "pd_at_pct", "re_at_pct", "ru_at_pct", "s_at_pct",
    "sc_at_pct", "si_at_pct", "sn_at_pct", "t_at_pct",
    "ta_at_pct", "ti_at_pct", "v_at_pct", "w_at_pct",
    "y_at_pct", "zn_at_pct", "zr_at_pct"
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

TARGET = "ys_tensile_mpa"
GROUP = "composition_canonical"


# ============================================================
# PREPARE DATA
# ============================================================

X = df[FEATURES].copy()

y = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

groups = df[GROUP].astype(str)

valid = y.notna()

X = X.loc[valid].reset_index(drop=True)
y = y.loc[valid].reset_index(drop=True)
groups = groups.loc[valid].reset_index(drop=True)

df_valid = df.loc[valid].reset_index(drop=True)


# ============================================================
# EXACT SAME GROUPED SPLIT AS FINAL YS MODEL
# ============================================================

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    gss.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx].reset_index(drop=True)
X_test = X.iloc[test_idx].reset_index(drop=True)

y_train = y.iloc[train_idx].reset_index(drop=True)
y_test = y.iloc[test_idx].reset_index(drop=True)

groups_train = groups.iloc[train_idx].reset_index(drop=True)
groups_test = groups.iloc[test_idx].reset_index(drop=True)

test_records = df_valid.iloc[test_idx].reset_index(drop=True)


print()
print(f"Training records      : {len(X_train)}")
print(f"Test records          : {len(X_test)}")
print(f"Train compositions    : {groups_train.nunique()}")
print(f"Test compositions     : {groups_test.nunique()}")

overlap = set(groups_train) & set(groups_test)

print(f"Composition overlap   : {len(overlap)}")


# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer([
    (
        "num",
        numeric_pipeline,
        ELEMENTS + NUMERIC_FEATURES
    ),
    (
        "cat",
        categorical_pipeline,
        CATEGORICAL_FEATURES
    )
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"Processed features    : {X_train_processed.shape[1]}")


# ============================================================
# BOOTSTRAP ENSEMBLE
# ============================================================

all_predictions = []

rng = np.random.RandomState(RANDOM_STATE)

print()
print("=" * 70)
print(f"TRAINING {N_MODELS} BOOTSTRAP XGBOOST MODELS")
print("=" * 70)

for model_id in range(N_MODELS):

    print(f"\nModel {model_id + 1}/{N_MODELS}")

    # Bootstrap sampling
    sample_indices = rng.choice(
        len(X_train_processed),
        size=len(X_train_processed),
        replace=True
    )

    X_boot = X_train_processed[sample_indices]

    y_boot = y_train.iloc[
        sample_indices
    ].values

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_STATE + model_id,
        n_jobs=4
    )

    model.fit(
        X_boot,
        y_boot
    )

    predictions = model.predict(
        X_test_processed
    )

    all_predictions.append(
        predictions
    )

    print(
        f"Prediction range: "
        f"{predictions.min():.2f} - "
        f"{predictions.max():.2f} MPa"
    )


# ============================================================
# UNCERTAINTY
# ============================================================

prediction_matrix = np.vstack(
    all_predictions
)

mean_prediction = prediction_matrix.mean(
    axis=0
)

std_prediction = prediction_matrix.std(
    axis=0,
    ddof=1
)

lower_95 = (
    mean_prediction -
    1.96 * std_prediction
)

upper_95 = (
    mean_prediction +
    1.96 * std_prediction
)


# ============================================================
# ERRORS
# ============================================================

actual = y_test.values

absolute_error = np.abs(
    mean_prediction - actual
)

relative_error = (
    absolute_error /
    np.maximum(np.abs(actual), 1e-8)
) * 100


# ============================================================
# SAVE PREDICTIONS
# ============================================================

results = test_records[
    [
        "record_id",
        "alloy_name",
        "composition_canonical"
    ]
].copy()

results["actual_ys_mpa"] = actual

results["predicted_ys_mpa"] = (
    mean_prediction
)

results["uncertainty_std_mpa"] = (
    std_prediction
)

results["lower_95_mpa"] = (
    lower_95
)

results["upper_95_mpa"] = (
    upper_95
)

results["absolute_error_mpa"] = (
    absolute_error
)

results["relative_error_pct"] = (
    relative_error
)

prediction_file = os.path.join(
    OUTPUT_DIR,
    "YS_ensemble_predictions.csv"
)

results.to_csv(
    prediction_file,
    index=False
)


# ============================================================
# METRICS
# ============================================================

mae = mean_absolute_error(
    actual,
    mean_prediction
)

rmse = np.sqrt(
    mean_squared_error(
        actual,
        mean_prediction
    )
)

r2 = r2_score(
    actual,
    mean_prediction
)

coverage = np.mean(
    (
        actual >= lower_95
    ) &
    (
        actual <= upper_95
    )
) * 100

mean_uncertainty = (
    std_prediction.mean()
)

median_uncertainty = (
    np.median(std_prediction)
)

if np.std(std_prediction) > 0:
    uncertainty_error_corr = np.corrcoef(
        std_prediction,
        absolute_error
    )[0, 1]
else:
    uncertainty_error_corr = np.nan


# ============================================================
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame({
    "metric": [
        "MAE_MPa",
        "RMSE_MPa",
        "R2",
        "95pct_interval_coverage",
        "mean_uncertainty_MPa",
        "median_uncertainty_MPa",
        "uncertainty_error_correlation"
    ],
    "value": [
        mae,
        rmse,
        r2,
        coverage,
        mean_uncertainty,
        median_uncertainty,
        uncertainty_error_corr
    ]
})

metrics_file = os.path.join(
    OUTPUT_DIR,
    "YS_uncertainty_metrics.csv"
)

metrics.to_csv(
    metrics_file,
    index=False
)


# ============================================================
# HIGH UNCERTAINTY ALLOYS
# ============================================================

high_uncertainty = results.sort_values(
    "uncertainty_std_mpa",
    ascending=False
)

high_uncertainty.head(50).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "YS_high_uncertainty_samples.csv"
    ),
    index=False
)


# ============================================================
# SAVE PREPROCESSOR
# ============================================================

with open(
    os.path.join(
        OUTPUT_DIR,
        "YS_uncertainty_preprocessor.pkl"
    ),
    "wb"
) as f:

    pickle.dump(
        preprocessor,
        f
    )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("UNCERTAINTY RESULTS")
print("=" * 70)

print(
    f"MAE                    : "
    f"{mae:.4f} MPa"
)

print(
    f"RMSE                   : "
    f"{rmse:.4f} MPa"
)

print(
    f"R²                     : "
    f"{r2:.4f}"
)

print(
    f"95% interval coverage  : "
    f"{coverage:.2f}%"
)

print(
    f"Mean uncertainty       : "
    f"{mean_uncertainty:.4f} MPa"
)

print(
    f"Median uncertainty     : "
    f"{median_uncertainty:.4f} MPa"
)

print(
    f"Uncertainty/Error corr : "
    f"{uncertainty_error_corr:.4f}"
)

print()
print("Files saved in:")
print(OUTPUT_DIR)

print()
print("Prediction file:")
print(prediction_file)

print("=" * 70)
