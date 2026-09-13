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
print("RANDOM FOREST - ELONGATION MODEL 2")
print("Composition + Experimental Conditions")
print("=" * 70)


# =========================================================
# 1. LOAD THE SAME CLEANED DATASET USED BY MODEL 1
# =========================================================

df = pd.read_csv(
    "Elongation_ML_Dataset.csv"
)

print("\nLoaded cleaned dataset:", df.shape)


# =========================================================
# 2. FEATURES
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

NUMERIC_FEATURES = [
    "test_temperature_c"
]

CATEGORICAL_FEATURES = [
    "test_type",
    "phase",
    "processing_method",
    "alloy_class"
]

TARGET = "elongation_tensile_pct"


# =========================================================
# 3. CONVERT DATA TYPES
# =========================================================

for col in ELEMENTS:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df[ELEMENTS] = df[ELEMENTS].fillna(0)

df["test_temperature_c"] = pd.to_numeric(
    df["test_temperature_c"],
    errors="coerce"
)

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)


# =========================================================
# 4. FEATURES / TARGET
# =========================================================

FEATURES = (
    ELEMENTS +
    NUMERIC_FEATURES +
    CATEGORICAL_FEATURES
)

X = df[FEATURES].copy()
y = df[TARGET].copy()

groups = df["composition_canonical"]


# =========================================================
# 5. SAME GROUPED SPLIT AS MODEL 1
# =========================================================

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
# 6. PREPROCESSING
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
# 7. RANDOM FOREST
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
# 8. TRAIN
# =========================================================

print("\nTraining Random Forest Model 2...")

model.fit(
    X_train,
    y_train
)

print("Training complete.")


# =========================================================
# 9. PREDICT
# =========================================================

y_pred = model.predict(
    X_test
)


# =========================================================
# 10. METRICS
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
print("ELONGATION MODEL 2 RESULTS")
print("=" * 70)

print(f"MAE  : {mae:.4f} %")
print(f"RMSE : {rmse:.4f} %")
print(f"R²   : {r2:.4f}")


# =========================================================
# 11. MODEL 1 COMPARISON
# =========================================================

model1_metrics = pd.read_csv(
    "Elongation_RandomForest_Metrics.csv"
)

model1_r2 = model1_metrics["R2"].iloc[0]
model1_mae = model1_metrics["MAE_pct"].iloc[0]
model1_rmse = model1_metrics["RMSE_pct"].iloc[0]

print("\n" + "=" * 70)
print("MODEL 1 vs MODEL 2")
print("=" * 70)

print(f"Model 1 R² : {model1_r2:.4f}")
print(f"Model 2 R² : {r2:.4f}")

print(
    f"\nR² improvement   : {r2 - model1_r2:+.4f}"
)

print(
    f"MAE improvement  : {model1_mae - mae:+.4f} %"
)

print(
    f"RMSE improvement : {model1_rmse - rmse:+.4f} %"
)


# =========================================================
# 12. SAVE MODEL
# =========================================================

joblib.dump(
    model,
    "Elongation_RandomForest_Model2.pkl"
)


# =========================================================
# 13. SAVE PREDICTIONS
# =========================================================

predictions = pd.DataFrame({
    "Actual_Elongation_pct": y_test.values,
    "Predicted_Elongation_pct": y_pred,
    "Error_pct": y_test.values - y_pred,
    "Absolute_Error_pct":
        np.abs(y_test.values - y_pred)
})

predictions.to_csv(
    "Elongation_Model2_Predictions.csv",
    index=False
)


# =========================================================
# 14. SAVE COMPARISON
# =========================================================

comparison = pd.DataFrame({
    "Model": [
        "RF - Composition Only",
        "RF - Composition + Conditions"
    ],
    "MAE_pct": [
        model1_mae,
        mae
    ],
    "RMSE_pct": [
        model1_rmse,
        rmse
    ],
    "R2": [
        model1_r2,
        r2
    ]
})

comparison.to_csv(
    "Elongation_Model_Comparison.csv",
    index=False
)


# =========================================================
# 15. FINAL
# =========================================================

print("\nSaved:")
print("Elongation_RandomForest_Model2.pkl")
print("Elongation_Model2_Predictions.csv")
print("Elongation_Model_Comparison.csv")

print("\n" + "=" * 70)
print("ELONGATION MODEL 2 COMPLETE")
print("=" * 70)