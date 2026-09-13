import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("YS_ML_Dataset.csv")

print("Data loaded successfully!")
print("Original shape:", df.shape)


# ============================================================
# 2. DEFINE FEATURES
# ============================================================

ELEMENTS = [
    "ag", "al", "b", "c", "ca", "co", "cr", "cu",
    "fe", "ga", "hf", "i", "li", "mg", "mn", "mo",
    "nb", "nd", "ni", "o", "pd", "re", "ru", "s",
    "sc", "si", "sn", "t", "ta", "ti", "v", "w", "y",
    "zn", "zr"
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

FEATURES = (
    ELEMENTS
    + NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)

TARGET = "yield_strength_mpa"


# ============================================================
# 3. REMOVE RECORDS WITH MISSING TARGET
# ============================================================

df = df.dropna(subset=[TARGET]).copy()

print("\nYield Strength records:", len(df))


# ============================================================
# 4. DEFINE X AND Y
# ============================================================

X = df[FEATURES]

y = df[TARGET]


# ============================================================
# 5. GROUPED TRAIN/TEST SPLIT
# ============================================================

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


print("\nModel 2: Composition + Experimental Conditions")
print("------------------------------------------------")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 6. CHECK COMPOSITION LEAKAGE
# ============================================================

train_groups = set(
    groups.iloc[train_idx]
)

test_groups = set(
    groups.iloc[test_idx]
)

overlap = train_groups.intersection(
    test_groups
)

print("\nComposition leakage check")
print("-------------------------")
print("Training compositions:", len(train_groups))
print("Testing compositions:", len(test_groups))
print("Composition overlap:", len(overlap))

if len(overlap) == 0:
    print("No composition leakage detected.")
else:
    print("WARNING: Composition leakage detected!")


# ============================================================
# 7. NUMERIC PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    )
])


# ============================================================
# 8. CATEGORICAL PREPROCESSING
# ============================================================

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
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])


# ============================================================
# 9. PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer([
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
])


# ============================================================
# 10. RANDOM FOREST MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=500,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 11. COMPLETE ML PIPELINE
# ============================================================

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
# 12. TRAIN MODEL
# ============================================================

print("\nTraining Model 2...")

pipeline.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# 13. MAKE PREDICTIONS
# ============================================================

y_pred = pipeline.predict(
    X_test
)


# ============================================================
# 14. CALCULATE METRICS
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
# 15. DISPLAY RESULTS
# ============================================================

print("\nModel 2 Results")
print("----------------")

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
# 16. COMPARE WITH MODEL 1
# ============================================================

MODEL1_R2 = 0.2095

improvement = r2 - MODEL1_R2

percentage_improvement = (
    improvement / MODEL1_R2
) * 100


print("\nModel Comparison")
print("----------------")

print(
    f"Model 1 R²: {MODEL1_R2:.4f}"
)

print(
    f"Model 2 R²: {r2:.4f}"
)

print(
    f"R² improvement: {improvement:.4f}"
)

print(
    f"Relative R² improvement: "
    f"{percentage_improvement:.2f}%"
)


# ============================================================
# 17. CREATE PREDICTION RESULTS TABLE
# ============================================================

results = X_test.copy()

results["actual_ys_mpa"] = y_test.values

results["predicted_ys_mpa"] = y_pred

results["absolute_error_mpa"] = np.abs(
    y_test.values - y_pred
)


# ============================================================
# 18. SAVE PREDICTIONS
# ============================================================

results.to_csv(
    "YS_Model2_Predictions.csv",
    index=False
)


# ============================================================
# 19. SAVE MODEL
# ============================================================

joblib.dump(
    pipeline,
    "YS_RandomForest_Model2.pkl"
)


# ============================================================
# 20. SAVE METRICS
# ============================================================

metrics = pd.DataFrame({
    "Model": [
        "Model 2 - Composition + Experimental Conditions"
    ],
    "MAE_MPa": [
        mae
    ],
    "RMSE_MPa": [
        rmse
    ],
    "R2": [
        r2
    ]
})

metrics.to_csv(
    "YS_Model2_Metrics.csv",
    index=False
)


# ============================================================
# 21. FINAL OUTPUT
# ============================================================

print("\nFiles saved:")
print("----------------")
print("YS_RandomForest_Model2.pkl")
print("YS_Model2_Predictions.csv")
print("YS_Model2_Metrics.csv")

print("\nModel 2 completed successfully.")