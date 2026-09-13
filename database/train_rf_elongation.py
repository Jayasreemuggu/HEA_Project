import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


print("=" * 70)
print("RANDOM FOREST - ELONGATION COMPOSITION-ONLY MODEL")
print("=" * 70)


# =========================================================
# 1. LOAD TRAIN / TEST DATA
# =========================================================

X_train = pd.read_csv(
    "X_train_Elongation.csv"
)

X_test = pd.read_csv(
    "X_test_Elongation.csv"
)

y_train = pd.read_csv(
    "y_train_Elongation.csv"
).iloc[:, 0]

y_test = pd.read_csv(
    "y_test_Elongation.csv"
).iloc[:, 0]


print("\nTraining data:", X_train.shape)
print("Testing data :", X_test.shape)


# =========================================================
# 2. CONVERT TO NUMERIC
# =========================================================

X_train = X_train.apply(
    pd.to_numeric,
    errors="coerce"
)

X_test = X_test.apply(
    pd.to_numeric,
    errors="coerce"
)

# Missing element = 0
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

y_train = pd.to_numeric(
    y_train,
    errors="coerce"
)

y_test = pd.to_numeric(
    y_test,
    errors="coerce"
)


# =========================================================
# 3. RANDOM FOREST
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


# =========================================================
# 4. TRAIN
# =========================================================

print("\nTraining Random Forest...")

rf_model.fit(
    X_train,
    y_train
)

print("Training complete.")


# =========================================================
# 5. PREDICTION
# =========================================================

y_pred = rf_model.predict(
    X_test
)


# =========================================================
# 6. METRICS
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
print("ELONGATION RANDOM FOREST RESULTS")
print("=" * 70)

print(f"MAE  : {mae:.4f} %")
print(f"RMSE : {rmse:.4f} %")
print(f"R²   : {r2:.4f}")


# =========================================================
# 7. SAVE MODEL
# =========================================================

joblib.dump(
    rf_model,
    "Elongation_RandomForest_Model.pkl"
)


# =========================================================
# 8. SAVE PREDICTIONS
# =========================================================

predictions = pd.DataFrame({
    "Actual_Elongation_pct": y_test.values,
    "Predicted_Elongation_pct": y_pred,
    "Error_pct": y_test.values - y_pred,
    "Absolute_Error_pct":
        np.abs(y_test.values - y_pred)
})

predictions.to_csv(
    "Elongation_RandomForest_Predictions.csv",
    index=False
)


# =========================================================
# 9. SAVE METRICS
# =========================================================

metrics = pd.DataFrame({
    "Model": [
        "RF - Composition Only"
    ],
    "MAE_pct": [
        mae
    ],
    "RMSE_pct": [
        rmse
    ],
    "R2": [
        r2
    ]
})

metrics.to_csv(
    "Elongation_RandomForest_Metrics.csv",
    index=False
)


# =========================================================
# 10. FINAL
# =========================================================

print("\nSaved:")
print("Elongation_RandomForest_Model.pkl")
print("Elongation_RandomForest_Predictions.csv")
print("Elongation_RandomForest_Metrics.csv")

print("\n" + "=" * 70)
print("ELONGATION RANDOM FOREST COMPLETE")
print("=" * 70)