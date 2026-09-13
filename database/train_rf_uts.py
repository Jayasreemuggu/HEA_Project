import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


print("=" * 70)
print("RANDOM FOREST - UTS COMPOSITION-ONLY MODEL")
print("=" * 70)

# =========================================================
# 1. LOAD TRAIN / TEST DATA
# =========================================================

X_train = pd.read_csv("X_train_UTS.csv")
X_test = pd.read_csv("X_test_UTS.csv")

y_train = pd.read_csv("y_train_UTS.csv").iloc[:, 0]
y_test = pd.read_csv("y_test_UTS.csv").iloc[:, 0]

print("\nTraining data:", X_train.shape)
print("Testing data :", X_test.shape)

# =========================================================
# 2. CONVERT COMPOSITION TO NUMERIC
# =========================================================

X_train = X_train.apply(
    pd.to_numeric,
    errors="coerce"
)

X_test = X_test.apply(
    pd.to_numeric,
    errors="coerce"
)

# Missing elemental concentration = 0
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# Ensure target is numeric
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
print("RANDOM FOREST UTS RESULTS")
print("=" * 70)

print(f"MAE  : {mae:.2f} MPa")
print(f"RMSE : {rmse:.2f} MPa")
print(f"R²   : {r2:.4f}")

# =========================================================
# 7. SAVE MODEL
# =========================================================

joblib.dump(
    rf_model,
    "UTS_RandomForest_Model.pkl"
)

print("\nSaved:")
print("UTS_RandomForest_Model.pkl")

# =========================================================
# 8. SAVE PREDICTIONS
# =========================================================

predictions = pd.DataFrame({
    "Actual_UTS_MPa": y_test.values,
    "Predicted_UTS_MPa": y_pred,
    "Error_MPa": y_test.values - y_pred,
    "Absolute_Error_MPa":
        np.abs(y_test.values - y_pred)
})

predictions.to_csv(
    "UTS_RandomForest_Predictions.csv",
    index=False
)

print("UTS_RandomForest_Predictions.csv")

# =========================================================
# 9. SAVE METRICS
# =========================================================

metrics = pd.DataFrame({
    "Model": ["RF - Composition Only"],
    "MAE_MPa": [mae],
    "RMSE_MPa": [rmse],
    "R2": [r2]
})

metrics.to_csv(
    "UTS_RandomForest_Metrics.csv",
    index=False
)

print("UTS_RandomForest_Metrics.csv")

print("\n" + "=" * 70)
print("UTS RANDOM FOREST TRAINING COMPLETE")
print("=" * 70)