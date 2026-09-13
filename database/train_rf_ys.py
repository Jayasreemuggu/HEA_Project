import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np


# Load data
X_train = pd.read_csv("X_train_YS.csv")
X_test = pd.read_csv("X_test_YS.csv")

y_train = pd.read_csv("y_train_YS.csv")["yield_strength_mpa"]
y_test = pd.read_csv("y_test_YS.csv")["yield_strength_mpa"]


print("Training Random Forest...")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# Random Forest model
model = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)


# Train
model.fit(X_train, y_train)


# Predictions
y_pred = model.predict(X_test)


# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)


print("\nRandom Forest Results")
print("---------------------")
print(f"MAE  : {mae:.2f} MPa")
print(f"RMSE : {rmse:.2f} MPa")
print(f"R²   : {r2:.4f}")


# Save model
joblib.dump(model, "YS_RandomForest_Model.pkl")

print("\nModel saved:")
print("YS_RandomForest_Model.pkl")