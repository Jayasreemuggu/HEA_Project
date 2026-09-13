import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


print("=" * 70)
print("RANDOM FOREST - HARDNESS MODEL 1")
print("Composition Only")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(
    "Hardness_ML_Dataset.csv"
)

print("\nLoaded dataset:", df.shape)


# ============================================================
# 2. ELEMENT FEATURES
# ============================================================

element_features = [
    "al", "b", "c", "co", "cr", "cu", "fe", "mn",
    "mo", "nb", "ni", "si", "ta", "ti", "v", "w",
    "zr", "ag", "ca", "ga", "hf", "i", "li", "mg",
    "nd", "o", "pd", "re", "ru", "s", "sc", "sn",
    "t", "y", "zn"
]


# ============================================================
# 3. PREPARE FEATURES
# ============================================================

X = df[element_features].copy()

y = pd.to_numeric(
    df["hardness_hv"],
    errors="coerce"
)


# Convert composition values to numeric
for col in element_features:

    X[col] = pd.to_numeric(
        X[col],
        errors="coerce"
    )


# Missing elements = 0
X = X.fillna(0)


# ============================================================
# 4. COMPOSITION CHECK
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


# ============================================================
# 5. GROUPED TRAIN / TEST SPLIT
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


# ============================================================
# 6. CHECK COMPOSITION LEAKAGE
# ============================================================

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
# 7. TRAIN RANDOM FOREST
# ============================================================

model = RandomForestRegressor(
    n_estimators=500,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)


print(
    "\nTraining Random Forest Model 1..."
)


model.fit(
    X_train,
    y_train
)


print(
    "Training complete."
)


# ============================================================
# 8. PREDICTION
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# 9. METRICS
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
print("HARDNESS MODEL 1 RESULTS")
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


# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "Feature": element_features,

    "Importance":
        model.feature_importances_

})


importance = importance.sort_values(
    "Importance",
    ascending=False
)


print("\nTop 15 Features:")

print(
    importance.head(15).to_string(
        index=False
    )
)


# ============================================================
# 11. SAVE FEATURE IMPORTANCE
# ============================================================

importance.to_csv(
    "Hardness_RF_Feature_Importance.csv",
    index=False
)


# ============================================================
# 12. SAVE PREDICTIONS
# ============================================================

predictions = pd.DataFrame({

    "Actual_Hardness_HV":
        y_test.values,

    "Predicted_Hardness_HV":
        y_pred,

    "Error":
        y_test.values - y_pred

})


predictions.to_csv(
    "Hardness_RandomForest_Predictions.csv",
    index=False
)


# ============================================================
# 13. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "Hardness_RandomForest_Model.pkl"
)


# ============================================================
# 14. SAVE METRICS
# ============================================================

metrics = pd.DataFrame({

    "Model": [
        "Random Forest - Composition Only"
    ],

    "R2": [
        r2
    ],

    "MAE_HV": [
        mae
    ],

    "RMSE_HV": [
        rmse
    ],

    "Training_Records": [
        len(X_train)
    ],

    "Testing_Records": [
        len(X_test)
    ],

    "Train_Compositions": [
        groups_train.nunique()
    ],

    "Test_Compositions": [
        groups_test.nunique()
    ],

    "Composition_Overlap": [
        len(overlap)
    ]

})


metrics.to_csv(
    "Hardness_RandomForest_Metrics.csv",
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\nSaved:")

print(
    "Hardness_RandomForest_Model.pkl"
)

print(
    "Hardness_RandomForest_Predictions.csv"
)

print(
    "Hardness_RF_Feature_Importance.csv"
)

print(
    "Hardness_RandomForest_Metrics.csv"
)


print("\n" + "=" * 70)
print("HARDNESS MODEL 1 COMPLETE")
print("=" * 70)