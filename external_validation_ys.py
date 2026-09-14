import os
import pickle
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

BASE = "/mnt/c/Users/jayam/Downloads/IITH_ML_project"

DATA = os.path.join(BASE, "HEA_MPEA_ML_Ready_Deduplicated.csv")
OUT_DIR = os.path.join(BASE, "evaluation", "results")
MODEL_OUT = os.path.join(BASE, "evaluation", "YS_External_Paper13_Model.pkl")
RESULT_OUT = os.path.join(OUT_DIR, "YS_External_Paper13_Predictions.csv")
METRIC_OUT = os.path.join(OUT_DIR, "YS_External_Paper13_Metrics.csv")

os.makedirs(OUT_DIR, exist_ok=True)

ELEMENTS = [
    "Al","B","Co","Cr","Cu","Fe","Mn","Mo","Nb","Ni","Si",
    "Ta","Ti","V","W","Zr","Ag","Ca","Ga","Hf","I","Li","Mg",
    "Nd","O","Pd","Re","Ru","S","Sc","Sn","T","Y","Zn"
]

NUMERIC = [
    "Test_Temperature_C",
    "VEC",
    "Atomic_Size_Mismatch",
    "Mixing_Enthalpy",
    "Mixing_Entropy",
    "Density_Exp_g_cm3",
    "Grain_Size_um",
    "Precipitate_Size_nm",
    "Matrix_Volume_pct",
]

CATEGORICAL = [
    "Test_Type",
    "Phase",
    "Processing_Method",
    "Alloy_Class",
    "Equilibrium_Condition",
    "Single_Multiphase",
    "Precipitate_Info",
]

FEATURES = [f"{e}_at_pct" for e in ELEMENTS] + NUMERIC + CATEGORICAL

df = pd.read_csv(DATA)

# Paper_13 is the completely unseen external dataset.
train = df[
    (df["Source_Paper"] != "Paper_13") &
    (df["YS_Tensile_MPa"].notna())
].copy()

test = df[
    (df["Source_Paper"] == "Paper_13") &
    (df["YS_Tensile_MPa"].notna())
].copy()

# Remove exact duplicate feature/target rows from training only.
train = train.drop_duplicates(subset=FEATURES + ["YS_Tensile_MPa"]).reset_index(drop=True)

X_train = train[FEATURES].copy()
y_train = train["YS_Tensile_MPa"].astype(float)

X_test = test[FEATURES].copy()
y_test = test["YS_Tensile_MPa"].astype(float)

# Same general preprocessing strategy as the final XGBoost models.
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipe, NUMERIC),
    ("categorical", categorical_pipe, CATEGORICAL),
    ("elements", SimpleImputer(strategy="constant", fill_value=0), [f"{e}_at_pct" for e in ELEMENTS]),
])

model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
)

Xtr = preprocessor.fit_transform(X_train)
Xte = preprocessor.transform(X_test)

model.fit(Xtr, y_train)

pred = model.predict(Xte)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("=" * 70)
print("EXTERNAL LITERATURE VALIDATION — YIELD STRENGTH")
print("=" * 70)
print(f"Training records:             {len(train)}")
print(f"External Paper_13 records:    {len(test)}")
print(f"Training compositions:        {train['Composition_Canonical'].nunique()}")
print(f"Paper_13 compositions:        {test['Composition_Canonical'].nunique()}")

overlap = (
    set(train["Composition_Canonical"].dropna()) &
    set(test["Composition_Canonical"].dropna())
)

print(f"Composition overlap:          {len(overlap)}")
print()
print(f"MAE:                           {mae:.4f} MPa")
print(f"RMSE:                          {rmse:.4f} MPa")
print(f"R²:                            {r2:.4f}")
print()
print("This is a true paper-level external test because")
print("Paper_13 compositions were completely unseen during training.")

# Save model + preprocessor.
with open(MODEL_OUT, "wb") as f:
    pickle.dump({
        "preprocessor": preprocessor,
        "model": model,
        "features": FEATURES,
        "train_sources": sorted(train["Source_Paper"].unique().tolist()),
        "external_source": "Paper_13",
    }, f)

# Save predictions.
results = test[
    [
        "Record_ID",
        "Alloy_Name",
        "Composition_Canonical",
        "Source_Paper",
        "YS_Tensile_MPa",
    ]
].copy()

results["Predicted_YS_MPa"] = pred
results["Error_MPa"] = pred - y_test.to_numpy()
results["Absolute_Error_MPa"] = np.abs(results["Error_MPa"])

results.to_csv(RESULT_OUT, index=False)

metrics = pd.DataFrame([{
    "Property": "YS",
    "External_Test_Source": "Paper_13",
    "Train_Records": len(train),
    "External_Test_Records": len(test),
    "Train_Compositions": train["Composition_Canonical"].nunique(),
    "External_Test_Compositions": test["Composition_Canonical"].nunique(),
    "Composition_Overlap": len(overlap),
    "MAE_MPa": mae,
    "RMSE_MPa": rmse,
    "R2": r2,
}])

metrics.to_csv(METRIC_OUT, index=False)

print()
print("Saved:")
print(MODEL_OUT)
print(RESULT_OUT)
print(METRIC_OUT)
