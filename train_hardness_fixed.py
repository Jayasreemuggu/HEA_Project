import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor


BASE = "/mnt/c/Users/jayam/Downloads/IITH_ML_project"

DATA_FILE = os.path.join(
    BASE,
    "HEA_MPEA_ML_Ready_Deduplicated.csv"
)

MODEL_FILE = os.path.join(
    BASE,
    "Hardness_ExtraTrees_Model.pkl"
)

print("=" * 70)
print("CORRECTED HARDNESS MODEL TRAINING")
print("=" * 70)

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)

# ============================================================
# ELEMENT FEATURES
# ============================================================

element_features = [
    "Al_at_pct",
    "B_at_pct",
    "C_at_pct",
    "Co_at_pct",
    "Cr_at_pct",
    "Cu_at_pct",
    "Fe_at_pct",
    "Mn_at_pct",
    "Mo_at_pct",
    "Nb_at_pct",
    "Ni_at_pct",
    "Si_at_pct",
    "Ta_at_pct",
    "Ti_at_pct",
    "V_at_pct",
    "W_at_pct",
    "Zr_at_pct",
    "Ag_at_pct",
    "Ca_at_pct",
    "Ga_at_pct",
    "Hf_at_pct",
    "I_at_pct",
    "Li_at_pct",
    "Mg_at_pct",
    "Nd_at_pct",
    "O_at_pct",
    "Pd_at_pct",
    "Re_at_pct",
    "Ru_at_pct",
    "S_at_pct",
    "Sc_at_pct",
    "Sn_at_pct",
    "T_at_pct",
    "Y_at_pct",
    "Zn_at_pct"
]

# ============================================================
# NUMERIC FEATURES
# ============================================================

numeric_features = [
    "Test_Temperature_C",
    "VEC",
    "Atomic_Size_Mismatch",
    "Mixing_Enthalpy",
    "Mixing_Entropy",
    "Density_Exp_g_cm3",
    "Density_Calc_g_cm3",
    "Grain_Size_um",
    "Precipitate_Size_nm",
    "Matrix_Volume_pct"
]

# ============================================================
# CATEGORICAL FEATURES
# ============================================================

categorical_features = [
    "Test_Type",
    "Phase",
    "Processing_Method",
    "Alloy_Class",
    "Equilibrium_Condition",
    "Single_Multiphase",
    "Precipitate_Info"
]

TARGET = "Hardness_HV"

GROUP = "Composition_Canonical"

# ============================================================
# CHECK TARGET ALTERNATIVE
# ============================================================

if TARGET not in df.columns:

    print(
        f"\n{TARGET} not found."
    )

    if "Hardness_HV" in df.columns:
        TARGET = "Hardness_HV"

    else:
        raise SystemExit(
            "Hardness_HV column not found."
        )

required_columns = (
    element_features
    + numeric_features
    + categorical_features
    + [TARGET, GROUP]
)

missing = [
    c for c in required_columns
    if c not in df.columns
]

if missing:

    print("\nERROR: Missing columns:")

    for c in missing:
        print(" -", c)

    raise SystemExit(1)

print("\nAll required columns found.")

# ============================================================
# CLEAN TARGET
# ============================================================

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

df = df.dropna(
    subset=[
        TARGET,
        GROUP
    ]
).copy()

# Remove invalid hardness values
df = df[
    np.isfinite(
        df[TARGET]
    )
].copy()

print("\nHardness records:", len(df))

print(
    "Unique compositions:",
    df[GROUP].nunique()
)

# ============================================================
# FEATURES
# ============================================================

all_features = (
    element_features
    + numeric_features
    + categorical_features
)

X = df[
    all_features
].copy()

y = df[
    TARGET
].astype(float)

groups = df[
    GROUP
]

# ============================================================
# GROUPED SPLIT
# ============================================================

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    gss.split(
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

print("\n" + "=" * 70)
print("GROUPED SPLIT")
print("=" * 70)

print(
    "Train records:",
    len(X_train)
)

print(
    "Test records:",
    len(X_test)
)

print(
    "Train compositions:",
    groups_train.nunique()
)

print(
    "Test compositions:",
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

if overlap:

    raise RuntimeError(
        "Composition leakage detected!"
    )

# ============================================================
# PREPROCESSING
# ============================================================

element_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    )
])

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
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])

preprocessor = ColumnTransformer(
    transformers=[
        (
            "elements",
            element_pipeline,
            element_features
        ),
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ],
    remainder="drop"
)

# ============================================================
# EXTRA TREES
# ============================================================

model_estimator = ExtraTreesRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=1.0,
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
        model_estimator
    )
])

# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("TRAINING EXTRA TREES")
print("=" * 70)

model.fit(
    X_train,
    y_train
)

# ============================================================
# FEATURE CHECK
# ============================================================

pre = model.named_steps[
    "preprocessor"
]

estimator = model.named_steps[
    "model"
]

feature_names = (
    pre.get_feature_names_out()
)

print("\n" + "=" * 70)
print("FEATURE CHECK")
print("=" * 70)

print(
    "Transformed feature count:",
    len(feature_names)
)

print(
    "ExtraTrees expected features:",
    estimator.n_features_in_
)

print("\nFirst 50 features:")

for i, name in enumerate(
    feature_names[:50]
):

    print(
        i,
        name
    )

# ============================================================
# TEST TRANSFORMATION
# ============================================================

X_test_transformed = (
    pre.transform(X_test)
)

print(
    "\nTransformed test matrix:",
    X_test_transformed.shape
)

varying = np.sum(
    np.ptp(
        X_test_transformed,
        axis=0
    ) > 1e-12
)

print(
    "Varying transformed features:",
    varying,
    "/",
    X_test_transformed.shape[1]
)

# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(
    X_test
)

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
print("CORRECTED HARDNESS RESULTS")
print("=" * 70)

print(
    f"MAE  : {mae:.4f} HV"
)

print(
    f"RMSE : {rmse:.4f} HV"
)

print(
    f"R2   : {r2:.4f}"
)

# ============================================================
# PREDICTION VARIATION
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION VARIATION CHECK")
print("=" * 70)

print(
    "Minimum:",
    float(y_pred.min())
)

print(
    "Maximum:",
    float(y_pred.max())
)

print(
    "Mean:",
    float(y_pred.mean())
)

print(
    "Median:",
    float(np.median(y_pred))
)

print(
    "Std:",
    float(y_pred.std())
)

print(
    "Unique predictions:",
    len(
        np.unique(
            np.round(
                y_pred,
                6
            )
        )
    )
)

# ============================================================
# TOP FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 FEATURE IMPORTANCES")
print("=" * 70)

importances = (
    estimator.feature_importances_
)

order = np.argsort(
    importances
)[::-1]

for idx in order[:20]:

    print(
        f"{idx:4d} "
        f"{feature_names[idx]:55s} "
        f"{importances[idx]:.6f}"
    )

# ============================================================
# SAVE
# ============================================================

joblib.dump(
    model,
    MODEL_FILE
)

print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(MODEL_FILE)

print("\nHARDNESS FIX COMPLETE")
