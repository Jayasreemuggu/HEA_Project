import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit

MODEL_PATH = "YS_XGBoost_Model.pkl"
DATA_PATH = "HEA_MPEA_ML_Ready_Deduplicated.csv"
OUT_DIR = "explainability/results"

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("YS XGBOOST SHAP ANALYSIS")
print("=" * 70)

# ------------------------------------------------------------
# Load
# ------------------------------------------------------------
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

TARGET = "YS_Tensile_MPa"
GROUP = "Composition_Canonical"

print(f"Dataset shape: {df.shape}")
print(f"Target: {TARGET}")

# ------------------------------------------------------------
# Get exact raw features expected by saved model
# ------------------------------------------------------------
preprocessor = model.named_steps["preprocessor"]
expected_features = list(preprocessor.feature_names_in_)

print(f"Raw features expected by model: {len(expected_features)}")

# ------------------------------------------------------------
# Map model lowercase names -> actual dataset columns
# ------------------------------------------------------------
column_map = {c.lower(): c for c in df.columns}

missing = []

for feature in expected_features:
    if feature.lower() not in column_map:
        missing.append(feature)

if missing:
    print("\nERROR: Missing features:")
    for x in missing:
        print(x)
    raise SystemExit(1)

# Create X using exact model feature order
actual_columns = [column_map[f.lower()] for f in expected_features]

data = df.dropna(subset=[TARGET]).copy()

X = data[actual_columns].copy()
y = data[TARGET].copy()

# Rename to exactly what the model expects
X.columns = expected_features

print(f"YS records: {len(data)}")

# ------------------------------------------------------------
# Exact grouped split used for final YS model
# ------------------------------------------------------------
groups = data[GROUP].fillna("UNKNOWN").astype(str)

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    gss.split(X, y, groups=groups)
)

X_test = X.iloc[test_idx].copy()
y_test = y.iloc[test_idx].copy()

print(f"Train records: {len(train_idx)}")
print(f"Test records: {len(test_idx)}")
print(
    f"Test compositions: "
    f"{groups.iloc[test_idx].nunique()}"
)

train_comps = set(groups.iloc[train_idx])
test_comps = set(groups.iloc[test_idx])

print(
    f"Composition overlap: "
    f"{len(train_comps.intersection(test_comps))}"
)

# ------------------------------------------------------------
# Transform using trained preprocessing
# ------------------------------------------------------------
X_test_transformed = preprocessor.transform(X_test)

print(
    f"Transformed test matrix: "
    f"{X_test_transformed.shape}"
)

try:
    feature_names = list(
        preprocessor.get_feature_names_out()
    )
except Exception:
    feature_names = [
        f"feature_{i}"
        for i in range(X_test_transformed.shape[1])
    ]

if hasattr(X_test_transformed, "toarray"):
    X_dense = X_test_transformed.toarray()
else:
    X_dense = np.asarray(X_test_transformed)

X_dense = pd.DataFrame(
    X_dense,
    columns=feature_names
)

# ------------------------------------------------------------
# SHAP
# ------------------------------------------------------------
print()
print("=" * 70)
print("CALCULATING SHAP")
print("=" * 70)

estimator = model.named_steps["model"]

# XGBoost rejects feature names containing [, ] or <.
# Use safe names only for the SHAP calculation.
# Original feature names are retained separately for reporting.
safe_feature_names = [
    f"f_{i}" for i in range(X_dense.shape[1])
]

X_shap = X_dense.copy()
X_shap.columns = safe_feature_names

explainer = shap.TreeExplainer(estimator)

shap_values = explainer.shap_values(X_shap)

shap_values = np.asarray(shap_values)

print(f"SHAP matrix: {shap_values.shape}")

# ------------------------------------------------------------
# Global SHAP importance
# ------------------------------------------------------------
mean_abs_shap = np.mean(
    np.abs(shap_values),
    axis=0
)

importance = pd.DataFrame({
    "Feature": feature_names,
    "Mean_Absolute_SHAP": mean_abs_shap
})

importance = importance.sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
).reset_index(drop=True)

importance["Rank"] = np.arange(
    1,
    len(importance) + 1
)

importance = importance[
    [
        "Rank",
        "Feature",
        "Mean_Absolute_SHAP"
    ]
]

print()
print("=" * 70)
print("TOP 20 SHAP FEATURES")
print("=" * 70)

print(
    importance.head(20).to_string(
        index=False
    )
)

# ------------------------------------------------------------
# Save importance
# ------------------------------------------------------------
importance.to_csv(
    f"{OUT_DIR}/YS_SHAP_Feature_Importance.csv",
    index=False
)

importance.head(20).to_csv(
    f"{OUT_DIR}/YS_SHAP_Top20.csv",
    index=False
)

# ------------------------------------------------------------
# Save individual SHAP values
# ------------------------------------------------------------
shap_df = pd.DataFrame(
    shap_values,
    columns=feature_names
)

shap_df.insert(
    0,
    "Sample_Index",
    X_test.index
)

shap_df.insert(
    1,
    "Actual_YS_MPa",
    y_test.values
)

predictions = model.predict(X_test)

shap_df.insert(
    2,
    "Predicted_YS_MPa",
    predictions
)

shap_df.to_csv(
    f"{OUT_DIR}/YS_SHAP_Values_Test.csv",
    index=False
)

# ------------------------------------------------------------
# Bar plot
# ------------------------------------------------------------
plt.figure(figsize=(10, 8))

shap.summary_plot(
    shap_values,
    X_shap,
    plot_type="bar",
    max_display=20,
    show=False
)

plt.title(
    "YS XGBoost - SHAP Feature Importance"
)

plt.tight_layout()

plt.savefig(
    f"{OUT_DIR}/YS_SHAP_Summary_Bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# Beeswarm
# ------------------------------------------------------------
plt.figure(figsize=(10, 8))

shap.summary_plot(
    shap_values,
    X_shap,
    max_display=20,
    show=False
)

plt.title(
    "YS XGBoost - SHAP Summary"
)

plt.tight_layout()

plt.savefig(
    f"{OUT_DIR}/YS_SHAP_Beeswarm.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# Final
# ------------------------------------------------------------
print()
print("=" * 70)
print("FILES CREATED")
print("=" * 70)

files = [
    "YS_SHAP_Feature_Importance.csv",
    "YS_SHAP_Top20.csv",
    "YS_SHAP_Values_Test.csv",
    "YS_SHAP_Summary_Bar.png",
    "YS_SHAP_Beeswarm.png"
]

for f in files:
    print(f"{OUT_DIR}/{f}")

print()
print("=" * 70)
print("YS SHAP ANALYSIS COMPLETE")
print("=" * 70)
