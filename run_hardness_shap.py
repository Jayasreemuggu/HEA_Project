import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit

MODEL_PATH = "Hardness_ExtraTrees_Model.pkl"
DATA_PATH = "HEA_MPEA_ML_Ready_Deduplicated.csv"
OUT_DIR = "explainability/results"

os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("HARDNESS EXTRATREES SHAP ANALYSIS")
print("=" * 70)

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

TARGET = "Hardness_HV"
GROUP = "Composition_Canonical"

print(f"Dataset shape: {df.shape}")
print(f"Target: {TARGET}")

preprocessor = model.named_steps["preprocessor"]
expected_features = list(preprocessor.feature_names_in_)

print(f"Raw features expected by model: {len(expected_features)}")

# ------------------------------------------------------------
# Match exact model features to dataset columns
# ------------------------------------------------------------
column_map = {c.lower(): c for c in df.columns}

missing = [
    f for f in expected_features
    if f.lower() not in column_map
]

if missing:
    print("\nERROR: Missing features:")
    for f in missing:
        print(f)
    raise SystemExit(1)

actual_columns = [
    column_map[f.lower()]
    for f in expected_features
]

# ------------------------------------------------------------
# Hardness dataset
# ------------------------------------------------------------
data = df.dropna(subset=[TARGET]).copy()

# Remove invalid all-zero compositions exactly as training did
element_cols_actual = [
    column_map[f.lower()]
    for f in expected_features
    if f.lower().endswith("_at_pct")
]

composition_sum = data[element_cols_actual].fillna(0).sum(axis=1)

valid_composition = composition_sum > 0

data = data.loc[valid_composition].copy()

X = data[actual_columns].copy()
y = data[TARGET].copy()

# Exact feature names expected by model
X.columns = expected_features

print(f"Hardness records: {len(data)}")
print(
    f"Unique compositions: "
    f"{data[GROUP].nunique()}"
)

# ------------------------------------------------------------
# Exact grouped split used for hardness training
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

train_comps = set(groups.iloc[train_idx])
test_comps = set(groups.iloc[test_idx])

print()
print("=" * 70)
print("GROUPED SPLIT")
print("=" * 70)

print(f"Train records: {len(train_idx)}")
print(f"Test records: {len(test_idx)}")
print(f"Train compositions: {len(train_comps)}")
print(f"Test compositions: {len(test_comps)}")
print(
    f"Composition overlap: "
    f"{len(train_comps.intersection(test_comps))}"
)

# ------------------------------------------------------------
# Transform
# ------------------------------------------------------------
X_test_transformed = preprocessor.transform(X_test)

print()
print("=" * 70)
print("FEATURE TRANSFORMATION")
print("=" * 70)

print(
    f"Transformed test matrix: "
    f"{X_test_transformed.shape}"
)

feature_names = list(
    preprocessor.get_feature_names_out()
)

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

# Safe feature names for compatibility
safe_feature_names = [
    f"f_{i}"
    for i in range(X_dense.shape[1])
]

X_shap = X_dense.copy()
X_shap.columns = safe_feature_names

explainer = shap.TreeExplainer(estimator)

shap_values = explainer.shap_values(X_shap)
shap_values = np.asarray(shap_values)

print(
    f"SHAP matrix: "
    f"{shap_values.shape}"
)

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
# Save feature importance
# ------------------------------------------------------------
importance.to_csv(
    f"{OUT_DIR}/Hardness_SHAP_Feature_Importance.csv",
    index=False
)

importance.head(20).to_csv(
    f"{OUT_DIR}/Hardness_SHAP_Top20.csv",
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
    "Actual_Hardness_HV",
    y_test.values
)

predictions = model.predict(X_test)

shap_df.insert(
    2,
    "Predicted_Hardness_HV",
    predictions
)

shap_df.to_csv(
    f"{OUT_DIR}/Hardness_SHAP_Values_Test.csv",
    index=False
)

# ------------------------------------------------------------
# SHAP bar plot
# ------------------------------------------------------------
shap.summary_plot(
    shap_values,
    X_shap,
    plot_type="bar",
    max_display=20,
    show=False
)

plt.title(
    "Hardness ExtraTrees - SHAP Feature Importance"
)

plt.tight_layout()

plt.savefig(
    f"{OUT_DIR}/Hardness_SHAP_Summary_Bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# SHAP beeswarm
# ------------------------------------------------------------
shap.summary_plot(
    shap_values,
    X_shap,
    max_display=20,
    show=False
)

plt.title(
    "Hardness ExtraTrees - SHAP Summary"
)

plt.tight_layout()

plt.savefig(
    f"{OUT_DIR}/Hardness_SHAP_Beeswarm.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# Complete
# ------------------------------------------------------------
print()
print("=" * 70)
print("FILES CREATED")
print("=" * 70)

for f in [
    "Hardness_SHAP_Feature_Importance.csv",
    "Hardness_SHAP_Top20.csv",
    "Hardness_SHAP_Values_Test.csv",
    "Hardness_SHAP_Summary_Bar.png",
    "Hardness_SHAP_Beeswarm.png"
]:
    print(f"{OUT_DIR}/{f}")

print()
print("=" * 70)
print("HARDNESS SHAP ANALYSIS COMPLETE")
print("=" * 70)
