import os
import glob
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA = "HEA_MPEA_ML_Ready_Deduplicated.csv"
FINAL_MODEL = "YS_XGBoost_Model.pkl"

MODEL_DIR = "uncertainty/models"
OUT_DIR = "uncertainty/results"

N_MODELS = 15
BASE_SEED = 1000

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("REBUILDING YS BOOTSTRAP ENSEMBLE")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------
df = pd.read_csv(DATA)

target = "YS_Tensile_MPa"
group_col = "Composition_Canonical"

data = df.dropna(subset=[target]).copy()
data = data.reset_index(drop=True)

print("YS records:", len(data))

# ------------------------------------------------------------
# 2. LOAD FINAL YS MODEL
# ------------------------------------------------------------
final_pipeline = joblib.load(FINAL_MODEL)

preprocessor = final_pipeline.named_steps["preprocessor"]
estimator = final_pipeline.named_steps["model"]

expected_features = list(
    preprocessor.feature_names_in_
)

column_map = {
    c.lower(): c
    for c in data.columns
}

actual_features = []

for feature in expected_features:
    if feature.lower() not in column_map:
        raise ValueError(
            f"Missing model feature: {feature}"
        )

    actual_features.append(
        column_map[feature.lower()]
    )

X = data[actual_features].copy()
X.columns = expected_features

y = data[target].astype(float)

groups = (
    data[group_col]
    .fillna("UNKNOWN")
    .astype(str)
)

# ------------------------------------------------------------
# 3. EXACT GROUPED SPLIT
# ------------------------------------------------------------
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

X_train = X.iloc[train_idx].copy()
y_train = y.iloc[train_idx].copy()

X_test = X.iloc[test_idx].copy()
y_test = y.iloc[test_idx].copy()

train_groups = groups.iloc[train_idx]
test_groups = groups.iloc[test_idx]

print("\nGrouped split:")
print("Train:", len(X_train))
print("Test :", len(X_test))
print(
    "Train compositions:",
    train_groups.nunique()
)
print(
    "Test compositions:",
    test_groups.nunique()
)
print(
    "Composition overlap:",
    len(
        set(train_groups) &
        set(test_groups)
    )
)

# ------------------------------------------------------------
# 4. FIT ONE PREPROCESSOR ON TRAIN DATA
# ------------------------------------------------------------
prep = clone(preprocessor)

X_train_Z = prep.fit_transform(
    X_train,
    y_train
)

X_test_Z = prep.transform(
    X_test
)

if hasattr(X_train_Z, "toarray"):
    X_train_Z = X_train_Z.toarray()

if hasattr(X_test_Z, "toarray"):
    X_test_Z = X_test_Z.toarray()

X_train_Z = np.asarray(X_train_Z)
X_test_Z = np.asarray(X_test_Z)

print(
    "\nTransformed feature count:",
    X_train_Z.shape[1]
)

# ------------------------------------------------------------
# 5. REBUILD 15 BOOTSTRAP MODELS
# ------------------------------------------------------------
models = []

print(
    "\nTraining",
    N_MODELS,
    "bootstrap models..."
)

for i in range(N_MODELS):

    seed = BASE_SEED + i

    rng = np.random.RandomState(seed)

    bootstrap_idx = rng.choice(
        len(X_train_Z),
        size=len(X_train_Z),
        replace=True
    )

    X_boot = X_train_Z[
        bootstrap_idx
    ]

    y_boot = y_train.iloc[
        bootstrap_idx
    ].to_numpy()

    model = clone(estimator)

    try:
        model.set_params(
            random_state=seed
        )
    except Exception:
        pass

    model.fit(
        X_boot,
        y_boot
    )

    model_path = (
        f"{MODEL_DIR}/"
        f"YS_Bootstrap_Model_{i+1:02d}.pkl"
    )

    # Save model together with the fitted preprocessor
    pipeline = {
        "preprocessor": prep,
        "model": model,
        "seed": seed
    }

    joblib.dump(
        pipeline,
        model_path
    )

    models.append(pipeline)

    print(
        f"[{i+1:02d}/{N_MODELS}] "
        f"saved -> {model_path}"
    )

# ------------------------------------------------------------
# 6. VALIDATION ENSEMBLE
# ------------------------------------------------------------
validation_predictions = []

for pipeline in models:

    prediction = pipeline["model"].predict(
        X_test_Z
    )

    validation_predictions.append(
        np.asarray(prediction)
    )

VP = np.vstack(
    validation_predictions
)

val_mean = VP.mean(axis=0)
val_std = VP.std(
    axis=0,
    ddof=1
)

mae = mean_absolute_error(
    y_test,
    val_mean
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        val_mean
    )
)

r2 = r2_score(
    y_test,
    val_mean
)

coverage = np.mean(
    (
        y_test.to_numpy()
        >= val_mean - 1.96 * val_std
    )
    &
    (
        y_test.to_numpy()
        <= val_mean + 1.96 * val_std
    )
)

print("\n" + "=" * 70)
print("BOOTSTRAP VALIDATION")
print("=" * 70)

print(
    f"MAE  : {mae:.4f} MPa"
)

print(
    f"RMSE : {rmse:.4f} MPa"
)

print(
    f"R²   : {r2:.4f}"
)

print(
    f"Mean uncertainty : "
    f"{val_std.mean():.4f} MPa"
)

print(
    f"Median uncertainty : "
    f"{np.median(val_std):.4f} MPa"
)

print(
    f"Nominal 95% coverage : "
    f"{coverage * 100:.2f}%"
)

validation_df = pd.DataFrame({
    "test_index":
        test_idx,
    "actual_ys_mpa":
        y_test.to_numpy(),
    "ensemble_predicted_ys_mpa":
        val_mean,
    "uncertainty_std_mpa":
        val_std,
    "lower_95_mpa":
        val_mean - 1.96 * val_std,
    "upper_95_mpa":
        val_mean + 1.96 * val_std,
    "absolute_error_mpa":
        np.abs(
            val_mean -
            y_test.to_numpy()
        )
})

validation_df.to_csv(
    f"{OUT_DIR}/"
    "YS_Bootstrap_Validation_Rebuilt.csv",
    index=False
)

# ------------------------------------------------------------
# 7. LOAD 5000 CANDIDATES
# ------------------------------------------------------------
candidate_file = (
    "predictions/results/"
    "HEA_MPEA_5000_Final_Predictions.csv"
)

cand = pd.read_csv(
    candidate_file
)

print(
    "\nCandidates loaded:",
    len(cand)
)

candidate_map = {
    c.lower(): c
    for c in cand.columns
}

candidate_data = {}

for feature in expected_features:

    key = feature.lower()

    if key in candidate_map:

        candidate_data[feature] = (
            cand[candidate_map[key]]
        )

    elif key.endswith("_at_pct"):

        candidate_data[feature] = np.zeros(len(cand))

    elif key == "test_temperature_c":

        candidate_data[feature] = np.full(
            len(cand), 25.0
        )

    elif key == "test_type":

        candidate_data[feature] = np.full(
            len(cand), "C", dtype=object
        )

    elif key == "phase":

        candidate_data[feature] = np.full(
            len(cand), "BCC", dtype=object
        )

    elif key == "processing_method":

        candidate_data[feature] = np.full(
            len(cand), "WROUGHT", dtype=object
        )

    elif key == "alloy_class":

        candidate_data[feature] = np.full(
            len(cand), "HEA", dtype=object
        )

    elif key == "precipitate_info":

        candidate_data[feature] = np.full(
            len(cand), "None", dtype=object
        )

    else:

        candidate_data[feature] = np.full(
            len(cand), np.nan
        )

X_candidates = pd.DataFrame(
    candidate_data
)

X_candidates_Z = prep.transform(
    X_candidates
)

if hasattr(
    X_candidates_Z,
    "toarray"
):
    X_candidates_Z = (
        X_candidates_Z.toarray()
    )

X_candidates_Z = np.asarray(
    X_candidates_Z
)

print(
    "Candidate transformed shape:",
    X_candidates_Z.shape
)

# ------------------------------------------------------------
# 8. PREDICT ALL 5000 WITH 15 MODELS
# ------------------------------------------------------------
candidate_predictions = []

print(
    "\nGenerating ensemble predictions..."
)

for i, pipeline in enumerate(
    models,
    start=1
):

    prediction = pipeline["model"].predict(
        X_candidates_Z
    )

    candidate_predictions.append(
        np.asarray(prediction)
    )

    print(
        f"Model {i:02d}/{N_MODELS} complete"
    )

CP = np.vstack(
    candidate_predictions
)

cand["YS_Ensemble_Mean_MPa"] = (
    CP.mean(axis=0)
)

cand["YS_Uncertainty_STD_MPa"] = (
    CP.std(
        axis=0,
        ddof=1
    )
)

cand["YS_Ensemble_Min_MPa"] = (
    CP.min(axis=0)
)

cand["YS_Ensemble_Max_MPa"] = (
    CP.max(axis=0)
)

cand["YS_Ensemble_Difference_MPa"] = (
    cand["YS_Ensemble_Mean_MPa"]
    - cand["Predicted_YS_MPa"]
)

# ------------------------------------------------------------
# 9. UNCERTAINTY-AWARE PERFORMANCE SCORE
# ------------------------------------------------------------
def minmax(s):

    s = pd.to_numeric(
        s,
        errors="coerce"
    )

    mn = s.min()
    mx = s.max()

    if mx == mn:
        return pd.Series(
            1.0,
            index=s.index
        )

    return (
        (s - mn) /
        (mx - mn)
    )

cand["YS_Normalized"] = minmax(
    cand["Predicted_YS_MPa"]
)

cand["UTS_Normalized"] = minmax(
    cand["Predicted_UTS_MPa"]
)

cand["Elongation_Normalized"] = minmax(
    cand["Predicted_Elongation_pct"]
)

cand["Hardness_Normalized"] = minmax(
    cand["Predicted_Hardness_HV"]
)

cand["Uncertainty_Normalized"] = minmax(
    cand["YS_Uncertainty_STD_MPa"]
)

cand["Uncertainty_Confidence"] = (
    1 -
    cand["Uncertainty_Normalized"]
)

cand["Performance_Score"] = (
    0.25 * cand["YS_Normalized"] +
    0.30 * cand["UTS_Normalized"] +
    0.30 * cand["Elongation_Normalized"] +
    0.15 * cand["Hardness_Normalized"]
)

# 20% penalty for uncertainty
cand["Uncertainty_Aware_Score"] = (
    0.80 * cand["Performance_Score"] +
    0.20 * cand["Uncertainty_Confidence"]
)

# ------------------------------------------------------------
# 10. RISK CATEGORY
# ------------------------------------------------------------
q25 = cand[
    "YS_Uncertainty_STD_MPa"
].quantile(0.25)

q75 = cand[
    "YS_Uncertainty_STD_MPa"
].quantile(0.75)

def risk(x):

    if x <= q25:
        return "Low"

    if x >= q75:
        return "High"

    return "Medium"

cand["Uncertainty_Risk"] = (
    cand[
        "YS_Uncertainty_STD_MPa"
    ].apply(risk)
)

# ------------------------------------------------------------
# 11. FINAL RANKING
# ------------------------------------------------------------
cand = cand.sort_values(
    "Uncertainty_Aware_Score",
    ascending=False
).reset_index(drop=True)

cand["Uncertainty_Aware_Rank"] = (
    np.arange(len(cand)) + 1
)

# ------------------------------------------------------------
# 12. HIGH PERFORMANCE + LOW UNCERTAINTY
# ------------------------------------------------------------
performance_threshold = cand[
    "Performance_Score"
].quantile(0.75)

uncertainty_threshold = cand[
    "YS_Uncertainty_STD_MPa"
].quantile(0.50)

cand[
    "High_Performance_Low_Uncertainty"
] = (
    (cand["Performance_Score"]
     >= performance_threshold)
    &
    (cand["YS_Uncertainty_STD_MPa"]
     <= uncertainty_threshold)
)

shortlist = cand[
    cand[
        "High_Performance_Low_Uncertainty"
    ]
].copy()

# ------------------------------------------------------------
# 13. PARETO SUBSET
# ------------------------------------------------------------
pareto = cand[
    cand["Pareto_Optimal"] == True
].copy()

pareto = pareto.sort_values(
    "Uncertainty_Aware_Score",
    ascending=False
)

# ------------------------------------------------------------
# 14. SAVE
# ------------------------------------------------------------
full_file = (
    f"{OUT_DIR}/"
    "HEA_MPEA_5000_Uncertainty_Aware_Ranking.csv"
)

top_file = (
    f"{OUT_DIR}/"
    "HEA_MPEA_Top50_Uncertainty_Aware.csv"
)

short_file = (
    f"{OUT_DIR}/"
    "HEA_MPEA_Low_Uncertainty_High_Performance.csv"
)

pareto_file = (
    f"{OUT_DIR}/"
    "HEA_MPEA_Pareto_Uncertainty_Ranked.csv"
)

cand.to_csv(
    full_file,
    index=False
)

cand.head(50).to_csv(
    top_file,
    index=False
)

shortlist.to_csv(
    short_file,
    index=False
)

pareto.to_csv(
    pareto_file,
    index=False
)

# ------------------------------------------------------------
# 15. FINAL OUTPUT
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("UNCERTAINTY-AWARE RANKING COMPLETE")
print("=" * 70)

print(
    "\nCandidate uncertainty statistics:"
)

print(
    cand[
        "YS_Uncertainty_STD_MPa"
    ].describe()
)

print(
    "\nLow uncertainty threshold:",
    round(
        uncertainty_threshold,
        3
    )
)

print(
    "High uncertainty threshold:",
    round(
        q75,
        3
    )
)

print(
    "\nHigh-performance + "
    "low-uncertainty candidates:",
    len(shortlist),
    f"({100 * len(shortlist) / len(cand):.2f}%)"
)

cols = [
    "candidate_id",
    "composition",
    "Predicted_YS_MPa",
    "Predicted_UTS_MPa",
    "Predicted_Elongation_pct",
    "Predicted_Hardness_HV",
    "YS_Ensemble_Mean_MPa",
    "YS_Uncertainty_STD_MPa",
    "Uncertainty_Risk",
    "Performance_Score",
    "Uncertainty_Aware_Score",
    "Uncertainty_Aware_Rank",
    "Pareto_Optimal"
]

print(
    "\nTOP 20 UNCERTAINTY-AWARE CANDIDATES"
)

print(
    cand.head(20)[cols].to_string(
        index=False
    )
)

print("\nFiles created:")
print(full_file)
print(top_file)
print(short_file)
print(pareto_file)

print(
    "\nBootstrap models:",
    len(
        glob.glob(
            f"{MODEL_DIR}/"
            "YS_Bootstrap_Model_*.pkl"
        )
    )
)

print(
    "\nIMPORTANT:"
)

print(
    "YS uncertainty is ensemble standard deviation "
    "and is used as a relative model-risk measure. "
    "It is NOT a calibrated 95% prediction interval."
)
