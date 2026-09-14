import pandas as pd
import numpy as np

from sqlalchemy import text
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor

from database.connection import engine

ELEMENTS = [
    "al_at_pct","b_at_pct","c_at_pct","co_at_pct","cr_at_pct",
    "cu_at_pct","fe_at_pct","mn_at_pct","mo_at_pct","nb_at_pct",
    "ni_at_pct","si_at_pct","ta_at_pct","ti_at_pct","v_at_pct",
    "w_at_pct","zr_at_pct","ag_at_pct","ca_at_pct","ga_at_pct",
    "hf_at_pct","i_at_pct","li_at_pct","mg_at_pct","nd_at_pct",
    "o_at_pct","pd_at_pct","re_at_pct","ru_at_pct","s_at_pct",
    "sc_at_pct","sn_at_pct","t_at_pct","y_at_pct","zn_at_pct"
]

NUMERIC_FEATURES = [
    "test_temperature_c","vec","atomic_size_mismatch",
    "mixing_enthalpy","mixing_entropy","density_exp_g_cm3",
    "density_calc_g_cm3","grain_size_um","precipitate_size_nm",
    "matrix_volume_pct"
]

CATEGORICAL_FEATURES = [
    "test_type","phase","processing_method","alloy_class",
    "equilibrium_condition","single_multiphase","precipitate_info"
]

FEATURES = ELEMENTS + NUMERIC_FEATURES + CATEGORICAL_FEATURES

print("=" * 80)
print("YS ENSEMBLE - GROUPED CROSS-VALIDATION WEIGHT SELECTION")
print("=" * 80)

df = pd.read_sql(
    text("""
        SELECT *
        FROM hea_mpea.dataset_v1_raw
        WHERE ys_tensile_mpa IS NOT NULL
    """),
    engine
)

df = df[
    FEATURES +
    ["ys_tensile_mpa", "composition_canonical"]
].copy()

df = df.dropna(
    subset=["composition_canonical"]
).reset_index(drop=True)

y = pd.to_numeric(
    df["ys_tensile_mpa"],
    errors="coerce"
)

valid = y.notna()

df = df.loc[valid].reset_index(drop=True)
y = y.loc[valid].reset_index(drop=True)

for e in ELEMENTS:
    df[e] = pd.to_numeric(
        df[e],
        errors="coerce"
    )

X = df[FEATURES].copy()

for e in ELEMENTS:
    X[e] = X[e].fillna(0)

groups = df[
    "composition_canonical"
].reset_index(drop=True)

# ============================================================
# FIXED TEST SET
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

X_dev = X.iloc[train_idx].reset_index(drop=True)
y_dev = y.iloc[train_idx].reset_index(drop=True)
groups_dev = groups.iloc[train_idx].reset_index(drop=True)

X_test = X.iloc[test_idx].reset_index(drop=True)
y_test = y.iloc[test_idx].reset_index(drop=True)
groups_test = groups.iloc[test_idx].reset_index(drop=True)

print("\nDevelopment records:", len(X_dev))
print("Test records       :", len(X_test))
print("Development groups :", groups_dev.nunique())
print("Test groups        :", groups_test.nunique())

print(
    "Group overlap:",
    len(
        set(groups_dev).intersection(
            set(groups_test)
        )
    )
)

# ============================================================
# PREPROCESS DEVELOPMENT DATA
# ============================================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "onehot",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])

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

X_dev_t = preprocessor.fit_transform(X_dev)
X_test_t = preprocessor.transform(X_test)

print("\nTransformed development:", X_dev_t.shape)
print("Transformed test       :", X_test_t.shape)

# ============================================================
# GROUPED 5-FOLD CV
# ============================================================

gkf = GroupKFold(n_splits=5)

weights = np.arange(
    0.0,
    1.01,
    0.05
)

cv_predictions = {
    float(w): np.zeros(len(X_dev))
    for w in weights
}

fold_numbers = np.zeros(
    len(X_dev),
    dtype=int
)

for fold, (tr_idx, val_idx) in enumerate(
    gkf.split(
        X_dev_t,
        y_dev,
        groups=groups_dev
    ),
    start=1
):

    print(f"\nTraining CV fold {fold}/5...")

    X_tr = X_dev_t[tr_idx]
    X_val = X_dev_t[val_idx]

    y_tr = y_dev.iloc[tr_idx]
    y_val = y_dev.iloc[val_idx]

    # --------------------------------------------------------
    # XGBoost MoreTrees
    # --------------------------------------------------------

    xgb = XGBRegressor(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    xgb.fit(
        X_tr,
        y_tr
    )

    pred_xgb = xgb.predict(
        X_val
    )

    # --------------------------------------------------------
    # ExtraTrees
    # --------------------------------------------------------

    et = ExtraTreesRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=1.0,
        random_state=42,
        n_jobs=-1
    )

    et.fit(
        X_tr,
        y_tr
    )

    pred_et = et.predict(
        X_val
    )

    # --------------------------------------------------------
    # Store predictions for every candidate weight
    # --------------------------------------------------------

    for w in weights:

        ensemble = (
            w * pred_xgb +
            (1.0 - w) * pred_et
        )

        cv_predictions[
            float(w)
        ][val_idx] = ensemble

    fold_numbers[val_idx] = fold

# ============================================================
# CV WEIGHT SELECTION
# ============================================================

cv_results = []

for w in weights:

    pred = cv_predictions[
        float(w)
    ]

    mae = mean_absolute_error(
        y_dev,
        pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_dev,
            pred
        )
    )

    r2 = r2_score(
        y_dev,
        pred
    )

    cv_results.append({
        "XGB_weight": float(w),
        "ET_weight": float(1.0 - w),
        "CV_MAE_MPa": mae,
        "CV_RMSE_MPa": rmse,
        "CV_R2": r2
    })

cv_df = pd.DataFrame(
    cv_results
)

best = cv_df.loc[
    cv_df["CV_R2"].idxmax()
]

best_weight = float(
    best["XGB_weight"]
)

print("\n" + "=" * 80)
print("CROSS-VALIDATION RESULTS")
print("=" * 80)

print(
    cv_df.sort_values(
        "CV_R2",
        ascending=False
    ).to_string(index=False)
)

print("\nSelected weight:")
print(
    f"XGBoost     : {best_weight:.2f}"
)
print(
    f"ExtraTrees  : {1.0 - best_weight:.2f}"
)
print(
    f"CV R²       : {best['CV_R2']:.4f}"
)
print(
    f"CV MAE      : {best['CV_MAE_MPa']:.2f} MPa"
)
print(
    f"CV RMSE     : {best['CV_RMSE_MPa']:.2f} MPa"
)

# ============================================================
# FINAL TRAINING ON ENTIRE DEVELOPMENT SET
# ============================================================

print("\n" + "=" * 80)
print("FINAL TRAINING ON DEVELOPMENT SET")
print("=" * 80)

xgb_final = XGBRegressor(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

et_final = ExtraTreesRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=1.0,
    random_state=42,
    n_jobs=-1
)

xgb_final.fit(
    X_dev_t,
    y_dev
)

et_final.fit(
    X_dev_t,
    y_dev
)

pred_xgb_test = xgb_final.predict(
    X_test_t
)

pred_et_test = et_final.predict(
    X_test_t
)

ensemble_test = (
    best_weight * pred_xgb_test +
    (1.0 - best_weight) * pred_et_test
)

# ============================================================
# FINAL TEST EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    ensemble_test
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        ensemble_test
    )
)

r2 = r2_score(
    y_test,
    ensemble_test
)

xgb_r2 = r2_score(
    y_test,
    pred_xgb_test
)

et_r2 = r2_score(
    y_test,
    pred_et_test
)

print("\n" + "=" * 80)
print("FINAL UNTOUCHED TEST RESULTS")
print("=" * 80)

print(
    f"XGBoost test R²    : {xgb_r2:.4f}"
)

print(
    f"ExtraTrees test R² : {et_r2:.4f}"
)

print(
    f"Ensemble test R²   : {r2:.4f}"
)

print(
    f"Ensemble MAE       : {mae:.2f} MPa"
)

print(
    f"Ensemble RMSE      : {rmse:.2f} MPa"
)

print(
    f"\nSelected weights: "
    f"XGB={best_weight:.2f}, "
    f"ET={1.0-best_weight:.2f}"
)

# ============================================================
# SAVE RESULTS
# ============================================================

cv_df.to_csv(
    "YS_Ensemble_GroupKFold_Weight_Selection.csv",
    index=False
)

pd.DataFrame({
    "record_index": test_idx,
    "actual_YS_MPa": y_test.values,
    "xgb_prediction_MPa": pred_xgb_test,
    "extratrees_prediction_MPa": pred_et_test,
    "ensemble_prediction_MPa": ensemble_test
}).to_csv(
    "YS_Ensemble_Final_Test_Predictions.csv",
    index=False
)

pd.DataFrame([{
    "XGB_weight": best_weight,
    "ET_weight": 1.0 - best_weight,
    "CV_R2": best["CV_R2"],
    "CV_MAE_MPa": best["CV_MAE_MPa"],
    "CV_RMSE_MPa": best["CV_RMSE_MPa"],
    "Test_R2": r2,
    "Test_MAE_MPa": mae,
    "Test_RMSE_MPa": rmse
}]).to_csv(
    "YS_Ensemble_Final_Metrics.csv",
    index=False
)

print("\nSaved:")
print("YS_Ensemble_GroupKFold_Weight_Selection.csv")
print("YS_Ensemble_Final_Test_Predictions.csv")
print("YS_Ensemble_Final_Metrics.csv")
