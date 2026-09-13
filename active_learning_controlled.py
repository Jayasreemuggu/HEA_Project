import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

BASE = "/mnt/c/Users/jayam/Downloads/IITH_ML_project"

DATA_FILE = os.path.join(
    BASE,
    "HEA_MPEA_ML_Ready_Deduplicated.csv"
)

OUT_DIR = os.path.join(
    BASE,
    "active_learning",
    "results"
)

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# CONTROLLED EXPERIMENT
# ============================================================

SEEDS = [13, 21, 42, 77, 101]

INITIAL_COMPOSITIONS = 200
ACQUIRE_COMPOSITIONS = 50
N_ROUNDS = 6

N_ENSEMBLE = 5

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

df = df[
    df["YS_Tensile_MPa"].notna()
].copy()

df = df.reset_index(drop=True)

groups = df["Composition_Canonical"].copy()

missing_groups = groups.isna()

groups.loc[missing_groups] = (
    "UNKNOWN_" +
    df.index[missing_groups].astype(str)
)

groups = groups.astype(str)

print("=" * 80)
print("CONTROLLED MULTI-SEED ACTIVE LEARNING")
print("=" * 80)

print("YS records:", len(df))
print(
    "Unique compositions:",
    groups.nunique()
)

# ============================================================
# FEATURES
# ============================================================

element_features = [
    c for c in df.columns
    if c.endswith("_at_pct")
]

numeric_features = [
    "Test_Temperature_C",
    "VEC",
    "Atomic_Size_Mismatch",
    "Mixing_Enthalpy",
    "Mixing_Entropy",
    "Density_Exp_g_cm3",
    "Grain_Size_um",
    "Precipitate_Size_nm",
    "Matrix_Volume_pct"
]

categorical_features = [
    "Test_Type",
    "Phase",
    "Processing_Method",
    "Alloy_Class",
    "Equilibrium_Condition",
    "Single_Multiphase",
    "Precipitate_Info"
]

FEATURES = (
    element_features
    + numeric_features
    + categorical_features
)

X = df[FEATURES].copy()

y = df["YS_Tensile_MPa"].astype(float)

# ============================================================
# PREPROCESSOR
# ============================================================

def make_preprocessor():

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

    return ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            numeric_features + element_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ])

# ============================================================
# MODEL
# ============================================================

def make_model(seed):

    return XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.03,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=-1
    )

# ============================================================
# TRAIN / EVALUATE
# ============================================================

def train_evaluate(
    train_idx,
    test_idx,
    seed
):

    prep = make_preprocessor()

    prep.fit(
        X.iloc[train_idx]
    )

    X_train_t = prep.transform(
        X.iloc[train_idx]
    )

    X_test_t = prep.transform(
        X.iloc[test_idx]
    )

    model = make_model(seed)

    model.fit(
        X_train_t,
        y.iloc[train_idx]
    )

    pred = model.predict(
        X_test_t
    )

    actual = y.iloc[
        test_idx
    ].values

    return {
        "MAE_MPa": mean_absolute_error(
            actual,
            pred
        ),
        "RMSE_MPa": np.sqrt(
            mean_squared_error(
                actual,
                pred
            )
        ),
        "R2": r2_score(
            actual,
            pred
        )
    }

# ============================================================
# UNCERTAINTY ACQUISITION
# ============================================================

def uncertainty_acquisition(
    train_idx,
    candidate_groups,
    seed
):

    candidate_idx = np.where(
        groups.isin(
            candidate_groups
        )
    )[0]

    # Keep only records belonging to the
    # candidate training pool.
    train_set = set(
        train_idx.tolist()
    )

    candidate_idx = np.array([
        i for i in candidate_idx
        if i not in train_set
    ])

    if len(candidate_idx) == 0:
        return []

    rng = np.random.RandomState(
        seed
    )

    predictions = []

    for ensemble_id in range(
        N_ENSEMBLE
    ):

        bootstrap_positions = rng.choice(
            len(train_idx),
            size=len(train_idx),
            replace=True
        )

        bootstrap_idx = train_idx[
            bootstrap_positions
        ]

        prep = make_preprocessor()

        prep.fit(
            X.iloc[bootstrap_idx]
        )

        X_train_t = prep.transform(
            X.iloc[bootstrap_idx]
        )

        X_candidate_t = prep.transform(
            X.iloc[candidate_idx]
        )

        model = make_model(
            seed + 1000 + ensemble_id
        )

        model.fit(
            X_train_t,
            y.iloc[bootstrap_idx]
        )

        predictions.append(
            model.predict(
                X_candidate_t
            )
        )

    predictions = np.asarray(
        predictions
    )

    uncertainty = predictions.std(
        axis=0
    )

    candidate_df = pd.DataFrame({
        "idx": candidate_idx,
        "group": groups.iloc[
            candidate_idx
        ].values,
        "uncertainty": uncertainty
    })

    # Aggregate record uncertainty at composition level.
    group_scores = (
        candidate_df
        .groupby("group")[
            "uncertainty"
        ]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    return group_scores.index.tolist()

# ============================================================
# MAIN EXPERIMENT
# ============================================================

records = []

for seed in SEEDS:

    print("\n" + "=" * 80)
    print(
        f"SEED {seed}"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Fixed composition-grouped test set
    # --------------------------------------------------------

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=seed
    )

    train_pool_idx, test_idx = next(
        gss.split(
            X,
            y,
            groups=groups
        )
    )

    train_pool_idx = np.asarray(
        train_pool_idx
    )

    test_idx = np.asarray(
        test_idx
    )

    test_groups = set(
        groups.iloc[
            test_idx
        ]
    )

    train_groups = (
        groups.iloc[
            train_pool_idx
        ]
        .drop_duplicates()
        .values
    )

    print(
        "Test records:",
        len(test_idx)
    )

    print(
        "Test compositions:",
        len(test_groups)
    )

    print(
        "Training-pool compositions:",
        len(train_groups)
    )

    print(
        "Composition overlap:",
        len(
            test_groups
            &
            set(train_groups)
        )
    )

    # --------------------------------------------------------
    # Initial composition pool
    # --------------------------------------------------------

    rng = np.random.RandomState(
        seed
    )

    initial_groups = rng.choice(
        train_groups,
        size=INITIAL_COMPOSITIONS,
        replace=False
    )

    initial_idx = np.where(
        groups.isin(
            initial_groups
        )
        &
        groups.index.isin(
            train_pool_idx
        )
    )[0]

    strategies = {
        "Random": initial_idx.copy(),
        "Uncertainty": initial_idx.copy()
    }

    # --------------------------------------------------------
    # Active-learning rounds
    # --------------------------------------------------------

    for round_id in range(
        N_ROUNDS + 1
    ):

        for strategy in [
            "Random",
            "Uncertainty"
        ]:

            selected_idx = np.unique(
                strategies[strategy]
            )

            metrics = train_evaluate(
                selected_idx,
                test_idx,
                seed + round_id
            )

            records.append({
                "Seed": seed,
                "Round": round_id,
                "Strategy": strategy,
                "Training_Records": len(
                    selected_idx
                ),
                "Training_Compositions": groups.iloc[
                    selected_idx
                ].nunique(),
                "Test_Records": len(
                    test_idx
                ),
                "Test_Compositions": len(
                    test_groups
                ),
                "MAE_MPa": metrics[
                    "MAE_MPa"
                ],
                "RMSE_MPa": metrics[
                    "RMSE_MPa"
                ],
                "R2": metrics[
                    "R2"
                ]
            })

            print(
                f"Round {round_id} | "
                f"{strategy:12s} | "
                f"Compositions="
                f"{groups.iloc[selected_idx].nunique():3d} | "
                f"Records="
                f"{len(selected_idx):4d} | "
                f"MAE="
                f"{metrics['MAE_MPa']:.2f} | "
                f"RMSE="
                f"{metrics['RMSE_MPa']:.2f} | "
                f"R2="
                f"{metrics['R2']:.4f}"
            )

            if round_id == N_ROUNDS:
                continue

            # ------------------------------------------------
            # Remaining compositions
            # ------------------------------------------------

            selected_groups = set(
                groups.iloc[
                    selected_idx
                ]
            )

            remaining_groups = [
                g for g in train_groups
                if g not in selected_groups
            ]

            n_acquire = min(
                ACQUIRE_COMPOSITIONS,
                len(remaining_groups)
            )

            # ------------------------------------------------
            # RANDOM
            # ------------------------------------------------

            if strategy == "Random":

                new_groups = rng.choice(
                    remaining_groups,
                    size=n_acquire,
                    replace=False
                )

            # ------------------------------------------------
            # UNCERTAINTY
            # ------------------------------------------------

            else:

                ranked_groups = (
                    uncertainty_acquisition(
                        selected_idx,
                        remaining_groups,
                        seed + round_id * 100
                    )
                )

                new_groups = ranked_groups[
                    :n_acquire
                ]

            new_idx = np.where(
                groups.isin(
                    new_groups
                )
                &
                groups.index.isin(
                    train_pool_idx
                )
            )[0]

            strategies[strategy] = np.concatenate([
                selected_idx,
                new_idx
            ])

# ============================================================
# SAVE RAW RESULTS
# ============================================================

raw_df = pd.DataFrame(
    records
)

raw_file = os.path.join(
    OUT_DIR,
    "YS_Active_Learning_Controlled_Raw.csv"
)

raw_df.to_csv(
    raw_file,
    index=False
)

# ============================================================
# MEAN ± STD
# ============================================================

summary = (
    raw_df
    .groupby(
        ["Round", "Strategy"]
    )
    .agg(
        Mean_MAE_MPa=(
            "MAE_MPa",
            "mean"
        ),
        Std_MAE_MPa=(
            "MAE_MPa",
            "std"
        ),
        Mean_RMSE_MPa=(
            "RMSE_MPa",
            "mean"
        ),
        Std_RMSE_MPa=(
            "RMSE_MPa",
            "std"
        ),
        Mean_R2=(
            "R2",
            "mean"
        ),
        Std_R2=(
            "R2",
            "std"
        ),
        Mean_Training_Records=(
            "Training_Records",
            "mean"
        ),
        Mean_Training_Compositions=(
            "Training_Compositions",
            "mean"
        )
    )
    .reset_index()
)

summary_file = os.path.join(
    OUT_DIR,
    "YS_Active_Learning_Controlled_Summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)

# ============================================================
# FINAL ROUND
# ============================================================

final = summary[
    summary["Round"] == N_ROUNDS
].copy()

final = final.sort_values(
    "Mean_R2",
    ascending=False
)

final_file = os.path.join(
    OUT_DIR,
    "YS_Active_Learning_Controlled_Final.csv"
)

final.to_csv(
    final_file,
    index=False
)

# ============================================================
# IMPROVEMENT
# ============================================================

random_row = final[
    final["Strategy"] == "Random"
].iloc[0]

unc_row = final[
    final["Strategy"] == "Uncertainty"
].iloc[0]

improvement = pd.DataFrame({
    "Metric": [
        "Mean_R2_Improvement",
        "Mean_MAE_Improvement_MPa",
        "Mean_RMSE_Improvement_MPa"
    ],
    "Value": [
        unc_row["Mean_R2"]
        - random_row["Mean_R2"],

        random_row["Mean_MAE_MPa"]
        - unc_row["Mean_MAE_MPa"],

        random_row["Mean_RMSE_MPa"]
        - unc_row["Mean_RMSE_MPa"]
    ]
})

improvement_file = os.path.join(
    OUT_DIR,
    "YS_Active_Learning_Controlled_Improvement.csv"
)

improvement.to_csv(
    improvement_file,
    index=False
)

# ============================================================
# R2 PLOT
# ============================================================

plt.figure(figsize=(9, 6))

for strategy in [
    "Random",
    "Uncertainty"
]:

    sub = summary[
        summary["Strategy"] == strategy
    ]

    plt.errorbar(
        sub["Mean_Training_Compositions"],
        sub["Mean_R2"],
        yerr=sub["Std_R2"],
        marker="o",
        capsize=4,
        label=strategy
    )

plt.xlabel(
    "Training Compositions"
)

plt.ylabel(
    "Mean R²"
)

plt.title(
    "Controlled Multi-Seed Active Learning"
)

plt.legend()

plt.grid(
    alpha=0.25
)

plt.tight_layout()

r2_plot = os.path.join(
    OUT_DIR,
    "YS_Active_Learning_Controlled_R2.png"
)

plt.savefig(
    r2_plot,
    dpi=200
)

plt.close()

# ============================================================
# MAE PLOT
# ============================================================

plt.figure(figsize=(9, 6))

for strategy in [
    "Random",
    "Uncertainty"
]:

    sub = summary[
        summary["Strategy"] == strategy
    ]

    plt.errorbar(
        sub["Mean_Training_Compositions"],
        sub["Mean_MAE_MPa"],
        yerr=sub["Std_MAE_MPa"],
        marker="o",
        capsize=4,
        label=strategy
    )

plt.xlabel(
    "Training Compositions"
)

plt.ylabel(
    "Mean MAE (MPa)"
)

plt.title(
    "Controlled Active Learning Error"
)

plt.legend()

plt.grid(
    alpha=0.25
)

plt.tight_layout()

mae_plot = os.path.join(
    OUT_DIR,
    "YS_Active_Learning_Controlled_MAE.png"
)

plt.savefig(
    mae_plot,
    dpi=200
)

plt.close()

# ============================================================
# PRINT
# ============================================================

print("\n" + "=" * 80)
print("CONTROLLED ACTIVE LEARNING COMPLETE")
print("=" * 80)

print("\nFINAL ROUND — MEAN ± STD")

for _, row in final.iterrows():

    print(
        f"{row['Strategy']:12s} | "
        f"MAE = {row['Mean_MAE_MPa']:.2f} "
        f"+/- {row['Std_MAE_MPa']:.2f} MPa | "
        f"RMSE = {row['Mean_RMSE_MPa']:.2f} "
        f"+/- {row['Std_RMSE_MPa']:.2f} MPa | "
        f"R2 = {row['Mean_R2']:.4f} "
        f"+/- {row['Std_R2']:.4f}"
    )

print("\nUncertainty vs Random:")

print(
    f"R2 improvement  : "
    f"{improvement.iloc[0]['Value']:.4f}"
)

print(
    f"MAE improvement : "
    f"{improvement.iloc[1]['Value']:.2f} MPa"
)

print(
    f"RMSE improvement: "
    f"{improvement.iloc[2]['Value']:.2f} MPa"
)

print("\nFiles created:")

for path in [
    raw_file,
    summary_file,
    final_file,
    improvement_file,
    r2_plot,
    mae_plot
]:
    print(path)

print("=" * 80)
