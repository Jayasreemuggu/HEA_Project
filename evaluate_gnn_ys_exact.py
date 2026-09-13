import numpy as np
import pandas as pd
import torch

from torch_geometric.loader import DataLoader

from gnn.hea_graph import build_graph
from gnn_model import HEAYieldStrengthGNN


# ============================================================
# CONFIG
# ============================================================

DATASET = "YS_ML_Dataset.csv"
TRAIN_X = "X_train_YS.csv"
TEST_X = "X_test_YS.csv"
TRAIN_Y = "y_train_YS.csv"
TEST_Y = "y_test_YS.csv"

MODEL_PATH = (
    "gnn_models/"
    "HEA_YieldStrength_GNN_EXACT_best.pt"
)

BATCH_SIZE = 32

ELEMENTS = [
    "ag", "al", "b", "c", "ca", "co", "cr", "cu",
    "fe", "ga", "hf", "i", "li", "mg", "mn", "mo",
    "nb", "nd", "ni", "o", "pd", "re", "ru", "s",
    "sc", "si", "sn", "t", "ta", "ti", "v", "w", "y",
    "zn", "zr"
]


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("HEA/MPEA GNN - EXACT-SPLIT EVALUATION")
print("=" * 70)

print("\nDevice:", device)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATASET)

X_train = pd.read_csv(TRAIN_X)
X_test = pd.read_csv(TEST_X)

y_train = pd.read_csv(TRAIN_Y)
y_test = pd.read_csv(TEST_Y)


# ============================================================
# CREATE COMPOSITION KEYS
# ============================================================

def make_key(row):
    return tuple(
        round(float(row[e]), 8)
        for e in ELEMENTS
    )


df["_composition_key"] = df.apply(
    make_key,
    axis=1
)

X_train["_composition_key"] = X_train.apply(
    make_key,
    axis=1
)

X_test["_composition_key"] = X_test.apply(
    make_key,
    axis=1
)


# ============================================================
# RECOVER ORIGINAL ROWS
# ============================================================

def recover_rows(
    x_split,
    y_split,
    full_df
):

    used = set()
    recovered = []

    y_values = y_split.iloc[:, 0].values

    for i in range(len(x_split)):

        key = x_split.iloc[i][
            "_composition_key"
        ]

        target = float(
            y_values[i]
        )

        candidates = full_df.index[
            full_df["_composition_key"] == key
        ].tolist()

        match = None

        for idx in candidates:

            if idx in used:
                continue

            full_target = float(
                full_df.loc[
                    idx,
                    "yield_strength_mpa"
                ]
            )

            if abs(
                full_target - target
            ) < 1e-8:

                match = idx
                break

        if match is None:

            raise RuntimeError(
                f"Could not recover split row {i}"
            )

        used.add(match)
        recovered.append(match)

    return full_df.loc[
        recovered
    ].copy()


train_df = recover_rows(
    X_train,
    y_train,
    df
)

test_df = recover_rows(
    X_test,
    y_test,
    df
)


# ============================================================
# VERIFY EXACT SPLIT
# ============================================================

print("\nTraining records:", len(train_df))
print("Testing records :", len(test_df))

train_groups = set(
    train_df[
        "composition_canonical"
    ]
)

test_groups = set(
    test_df[
        "composition_canonical"
    ]
)

overlap = train_groups & test_groups

print(
    "Composition overlap:",
    len(overlap)
)

if len(overlap) != 0:
    raise RuntimeError(
        "Composition leakage detected!"
    )


# ============================================================
# NORMALIZATION PARAMETERS
# ============================================================

target_mean = train_df[
    "yield_strength_mpa"
].mean()

target_std = train_df[
    "yield_strength_mpa"
].std()

temperature_values = pd.to_numeric(
    train_df[
        "test_temperature_c"
    ],
    errors="coerce"
)

temperature_mean = temperature_values.mean()
temperature_std = temperature_values.std()

if pd.isna(temperature_mean):
    temperature_mean = 25.0

if pd.isna(temperature_std) or temperature_std == 0:
    temperature_std = 100.0


# ============================================================
# BUILD TEST GRAPHS
# ============================================================

test_graphs = []

for _, row in test_df.iterrows():

    graph = build_graph(
        row,
        target_mean=target_mean,
        target_std=target_std,
        temperature_mean=temperature_mean,
        temperature_std=temperature_std
    )

    if graph is not None:
        test_graphs.append(graph)


test_loader = DataLoader(
    test_graphs,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

model = HEAYieldStrengthGNN(
    node_features=checkpoint["node_features"],
    hidden_dim=checkpoint["hidden_dim"],
    condition_dim=checkpoint["condition_dim"],
    dropout=checkpoint["dropout"]
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()


# ============================================================
# PREDICTIONS
# ============================================================

predictions = []
actuals = []

with torch.no_grad():

    for batch in test_loader:

        batch = batch.to(device)

        pred = model(
            batch.x,
            batch.edge_index,
            batch.batch,
            batch.conditions
        )

        predictions.extend(
            pred.cpu().numpy()
        )

        actuals.extend(
            batch.y.cpu().numpy()
        )


predictions = np.array(predictions)
actuals = np.array(actuals)


# ============================================================
# CONVERT TO MPa
# ============================================================

predictions_mpa = (
    predictions * target_std
    + target_mean
)

actuals_mpa = (
    actuals * target_std
    + target_mean
)


# ============================================================
# METRICS
# ============================================================

errors = (
    actuals_mpa -
    predictions_mpa
)

mae = np.mean(
    np.abs(errors)
)

rmse = np.sqrt(
    np.mean(errors ** 2)
)

ss_res = np.sum(
    errors ** 2
)

ss_tot = np.sum(
    (
        actuals_mpa -
        np.mean(actuals_mpa)
    ) ** 2
)

r2 = (
    1 -
    ss_res / ss_tot
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL EXACT-SPLIT GNN RESULTS")
print("=" * 70)

print(
    f"\nMAE  : {mae:.2f} MPa"
)

print(
    f"RMSE : {rmse:.2f} MPa"
)

print(
    f"R²   : {r2:.4f}"
)

print(
    "\nTest records:",
    len(actuals_mpa)
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

results = test_df[
    [
        "record_id",
        "alloy_name",
        "composition_canonical",
        "yield_strength_mpa"
    ]
].copy()

results["GNN_Predicted_YS_MPa"] = predictions_mpa

results["Error_MPa"] = (
    results["yield_strength_mpa"]
    - results["GNN_Predicted_YS_MPa"]
)

results["Absolute_Error_MPa"] = np.abs(
    results["Error_MPa"]
)

results.to_csv(
    "GNN_YieldStrength_EXACT_Predictions.csv",
    index=False
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame(
    {
        "Model": [
            "HEA/MPEA GNN"
        ],
        "MAE_MPa": [
            mae
        ],
        "RMSE_MPa": [
            rmse
        ],
        "R2": [
            r2
        ],
        "Test_Records": [
            len(actuals_mpa)
        ],
        "Test_Compositions": [
            test_df[
                "composition_canonical"
            ].nunique()
        ]
    }
)

metrics.to_csv(
    "GNN_YieldStrength_EXACT_Metrics.csv",
    index=False
)


print("\nSaved:")
print(
    "GNN_YieldStrength_EXACT_Predictions.csv"
)
print(
    "GNN_YieldStrength_EXACT_Metrics.csv"
)

print("\n" + "=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)
