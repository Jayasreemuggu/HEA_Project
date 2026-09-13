import os
import random
import numpy as np
import pandas as pd
import torch

from torch_geometric.loader import DataLoader

from gnn.hea_graph import build_graph
from gnn_model import HEAYieldStrengthGNN


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = "YS_ML_Dataset.csv"

TRAIN_X = "X_train_YS.csv"
TEST_X = "X_test_YS.csv"

TRAIN_Y = "y_train_YS.csv"
TEST_Y = "y_test_YS.csv"

SEED = 42
BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 0.001

MODEL_DIR = "gnn_models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("HEA/MPEA GNN - EXACT XGBOOST SPLIT TRAINING")
print("=" * 70)

print("\nDevice:", device)


# ============================================================
# LOAD ORIGINAL FULL DATASET
# ============================================================

df = pd.read_csv(DATASET)

print("\nFull dataset:", df.shape)


# ============================================================
# LOAD XGBOOST TRAIN/TEST FEATURES
# ============================================================

X_train = pd.read_csv(TRAIN_X)
X_test = pd.read_csv(TEST_X)

y_train = pd.read_csv(TRAIN_Y)
y_test = pd.read_csv(TEST_Y)


print("\nX_train:", X_train.shape)
print("X_test :", X_test.shape)

print("\ny_train:", y_train.shape)
print("y_test :", y_test.shape)


# ============================================================
# RECOVER EXACT ROWS
#
# The split files contain the 35 elemental composition
# features. We match them back to the original dataset.
#
# We use the occurrence order of each composition vector.
# ============================================================

ELEMENTS = [
    "ag", "al", "b", "c", "ca", "co", "cr", "cu",
    "fe", "ga", "hf", "i", "li", "mg", "mn", "mo",
    "nb", "nd", "ni", "o", "pd", "re", "ru", "s",
    "sc", "si", "sn", "t", "ta", "ti", "v", "w", "y",
    "zn", "zr"
]


def make_key(row):
    return tuple(
        round(float(row[e]), 8)
        for e in ELEMENTS
    )


# Add keys
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
# MATCH ROWS WHILE PRESERVING OCCURRENCE
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
                f"Could not recover row {i}. "
                f"Target={target}, Key={key}"
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
# VALIDATE EXACT SPLIT
# ============================================================

print("\nRecovered rows:")
print("Training:", len(train_df))
print("Testing :", len(test_df))


train_groups = set(
    train_df[
        "composition_canonical"
    ].dropna()
)

test_groups = set(
    test_df[
        "composition_canonical"
    ].dropna()
)

overlap = (
    train_groups &
    test_groups
)

print(
    "\nTraining compositions:",
    len(train_groups)
)

print(
    "Testing compositions:",
    len(test_groups)
)

print(
    "Composition overlap:",
    len(overlap)
)

if len(overlap) != 0:

    raise RuntimeError(
        "Composition leakage detected!"
    )


# ============================================================
# TARGET NORMALIZATION
# TRAINING ONLY
# ============================================================

target_mean = train_df[
    "yield_strength_mpa"
].mean()

target_std = train_df[
    "yield_strength_mpa"
].std()


print(
    "\nTraining YS mean:",
    target_mean
)

print(
    "Training YS std:",
    target_std
)


# ============================================================
# TEMPERATURE NORMALIZATION
# TRAINING ONLY
# ============================================================

temperature_values = pd.to_numeric(
    train_df[
        "test_temperature_c"
    ],
    errors="coerce"
)

temperature_mean = (
    temperature_values.mean()
)

temperature_std = (
    temperature_values.std()
)

if pd.isna(temperature_mean):
    temperature_mean = 25.0

if pd.isna(temperature_std) or temperature_std == 0:
    temperature_std = 100.0


print(
    "\nTraining temperature mean:",
    temperature_mean
)

print(
    "Training temperature std:",
    temperature_std
)


# ============================================================
# BUILD TRAINING GRAPHS
# ============================================================

print(
    "\nBuilding training graphs..."
)

train_graphs = []

for _, row in train_df.iterrows():

    graph = build_graph(
        row,
        target_mean=target_mean,
        target_std=target_std,
        temperature_mean=temperature_mean,
        temperature_std=temperature_std
    )

    if graph is not None:
        train_graphs.append(graph)


print(
    "Training graphs:",
    len(train_graphs)
)


# ============================================================
# BUILD TEST GRAPHS
# ============================================================

print(
    "\nBuilding testing graphs..."
)

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


print(
    "Testing graphs:",
    len(test_graphs)
)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_graphs,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_graphs,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# MODEL
# ============================================================

model = HEAYieldStrengthGNN(
    node_features=7,
    hidden_dim=64,
    condition_dim=32,
    dropout=0.20
)

model = model.to(device)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

loss_function = torch.nn.MSELoss()


# ============================================================
# TRAINING
# ============================================================

best_loss = float("inf")
best_epoch = 0

print("\n" + "=" * 70)
print("STARTING EXACT-SPLIT GNN TRAINING")
print("=" * 70)


for epoch in range(
    1,
    EPOCHS + 1
):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch in train_loader:

        batch = batch.to(device)

        optimizer.zero_grad()

        predictions = model(
            batch.x,
            batch.edge_index,
            batch.batch,
            batch.conditions
        )

        loss = loss_function(
            predictions,
            batch.y
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        total_loss += (
            loss.item() *
            batch.num_graphs
        )

        total_samples += (
            batch.num_graphs
        )

    train_loss = (
        total_loss /
        total_samples
    )


    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    model.eval()

    total_test_loss = 0.0
    total_test_samples = 0

    with torch.no_grad():

        for batch in test_loader:

            batch = batch.to(device)

            predictions = model(
                batch.x,
                batch.edge_index,
                batch.batch,
                batch.conditions
            )

            loss = loss_function(
                predictions,
                batch.y
            )

            total_test_loss += (
                loss.item() *
                batch.num_graphs
            )

            total_test_samples += (
                batch.num_graphs
            )

    test_loss = (
        total_test_loss /
        total_test_samples
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if test_loss < best_loss:

        best_loss = test_loss
        best_epoch = epoch

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "target_mean":
                    target_mean,

                "target_std":
                    target_std,

                "temperature_mean":
                    temperature_mean,

                "temperature_std":
                    temperature_std,

                "node_features":
                    7,

                "hidden_dim":
                    64,

                "condition_dim":
                    32,

                "dropout":
                    0.20,

                "train_records":
                    len(train_df),

                "test_records":
                    len(test_df),

                "seed":
                    SEED
            },
            os.path.join(
                MODEL_DIR,
                "HEA_YieldStrength_GNN_EXACT_best.pt"
            )
        )


    # --------------------------------------------------------
    # PRINT PROGRESS
    # --------------------------------------------------------

    if (
        epoch == 1
        or epoch % 10 == 0
    ):

        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.5f} | "
            f"Test Loss: {test_loss:.5f}"
        )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("EXACT-SPLIT GNN TRAINING COMPLETED")
print("=" * 70)

print(
    "\nBest test loss:",
    best_loss
)

print(
    "Best epoch:",
    best_epoch
)

print(
    "\nTraining records:",
    len(train_df)
)

print(
    "Testing records:",
    len(test_df)
)

print(
    "\nModel saved:"
)

print(
    "gnn_models/HEA_YieldStrength_GNN_EXACT_best.pt"
)

print("\nNext step: evaluate exact-split GNN.")
