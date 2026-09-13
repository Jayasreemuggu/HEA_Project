import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


print("=" * 70)
print("ELONGATION - GROUPED TRAIN / TEST SPLIT")
print("=" * 70)


# =========================================================
# 1. LOAD DATA
# =========================================================

df = pd.read_csv(
    "Elongation_ML_Dataset.csv"
)

print("\nLoaded dataset:", df.shape)


# =========================================================
# 2. ELEMENT FEATURES
# =========================================================

ELEMENTS = [
    "al_at_pct", "b_at_pct", "c_at_pct", "co_at_pct",
    "cr_at_pct", "cu_at_pct", "fe_at_pct", "mn_at_pct",
    "mo_at_pct", "nb_at_pct", "ni_at_pct", "si_at_pct",
    "ta_at_pct", "ti_at_pct", "v_at_pct", "w_at_pct",
    "zr_at_pct", "ag_at_pct", "ca_at_pct", "ga_at_pct",
    "hf_at_pct", "i_at_pct", "li_at_pct", "mg_at_pct",
    "nd_at_pct", "o_at_pct", "pd_at_pct", "re_at_pct",
    "ru_at_pct", "s_at_pct", "sc_at_pct", "sn_at_pct",
    "t_at_pct", "y_at_pct", "zn_at_pct"
]

TARGET = "elongation_tensile_pct"


# =========================================================
# 3. FEATURES / TARGET
# =========================================================

X = df[ELEMENTS].copy()

y = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

groups = df["composition_canonical"]


# =========================================================
# 4. GROUPED SPLIT
# =========================================================

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


X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()

y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()


# =========================================================
# 5. CHECK COMPOSITION OVERLAP
# =========================================================

train_compositions = set(
    df.iloc[train_idx]["composition_canonical"]
)

test_compositions = set(
    df.iloc[test_idx]["composition_canonical"]
)

overlap = train_compositions.intersection(
    test_compositions
)


# =========================================================
# 6. RESULTS
# =========================================================

print("\n" + "=" * 70)
print("GROUPED SPLIT RESULTS")
print("=" * 70)

print("Train records:", len(X_train))
print("Test records :", len(X_test))

print(
    "Unique train compositions:",
    len(train_compositions)
)

print(
    "Unique test compositions:",
    len(test_compositions)
)

print(
    "Composition overlap:",
    len(overlap)
)


# =========================================================
# 7. SAVE
# =========================================================

X_train.to_csv(
    "X_train_Elongation.csv",
    index=False
)

X_test.to_csv(
    "X_test_Elongation.csv",
    index=False
)

y_train.to_csv(
    "y_train_Elongation.csv",
    index=False
)

y_test.to_csv(
    "y_test_Elongation.csv",
    index=False
)


print("\nSaved:")
print("1. X_train_Elongation.csv")
print("2. X_test_Elongation.csv")
print("3. y_train_Elongation.csv")
print("4. y_test_Elongation.csv")

print("\n" + "=" * 70)
print("ELONGATION GROUPED SPLIT COMPLETE")
print("=" * 70)