import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

print("=" * 70)
print("UTS - GROUPED TRAIN / TEST SPLIT")
print("=" * 70)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv("UTS_ML_Dataset.csv")

print("\nLoaded dataset:", df.shape)

# ---------------------------------------------------------
# 2. FEATURES
# ---------------------------------------------------------

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

TARGET = "uts_tensile_mpa"

# ---------------------------------------------------------
# 3. PREPARE X, y AND GROUPS
# ---------------------------------------------------------

X = df[ELEMENTS].copy()

y = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

groups = df["composition_canonical"]

# ---------------------------------------------------------
# 4. GROUPED SPLIT
# ---------------------------------------------------------

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    gss.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

groups_train = groups.iloc[train_idx]
groups_test = groups.iloc[test_idx]

# ---------------------------------------------------------
# 5. CHECK SPLIT
# ---------------------------------------------------------

print("\nTrain records:", len(X_train))
print("Test records :", len(X_test))

print(
    "Unique train compositions:",
    groups_train.nunique()
)

print(
    "Unique test compositions:",
    groups_test.nunique()
)

overlap = set(groups_train).intersection(
    set(groups_test)
)

print(
    "Composition overlap:",
    len(overlap)
)

# ---------------------------------------------------------
# 6. SAVE FILES
# ---------------------------------------------------------

X_train.to_csv(
    "X_train_UTS.csv",
    index=False
)

X_test.to_csv(
    "X_test_UTS.csv",
    index=False
)

y_train.to_csv(
    "y_train_UTS.csv",
    index=False
)

y_test.to_csv(
    "y_test_UTS.csv",
    index=False
)

print("\nSaved:")
print("1. X_train_UTS.csv")
print("2. X_test_UTS.csv")
print("3. y_train_UTS.csv")
print("4. y_test_UTS.csv")

print("\n" + "=" * 70)
print("UTS GROUPED SPLIT COMPLETE")
print("=" * 70)