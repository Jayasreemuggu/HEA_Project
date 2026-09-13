import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# Load YS dataset
df = pd.read_csv("YS_ML_Dataset.csv")

# 35 elemental composition features
ELEMENTS = [
    "ag", "al", "b", "c", "ca", "co", "cr", "cu",
    "fe", "ga", "hf", "i", "li", "mg", "mn", "mo",
    "nb", "nd", "ni", "o", "pd", "re", "ru", "s",
    "sc", "si", "sn", "t", "ta", "ti", "v", "w", "y",
    "zn", "zr"
]

TARGET = "yield_strength_mpa"

X = df[ELEMENTS]
y = df[TARGET]

# Use canonical composition as the grouping variable
groups = df["composition_canonical"]

# 80/20 grouped split
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()

y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()

print("Train/Test Split")
print("----------------")

print("Total records:", len(df))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining YS mean:", y_train.mean())
print("Testing YS mean:", y_test.mean())

print("\nUnique compositions:")
print("Training:", groups.iloc[train_idx].nunique())
print("Testing:", groups.iloc[test_idx].nunique())

# Verify no composition leakage
train_groups = set(groups.iloc[train_idx])
test_groups = set(groups.iloc[test_idx])

overlap = train_groups.intersection(test_groups)

print("\nComposition overlap:", len(overlap))

if len(overlap) == 0:
    print("No composition leakage detected.")
else:
    print("WARNING: Composition leakage detected!")

# Save
X_train.to_csv("X_train_YS.csv", index=False)
X_test.to_csv("X_test_YS.csv", index=False)

y_train.to_csv("y_train_YS.csv", index=False)
y_test.to_csv("y_test_YS.csv", index=False)

print("\nFiles saved:")
print("X_train_YS.csv")
print("X_test_YS.csv")
print("y_train_YS.csv")
print("y_test_YS.csv")