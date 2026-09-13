import pandas as pd
import numpy as np

from database.connection import engine


print("=" * 70)
print("ELONGATION DATASET PREPARATION")
print("=" * 70)


# =========================================================
# 1. LOAD DATA FROM POSTGRESQL
# =========================================================

query = """
SELECT *
FROM hea_mpea.dataset_v1_raw
WHERE elongation_tensile_pct IS NOT NULL
"""

print("\nLoading Elongation data from PostgreSQL...")

df = pd.read_sql(query, engine)

print("Original shape:", df.shape)


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


# =========================================================
# 3. CONVERT ELEMENTS TO NUMERIC
# =========================================================

for col in ELEMENTS:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# Missing elemental concentration = 0
df[ELEMENTS] = df[ELEMENTS].fillna(0)


# =========================================================
# 4. COMPOSITION CHECK
# =========================================================

composition_sum = df[ELEMENTS].sum(axis=1)

print("\nComposition sum statistics:")
print(composition_sum.describe())

invalid = (
    (composition_sum < 99) |
    (composition_sum > 101)
).sum()

print(
    "\nRecords with composition sum outside 99-101:",
    invalid
)


# =========================================================
# 5. TARGET CONVERSION
# =========================================================

TARGET = "elongation_tensile_pct"

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

df = df[
    df[TARGET].notna()
].copy()


# =========================================================
# 6. SELECT ML FEATURES
# =========================================================

FEATURES = (
    ELEMENTS +
    [
        TARGET,
        "composition_canonical",
        "test_type",
        "test_temperature_c",
        "phase",
        "processing_method",
        "alloy_class"
    ]
)

ml_df = df[FEATURES].copy()


# =========================================================
# 7. REMOVE EXACT DUPLICATES
# =========================================================

before = len(ml_df)

ml_df = ml_df.drop_duplicates()

after = len(ml_df)

print("\nExact duplicate rows removed:", before - after)


# =========================================================
# 8. FINAL STATISTICS
# =========================================================

print("\nFinal Elongation records:", len(ml_df))

print("\nElongation statistics:")
print(
    ml_df[TARGET].describe()
)

print(
    "\nUnique compositions:",
    ml_df["composition_canonical"].nunique()
)


# =========================================================
# 9. SAVE DATASET
# =========================================================

output_file = "Elongation_ML_Dataset.csv"

ml_df.to_csv(
    output_file,
    index=False
)

print("\nSaved:")
print(output_file)

print("\n" + "=" * 70)
print("ELONGATION DATASET PREPARATION COMPLETE")
print("=" * 70)