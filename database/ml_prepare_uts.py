import pandas as pd
from sqlalchemy import text

from database.connection import engine


print("=" * 70)
print("UTS DATASET PREPARATION")
print("=" * 70)

# =========================================================
# 1. LOAD UTS RECORDS
# =========================================================

query = """
SELECT *
FROM hea_mpea.dataset_v1_raw
WHERE uts_tensile_mpa IS NOT NULL
"""

print("\nLoading UTS data from PostgreSQL...")

df = pd.read_sql(
    text(query),
    engine
)

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

for element in ELEMENTS:

    df[element] = pd.to_numeric(
        df[element],
        errors="coerce"
    )


# =========================================================
# 4. COMPOSITION CHECK
# =========================================================

composition_sum = (
    df[ELEMENTS]
    .fillna(0)
    .sum(axis=1)
)

print("\nComposition sum statistics:")
print(composition_sum.describe())

print(
    "\nRecords with composition sum outside 99-101:",
    ((composition_sum < 99) | (composition_sum > 101)).sum()
)


# =========================================================
# 5. FILL ABSENT ELEMENTS WITH ZERO
# =========================================================

for element in ELEMENTS:

    df[element] = df[element].fillna(0)


# =========================================================
# 6. SELECT ML COLUMNS
# =========================================================

ML_COLUMNS = ELEMENTS + [
    "uts_tensile_mpa",
    "composition_canonical",
    "test_type",
    "test_temperature_c",
    "phase",
    "processing_method",
    "alloy_class"
]

ml_df = df[ML_COLUMNS].copy()


# =========================================================
# 7. REMOVE MISSING TARGETS
# =========================================================

ml_df["uts_tensile_mpa"] = pd.to_numeric(
    ml_df["uts_tensile_mpa"],
    errors="coerce"
)

ml_df = ml_df.dropna(
    subset=["uts_tensile_mpa"]
).reset_index(drop=True)


# =========================================================
# 8. TARGET STATISTICS
# =========================================================

print("\nFinal UTS records:", len(ml_df))

print("\nUTS statistics:")
print(
    ml_df["uts_tensile_mpa"].describe()
)


# =========================================================
# 9. UNIQUE COMPOSITIONS
# =========================================================

print(
    "\nUnique compositions:",
    ml_df["composition_canonical"].nunique()
)


# =========================================================
# 10. SAVE
# =========================================================

output_file = "UTS_ML_Dataset.csv"

ml_df.to_csv(
    output_file,
    index=False
)

print("\nSaved:")
print(output_file)

print("\n" + "=" * 70)
print("UTS DATASET PREPARATION COMPLETE")
print("=" * 70)