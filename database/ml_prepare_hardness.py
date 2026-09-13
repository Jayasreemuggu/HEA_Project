import pandas as pd
import numpy as np

from sqlalchemy import text
from database.connection import engine


print("=" * 70)
print("HARDNESS DATA PREPARATION")
print("Experimental Hardness (HV)")
print("=" * 70)


# ============================================================
# 1. LOAD DATA FROM POSTGRESQL
# ============================================================

query = """
SELECT
    d.*,

    c.al, c.b, c.c, c.co, c.cr, c.cu, c.fe, c.mn,
    c.mo, c.nb, c.ni, c.si, c.ta, c.ti, c.v, c.w,
    c.zr, c.ag, c.ca, c.ga, c.hf, c.i, c.li, c.mg,
    c.nd, c.o, c.pd, c.re, c.ru, c.s, c.sc, c.sn,
    c.t, c.y, c.zn

FROM hea_mpea.dataset_v1_raw d

LEFT JOIN hea_mpea.alloys a
    ON d.composition_canonical = a.composition_canonical

LEFT JOIN hea_mpea.compositions c
    ON a.alloy_id = c.alloy_id

WHERE d.hardness_hv IS NOT NULL
"""

with engine.connect() as conn:
    df = pd.read_sql(
        text(query),
        conn
    )

print("\nOriginal Hardness records:", df.shape)


# ============================================================
# 2. ELEMENT FEATURES
# ============================================================

element_features = [
    "al", "b", "c", "co", "cr", "cu", "fe", "mn",
    "mo", "nb", "ni", "si", "ta", "ti", "v", "w",
    "zr", "ag", "ca", "ga", "hf", "i", "li", "mg",
    "nd", "o", "pd", "re", "ru", "s", "sc", "sn",
    "t", "y", "zn"
]


# ============================================================
# 3. CHECK COMPOSITION
# ============================================================

for col in element_features:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


composition_sum = df[
    element_features
].sum(axis=1)


print("\nComposition sum statistics:")
print(composition_sum.describe())


invalid = (
    (composition_sum < 99)
    |
    (composition_sum > 101)
).sum()


print(
    "\nRecords with composition sum outside 99-101:",
    invalid
)


# ============================================================
# 4. REMOVE INVALID COMPOSITIONS
# ============================================================

df = df[
    (composition_sum >= 99)
    &
    (composition_sum <= 101)
].copy()


print(
    "Records after composition validation:",
    len(df)
)


# ============================================================
# 5. REMOVE EXACT DUPLICATES
# ============================================================

before = len(df)

df = df.drop_duplicates()

removed = before - len(df)

print(
    "Exact duplicate rows removed:",
    removed
)


# ============================================================
# 6. TARGET STATISTICS
# ============================================================

df["hardness_hv"] = pd.to_numeric(
    df["hardness_hv"],
    errors="coerce"
)


print("\nHardness statistics:")

print(
    df["hardness_hv"].describe()
)


# ============================================================
# 7. UNIQUE COMPOSITIONS
# ============================================================

print(
    "\nUnique compositions:",
    df["composition_canonical"].nunique()
)


# ============================================================
# 8. MISSING COMPOSITION VALUES
# ============================================================

missing_composition = df[
    element_features
].isna().all(axis=1).sum()


print(
    "Records with completely missing composition:",
    missing_composition
)


# ============================================================
# 9. FILL MISSING ELEMENTS WITH ZERO
# ============================================================

df[element_features] = df[
    element_features
].fillna(0)


# ============================================================
# 10. FINAL COLUMN SELECTION
# ============================================================

final_columns = [
    "record_id",
    "composition_canonical",
    "hardness_hv"
] + element_features


# Keep only columns that exist
final_columns = [
    c for c in final_columns
    if c in df.columns
]


df_final = df[
    final_columns
].copy()


# ============================================================
# 11. SAVE DATASET
# ============================================================

output_file = "Hardness_ML_Dataset.csv"

df_final.to_csv(
    output_file,
    index=False
)


print("\n" + "=" * 70)
print("HARDNESS DATASET READY")
print("=" * 70)

print(
    "Final records:",
    len(df_final)
)

print(
    "Final columns:",
    len(df_final.columns)
)

print(
    "\nSaved:",
    output_file
)

print("=" * 70)