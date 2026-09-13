from connection import engine
from sqlalchemy import text
import pandas as pd


ELEMENTS = [
    "ag", "al", "b", "c", "ca", "co", "cr", "cu",
    "fe", "ga", "hf", "i", "li", "mg", "mn", "mo",
    "nb", "nd", "ni", "o", "pd", "re", "ru", "s",
    "sc", "si", "sn", "t", "ta", "ti", "v", "w", "y",
    "zn", "zr"
]


def load_data():

    composition_sql = ", ".join(
        f"c.{element}" for element in ELEMENTS
    )

    query = f"""
    SELECT
        e.record_id,
        a.alloy_name,
        a.composition_canonical,
        a.alloy_class,

        {composition_sql},

        e.test_type,
        e.test_temperature_c,
        e.processing_method,
        e.phase,

        p.yield_strength_mpa,
        p.ultimate_tensile_strength_mpa,
        p.elongation_percent,
        p.hardness_hv,
        p.hardness_gpa

    FROM hea_mpea.experimental_records e

    JOIN hea_mpea.alloys a
        ON e.alloy_id = a.alloy_id

    LEFT JOIN hea_mpea.compositions c
        ON a.alloy_id = c.alloy_id

    JOIN hea_mpea.properties p
        ON e.record_id = p.record_id

    ORDER BY e.record_id;
    """

    with engine.connect() as connection:
        df = pd.read_sql(text(query), connection)

    return df


if __name__ == "__main__":

    df = load_data()

    print("Data loaded successfully!")
    print("Original shape:", df.shape)

    # --------------------------------
    # Yield Strength dataset
    # --------------------------------

    ys_df = df.dropna(
        subset=["yield_strength_mpa"]
    ).copy()

    print("\nYield Strength Dataset")
    print("----------------------")
    print("Records with YS:", len(ys_df))

    # --------------------------------
    # Check whether composition exists
    # --------------------------------

    ys_df["composition_available"] = (
        ys_df[ELEMENTS].notna().any(axis=1)
    )

    missing_composition = (
        ~ys_df["composition_available"]
    ).sum()

    print(
        "Records with completely missing composition:",
        missing_composition
    )

    # Keep only records with composition information
    ys_ml = ys_df[
        ys_df["composition_available"]
    ].copy()

    # --------------------------------
    # Convert absent elements to 0
    # --------------------------------

    ys_ml[ELEMENTS] = ys_ml[ELEMENTS].fillna(0)

    print(
        "Final YS ML records:",
        len(ys_ml)
    )

    # --------------------------------
    # Verify composition sums
    # --------------------------------

    ys_ml["composition_sum"] = (
        ys_ml[ELEMENTS].sum(axis=1)
    )

    print("\nComposition Sum Statistics")
    print("--------------------------")
    print(
        ys_ml["composition_sum"].describe()
    )

    # --------------------------------
    # Save dataset
    # --------------------------------

    ys_ml.to_csv(
        "YS_ML_Dataset.csv",
        index=False
    )

    print("\nSaved:")
    print("YS_ML_Dataset.csv")

    # --------------------------------
    # YS statistics
    # --------------------------------

    print("\nYield Strength Statistics")
    print("-------------------------")
    print(
        ys_ml["yield_strength_mpa"].describe()
    )

    # --------------------------------
    # Missing composition values
    # --------------------------------

    print("\nMissing values after composition processing:")

    print(
        ys_ml[ELEMENTS].isna().sum()
    )