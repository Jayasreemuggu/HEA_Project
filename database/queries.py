from connection import engine
from sqlalchemy import text
import pandas as pd


def get_ml_data():

    element_columns = [
        "ag", "al", "b", "c", "ca", "co", "cr", "cu",
        "fe", "ga", "hf", "i", "li", "mg", "mn", "mo",
        "nb", "nd", "ni", "o", "pd", "re", "ru", "s",
        "sc", "si", "sn", "t", "ta", "ti", "v", "w", "y",
        "zn", "zr"
    ]

    composition_sql = ", ".join(
        f"c.{element}" for element in element_columns
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
        p.hardness_gpa,
        p.youngs_modulus_gpa

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

    df = get_ml_data()

    print("Data retrieved successfully!")
    print("Shape:", df.shape)

    print("\nNumber of columns:", len(df.columns))

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 records:")
    print(df.head())