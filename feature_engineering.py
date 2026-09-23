import pandas as pd
import numpy as np


ELEMENTS = [
    "Ag","Al","B","C","Ca","Co","Cr","Cu","Fe","Ga","Hf","I",
    "Li","Mg","Mn","Mo","Nb","Nd","Ni","O","Pd","Re","Ru","S",
    "Sc","Si","Sn","T","Ta","Ti","V","W","Y","Zn","Zr"
]

MI_FEATURES = [
    "MI_n_elements",
    "MI_max_fraction",
    "MI_min_nonzero_fraction",
    "MI_top2_fraction",
    "MI_top3_fraction",
    "MI_fraction_std",
    "MI_fraction_range",
    "MI_config_entropy_R",
    "MI_weighted_atomic_mass",
    "MI_weighted_electronegativity",
    "MI_weighted_atomic_radius",
    "MI_weighted_VEC",
    "MI_radius_std",
    "MI_en_std",
    "MI_mass_std",
    "MI_atomic_size_mismatch",
    "MI_pair_fraction_product_sum",
    "MI_pair_electronegativity_interaction",
    "MI_pair_radius_interaction"
]

CONDITION_NUMERIC = [
    "Test_Temperature_C",
    "Grain_Size_um",
    "Density_Calc_g_cm3",
    "Youngs_Modulus_Calc_GPa",
    "Precipitate_Size_nm",
    "Matrix_Volume_pct"
]

CONDITION_CATEGORICAL = [
    "Phase",
    "Processing_Method",
    "Equilibrium_Condition",
    "Single_Multiphase",
    "Test_Type",
    "Precipitate_Info"
]

INTERACTION_CONDITIONS = [
    "Test_Temperature_C",
    "Grain_Size_um",
    "Density_Calc_g_cm3",
    "Precipitate_Size_nm",
    "Matrix_Volume_pct"
]


def create_330_features(df):
    """
    Reproduce the exact feature engineering used by
    experiment_controlled_interactions.py and train_production_ys.py.

    Input:
        DataFrame containing raw/base HEA composition, MI, ACF,
        and experimental condition columns.

    Output:
        DataFrame containing the engineered model input features.
    """
    df = df.copy()

    elements = [c for c in ELEMENTS if c in df.columns]
    mi_features = [c for c in MI_FEATURES if c in df.columns]
    acf_features = [c for c in df.columns if c.startswith("ACF_")]

    condition_numeric = [
        c for c in CONDITION_NUMERIC
        if c in df.columns
    ]

    condition_categorical = [
        c for c in CONDITION_CATEGORICAL
        if c in df.columns
    ]

    temp = pd.to_numeric(
        df["Test_Temperature_C"],
        errors="coerce"
    )

    df["H_TEMP_MISSING"] = temp.isna().astype(int)
    df["H_TEMP_DELTA"] = temp - 25.0
    df["H_TEMP_SQUARED"] = df["H_TEMP_DELTA"] ** 2
    df["H_TEMP_CUBED"] = df["H_TEMP_DELTA"] ** 3

    condition_numeric += [
        "H_TEMP_MISSING",
        "H_TEMP_DELTA",
        "H_TEMP_SQUARED",
        "H_TEMP_CUBED"
    ]

    for c in [
        "Grain_Size_um",
        "Density_Calc_g_cm3",
        "Youngs_Modulus_Calc_GPa",
        "Precipitate_Size_nm",
        "Matrix_Volume_pct"
    ]:
        name = f"H_MISSING_{c}"
        df[name] = df[c].isna().astype(int)
        condition_numeric.append(name)

    interaction_conditions = [
        c for c in INTERACTION_CONDITIONS
        if c in df.columns
    ]

    interaction_features = []

    for element in elements:
        for condition in interaction_conditions:
            name = f"INT_{element}_x_{condition}"

            e = pd.to_numeric(
                df[element],
                errors="coerce"
            )

            c = pd.to_numeric(
                df[condition],
                errors="coerce"
            ).fillna(0.0)

            df[name] = e * c
            interaction_features.append(name)

    base_features = list(dict.fromkeys(
        elements
        + mi_features
        + acf_features
        + condition_numeric
        + condition_categorical
    ))

    features = list(dict.fromkeys(
        base_features + interaction_features
    ))

    return df[features].copy()

