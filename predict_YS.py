import os
import joblib
import numpy as np
import pandas as pd

PROJECT = r"C:\Users\jayam\Downloads\IITH_ML_project"
MODEL_PATH = os.path.join(
    PROJECT, "models", "YS_FINAL_PRODUCTION", "YS_FINAL_PRODUCTION_MODEL.pkl"
)

# ------------------------------------------------------------
# LOAD PRODUCTION MODEL
# ------------------------------------------------------------
artifact = joblib.load(MODEL_PATH)

# ------------------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------------------
def build_features(df):
    phase_cols = [
        "PHASE_BCC_IND","PHASE_FCC_IND","PHASE_HCP_IND","PHASE_B2_IND",
        "PHASE_LAVES_IND","PHASE_SIGMA_IND","PHASE_L12_IND",
        "PHASE_COMPLEX_IND","PHASE_TEST_TENSILE",
        "PHASE_TEST_COMPRESSION"
    ]

    descriptor_cols = [
        "Density_Exp_g_cm3","Density_Calc_g_cm3",
        "MI_weighted_atomic_mass",
        "MI_weighted_electronegativity",
        "MI_weighted_atomic_radius",
        "MI_weighted_VEC",
        "MI_atomic_size_mismatch",
        "MI_pair_electronegativity_interaction",
        "ACF_atomic_mass_mean","ACF_atomic_mass_var",
        "ACF_atomic_mass_std","ACF_atomic_mass_skew",
        "ACF_atomic_mass_kurtosis","ACF_atomic_mass_min",
        "ACF_atomic_mass_max","ACF_atomic_mass_range",
        "ACF_atomic_mass_q25","ACF_atomic_mass_q50",
        "ACF_atomic_mass_q75","ACF_atomic_radius_mean"
    ]

    exclude = {
        "YS_Tensile_MPa",
        "Record_ID",
        "Composition_Canonical",
        "Predicted_Solidus_C"
    }

    base_cols = [
        c for c in df.columns
        if c not in exclude and c not in phase_cols
    ]

    base_cols = base_cols[:149]

    X = df[base_cols].copy()

    phase = df["Phase"].fillna("").astype(str).str.upper()
    test = df["Test_Type"].fillna("").astype(str).str.upper()

    X["PHASE_BCC_IND"] = phase.str.contains(
        "BCC", regex=False
    ).astype(float)

    X["PHASE_FCC_IND"] = phase.str.contains(
        "FCC", regex=False
    ).astype(float)

    X["PHASE_HCP_IND"] = phase.str.contains(
        "HCP", regex=False
    ).astype(float)

    X["PHASE_B2_IND"] = phase.str.contains(
        "B2", regex=False
    ).astype(float)

    X["PHASE_LAVES_IND"] = phase.str.contains(
        "LAVES", regex=False
    ).astype(float)

    X["PHASE_SIGMA_IND"] = phase.str.contains(
        "SIGMA", regex=False
    ).astype(float)

    X["PHASE_L12_IND"] = phase.str.contains(
        "L12", regex=False
    ).astype(float)

    X["PHASE_COMPLEX_IND"] = phase.str.contains(
        "COMPLEX", regex=False
    ).astype(float)

    X["PHASE_TEST_TENSILE"] = test.str.contains(
        "TENSILE", regex=False
    ).astype(float)

    X["PHASE_TEST_COMPRESSION"] = test.str.contains(
        "COMPRESSION", regex=False
    ).astype(float)

    for p in phase_cols:
        for d in descriptor_cols:
            X[f"{p}__{d}"] = (
                X[p] * pd.to_numeric(
                    df[d], errors="coerce"
                )
            )

    if X.shape[1] != 359:
        raise ValueError(
            f"Expected 359 features, got {X.shape[1]}"
        )

    return X


# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------
def predict(df):
    X = build_features(df)

    base = artifact["base_model"].predict(X)

    mid1 = artifact["mid1_model"].predict(X)
    mid2 = artifact["mid2_model"].predict(X)

    final = base.copy()

    m1 = (base >= 800) & (base < 1200)
    m2 = (base >= 1200) & (base < 1580)

    final[m1] = (
        0.20 * base[m1] +
        0.80 * mid1[m1]
    )

    final[m2] = (
        0.20 * base[m2] +
        0.80 * mid2[m2]
    )

    residual = np.zeros(len(X))

    masks = [
        ("0_800", base < 800, 0.95),
        ("800_1200",
         (base >= 800) & (base < 1200), 0.30),
        ("1200_1580",
         (base >= 1200) & (base < 1580), 0.30),
        ("1580_1800",
         (base >= 1580) & (base < 1800), 0.50),
        ("1800_PLUS", base >= 1800, 0.30)
    ]

    for name, mask, alpha in masks:
        if mask.any():
            residual[mask] = (
                artifact["residual_models"][name]
                .predict(X.loc[mask])
            )
            final[mask] += alpha * residual[mask]

    return pd.DataFrame({
        "Base_Prediction": base,
        "Residual_Correction": residual,
        "Final_Prediction_MPa": final
    })


# ------------------------------------------------------------
# COMMAND-LINE USAGE
# ------------------------------------------------------------
if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:
        print(
            "Usage: python predict_ys.py <input_csv>"
        )
        raise SystemExit(1)

    input_path = sys.argv[1]

    df = pd.read_csv(input_path)

    predictions = predict(df)

    output_path = os.path.splitext(input_path)[0] + "_YS_Predictions.csv"

    predictions.to_csv(output_path, index=False)

    print("=" * 80)
    print("YS PRODUCTION INFERENCE COMPLETE")
    print("=" * 80)
    print("Input:", input_path)
    print("Rows:", len(df))
    print("Output:", output_path)
    print()
    print(predictions.describe())
