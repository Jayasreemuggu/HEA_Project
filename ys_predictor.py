import os
import tempfile
import pandas as pd
import joblib

PROJECT = r"C:\Users\jayam\Downloads\IITH_ML_project"
MODEL_PATH = os.path.join(PROJECT, "models", "YS_330_Controlled_Interactions_Production.pkl")

ELEMENTS = [
    "Ag","Al","B","C","Ca","Co","Cr","Cu","Fe","Ga","Hf","I",
    "Li","Mg","Mn","Mo","Nb","Nd","Ni","O","Pd","Re","Ru","S",
    "Sc","Si","Sn","T","Ta","Ti","V","W","Y","Zn","Zr"
]

def predict_ys(composition, conditions=None):
    conditions = conditions or {}

    row = {e: float(composition.get(e, 0.0)) for e in ELEMENTS}

    total = sum(row.values())
    if total <= 0:
        raise ValueError("Composition must contain at least one non-zero element.")

    if abs(total - 100.0) > 1e-6:
        raise ValueError(f"Composition must sum to 100 at.%. Current total: {total:.6f}")

    for key in [
        "Test_Temperature_C",
        "Grain_Size_um",
        "Density_Calc_g_cm3",
        "Youngs_Modulus_Calc_GPa",
        "Precipitate_Size_nm",
        "Matrix_Volume_pct",
        "Phase",
        "Processing_Method",
        "Equilibrium_Condition",
        "Single_Multiphase",
        "Test_Type",
        "Precipitate_Info"
    ]:
        row[key] = conditions.get(key)

    # Temporary one-row dataset.
    # The original descriptor formulas are used unchanged.
    with tempfile.TemporaryDirectory() as tmp:
        base_path = os.path.join(tmp, "HEA_MPEA_ML_Ready_Deduplicated.csv")
        mi_path = os.path.join(tmp, "YS_Materials_Informatics_Features.csv")
        acf_path = os.path.join(tmp, "YS_Advanced_Composition_Features.csv")

        base = pd.DataFrame([row])
        base["YS_Tensile_MPa"] = float("nan")

        # Add the column names expected by the original MI builder.
        for e in ELEMENTS:
            base[f"{e}_at_pct"] = base[e]

        base.to_csv(base_path, index=False)

        # Execute the exact MI builder on the temporary one-row dataset.
        mi_source = open(
            os.path.join(PROJECT, "build_ys_materials_features.py"),
            encoding="utf-8"
        ).read()

        mi_source = mi_source.replace(
            'INPUT = "HEA_MPEA_ML_Ready_Deduplicated.csv"',
            f'INPUT = r"{base_path}"'
        ).replace(
            'OUTPUT = "YS_Materials_Informatics_Features.csv"',
            f'OUTPUT = r"{mi_path}"'
        )

        exec(
            compile(mi_source, "build_ys_materials_features.py", "exec"),
            {"__name__": "__main__"}
        )

        mi = pd.read_csv(mi_path, low_memory=False)

        # Convert the elemental at.% columns to the names expected by
        # the advanced ACF builder.
        acf_source = open(
            os.path.join(PROJECT, "build_advanced_composition_features.py"),
            encoding="utf-8"
        ).read()

        acf_source = acf_source.replace(
            'ROOT = r"C:\\Users\\jayam\\Downloads\\IITH_ML_project"',
            f'ROOT = r"{tmp}"'
        )

        mi.to_csv(
            os.path.join(tmp, "YS_Materials_Informatics_Features.csv"),
            index=False
        )

        exec(
            compile(acf_source, "build_advanced_composition_features.py", "exec"),
            {"__name__": "__main__"}
        )

        engineered = pd.read_csv(acf_path, low_memory=False)

        # The model's exact 330-feature engineering.
        from feature_engineering import create_330_features

        X = create_330_features(engineered)

        model = joblib.load(MODEL_PATH)

        expected = list(model.feature_names_in_)

        if list(X.columns) != expected:
            raise RuntimeError(
                f"Feature schema mismatch: generated={len(X.columns)}, "
                f"expected={len(expected)}"
            )

        prediction = float(model.predict(X)[0])

        return prediction


if __name__ == "__main__":
    test_composition = {
        "Co": 20,
        "Cr": 20,
        "Fe": 20,
        "Ni": 20,
        "Mn": 20
    }

    prediction = predict_ys(test_composition, {
        "Test_Temperature_C": 25,
        "Phase": "FCC",
        "Test_Type": "T"
    })

    print("DEPLOYMENT_TEST_SUCCESS")
    print("PREDICTED_YS_MPa:", prediction)

