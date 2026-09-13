import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit

DATA = "HEA_MPEA_ML_Ready_Deduplicated.csv"
OUT = "explainability/results"
os.makedirs(OUT, exist_ok=True)

CONFIG = {
    "YS": ("YS_XGBoost_Model.pkl", "YS_Tensile_MPa"),
    "UTS": ("UTS_XGBoost_Model.pkl", "UTS_Tensile_MPa"),
    "Elongation": ("Elongation_XGBoost_Model.pkl", "Elongation_Tensile_pct"),
    "Hardness": ("Hardness_ExtraTrees_Model.pkl", "Hardness_HV"),
}

df = pd.read_csv(DATA)

def canonical_feature(x):
    x = x.lower()
    if "__" in x:
        x = x.split("__", 1)[1]

    # Keep element names
    if x.endswith("_at_pct"):
        return x

    for name in [
        "test_temperature_c",
        "density_calc_g_cm3",
        "density_exp_g_cm3",
        "grain_size_um",
        "precipitate_size_nm",
        "matrix_volume_pct",
        "atomic_size_mismatch",
        "mixing_enthalpy",
        "mixing_entropy",
        "vec",
    ]:
        if x.startswith(name):
            return name

    return x

summary = []

for prop, (model_file, target) in CONFIG.items():

    print("\n" + "="*70)
    print(prop)
    print("="*70)

    model = joblib.load(model_file)
    prep = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]

    expected = list(prep.feature_names_in_)
    colmap = {c.lower(): c for c in df.columns}

    actual = [colmap[x.lower()] for x in expected]

    data = df.dropna(subset=[target]).copy()

    # Match final Hardness preprocessing
    if prop == "Hardness":
        elems = [
            colmap[x.lower()]
            for x in expected
            if x.lower().endswith("_at_pct")
        ]
        data = data.loc[
            data[elems].fillna(0).sum(axis=1) > 0
        ].copy()

    # Match final Elongation duplicate cleaning
    if prop == "Elongation":
        key_candidates = [
            "Core_Duplicate_Key",
            "core_duplicate_key"
        ]
        key = next(
            (colmap[x.lower()] for x in key_candidates
             if x.lower() in colmap),
            None
        )

        if key:
            before = len(data)
            data = data.drop_duplicates(
                subset=[key, target],
                keep="first"
            ).copy()
            print("Elongation duplicate removal:", before, "->", len(data))

    X = data[actual].copy()
    X.columns = expected
    y = data[target]

    groups = data["Composition_Canonical"].fillna("UNKNOWN")

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    train_idx, test_idx = next(
        gss.split(X, y, groups=groups)
    )

    X_test = X.iloc[test_idx].copy()

    Z = prep.transform(X_test)

    if hasattr(Z, "toarray"):
        Z = Z.toarray()

    Z = np.asarray(Z)

    names = list(prep.get_feature_names_out())

    Zdf = pd.DataFrame(Z, columns=names)

    # Safe names for tree SHAP
    Zsafe = Zdf.copy()
    Zsafe.columns = [f"f_{i}" for i in range(Zsafe.shape[1])]

    explainer = shap.TreeExplainer(estimator)
    sv = np.asarray(explainer.shap_values(Zsafe))

    # -------------------------------------------------------
    # Direction analysis
    # -------------------------------------------------------
    rows = []

    for i, name in enumerate(names):

        values = Zdf.iloc[:, i].values
        shap_i = sv[:, i]

        if np.std(values) == 0:
            continue

        corr = np.corrcoef(values, shap_i)[0, 1]

        if not np.isfinite(corr):
            continue

        rows.append({
            "Feature": name,
            "Material_Feature": canonical_feature(name),
            "Mean_Absolute_SHAP": np.mean(np.abs(shap_i)),
            "Mean_SHAP": np.mean(shap_i),
            "Correlation": corr,
        })

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "Mean_Absolute_SHAP",
        ascending=False
    )

    result.to_csv(
        f"{OUT}/{prop}_SHAP_Final_Direction.csv",
        index=False
    )

    # -------------------------------------------------------
    # Top 15 directional features
    # -------------------------------------------------------
    top = result.head(15).copy()

    top["Direction"] = np.where(
        top["Correlation"] > 0,
        "Positive",
        "Negative"
    )

    top.to_csv(
        f"{OUT}/{prop}_SHAP_Final_Top15_Direction.csv",
        index=False
    )

    # -------------------------------------------------------
    # Direction plot
    # -------------------------------------------------------
    plot = top.sort_values(
        "Mean_Absolute_SHAP"
    )

    plt.figure(figsize=(9, 6))

    plt.barh(
        plot["Material_Feature"],
        plot["Correlation"]
    )

    plt.axvline(
        0,
        linewidth=1
    )

    plt.xlabel(
        "Feature–SHAP Correlation"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        f"{prop}: SHAP Direction Analysis"
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUT}/{prop}_SHAP_Direction_Plot.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Final split: train={len(train_idx)}, "
        f"test={len(test_idx)}"
    )

    print(
        f"Unique compositions: "
        f"train={groups.iloc[train_idx].nunique()}, "
        f"test={groups.iloc[test_idx].nunique()}"
    )

    print(
        f"Transformed features: {len(names)}"
    )

    print("\nTOP 15:")
    print(
        top[
            [
                "Material_Feature",
                "Mean_Absolute_SHAP",
                "Correlation",
                "Direction"
            ]
        ].to_string(index=False)
    )

    summary.append({
        "Property": prop,
        "Train_Samples": len(train_idx),
        "Test_Samples": len(test_idx),
        "Unique_Train_Compositions":
            groups.iloc[train_idx].nunique(),
        "Unique_Test_Compositions":
            groups.iloc[test_idx].nunique(),
        "Transformed_Features": len(names),
    })

pd.DataFrame(summary).to_csv(
    f"{OUT}/SHAP_Final_Summary.csv",
    index=False
)

print("\n" + "="*70)
print("FINAL SHAP ANALYSIS COMPLETE")
print("="*70)

print("\nFiles created:")
for prop in CONFIG:
    print(f"{prop}_SHAP_Final_Direction.csv")
    print(f"{prop}_SHAP_Final_Top15_Direction.csv")
    print(f"{prop}_SHAP_Direction_Plot.png")

print("SHAP_Final_Summary.csv")
