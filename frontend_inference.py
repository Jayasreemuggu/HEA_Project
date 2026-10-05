import numpy as np
import pandas as pd

def generate_materials_features(df):

    elements = [
        "Ag","Al","B","C","Ca","Co","Cr","Cu","Fe","Ga","Hf","I",
        "Li","Mg","Mn","Mo","Nb","Nd","Ni","O","Pd","Re","Ru","S",
        "Sc","Si","Sn","T","Ta","Ti","V","W","Y","Zn","Zr"
    ]

    # Elemental properties.
    # Values are standard approximate elemental descriptors used
    # for materials-informatics feature construction.
    props = {
        "atomic_mass": {
            "Ag":107.8682,"Al":26.9815,"B":10.81,"C":12.011,"Ca":40.078,
            "Co":58.9332,"Cr":51.9961,"Cu":63.546,"Fe":55.845,"Ga":69.723,
            "Hf":178.49,"I":126.9045,"Li":6.94,"Mg":24.305,"Mn":54.938,
            "Mo":95.95,"Nb":92.9064,"Nd":144.242,"Ni":58.6934,"O":15.999,
            "Pd":106.42,"Re":186.207,"Ru":101.07,"S":32.06,"Sc":44.9559,
            "Si":28.085,"Sn":118.710,"T":0.0,"Ta":180.9479,"Ti":47.867,
            "V":50.9415,"W":183.84,"Y":88.9058,"Zn":65.38,"Zr":91.224
        },
        "atomic_radius": {
            "Ag":144,"Al":143,"B":87,"C":67,"Ca":197,
            "Co":125,"Cr":128,"Cu":128,"Fe":126,"Ga":135,
            "Hf":159,"I":140,"Li":152,"Mg":160,"Mn":127,
            "Mo":139,"Nb":146,"Nd":181,"Ni":124,"O":48,
            "Pd":137,"Re":137,"Ru":134,"S":88,"Sc":162,
            "Si":111,"Sn":145,"T":170,"Ta":146,"Ti":147,
            "V":134,"W":137,"Y":180,"Zn":134,"Zr":160
        },
        "electronegativity": {
            "Ag":1.93,"Al":1.61,"B":2.04,"C":2.55,"Ca":1.00,
            "Co":1.88,"Cr":1.66,"Cu":1.90,"Fe":1.83,"Ga":1.81,
            "Hf":1.30,"I":2.66,"Li":0.98,"Mg":1.31,"Mn":1.55,
            "Mo":2.16,"Nb":1.60,"Nd":1.14,"Ni":1.91,"O":3.44,
            "Pd":2.20,"Re":1.90,"Ru":2.20,"S":2.58,"Sc":1.36,
            "Si":1.90,"Sn":1.96,"T":0.0,"Ta":1.50,"Ti":1.54,
            "V":1.63,"W":2.36,"Y":1.22,"Zn":1.65,"Zr":1.33
        },
        "VEC": {
            "Ag":11,"Al":3,"B":3,"C":4,"Ca":2,
            "Co":9,"Cr":6,"Cu":11,"Fe":8,"Ga":3,
            "Hf":4,"I":7,"Li":1,"Mg":2,"Mn":7,
            "Mo":6,"Nb":5,"Nd":4,"Ni":10,"O":6,
            "Pd":10,"Re":7,"Ru":8,"S":6,"Sc":3,
            "Si":4,"Sn":4,"T":0,"Ta":5,"Ti":4,
            "V":5,"W":6,"Y":3,"Zn":12,"Zr":4
        }
    }

    # Chemically useful element groups.
    refractory = {"Hf","Mo","Nb","Re","Ta","Ti","V","W","Zr"}
    transition = {
        "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu",
        "Zn","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd",
        "Ag","Hf","Ta","W","Re"
    }
    early_transition = {"Sc","Ti","V","Cr","Y","Zr","Nb","Mo","Hf","Ta","W"}
    late_transition = {"Mn","Fe","Co","Ni","Cu","Ru","Rh","Pd","Ag","Re"}

    # Ensure complete numerical composition representation.
    comp = df[elements].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Normalize composition safely.
    row_sum = comp.sum(axis=1)
    w = comp.div(row_sum.replace(0, np.nan), axis=0).fillna(0.0)

    new = pd.DataFrame(index=df.index)

    def weighted_stats(prefix, values):

        values = np.asarray(values, dtype=float)

        mean = np.sum(w.values * values[None, :], axis=1)

        var = np.sum(
            w.values * (values[None, :] - mean[:, None])**2,
            axis=1
        )

        std = np.sqrt(np.maximum(var, 0))

        centered = values[None, :] - mean[:, None]

        skew = np.sum(
            w.values * centered**3,
            axis=1
        ) / np.maximum(std**3, 1e-12)

        kurt = np.sum(
            w.values * centered**4,
            axis=1
        ) / np.maximum(std**4, 1e-12) - 3.0

        weighted_min = np.min(
            np.where(w.values > 0, values[None, :], np.inf),
            axis=1
        )

        weighted_max = np.max(
            np.where(w.values > 0, values[None, :], -np.inf),
            axis=1
        )

        new[f"ACF_{prefix}_mean"] = mean
        new[f"ACF_{prefix}_var"] = var
        new[f"ACF_{prefix}_std"] = std
        new[f"ACF_{prefix}_skew"] = skew
        new[f"ACF_{prefix}_kurtosis"] = kurt
        new[f"ACF_{prefix}_min"] = weighted_min
        new[f"ACF_{prefix}_max"] = weighted_max
        new[f"ACF_{prefix}_range"] = weighted_max - weighted_min

        # Quantile-like composition-weighted thresholds
        order = np.argsort(values)

        sorted_values = values[order]
        sorted_w = w.values[:, order]
        cumulative = np.cumsum(sorted_w, axis=1)

        for q, label in [
            (0.25, "q25"),
            (0.50, "q50"),
            (0.75, "q75")
        ]:
            idx = np.argmax(cumulative >= q, axis=1)
            new[f"ACF_{prefix}_{label}"] = sorted_values[idx]

    for pname, pvalues in props.items():
        values = np.array([pvalues[e] for e in elements], dtype=float)
        weighted_stats(pname, values)

    # ---------------------------------------------------------
    # Dominant-element descriptors
    # ---------------------------------------------------------
    dominant_idx = np.argmax(w.values, axis=1)

    new["ACF_dominant_fraction"] = np.max(w.values, axis=1)

    second_sorted = np.sort(w.values, axis=1)
    new["ACF_second_fraction"] = second_sorted[:, -2]

    new["ACF_top2_fraction"] = (
        second_sorted[:, -1] + second_sorted[:, -2]
    )

    new["ACF_top3_fraction"] = (
        second_sorted[:, -1]
        + second_sorted[:, -2]
        + second_sorted[:, -3]
    )

    # ---------------------------------------------------------
    # Property-threshold fractions
    # ---------------------------------------------------------
    for pname, pvalues in props.items():

        values = np.array(
            [pvalues[e] for e in elements],
            dtype=float
        )

        mean = np.sum(
            w.values * values[None, :],
            axis=1
        )

        new[f"ACF_{pname}_above_mean_fraction"] = np.sum(
            w.values * (values[None, :] > mean[:, None]),
            axis=1
        )

        new[f"ACF_{pname}_below_mean_fraction"] = np.sum(
            w.values * (values[None, :] < mean[:, None]),
            axis=1
        )

    # ---------------------------------------------------------
    # Entropy-weighted descriptors
    # ---------------------------------------------------------
    entropy = -np.sum(
        np.where(
            w.values > 0,
            w.values * np.log(np.maximum(w.values, 1e-12)),
            0
        ),
        axis=1
    )

    new["ACF_config_entropy_normalized"] = (
        entropy / np.log(len(elements))
    )

    for pname, pvalues in props.items():

        values = np.array(
            [pvalues[e] for e in elements],
            dtype=float
        )

        weighted_mean = np.sum(
            w.values * values[None, :],
            axis=1
        )

        new[f"ACF_entropy_weighted_{pname}"] = (
            weighted_mean * (1.0 + entropy)
        )

    # ---------------------------------------------------------
    # Chemical group fractions
    # ---------------------------------------------------------
    groups = {
        "refractory": refractory,
        "transition": transition,
        "early_transition": early_transition,
        "late_transition": late_transition
    }

    for gname, group in groups.items():

        cols = [
            i for i, e in enumerate(elements)
            if e in group
        ]

        new[f"ACF_{gname}_fraction"] = (
            w.iloc[:, cols].sum(axis=1)
        )

    # ---------------------------------------------------------
    # Light/heavy and size/electronegativity fractions
    # ---------------------------------------------------------
    mass = np.array(
        [props["atomic_mass"][e] for e in elements]
    )

    radius = np.array(
        [props["atomic_radius"][e] for e in elements]
    )

    en = np.array(
        [props["electronegativity"][e] for e in elements]
    )

    mass_median = np.median(mass)
    radius_median = np.median(radius)
    en_median = np.median(en)

    new["ACF_heavy_element_fraction"] = np.sum(
        w.values * (mass[None, :] > mass_median),
        axis=1
    )

    new["ACF_light_element_fraction"] = np.sum(
        w.values * (mass[None, :] < mass_median),
        axis=1
    )

    new["ACF_large_radius_fraction"] = np.sum(
        w.values * (radius[None, :] > radius_median),
        axis=1
    )

    new["ACF_small_radius_fraction"] = np.sum(
        w.values * (radius[None, :] < radius_median),
        axis=1
    )

    new["ACF_high_EN_fraction"] = np.sum(
        w.values * (en[None, :] > en_median),
        axis=1
    )

    new["ACF_low_EN_fraction"] = np.sum(
        w.values * (en[None, :] < en_median),
        axis=1
    )

    # ---------------------------------------------------------
    # Nonlinear composition descriptors
    # ---------------------------------------------------------
    new["ACF_fraction_squared_sum"] = np.sum(
        w.values**2,
        axis=1
    )

    new["ACF_fraction_cubed_sum"] = np.sum(
        w.values**3,
        axis=1
    )

    new["ACF_fraction_fourth_sum"] = np.sum(
        w.values**4,
        axis=1
    )

    new["ACF_effective_component_number"] = (
        1.0 / np.maximum(
            new["ACF_fraction_squared_sum"],
            1e-12
        )
    )

    # ---------------------------------------------------------
    # Interaction between composition complexity and properties
    # ---------------------------------------------------------
    new["ACF_entropy_x_radius_std"] = (
        entropy * new["ACF_atomic_radius_std"]
    )

    new["ACF_entropy_x_EN_std"] = (
        entropy * new["ACF_electronegativity_std"]
    )

    new["ACF_entropy_x_mass_std"] = (
        entropy * new["ACF_atomic_mass_std"]
    )

    new["ACF_refractory_x_radius_std"] = (
        new["ACF_refractory_fraction"]
        * new["ACF_atomic_radius_std"]
    )

    new["ACF_transition_x_EN_std"] = (
        new["ACF_transition_fraction"]
        * new["ACF_electronegativity_std"]
    )

    # ---------------------------------------------------------
    # Clean infinite values
    # ---------------------------------------------------------
    new = new.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Generate the 19 MI descriptors using the exact MI property tables.
    mi_atomic_mass = {
        "Ag":107.868,"Al":26.982,"B":10.811,"C":12.011,"Ca":40.078,
        "Co":58.933,"Cr":51.996,"Cu":63.546,"Fe":55.845,"Ga":69.723,
        "Hf":178.49,"I":126.904,"Li":6.94,"Mg":24.305,"Mn":54.938,
        "Mo":95.95,"Nb":92.906,"Nd":144.242,"Ni":58.693,"O":15.999,
        "Pd":106.42,"Re":186.207,"Ru":101.07,"S":32.06,"Sc":44.956,
        "Si":28.085,"Sn":118.710,"T":0.0,"Ta":180.948,"Ti":47.867,
        "V":50.942,"W":183.84,"Y":88.906,"Zn":65.38,"Zr":91.224
    }

    mi_electronegativity = {
        "Ag":1.93,"Al":1.61,"B":2.04,"C":2.55,"Ca":1.00,
        "Co":1.88,"Cr":1.66,"Cu":1.90,"Fe":1.83,"Ga":1.81,
        "Hf":1.30,"I":2.66,"Li":0.98,"Mg":1.31,"Mn":1.55,
        "Mo":2.16,"Nb":1.60,"Nd":1.14,"Ni":1.91,"O":3.44,
        "Pd":2.20,"Re":1.90,"Ru":2.20,"S":2.58,"Sc":1.36,
        "Si":1.90,"Sn":1.96,"T":0.0,"Ta":1.50,"Ti":1.54,
        "V":1.63,"W":2.36,"Y":1.22,"Zn":1.65,"Zr":1.33
    }

    mi_atomic_radius = {
        "Ag":144,"Al":143,"B":85,"C":70,"Ca":197,"Co":125,"Cr":128,
        "Cu":128,"Fe":126,"Ga":135,"Hf":159,"I":140,"Li":152,"Mg":160,
        "Mn":127,"Mo":139,"Nb":146,"Nd":181,"Ni":124,"O":60,"Pd":137,
        "Re":137,"Ru":134,"S":100,"Sc":162,"Si":111,"Sn":145,"T":170,
        "Ta":146,"Ti":147,"V":134,"W":139,"Y":180,"Zn":134,"Zr":160
    }

    mi_vec = {
        "Ag":11,"Al":3,"B":3,"C":4,"Ca":2,"Co":9,"Cr":6,"Cu":11,
        "Fe":8,"Ga":3,"Hf":4,"I":7,"Li":1,"Mg":2,"Mn":7,"Mo":6,
        "Nb":5,"Nd":4,"Ni":10,"O":6,"Pd":10,"Re":7,"Ru":8,"S":6,
        "Sc":3,"Si":4,"Sn":4,"T":0,"Ta":5,"Ti":4,"V":5,"W":6,
        "Y":3,"Zn":12,"Zr":4
    }

    mi = pd.DataFrame(index=df.index)

    for e in elements:
        if e in df.columns:
            col = e
        elif f"{e}_at_pct" in df.columns:
            col = f"{e}_at_pct"
        else:
            col = None

        if col is not None:
            mi[e] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            mi[e] = 0.0

    row_sum = mi[elements].sum(axis=1)
    P = mi[elements].div(row_sum.replace(0, np.nan), axis=0)

    mi["MI_n_elements"] = (mi[elements] > 1e-10).sum(axis=1)
    mi["MI_max_fraction"] = P.max(axis=1)
    mi["MI_min_nonzero_fraction"] = P.replace(0, np.nan).min(axis=1)
    mi["MI_top2_fraction"] = P.apply(lambda r: r.nlargest(2).sum(), axis=1)
    mi["MI_top3_fraction"] = P.apply(lambda r: r.nlargest(3).sum(), axis=1)
    mi["MI_fraction_std"] = P.std(axis=1)
    mi["MI_fraction_range"] = P.max(axis=1) - P.replace(0, np.nan).min(axis=1)

    P_safe = P.clip(lower=1e-12)
    mi["MI_config_entropy_R"] = -(P_safe * np.log(P_safe)).sum(axis=1)

    for name, values in [
        ("atomic_mass", mi_atomic_mass),
        ("electronegativity", mi_electronegativity),
        ("atomic_radius", mi_atomic_radius),
        ("VEC", mi_vec),
    ]:
        v = np.array([values[e] for e in elements], dtype=float)
        mi[f"MI_weighted_{name}"] = (P * v).sum(axis=1)

    radius_vec = np.array([mi_atomic_radius[e] for e in elements], dtype=float)
    en_vec = np.array([mi_electronegativity[e] for e in elements], dtype=float)
    mass_vec = np.array([mi_atomic_mass[e] for e in elements], dtype=float)

    mean_r = (P * radius_vec).sum(axis=1)
    mean_en = (P * en_vec).sum(axis=1)
    mean_mass = (P * mass_vec).sum(axis=1)

    mi["MI_radius_std"] = np.sqrt(
        (P * (radius_vec - mean_r.values[:, None]) ** 2).sum(axis=1)
    )
    mi["MI_en_std"] = np.sqrt(
        (P * (en_vec - mean_en.values[:, None]) ** 2).sum(axis=1)
    )
    mi["MI_mass_std"] = np.sqrt(
        (P * (mass_vec - mean_mass.values[:, None]) ** 2).sum(axis=1)
    )
    mi["MI_atomic_size_mismatch"] = np.sqrt(
        (P * (1.0 - radius_vec / mean_r.values[:, None]) ** 2).sum(axis=1)
    ) * 100

    arr = P.to_numpy()
    pair_products = []
    pair_en_diff = []
    pair_radius_diff = []

    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            pair_products.append(arr[:, i] * arr[:, j])
            pair_en_diff.append(
                arr[:, i] * arr[:, j] * abs(en_vec[i] - en_vec[j])
            )
            pair_radius_diff.append(
                arr[:, i] * arr[:, j] * abs(radius_vec[i] - radius_vec[j])
            )

    mi["MI_pair_fraction_product_sum"] = np.nansum(
        np.vstack(pair_products), axis=0
    )
    mi["MI_pair_electronegativity_interaction"] = np.nansum(
        np.vstack(pair_en_diff), axis=0
    )
    mi["MI_pair_radius_interaction"] = np.nansum(
        np.vstack(pair_radius_diff), axis=0
    )

    mi = mi.drop(columns=elements)

    result = pd.concat([mi, new], axis=1)
    result = result.replace([np.inf, -np.inf], np.nan)

    base_order = [
        "Test_Temperature_C","Grain_Size_um","Density_Exp_g_cm3",
        "Density_Calc_g_cm3","Precipitate_Size_nm","Matrix_Volume_pct",
        "Youngs_Modulus_Exp_GPa","Youngs_Modulus_Calc_GPa",
        "Processing_Method","Phase","Alloy_Class","Equilibrium_Condition",
        "Single_Multiphase","Test_Type","Precipitate_Info"
    ] + elements + [
        "MI_n_elements","MI_max_fraction","MI_min_nonzero_fraction",
        "MI_top2_fraction","MI_top3_fraction","MI_fraction_std",
        "MI_fraction_range","MI_config_entropy_R",
        "MI_weighted_atomic_mass","MI_weighted_electronegativity",
        "MI_weighted_atomic_radius","MI_weighted_VEC","MI_radius_std",
        "MI_en_std","MI_mass_std","MI_atomic_size_mismatch",
        "MI_pair_fraction_product_sum",
        "MI_pair_electronegativity_interaction",
        "MI_pair_radius_interaction"
    ] + [c for c in result.columns if c.startswith("ACF_")]

    missing_base = [c for c in base_order if c not in df.columns and c not in result.columns]
    if missing_base:
        raise ValueError(f"Missing base features: {missing_base}")

    base = pd.DataFrame(index=df.index)

    for c in base_order:
        if c in result.columns:
            base[c] = result[c]
        elif c in df.columns:
            base[c] = df[c]
        else:
            base[c] = np.nan

    phase = df.get("Phase", pd.Series("", index=df.index)).fillna("").astype(str).str.upper()
    test = df.get("Test_Type", pd.Series("", index=df.index)).fillna("").astype(str).str.upper()

    base["PHASE_BCC_IND"] = phase.str.contains("BCC", regex=False).astype(float)
    base["PHASE_FCC_IND"] = phase.str.contains("FCC", regex=False).astype(float)
    base["PHASE_HCP_IND"] = phase.str.contains("HCP", regex=False).astype(float)
    base["PHASE_B2_IND"] = phase.str.contains("B2", regex=False).astype(float)
    base["PHASE_LAVES_IND"] = phase.str.contains("LAVES", regex=False).astype(float)
    base["PHASE_SIGMA_IND"] = phase.str.contains("SIGMA", regex=False).astype(float)
    base["PHASE_L12_IND"] = phase.str.contains("L12", regex=False).astype(float)
    base["PHASE_COMPLEX_IND"] = phase.str.contains("COMPLEX", regex=False).astype(float)
    base["PHASE_TEST_TENSILE"] = test.str.contains("TENSILE", regex=False).astype(float)
    base["PHASE_TEST_COMPRESSION"] = test.str.contains("COMPRESSION", regex=False).astype(float)

    phase_cols = [
        "PHASE_BCC_IND","PHASE_FCC_IND","PHASE_HCP_IND","PHASE_B2_IND",
        "PHASE_LAVES_IND","PHASE_SIGMA_IND","PHASE_L12_IND",
        "PHASE_COMPLEX_IND","PHASE_TEST_TENSILE","PHASE_TEST_COMPRESSION"
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

    for pcol in phase_cols:
        for dcol in descriptor_cols:
            base[f"{pcol}__{dcol}"] = (
                base[pcol] * pd.to_numeric(base[dcol], errors="coerce")
            )

    if base.shape[1] != 359:
        raise ValueError(f"Expected 359 features, got {base.shape[1]}")

    return base.replace([np.inf, -np.inf], np.nan)
