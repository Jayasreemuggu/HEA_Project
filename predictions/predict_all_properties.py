import os
import warnings
import joblib
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = "/mnt/c/Users/jayam/Downloads/IITH_ML_project"

CANDIDATE_FILE = os.path.join(
    BASE_DIR,
    "uncertainty",
    "candidates",
    "results",
    "HEA_MPEA_Candidate_Compositions.csv"
)

MODEL_DIR = BASE_DIR

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "predictions",
    "results"
)

MODELS = {
    "YS": "YS_XGBoost_Model.pkl",
    "UTS": "UTS_XGBoost_Model.pkl",
    "Elongation": "Elongation_XGBoost_Model.pkl",
    "Hardness": "Hardness_ExtraTrees_Model.pkl",
}

# All elemental features used by the trained models
ALL_ELEMENTS = [
    "al", "b", "c", "co", "cr", "cu", "fe", "mn",
    "mo", "nb", "ni", "si", "ta", "ti", "v", "w",
    "zr", "ag", "ca", "ga", "hf", "i", "li", "mg",
    "nd", "o", "pd", "re", "ru", "s", "sc", "sn",
    "t", "y", "zn"
]


def get_element_symbol(feature):

    """
    Convert either:

        al_at_pct -> al
        al        -> al

    into the chemical symbol used by the candidate dataset.
    """

    feature = feature.strip().lower()

    if feature.endswith("_at_pct"):
        return feature[:-7]

    if feature in ALL_ELEMENTS:
        return feature

    return None


def build_prediction_matrix(candidates, model):

    # ---------------------------------------------------------
    # Determine the expected raw feature schema
    # ---------------------------------------------------------

    if isinstance(model, dict) and "preprocessor" in model:
        # Bootstrap YS model:
        # {
        #     "preprocessor": ...,
        #     "model": ...,
        #     "seed": ...
        # }
        features = [
            "al_at_pct",
            "b_at_pct",
            "c_at_pct",
            "co_at_pct",
            "cr_at_pct",
            "cu_at_pct",
            "fe_at_pct",
            "mn_at_pct",
            "mo_at_pct",
            "nb_at_pct",
            "ni_at_pct",
            "si_at_pct",
            "ta_at_pct",
            "ti_at_pct",
            "v_at_pct",
            "w_at_pct",
            "zr_at_pct",
            "ag_at_pct",
            "ca_at_pct",
            "ga_at_pct",
            "hf_at_pct",
            "i_at_pct",
            "li_at_pct",
            "mg_at_pct",
            "nd_at_pct",
            "o_at_pct",
            "pd_at_pct",
            "re_at_pct",
            "ru_at_pct",
            "s_at_pct",
            "sc_at_pct",
            "sn_at_pct",
            "t_at_pct",
            "y_at_pct",
            "zn_at_pct",
            "test_temperature_c",
            "vec",
            "atomic_size_mismatch",
            "mixing_enthalpy",
            "mixing_entropy",
            "density_exp_g_cm3",
            "density_calc_g_cm3",
            "grain_size_um",
            "precipitate_size_nm",
            "matrix_volume_pct",
            "test_type",
            "phase",
            "processing_method",
            "alloy_class",
            "equilibrium_condition",
            "single_multiphase",
            "precipitate_info",
        ]

    elif hasattr(model, "feature_names_in_"):
        # Normal sklearn Pipeline / fitted estimator
        features = list(model.feature_names_in_)

    elif isinstance(model, dict) and "model" in model:
        estimator = model["model"]

        if hasattr(estimator, "feature_names_in_"):
            features = list(estimator.feature_names_in_)

        elif "preprocessor" in model and hasattr(
            model["preprocessor"], "feature_names_in_"
        ):
            features = list(model["preprocessor"].feature_names_in_)

        else:
            raise AttributeError(
                "Cannot determine feature names from saved model."
            )

    else:
        raise AttributeError(
            "Model has no usable feature schema."
        )

    # ---------------------------------------------------------
    # Create raw prediction matrix
    # ---------------------------------------------------------

    X = pd.DataFrame(index=candidates.index)

    candidate_lookup = {
        str(col).strip().lower(): col
        for col in candidates.columns
    }

    for feature in features:

        # -----------------------------------------------------
        # ELEMENTAL FEATURES
        # -----------------------------------------------------

        symbol = get_element_symbol(feature)

        if symbol is not None:

            if symbol in candidate_lookup:

                source_col = candidate_lookup[symbol]

                X[feature] = pd.to_numeric(
                    candidates[source_col],
                    errors="coerce"
                )

            else:

                # Candidate generator does not use this element.
                # Zero is correct for generated compositions.
                X[feature] = 0.0

            continue

        # -----------------------------------------------------
        # CONDITIONS
        # -----------------------------------------------------

        if feature == "test_temperature_c":
            X[feature] = 25.0

        elif feature == "test_type":
            X[feature] = "T"

        elif feature == "phase":
            X[feature] = "UNKNOWN"

        elif feature == "processing_method":
            X[feature] = "CAST"

        elif feature == "alloy_class":
            X[feature] = "HEA"

        elif feature == "equilibrium_condition":
            X[feature] = "UNKNOWN"

        elif feature == "single_multiphase":
            X[feature] = "UNKNOWN"

        elif feature == "precipitate_info":
            X[feature] = "UNKNOWN"

        # -----------------------------------------------------
        # PHYSICAL / MICROSTRUCTURAL FEATURES
        # -----------------------------------------------------

        elif feature in [
            "vec",
            "atomic_size_mismatch",
            "mixing_enthalpy",
            "mixing_entropy",
            "density_exp_g_cm3",
            "density_calc_g_cm3",
            "grain_size_um",
            "precipitate_size_nm",
            "matrix_volume_pct",
        ]:

            # Unavailable for hypothetical candidates.
            # The fitted sklearn preprocessor performs imputation.
            X[feature] = np.nan

        else:

            X[feature] = np.nan

    return X

def check_element_mapping(X, model_name):

    print("\nELEMENT MAPPING CHECK")
    print("-" * 70)

    mapped_features = []

    for feature in X.columns:

        symbol = get_element_symbol(feature)

        if symbol is not None:
            mapped_features.append(feature)

    for feature in mapped_features:

        values = X[feature]

        print(
            f"{feature:18s} "
            f"min={values.min():7.2f} "
            f"max={values.max():7.2f} "
            f"unique={values.nunique():5d} "
            f"missing={values.isna().sum():5d}"
        )

    print(
        f"\n{model_name}: "
        f"{len(mapped_features)} elemental features detected."
    )


print("=" * 70)
print("HEA/MPEA FOUR-PROPERTY CANDIDATE PREDICTION")
print("=" * 70)

# ------------------------------------------------------------
# LOAD CANDIDATES
# ------------------------------------------------------------

candidates = pd.read_csv(CANDIDATE_FILE)

print(f"\nCandidate dataset: {candidates.shape}")

if len(candidates) != 5000:

    print(
        f"WARNING: Expected 5000 candidates, "
        f"found {len(candidates)}"
    )

# ------------------------------------------------------------
# INITIAL RESULT TABLE
# ------------------------------------------------------------

results = candidates[
    [
        "candidate_id",
        "composition",
        "n_elements"
    ]
].copy()

# ------------------------------------------------------------
# PREDICT FOUR PROPERTIES
# ------------------------------------------------------------

for property_name, model_file in MODELS.items():

    print("\n" + "=" * 70)
    print(property_name.upper() + " MODEL")
    print("=" * 70)

    model_path = os.path.join(
        MODEL_DIR,
        model_file
    )

    print("Loading:", model_path)

    model = joblib.load(model_path)

    print(
        "Model type:",
        type(model).__name__
    )

    features = list(
        model.feature_names_in_
    )

    print(
        "Expected features:",
        len(features)
    )

    X_candidate = build_prediction_matrix(
        candidates,
        model
    )

    print(
        "Prediction matrix:",
        X_candidate.shape
    )

    # --------------------------------------------------------
    # ELEMENT VALIDATION
    # --------------------------------------------------------

    check_element_mapping(
        X_candidate,
        property_name
    )

    # --------------------------------------------------------
    # SHOW FIRST CANDIDATE
    # --------------------------------------------------------

    if property_name in ["YS", "UTS", "Elongation", "Hardness"]:

        print("\nFirst candidate elemental values:")

        first_values = []

        for feature in features:

            symbol = get_element_symbol(feature)

            if symbol is not None:

                first_values.append(
                    (feature, X_candidate.iloc[0][feature])
                )

        print(first_values)

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    predictions = model.predict(
        X_candidate
    )

    predictions = np.asarray(
        predictions,
        dtype=float
    )

    print(
        f"\nPrediction range: "
        f"{predictions.min():.2f} - "
        f"{predictions.max():.2f}"
    )

    print(
        "Unique predictions:",
        len(np.unique(predictions))
    )

    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    if property_name == "YS":

        results[
            "YS_predicted_MPa"
        ] = predictions

    elif property_name == "UTS":

        results[
            "UTS_predicted_MPa"
        ] = predictions

    elif property_name == "Elongation":

        results[
            "Elongation_predicted_pct"
        ] = predictions

    elif property_name == "Hardness":

        results[
            "Hardness_predicted_HV"
        ] = predictions


# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PREDICTION VALIDATION")
print("=" * 70)

prediction_columns = [
    "YS_predicted_MPa",
    "UTS_predicted_MPa",
    "Elongation_predicted_pct",
    "Hardness_predicted_HV"
]

for column in prediction_columns:

    values = results[column]

    print("\n" + column)

    print(
        "  Missing:",
        values.isna().sum()
    )

    print(
        "  Minimum:",
        values.min()
    )

    print(
        "  Maximum:",
        values.max()
    )

    print(
        "  Median:",
        values.median()
    )

    print(
        "  Unique:",
        values.nunique()
    )

# ------------------------------------------------------------
# VALID FLAG
# ------------------------------------------------------------

results["valid_prediction"] = (
    results[prediction_columns]
    .notna()
    .all(axis=1)
)

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

output_file = os.path.join(
    OUTPUT_DIR,
    "HEA_MPEA_5000_Four_Property_Predictions.csv"
)

results.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)

print(
    "Total candidates:",
    len(results)
)

print(
    "Valid predictions:",
    results["valid_prediction"].sum()
)

print("\nOutput:")
print(output_file)

# ------------------------------------------------------------
# TOP 100
# ------------------------------------------------------------

for column in prediction_columns:

    top = (
        results
        .sort_values(
            column,
            ascending=False
        )
        .head(100)
    )

    if column.startswith("YS_"):
        name = "YS"

    elif column.startswith("UTS_"):
        name = "UTS"

    elif column.startswith("Elongation_"):
        name = "Elongation"

    else:
        name = "Hardness"

    top_file = os.path.join(
        OUTPUT_DIR,
        f"Top100_{name}_Candidates.csv"
    )

    top.to_csv(
        top_file,
        index=False
    )

print("\nTop-100 files created.")
