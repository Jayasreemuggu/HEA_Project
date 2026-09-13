import re
import pandas as pd
import torch

from torch_geometric.data import Data
from itertools import combinations

try:
    from gnn.element_features import get_element_features
except ModuleNotFoundError:
    from element_features import get_element_features


# ============================================================
# ELEMENTS
# ============================================================

ELEMENTS = [
    "Ag", "Al", "B", "C", "Ca", "Co", "Cr", "Cu",
    "Fe", "Ga", "Hf", "I", "Li", "Mg", "Mn", "Mo",
    "Nb", "Nd", "Ni", "O", "Pd", "Re", "Ru", "S",
    "Sc", "Si", "Sn", "Ta", "Ti", "V", "W", "Y",
    "Zn", "Zr"
]


# ============================================================
# CONDITION CATEGORIES
# ============================================================

TEST_TYPES = [
    "C",
    "T",
    "UNKNOWN"
]


PROCESSING_TYPES = [
    "CAST",
    "WROUGHT",
    "ANNEAL",
    "POWDER",
    "ROLLING",
    "FORGING",
    "HEAT_TREATED",
    "QUENCHED",
    "HOMOGENIZED",
    "AS_CAST",
    "OTHER",
    "UNKNOWN"
]


PHASE_TYPES = [
    "BCC",
    "FCC",
    "HCP",
    "BCC_FCC",
    "BCC_LAVES",
    "BCC_SEC",
    "FCC_BCC",
    "FCC_SEC",
    "LAVES",
    "SIGMA",
    "INTERMETALLIC",
    "MULTIPHASE",
    "AMORPHOUS",
    "OTHER",
    "UNKNOWN"
]


# ============================================================
# ONE-HOT ENCODING
# ============================================================

def one_hot(value, categories):

    vector = [0.0] * len(categories)

    if value in categories:
        vector[categories.index(value)] = 1.0
    else:
        vector[-1] = 1.0

    return vector


# ============================================================
# TEST TYPE NORMALIZATION
# ============================================================

def normalize_test_type(value):

    if pd.isna(value):
        return "UNKNOWN"

    value = str(value).strip().upper()

    if value == "C":
        return "C"

    if value == "T":
        return "T"

    return "UNKNOWN"


# ============================================================
# PROCESSING NORMALIZATION
# ============================================================

def normalize_processing(value):

    if pd.isna(value):
        return "UNKNOWN"

    text = str(value).strip().upper()

    if text in ["", "NAN", "NONE"]:
        return "UNKNOWN"

    # CAST
    if "CAST" in text:
        if "AS-CAST" in text or "AS CAST" in text:
            return "AS_CAST"
        return "CAST"

    # WROUGHT
    if "WROUGHT" in text:
        return "WROUGHT"

    # ANNEAL
    if "ANNEAL" in text or "RECRYSTALL" in text:
        return "ANNEAL"

    # POWDER
    if "POWDER" in text or "SINTER" in text:
        return "POWDER"

    # ROLLING
    if (
        "ROLL" in text
        or "COLD ROL" in text
        or "HOT ROL" in text
        or "CRYOROLL" in text
    ):
        return "ROLLING"

    # FORGING
    if "FORG" in text:
        return "FORGING"

    # HEAT TREATMENT
    if (
        "HEAT TREAT" in text
        or "SOLUTION TREAT" in text
        or "AGING" in text
        or "AGED" in text
    ):
        return "HEAT_TREATED"

    # QUENCHING
    if "QUENCH" in text:
        return "QUENCHED"

    # HOMOGENIZATION
    if "HOMOGEN" in text:
        return "HOMOGENIZED"

    return "OTHER"


# ============================================================
# PHASE NORMALIZATION
# ============================================================

def normalize_phase(value):

    if pd.isna(value):
        return "UNKNOWN"

    text = str(value).strip().upper()

    if text in ["", "NAN", "NONE"]:
        return "UNKNOWN"

    # Remove classification descriptions
    text = re.sub(
        r"\(.*?CLASSIFICATION.*?\)",
        "",
        text
    )

    # Normalize separators
    text = text.replace("–", "-")
    text = text.replace("—", "-")

    has_bcc = "BCC" in text or "BODY-CENTERED CUBIC" in text
    has_fcc = "FCC" in text or "FACE-CENTERED CUBIC" in text
    has_hcp = "HCP" in text or "HEXAGONAL CLOSE" in text

    has_laves = "LAVES" in text
    has_sigma = "SIGMA" in text
    has_sec = (
        "SEC." in text
        or "SECONDARY" in text
        or "SEC " in text
    )

    # Specific multiphase cases
    if has_bcc and has_fcc:
        return "BCC_FCC"

    if has_bcc and has_laves:
        return "BCC_LAVES"

    if has_bcc and has_sec:
        return "BCC_SEC"

    if has_fcc and has_bcc:
        return "FCC_BCC"

    if has_fcc and has_sec:
        return "FCC_SEC"

    if has_laves:
        return "LAVES"

    if has_sigma:
        return "SIGMA"

    if has_bcc and has_fcc:
        return "MULTIPHASE"

    if has_bcc:
        return "BCC"

    if has_fcc:
        return "FCC"

    if has_hcp:
        return "HCP"

    if (
        "INTERMETALLIC" in text
        or "INTERMETAL" in text
    ):
        return "INTERMETALLIC"

    if "AMORPH" in text:
        return "AMORPHOUS"

    # Multiple phases separated by common delimiters
    phase_tokens = [
        "BCC", "FCC", "HCP", "LAVES",
        "SIGMA", "INTERMETALLIC"
    ]

    matches = sum(
        token in text
        for token in phase_tokens
    )

    if matches > 1:
        return "MULTIPHASE"

    return "OTHER"


# ============================================================
# TEMPERATURE FEATURES
# ============================================================

def temperature_features(value, mean=25.0, std=100.0):

    if pd.isna(value):

        return [
            0.0,
            1.0
        ]

    temperature = float(value)

    normalized = (
        temperature - mean
    ) / max(std, 1e-8)

    return [
        normalized,
        0.0
    ]


# ============================================================
# CONDITION VECTOR
#
# 2  = temperature
# 3  = test type
# 12 = processing
# 15 = phase
#
# TOTAL = 32
# ============================================================

def build_condition_vector(
    row,
    temperature_mean=25.0,
    temperature_std=100.0
):

    temperature = temperature_features(
        row.get("test_temperature_c"),
        temperature_mean,
        temperature_std
    )

    test_type = normalize_test_type(
        row.get("test_type")
    )

    processing = normalize_processing(
        row.get("processing_method")
    )

    phase = normalize_phase(
        row.get("phase")
    )

    vector = (
        temperature
        + one_hot(
            test_type,
            TEST_TYPES
        )
        + one_hot(
            processing,
            PROCESSING_TYPES
        )
        + one_hot(
            phase,
            PHASE_TYPES
        )
    )

    assert len(vector) == 32

    return torch.tensor(
        vector,
        dtype=torch.float
    )


# ============================================================
# COMPOSITION PARSER
# ============================================================

def parse_composition(
    composition_canonical
):

    if pd.isna(composition_canonical):
        return {}

    composition = {}

    text = str(
        composition_canonical
    )

    parts = text.split(";")

    for part in parts:

        part = part.strip()

        if ":" not in part:
            continue

        element, fraction = part.split(
            ":",
            1
        )

        element = element.strip()

        try:
            fraction = float(
                fraction.strip()
            )
        except ValueError:
            continue

        if element in ELEMENTS:
            composition[element] = fraction

    return composition


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

def build_graph(
    row,
    target_mean=None,
    target_std=None,
    temperature_mean=25.0,
    temperature_std=100.0
):

    composition = parse_composition(
        row["composition_canonical"]
    )

    if len(composition) < 2:
        return None

    elements = list(
        composition.keys()
    )

    # --------------------------------------------------------
    # Node features
    #
    # 6 elemental descriptors
    # + composition fraction
    #
    # = 7 features
    # --------------------------------------------------------

    node_features = []

    for element in elements:

        features = get_element_features(
            element
        )

        fraction = (
            composition[element] / 100.0
        )

        features = features + [
            fraction
        ]

        node_features.append(
            features
        )

    x = torch.tensor(
        node_features,
        dtype=torch.float
    )

    # --------------------------------------------------------
    # Edges
    #
    # Complete directed graph
    # --------------------------------------------------------

    edge_pairs = []

    for i, j in combinations(
        range(len(elements)),
        2
    ):

        edge_pairs.append(
            [i, j]
        )

        edge_pairs.append(
            [j, i]
        )

    if edge_pairs:

        edge_index = torch.tensor(
            edge_pairs,
            dtype=torch.long
        ).t().contiguous()

    else:

        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long
        )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    target = float(
        row["yield_strength_mpa"]
    )

    if (
        target_mean is not None
        and target_std is not None
    ):

        target = (
            target - target_mean
        ) / max(
            target_std,
            1e-8
        )

    y = torch.tensor(
        [target],
        dtype=torch.float
    )

    # --------------------------------------------------------
    # Conditions
    # --------------------------------------------------------

    conditions = build_condition_vector(
        row,
        temperature_mean,
        temperature_std
    ).unsqueeze(0)

    # --------------------------------------------------------
    # Create PyG Data object
    # --------------------------------------------------------

    graph = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        conditions=conditions
    )

    graph.element_names = elements

    graph.record_id = int(
        row["record_id"]
    )

    return graph


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("HEA/MPEA GRAPH CONSTRUCTION TEST")
    print("=" * 60)

    dataset_path = "YS_ML_Dataset.csv"

    df = pd.read_csv(
        dataset_path
    )

    print(
        "\nDataset shape:",
        df.shape
    )

    row = df.iloc[0]

    graph = build_graph(
        row,
        target_mean=df[
            "yield_strength_mpa"
        ].mean(),
        target_std=df[
            "yield_strength_mpa"
        ].std()
    )

    if graph is None:

        print(
            "\nGraph construction failed."
        )

    else:

        print(
            "\nRecord ID:",
            row["record_id"]
        )

        print(
            "Alloy:",
            row["alloy_name"]
        )

        print(
            "Composition:",
            row["composition_canonical"]
        )

        print(
            "\nGraph:"
        )

        print(graph)

        print(
            "\nElements:"
        )

        print(
            graph.element_names
        )

        print(
            "\nNode feature shape:"
        )

        print(
            graph.x.shape
        )

        print(
            "\nEdge shape:"
        )

        print(
            graph.edge_index.shape
        )

        print(
            "\nCondition shape:"
        )

        print(
            graph.conditions.shape
        )

        print(
            "\nCondition vector length:"
        )

        print(
            graph.conditions.shape[1]
        )

        print(
            "\nYield strength:"
        )

        print(
            row["yield_strength_mpa"]
        )

        print(
            "\nNormalized target:"
        )

        print(
            graph.y
        )

    print("\n" + "=" * 60)
    print("GRAPH TEST COMPLETED")
    print("=" * 60)
