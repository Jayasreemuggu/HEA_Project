import os
import sys
import itertools
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# SETTINGS
# ============================================================

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "candidates",
    "results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
N_CANDIDATES = 5000

# Candidate elements selected from the main HEA/MPEA dataset.
# These are common elements with substantial representation.
ELEMENTS = [
    "Al",
    "Co",
    "Cr",
    "Cu",
    "Fe",
    "Mn",
    "Mo",
    "Nb",
    "Ni",
    "Ti",
    "V",
    "W"
]

# Number of principal elements.
MIN_ELEMENTS = 4
MAX_ELEMENTS = 6

# Minimum and maximum atomic percentage.
MIN_AT_PCT = 5.0
MAX_AT_PCT = 45.0

rng = np.random.default_rng(RANDOM_STATE)


# ============================================================
# GENERATE ONE COMPOSITION
# ============================================================

def generate_composition(elements):

    n = rng.integers(
        MIN_ELEMENTS,
        MAX_ELEMENTS + 1
    )

    selected = list(
        rng.choice(
            elements,
            size=n,
            replace=False
        )
    )

    # Dirichlet creates positive fractions summing to 1.
    fractions = rng.dirichlet(
        np.ones(n) * 2.0
    )

    composition = fractions * 100.0

    # Reject if any element is outside allowed range.
    if np.any(composition < MIN_AT_PCT):
        return None

    if np.any(composition > MAX_AT_PCT):
        return None

    return dict(
        zip(
            selected,
            composition
        )
    )


# ============================================================
# CANONICAL COMPOSITION
# ============================================================

def canonical_composition(comp):

    parts = []

    for element in sorted(comp.keys()):

        value = comp[element]

        if value > 0:

            parts.append(
                f"{element}{value:.2f}"
            )

    return "".join(parts)


# ============================================================
# GENERATE CANDIDATES
# ============================================================

candidates = []
seen = set()

attempts = 0

print("=" * 70)
print("HEA/MPEA CANDIDATE GENERATION")
print("=" * 70)

while len(candidates) < N_CANDIDATES:

    attempts += 1

    comp = generate_composition(ELEMENTS)

    if comp is None:
        continue

    # Normalize to exactly 100 at.%
    total = sum(comp.values())

    comp = {
        element: value / total * 100.0
        for element, value in comp.items()
    }

    # Round to 2 decimals.
    comp = {
        element: round(value, 2)
        for element, value in comp.items()
    }

    # Correct rounding difference.
    total = sum(comp.values())

    largest_element = max(
        comp,
        key=comp.get
    )

    comp[largest_element] = round(
        comp[largest_element] +
        (100.0 - total),
        2
    )

    # Validate.
    total = sum(comp.values())

    if abs(total - 100.0) > 0.01:
        continue

    if any(
        value < MIN_AT_PCT or
        value > MAX_AT_PCT
        for value in comp.values()
    ):
        continue

    canonical = canonical_composition(comp)

    if canonical in seen:
        continue

    seen.add(canonical)

    row = {
        "candidate_id":
            f"CAND_{len(candidates)+1:05d}",
        "composition":
            canonical,
        "n_elements":
            len(comp)
    }

    for element in ELEMENTS:
        row[element] = comp.get(
            element,
            0.0
        )

    candidates.append(row)


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(candidates)

element_columns = ELEMENTS

df["composition_sum"] = df[
    element_columns
].sum(axis=1)

df["valid_composition"] = np.isclose(
    df["composition_sum"],
    100.0,
    atol=0.01
)


# ============================================================
# SAVE
# ============================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "HEA_MPEA_Candidate_Compositions.csv"
)

df.to_csv(
    output_file,
    index=False
)


# ============================================================
# REPORT
# ============================================================

print()
print(f"Candidates generated : {len(df)}")
print(f"Generation attempts  : {attempts}")
print(
    f"Valid compositions   : "
    f"{df['valid_composition'].sum()}"
)

print(
    f"Unique compositions  : "
    f"{df['composition'].nunique()}"
)

print(
    f"Element count range  : "
    f"{df['n_elements'].min()} - "
    f"{df['n_elements'].max()}"
)

print()
print("Example candidates:")
print(
    df[
        [
            "candidate_id",
            "composition",
            "n_elements",
            "composition_sum"
        ]
    ].head(10).to_string(index=False)
)

print()
print("Saved:")
print(output_file)

print("=" * 70)
