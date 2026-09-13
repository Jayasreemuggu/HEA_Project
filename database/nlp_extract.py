import os
import re
import json
import pandas as pd
import pymupdf


# ================================================================
# CONFIGURATION
# ================================================================

PDF_FOLDER = "papers"
OUTPUT_FILE = "literature_records_v3.csv"
TEXT_FOLDER = "extracted_text_v3"

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(TEXT_FOLDER, exist_ok=True)


# ================================================================
# METALLIC ELEMENTS
# ================================================================

METALLIC_ELEMENTS = {
    "Al", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
    "Zn", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag",
    "Cd", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
    "Li", "Mg", "Ca", "Sr", "Ba", "La", "Ce", "Pr", "Nd",
    "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Th", "U"
}


# ================================================================
# ELEMENTS COMMONLY FOUND IN NON-ALLOY COMPOUNDS
# ================================================================

NON_METALLIC_ELEMENTS = {
    "H", "B", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "Si"
}


# ================================================================
# KNOWN NON-COMPOSITION TERMS
# ================================================================

BLACKLIST = {
    "MPEA", "MPEAs",
    "HEA", "HEAs",
    "CCA", "CCAs",
    "RHEA", "RHEAs",
    "RCCA", "RCCAs",

    "USA",
    "DOI",
    "GPa",
    "MPa",
    "HV",
    "UTS",
    "YS",

    "SEM",
    "TEM",
    "EDS",
    "XRD",
    "FCC",
    "BCC",
    "HCP",
    "Laves",
    "B2",
    "L12",

    "ROM",
    "VEC",
    "API",
    "LLM",
    "LLMs",
    "FIB",
    "ML",
    "URL",

    "OPEN",
    "OTHER",
    "CAST",
    "POWDER",
    "WROUGHT",
    "ANNEAL",

    "CNRS",
    "UPEC",
    "JHU",
    "ICMCB",
    "INP",
    "AFB",
    "LSPM",

    "OK",
    "BY",
    "NC",
    "CI",
    "BC",
    "BN",
    "BF",
    "BY",
    "O.N",
    "D.B"
}


# ================================================================
# KNOWN COMPOUND / CERAMIC / SOLVENT PATTERNS
# ================================================================

COMPOUND_PATTERNS = [
    r"^SiC$",
    r"^Ni\d+(?:\.\d+)?Ti$",
    r"^C\d+H\d+O\d*$",
    r"^H\d+C\d+O\d*$",
    r"^HCIO\d+$",
    r"^Al\d+O\d+$",
    r"^TiO\d+$",
    r"^ZrO\d+$",
    r"^NbC$",
    r"^TaC$",
    r"^WC$"
]


# ================================================================
# TOKENIZE CHEMICAL FORMULA
# ================================================================

def tokenize_formula(candidate):

    pattern = r"([A-Z][a-z]?)(\d*(?:\.\d+)?)"

    matches = re.findall(
        pattern,
        candidate
    )

    if not matches:
        return []

    reconstructed = ""

    tokens = []

    for element, amount in matches:

        reconstructed += element + amount

        tokens.append(
            (element, amount)
        )

    if reconstructed != candidate:
        return []

    return tokens


# ================================================================
# VALIDATE COMPOSITION
# ================================================================

def is_valid_composition(candidate):

    candidate = candidate.strip()

    if not candidate:
        return False

    if candidate in BLACKLIST:
        return False

    # ------------------------------------------------------------
    # Length limits
    # ------------------------------------------------------------

    if len(candidate) < 4:
        return False

    if len(candidate) > 80:
        return False

    # ------------------------------------------------------------
    # Must start with a capital letter
    # ------------------------------------------------------------

    if not re.match(
        r"^[A-Z]",
        candidate
    ):
        return False

    # ------------------------------------------------------------
    # Reject obvious compounds
    # ------------------------------------------------------------

    for pattern in COMPOUND_PATTERNS:

        if re.match(
            pattern,
            candidate,
            flags=re.IGNORECASE
        ):
            return False

    # ------------------------------------------------------------
    # Tokenize
    # ------------------------------------------------------------

    tokens = tokenize_formula(
        candidate
    )

    if len(tokens) < 2:
        return False

    elements = [
        element
        for element, amount in tokens
    ]

    # ------------------------------------------------------------
    # Every element must be known
    # ------------------------------------------------------------

    for element in elements:

        if element not in METALLIC_ELEMENTS:

            return False

    # ------------------------------------------------------------
    # At least 3 metallic elements
    #
    # This is intentionally stricter for HEA/MPEA extraction.
    # ------------------------------------------------------------

    if len(set(elements)) < 3:
        return False

    # ------------------------------------------------------------
    # Reject repeated-only structures / simple compounds
    # ------------------------------------------------------------

    if len(elements) == 2:
        return False

    # ------------------------------------------------------------
    # Count elements with explicit numeric composition
    # ------------------------------------------------------------

    numeric_count = 0

    for element, amount in tokens:

        if amount != "":
            numeric_count += 1

    # ------------------------------------------------------------
    # HEA formulas commonly appear either as:
    #
    # AlCoCrFeNi
    # Al20Co20Cr20Fe20Ni20
    # Al0.5CrNbTi2V0.5
    #
    # At least 3 elements is sufficient.
    # ------------------------------------------------------------

    return True


# ================================================================
# FIND HEA/MPEA COMPOSITIONS
# ================================================================

def find_compositions(text):

    pattern = (
        r"\b"
        r"(?:"
        r"[A-Z][a-z]?"
        r"(?:\d+(?:\.\d+)?)?"
        r"){3,}"
        r"\b"
    )

    candidates = re.findall(
        pattern,
        text
    )

    valid = []

    for candidate in candidates:

        if is_valid_composition(
            candidate
        ):

            valid.append(
                candidate
            )

    return list(
        dict.fromkeys(valid)
    )


# ================================================================
# TEMPERATURES
# ================================================================

def find_temperatures(text):

    pattern = (
        r"-?\d+(?:\.\d+)?"
        r"\s*(?:°\s*C|°C|K)\b"
    )

    return list(
        dict.fromkeys(
            re.findall(
                pattern,
                text,
                flags=re.IGNORECASE
            )
        )
    )


# ================================================================
# STRENGTH
# ================================================================

def find_strength_values(text):

    pattern = (
        r"\b"
        r"\d+(?:\.\d+)?"
        r"\s*(?:MPa|GPa)"
        r"\b"
    )

    return list(
        dict.fromkeys(
            re.findall(
                pattern,
                text,
                flags=re.IGNORECASE
            )
        )
    )


# ================================================================
# HARDNESS
# ================================================================

def find_hardness_values(text):

    pattern = (
        r"\b"
        r"\d+(?:\.\d+)?"
        r"\s*(?:HV|HB|HRC)"
        r"\b"
    )

    return list(
        dict.fromkeys(
            re.findall(
                pattern,
                text,
                flags=re.IGNORECASE
            )
        )
    )


# ================================================================
# ELONGATION
# ================================================================

def find_elongation_values(text):

    pattern = (
        r"\b"
        r"\d+(?:\.\d+)?"
        r"\s*%"
    )

    return list(
        dict.fromkeys(
            re.findall(
                pattern,
                text
            )
        )
    )


# ================================================================
# PHASES
# ================================================================

def find_phases(text):

    phase_patterns = {

        "FCC":
            r"\bFCC\b",

        "BCC":
            r"\bBCC\b",

        "HCP":
            r"\bHCP\b",

        "Laves":
            r"\bLaves\b",

        "B2":
            r"\bB2\b",

        "L12":
            r"\bL12\b",

        "Sigma":
            r"\bSigma(?:\s+phase)?\b",

        "mu_phase":
            r"\bmu\s+phase\b|\bμ\s*phase\b"
    }

    found = []

    for phase, pattern in phase_patterns.items():

        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        ):

            found.append(phase)

    return found


# ================================================================
# PROCESSING
# ================================================================

def find_processing_methods(text):

    processing_patterns = {

        "casting":
            r"\bcast(?:ing)?\b",

        "forging":
            r"\bforg(?:ed|ing)\b",

        "rolling":
            r"\broll(?:ed|ing)\b",

        "annealing":
            r"\banneal(?:ed|ing)\b",

        "heat_treatment":
            r"\bheat[- ]treatment\b",

        "sintering":
            r"\bsinter(?:ed|ing)\b",

        "mechanical_alloying":
            r"\bmechanical alloying\b",

        "powder_metallurgy":
            r"\bpowder metallurgy\b",

        "cold_rolling":
            r"\bcold[- ]rolled\b",

        "hot_rolling":
            r"\bhot[- ]rolled\b",

        "HIP":
            r"\bHIP\b",

        "SPS":
            r"\bSPS\b"
    }

    found = []

    for method, pattern in processing_patterns.items():

        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        ):

            found.append(method)

    return found


# ================================================================
# CLEAN CONTEXT
# ================================================================

def clean_context(text):

    return " ".join(
        text.split()
    )[:2500]


# ================================================================
# PROCESS PDF
# ================================================================

def process_pdf(pdf_path):

    document = pymupdf.open(
        pdf_path
    )

    all_text = ""

    page_records = []

    for page_number, page in enumerate(
        document,
        start=1
    ):

        text = page.get_text()

        all_text += "\n" + text

        compositions = find_compositions(
            text
        )

        temperatures = find_temperatures(
            text
        )

        strengths = find_strength_values(
            text
        )

        hardness = find_hardness_values(
            text
        )

        elongation = find_elongation_values(
            text
        )

        phases = find_phases(
            text
        )

        processing = find_processing_methods(
            text
        )

        if (
            compositions
            or temperatures
            or strengths
            or hardness
            or elongation
            or phases
            or processing
        ):

            page_records.append({

                "page":
                    page_number,

                "compositions":
                    json.dumps(
                        compositions,
                        ensure_ascii=False
                    ),

                "temperatures":
                    json.dumps(
                        temperatures,
                        ensure_ascii=False
                    ),

                "strength_values":
                    json.dumps(
                        strengths,
                        ensure_ascii=False
                    ),

                "hardness_values":
                    json.dumps(
                        hardness,
                        ensure_ascii=False
                    ),

                "elongation_values":
                    json.dumps(
                        elongation,
                        ensure_ascii=False
                    ),

                "phases":
                    json.dumps(
                        phases,
                        ensure_ascii=False
                    ),

                "processing_methods":
                    json.dumps(
                        processing,
                        ensure_ascii=False
                    ),

                "context":
                    clean_context(text)
            })

    document.close()

    return all_text, page_records


# ================================================================
# MAIN
# ================================================================

print("=" * 70)
print("NLP LITERATURE EXTRACTION V3")
print("=" * 70)


pdf_files = sorted([
    file
    for file in os.listdir(PDF_FOLDER)
    if file.lower().endswith(".pdf")
])


print(
    "\nPDF files found:",
    len(pdf_files)
)


records = []


for index, filename in enumerate(
    pdf_files,
    start=1
):

    print(
        f"\nProcessing {index}/{len(pdf_files)}: {filename}"
    )

    pdf_path = os.path.join(
        PDF_FOLDER,
        filename
    )

    text, page_records = process_pdf(
        pdf_path
    )

    print(
        "Characters extracted:",
        len(text)
    )

    print(
        "Relevant pages:",
        len(page_records)
    )

    # Save raw text
    text_filename = (
        os.path.splitext(filename)[0]
        + ".txt"
    )

    text_path = os.path.join(
        TEXT_FOLDER,
        text_filename
    )

    with open(
        text_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(text)

    # Add records
    for page_record in page_records:

        records.append({

            "paper_file":
                filename,

            "page":
                page_record["page"],

            "compositions":
                page_record["compositions"],

            "temperatures":
                page_record["temperatures"],

            "strength_values":
                page_record["strength_values"],

            "hardness_values":
                page_record["hardness_values"],

            "elongation_values":
                page_record["elongation_values"],

            "phases":
                page_record["phases"],

            "processing_methods":
                page_record["processing_methods"],

            "context":
                page_record["context"]
        })


# ================================================================
# SAVE CSV
# ================================================================

df = pd.DataFrame(
    records
)

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ================================================================
# UNIQUE COMPOSITIONS
# ================================================================

unique_compositions = set()

for values in df["compositions"].dropna():

    try:

        compositions = json.loads(
            values
        )

        for composition in compositions:

            unique_compositions.add(
                composition
            )

    except Exception:
        pass


# ================================================================
# SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("NLP V3 EXTRACTION COMPLETED")
print("=" * 70)

print(
    "Papers processed:",
    len(pdf_files)
)

print(
    "Relevant page records:",
    len(df)
)

print(
    "Unique HEA/MPEA-like compositions:",
    len(unique_compositions)
)

print(
    "Output:",
    OUTPUT_FILE
)

print(
    "Text folder:",
    TEXT_FOLDER
)