import sys
import os
import importlib.util

import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


PROJECT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT)

# ------------------------------------------------------------------
# Production prediction module
# ------------------------------------------------------------------

PREDICT_PATH = os.path.join(PROJECT, "predict_ys.py")

spec = importlib.util.spec_from_file_location("predict_ys", PREDICT_PATH)
predict_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predict_module)

predict = predict_module.predict

# ------------------------------------------------------------------
# Frontend feature-generation module
# ------------------------------------------------------------------

FEATURE_PATH = os.path.join(PROJECT, "frontend_inference.py")

feature_spec = importlib.util.spec_from_file_location(
    "frontend_inference",
    FEATURE_PATH
)
feature_module = importlib.util.module_from_spec(feature_spec)
feature_spec.loader.exec_module(feature_module)

generate_materials_features = feature_module.generate_materials_features

# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

MODEL_PATH = os.path.join(
    PROJECT,
    "models",
    "YS_FINAL_PRODUCTION",
    "YS_FINAL_PRODUCTION_MODEL.pkl"
)

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Production model not found: {MODEL_PATH}")

artifact = joblib.load(MODEL_PATH)

# ------------------------------------------------------------------
# FastAPI
# ------------------------------------------------------------------

app = FastAPI(
    title="HEA/MPEA Yield Strength Prediction API",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Request models
# ------------------------------------------------------------------

class PredictionRequest(BaseModel):
    features: dict


class MaterialPredictionRequest(BaseModel):
    composition: dict[str, float] = Field(
        ...,
        description="Elemental composition in atomic percent."
    )

    Test_Temperature_C: float | None = None
    Grain_Size_um: float | None = None
    Density_Exp_g_cm3: float | None = None
    Density_Calc_g_cm3: float | None = None
    Precipitate_Size_nm: float | None = None
    Matrix_Volume_pct: float | None = None
    Youngs_Modulus_Exp_GPa: float | None = None
    Youngs_Modulus_Calc_GPa: float | None = None

    Processing_Method: str | None = None
    Phase: str | None = None
    Alloy_Class: str | None = None
    Equilibrium_Condition: str | None = None
    Single_Multiphase: str | None = None
    Test_Type: str | None = None
    Precipitate_Info: str | None = None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_material_dataframe(request: MaterialPredictionRequest) -> pd.DataFrame:
    row = {}

    # Composition
    # The feature generator expects all 35 elemental columns.
    # Elements omitted by the user are treated as 0 at.%.
    elements = [
        "Ag", "Al", "B", "C", "Ca", "Co", "Cr", "Cu", "Fe", "Ga",
        "Hf", "I", "Li", "Mg", "Mn", "Mo", "Nb", "Nd", "Ni", "O",
        "Pd", "Re", "Ru", "S", "Sc", "Si", "Sn", "T", "Ta", "Ti",
        "V", "W", "Y", "Zn", "Zr"
    ]

    for element in elements:
        row[element] = float(request.composition.get(element, 0.0))

    # Experimental / physical features
    optional_numeric = [
        "Test_Temperature_C",
        "Grain_Size_um",
        "Density_Exp_g_cm3",
        "Density_Calc_g_cm3",
        "Precipitate_Size_nm",
        "Matrix_Volume_pct",
        "Youngs_Modulus_Exp_GPa",
        "Youngs_Modulus_Calc_GPa",
    ]

    for column in optional_numeric:
        row[column] = getattr(request, column)

    # Categorical features
    categorical = [
        "Processing_Method",
        "Phase",
        "Alloy_Class",
        "Equilibrium_Condition",
        "Single_Multiphase",
        "Test_Type",
        "Precipitate_Info",
    ]

    for column in categorical:
        row[column] = getattr(request, column)

    return pd.DataFrame([row])


def run_prediction(df: pd.DataFrame):
    features = generate_materials_features(df)

    if features.shape[1] != 359:
        raise ValueError(
            f"Feature generation produced {features.shape[1]} features; "
            "production model requires exactly 359."
        )

    result = predict(features)

    return {
        "base_prediction_mpa": float(
            result["Base_Prediction"].iloc[0]
        ),
        "residual_correction_mpa": float(
            result["Residual_Correction"].iloc[0]
        ),
        "yield_strength_prediction_mpa": float(
            result["Final_Prediction_MPa"].iloc[0]
        )
    }


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "HEA/MPEA Yield Strength Prediction API",
        "status": "running",
        "model": "YS_FINAL_PRODUCTION_MODEL"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "raw_features": artifact["raw_features"],
        "training_rows": artifact["training_rows"]
    }


# Existing 359-feature endpoint
@app.post("/predict")
def prediction(request: PredictionRequest):
    try:
        df = pd.DataFrame([request.features])
        result = predict(df)

        return {
            "base_prediction_mpa": float(
                result["Base_Prediction"].iloc[0]
            ),
            "residual_correction_mpa": float(
                result["Residual_Correction"].iloc[0]
            ),
            "yield_strength_prediction_mpa": float(
                result["Final_Prediction_MPa"].iloc[0]
            )
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# User-friendly composition endpoint
@app.post("/predict-material")
def predict_material(request: MaterialPredictionRequest):
    try:
        df = make_material_dataframe(request)
        return run_prediction(df)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
