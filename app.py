import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

import importlib.util

PREDICT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predict_ys.py')
spec = importlib.util.spec_from_file_location('predict_ys', PREDICT_PATH)
predict_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predict_module)
predict = predict_module.predict

app = FastAPI(
    title="HEA/MPEA Yield Strength Prediction API",
    version="1.0.0"
)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "YS_FINAL_PRODUCTION",
    "YS_FINAL_PRODUCTION_MODEL.pkl"
)

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Production model not found: {MODEL_PATH}")

artifact = joblib.load(MODEL_PATH)


class PredictionRequest(BaseModel):
    features: dict


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


@app.post("/predict")
def prediction(request: PredictionRequest):
    try:
        df = pd.DataFrame([request.features])
        result = predict(df)

        return {
            "base_prediction_mpa": float(result["Base_Prediction"].iloc[0]),
            "residual_correction_mpa": float(result["Residual_Correction"].iloc[0]),
            "yield_strength_prediction_mpa": float(
                result["Final_Prediction_MPa"].iloc[0]
            )
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

