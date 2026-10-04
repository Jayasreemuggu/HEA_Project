# AI-Assisted HEA/MPEA Materials Discovery using Machine Learning

An end-to-end machine-learning system for predicting tensile yield strength of High-Entropy Alloys (HEAs) and Multi-Principal Element Alloys (MPEAs).

## Project Overview

The system uses experimental alloy data, materials descriptors, phase-aware feature engineering, XGBoost regression, specialist models, residual correction, and a FastAPI inference service.

### Dataset

- 1,941 experimental records
- 729 unique canonical alloy compositions
- Target: `YS_Tensile_MPa`
- 359 production features
- 352 numeric features
- 7 categorical features

## Production Model

The final production system contains:

1. Full-data XGBoost base model
2. 800–1200 MPa specialist XGBoost model
3. 1200–1580 MPa specialist XGBoost model
4. Five range-specific residual XGBoost models
5. Frozen range-specific residual correction weights

## Frozen OOF Performance

| Metric | Result |
|---|---:|
| R² | **0.950940360** |
| MAE | **80.454415 MPa** |
| RMSE | **122.959738 MPa** |

These are the frozen 5-fold out-of-fold results on the 1,941-row modelling dataset.

## Locked Test Evaluation

The locked test set contains 428 samples with composition-based inputs but does not contain all inputs required by the 359-feature production model. Therefore, the production model was not evaluated by artificially filling unavailable features.

A separate composition-only model was evaluated:

| Metric | Result |
|---|---:|
| R² | **0.564540388** |
| MAE | **257.604275 MPa** |
| RMSE | **352.263569 MPa** |

## Inference Pipeline

``text
Input alloy data
      ↓
359-feature reconstruction
      ↓
Base XGBoost
      ↓
Midrange specialist correction
      ↓
Range-specific residual correction
      ↓
Final yield-strength prediction
``

## FastAPI

The API is implemented in `app.py`.

Start the API:

``powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000
``

Endpoints: `/`, `/health`, and `/predict`.

Swagger documentation: `http://127.0.0.1:8000/docs`

## Production Artifact

``text
models/YS_FINAL_PRODUCTION/YS_FINAL_PRODUCTION_MODEL.pkl
``

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- Joblib
- FastAPI
- Uvicorn

## Disclaimer

Model predictions are statistical estimates generated from experimental materials data. They should not be treated as experimentally validated material properties. Predictions outside the training distribution may have higher uncertainty and require experimental validation.
