# AI-Assisted HEA/MPEA Materials Discovery using Machine Learning

An end-to-end machine-learning system for predicting tensile yield strength of High-Entropy Alloys (HEAs) and Multi-Principal Element Alloys (MPEAs) using composition, material descriptors, phase information, and processing conditions.

## Project Overview

The system uses experimental alloy data, materials descriptors, phase-aware feature engineering, XGBoost regression, specialist models, residual correction, and a production inference application built with FastAPI and Next.js.

The final production pipeline uses **359 raw features** to predict tensile yield strength.

## Dataset

- **1,941** experimental records
- **729** unique canonical alloy compositions
- Target: `YS_Tensile_MPa`
- **359** production features
- **352** numeric features
- **7** categorical features

The seven categorical production features are:

- `Phase`
- `Processing_Method`
- `Alloy_Class`
- `Equilibrium_Condition`
- `Single_Multiphase`
- `Test_Type`
- `Precipitate_Info`

## Feature Engineering

The production feature pipeline combines:

- Elemental composition features
- Physical material properties
- Materials-informatics descriptors
- Atomic correlation features
- Phase indicators
- Test-condition indicators
- Feature interactions

### Phase and Test Indicators

The production pipeline derives indicators for:

- BCC
- FCC
- HCP
- B2
- Laves
- Sigma
- L12
- Complex phase
- Tensile testing
- Compression testing

The final feature construction produces exactly **359 raw production features**.

## Production Model

The final production system contains:

1. Full-data XGBoost base model
2. 800–1200 MPa specialist XGBoost model
3. 1200–1580 MPa specialist XGBoost model
4. Five range-specific residual XGBoost models
5. Frozen range-specific residual correction weights

### Base XGBoost Configuration

```text
n_estimators      = 1100
max_depth         = 4
learning_rate     = 0.025
subsample         = 0.85
colsample_bytree  = 0.85
min_child_weight  = 5
random_state      = 42
n_jobs             = -1
```

### Specialist Models

Two specialist models are applied to important prediction ranges:

- **800–1200 MPa**
- **1200–1580 MPa**

The specialist prediction is blended with the base prediction using the frozen production blending configuration.

### Residual Correction

The production system applies cross-fitted residual correction using five prediction ranges:

| Prediction Range | Residual Weight |
|---|---:|
| `< 800 MPa` | 0.95 |
| `800–1200 MPa` | 0.30 |
| `1200–1580 MPa` | 0.30 |
| `1580–1800 MPa` | 0.50 |
| `>= 1800 MPa` | 0.30 |

This correction is applied after the base and specialist predictions.

## Frozen OOF Performance

The final frozen production architecture was evaluated using **5-fold out-of-fold validation** on the complete 1,941-row modelling dataset.

| Metric | Result |
|---|---:|
| **R²** | **0.950940360** |
| **MAE** | **80.454415 MPa** |
| **RMSE** | **122.959738 MPa** |

### Fold-wise R²

| Fold | R² |
|---|---:|
| Fold 1 | 0.949603 |
| Fold 2 | 0.947568 |
| Fold 3 | 0.941603 |
| Fold 4 | 0.953869 |
| Fold 5 | 0.949663 |

These results are **out-of-fold validation results**, not in-sample training scores.

## Locked Test Evaluation

The locked test set contains **428 samples**.

The locked test data provides composition-based inputs but does not contain all the material, phase, processing, and experimental inputs required by the 359-feature production model.

Therefore, the production model was **not** evaluated by artificially filling unavailable features.

A separate composition-only model was evaluated on the locked test set:

| Metric | Result |
|---|---:|
| **R²** | **0.564540388** |
| **MAE** | **257.604275 MPa** |
| **RMSE** | **352.263569 MPa** |

This evaluation is kept separate from the 359-feature production model.

## Inference Pipeline

```text
Alloy composition
       ↓
Material and processing inputs
       ↓
359-feature reconstruction
       ↓
Base XGBoost prediction
       ↓
Midrange specialist correction
       ↓
Range-specific residual correction
       ↓
Final yield-strength prediction (MPa)
```

## Production Feature Architecture

The production feature construction consists of:

```text
149 base features
       +
10 phase/test indicators
       +
200 indicator × descriptor interactions
       =
359 raw production features
```

The 10 indicators are:

```text
PHASE_BCC_IND
PHASE_FCC_IND
PHASE_HCP_IND
PHASE_B2_IND
PHASE_LAVES_IND
PHASE_SIGMA_IND
PHASE_L12_IND
PHASE_COMPLEX_IND
PHASE_TEST_TENSILE
PHASE_TEST_COMPRESSION
```

The production preprocessing pipeline transforms these raw features into the model-ready representation.

## Application Architecture

The project contains a complete prediction application:

```text
Next.js Frontend
       ↓
Next.js API Proxy
       ↓
FastAPI Backend
       ↓
Feature Generation
       ↓
359-Feature Production Pipeline
       ↓
XGBoost Ensemble
       ↓
Yield Strength Prediction
```

## Frontend

The frontend is built using:

- Next.js
- React
- TypeScript
- Tailwind CSS

The interface allows users to enter:

- Elemental composition
- Test temperature
- Grain size
- Experimental density
- Calculated density
- Precipitate size
- Matrix volume percentage
- Young's modulus
- Processing method
- Phase
- Alloy class
- Equilibrium condition
- Single/multiphase information
- Test type
- Precipitate information

The application then sends the material information to the FastAPI prediction service and displays the predicted yield strength.

## FastAPI Backend

The backend provides production inference through FastAPI.

### Main Endpoints

```text
GET  /
GET  /health
POST /predict
POST /predict-material
```

### Health Check

```text
GET /health
```

Returns information about:

- API status
- Model loading status
- Number of raw production features
- Training dataset size

Example:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "raw_features": 359,
  "training_rows": 1941
}
```

### Material Prediction

```text
POST /predict-material
```

The endpoint accepts alloy composition and optional material/processing information and returns:

```json
{
  "base_prediction_mpa": 467.3251953125,
  "residual_correction_mpa": 7.1741905212,
  "yield_strength_prediction_mpa": 474.1406860352
}
```

## Deployment

The system is deployed using:

```text
Frontend:
Next.js + Vercel

Backend:
FastAPI + AWS Elastic Beanstalk

Model:
XGBoost + scikit-learn + Joblib
```

### Live Application

Frontend:

https://heaproject-nextfrontend-y6bw.vercel.app/

Backend:

http://hea-ml-api-v2.eu-north-1.elasticbeanstalk.com/

## Production Artifact

The main production model is stored at:

```text
models/YS_FINAL_PRODUCTION/YS_FINAL_PRODUCTION_MODEL.pkl
```

The artifact contains:

- Base XGBoost model
- Midrange specialist models
- Residual correction models
- Feature configuration
- Categorical feature configuration
- Prediction range rules
- Residual correction rules
- Production metadata

## Project Structure

```text
HEA_Project/
│
├── models/
│   └── YS_FINAL_PRODUCTION/
│       └── YS_FINAL_PRODUCTION_MODEL.pkl
│
├── aws_backend/
│   ├── app.py
│   ├── predict_YS.py
│   ├── frontend_inference.py
│   ├── requirements.txt
│   └── Procfile
│
├── next_frontend/
│   ├── app/
│   ├── components/
│   ├── public/
│   ├── package.json
│   └── ...
│
├── YS_ML_Advanced_Composition_Features.csv
├── YS_FINAL_FROZEN_OOF.csv
├── YS_OFFICIAL_0904626_OOF.csv
└── README.md
```

## Technologies

### Machine Learning

- Python
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- Joblib

### Materials Informatics

- Elemental composition features
- Material-property descriptors
- Atomic correlation features
- Phase-aware features
- Composition-property interactions

### Backend

- FastAPI
- Uvicorn
- Pydantic

### Frontend

- Next.js
- React
- TypeScript
- Tailwind CSS

### Deployment

- AWS Elastic Beanstalk
- Vercel
- GitHub

## Key Results

The final production model achieved:

```text
R²   = 0.950940360
MAE  = 80.454415 MPa
RMSE = 122.959738 MPa
```

using 5-fold out-of-fold validation on 1,941 experimental records.

The system demonstrates an end-to-end workflow from:

```text
Materials Data
      ↓
Feature Engineering
      ↓
Machine Learning
      ↓
Model Validation
      ↓
Production Inference
      ↓
FastAPI Backend
      ↓
Next.js Web Application
      ↓
Cloud Deployment
```

## Limitations

- The model is trained on the available experimental dataset and may not generalize equally well to unseen alloy systems.
- The locked test set does not contain all 359 production inputs.
- Composition-only evaluation therefore represents a separate model and should not be compared directly with the full-feature OOF score.
- Predictions outside the training distribution may have higher uncertainty.
- Experimental conditions can strongly influence measured yield strength.

## Disclaimer

Model predictions are statistical estimates generated from experimental materials data. They should not be treated as experimentally validated material properties.

Predictions should be experimentally validated before being used for material selection, manufacturing decisions, structural design, or other engineering applications.