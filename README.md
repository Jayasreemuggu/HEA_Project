# AI-Assisted HEA/MPEA Materials Discovery using Machine Learning

An end-to-end machine learning project for predicting tensile yield strength of High-Entropy Alloys (HEAs) and Multi-Principal Element Alloys (MPEAs).

The project combines structured materials data, feature engineering, ensemble machine learning, uncertainty quantification, explainable AI, and an interactive prediction dashboard.

## Project Overview

The primary objective is to develop a machine learning model that estimates alloy tensile yield strength from composition, material descriptors, phase information, processing conditions, and testing conditions.

### Dataset

- **1,941** experimental records used for yield-strength modelling
- **729** unique canonical alloy compositions
- Target variable: `YS_Tensile_MPa`
- **359 engineered features**
- Composition, phase, processing, microstructural, and material-descriptor features
- Reproducible preprocessing and feature-engineering pipeline

## Machine Learning Pipeline

    Experimental Alloy Data
            ↓
    Data Cleaning & Preprocessing
            ↓
    Feature Engineering
            ↓
    Phase Features + Feature Interactions
            ↓
    359-Feature Representation
            ↓
    Model Training
            ↓
    ExtraTrees + HistGradientBoosting + XGBoost
            ↓
    Weighted Ensemble
            ↓
    Yield Strength Prediction
            ↓
    Uncertainty Quantification
            ↓
    SHAP Explainability
            ↓
    Interactive Streamlit Dashboard

## Final Model

The final prediction system combines three tree-based regression models:

| Model | Role |
|---|---|
| ExtraTreesRegressor | Nonlinear ensemble learner |
| HistGradientBoostingRegressor | Gradient-boosting learner |
| XGBoost | Additional gradient-boosting model |

### Ensemble Weights

- ExtraTrees: **40%**
- HistGradientBoosting: **30%**
- XGBoost: **30%**

The ensemble combines predictions from the three models to produce the final yield-strength estimate.

## Model Performance

The primary project dashboard reports performance using **5-fold random cross-validation**.

| Metric | Result |
|---|---:|
| R² | **0.842** |
| MAE | **133.12 MPa** |
| RMSE | **220.72 MPa** |

These values represent the primary random cross-validation evaluation used in the project dashboard.

A stricter composition-grouped validation was also performed to reduce composition leakage between training and validation sets. This produced lower performance, demonstrating the effect of evaluating the model on unseen alloy compositions.

## Feature Engineering

The final machine-learning representation contains **359 engineered features**.

Features include:

- Alloy composition descriptors
- Atomic radius and size statistics
- Electronegativity descriptors
- Atomic mass statistics
- Valence-electron-related descriptors
- Material-property descriptors
- Phase-related features
- Processing-method information
- Testing temperature
- Grain size
- Composition-derived descriptors
- Composition-phase interaction features
- Phase-material descriptor interactions

Categorical and numerical variables are processed through a reproducible preprocessing pipeline with missing-value handling and categorical encoding.

## Explainable AI

**SHAP (SHapley Additive exPlanations)** is used to investigate how input features contribute to model predictions.

The explainability workflow includes:

- Global feature importance
- Aggregated feature importance
- SHAP-based feature contribution analysis
- Feature dependence analysis

Important model features include testing temperature, material descriptors, phase-interaction features, atomic-size-related descriptors, processing information, and composition-derived features.

SHAP describes **model behaviour** and should not be interpreted as proof of causal physical relationships.

## Uncertainty Quantification

Conformal prediction was implemented using grouped out-of-fold predictions to estimate prediction intervals.

| Prediction Interval | Empirical Coverage |
|---|---:|
| 80% | **80.11%** |
| 90% | **90.11%** |
| 95% | **95.11%** |

The 95% conformal interval uses a residual-based calibration value of approximately **563.71 MPa**.

Coverage is not uniform across the complete yield-strength range, particularly in the high-strength tail. Therefore, the intervals should be interpreted as model-based statistical uncertainty estimates rather than guarantees.

## Interactive Dashboard

The project includes a Streamlit dashboard containing:

- Project Overview
- Model Performance
- Prediction & Uncertainty
- Production Prediction
- Scenario Analysis
- Uncertainty Analysis
- Explainable AI / SHAP

Run the application with:

    streamlit run app.py

## Repository Structure

    ├── app.py
    ├── feature_engineering.py
    ├── ys_predictor.py
    ├── frontend/
    │   ├── index.html
    │   └── script.js
    │
    ├── models/
    │   └── YS_Final_359/
    │       ├── YS_Final_359_ET.pkl
    │       ├── YS_Final_359_HGB.pkl
    │       ├── YS_Final_359_XGB.pkl
    │       ├── YS_Final_359_Preprocessor.pkl
    │       └── YS_Final_359_Metadata.pkl
    │
    ├── YS_ML_Advanced_Composition_Features.csv
    ├── YS_CONFORMAL_INTERVAL_RESULTS.csv
    ├── YS_GROUPED_OOF_PREDICTIONS_WITH_UNCERTAINTY.csv
    ├── YS_UNCERTAINTY_BY_TARGET_RANGE.csv
    ├── YS_SHAP_CORRECTED_ORIGINAL.csv
    ├── YS_SHAP_CORRECTED_TOP25.png
    ├── README.md
    └── requirements.txt

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- ExtraTreesRegressor
- HistGradientBoostingRegressor
- XGBoost
- SHAP
- Conformal Prediction
- Streamlit
- Git
- Git LFS

## Key Machine Learning Concepts Demonstrated

- Data preprocessing
- Feature engineering
- Nonlinear regression
- Ensemble learning
- Cross-validation
- Model comparison
- Feature interaction engineering
- Uncertainty quantification
- Explainable AI
- Model deployment
- Interactive ML dashboard

## Reproducibility

The repository contains the final preprocessing pipeline, trained model artifacts, feature-engineering code, evaluation outputs, and dashboard required to reproduce and use the final prediction workflow.

Large model files are managed using **Git LFS**.

## Disclaimer

Model predictions are statistical estimates generated from the available experimental data.

They should not be treated as experimentally validated material properties. Predictions outside the training distribution may have higher uncertainty and require experimental validation.
