# Project Status

## Completed

- Multi-source HEA/MPEA experimental dataset
- Data cleaning and deduplication
- Composition-grouped validation
- Random Forest baseline
- Extra Trees model
- XGBoost models
- GCN/GAT deep-learning benchmark
- Four-property prediction
- SHAP explainability
- Bootstrap uncertainty estimation
- Uncertainty-guided active learning
- 5,000 hypothetical candidate generation
- Multi-property screening
- Pareto optimization
- 35-dimensional novelty analysis
- Uncertainty-aware candidate ranking
- Final project figures
- Reproducible validation script

## Final Models

- Yield Strength: XGBoost
- UTS: XGBoost
- Elongation: XGBoost
- Hardness: Extra Trees
- Deep-learning benchmark: GCN

## Important Interpretation

The hypothetical candidate properties are machine-learning predictions within the explored composition space. They are not experimentally validated material properties.

Bootstrap uncertainty is an empirical model-disagreement measure and is not treated as a calibrated confidence interval.

SHAP values indicate model influence and should not automatically be interpreted as causal effects.

Pareto optimality indicates non-dominated trade-offs among the selected objectives.

35-dimensional novelty measures distance in the available experimental composition space and does not by itself establish experimental novelty.
