# AI-Assisted HEA/MPEA Materials Discovery using Machine Learning

An AI-assisted materials-informatics workflow for predicting mechanical properties and screening High-Entropy Alloys (HEAs) and Multi-Principal Element Alloys (MPEAs).

## Results

- 2,748 experimental HEA/MPEA records
- XGBoost + ExtraTrees Ensemble: Yield Strength R² = 0.7138
- XGBoost: UTS R² = 0.7260
- XGBoost: Elongation R² = 0.3445
- Extra Trees: Hardness R² = 0.6930
- 5,000 hypothetical alloy candidates screened
- 208 Pareto candidates identified
- 277 high-performance/high-novelty/low-uncertainty candidates

## Methods

Machine Learning, XGBoost, Extra Trees, Graph Neural Networks, SHAP, bootstrap uncertainty quantification, active learning, novelty detection, and multi-objective Pareto screening.

## Disclaimer

Candidate properties are machine-learning predictions within the explored composition space and require experimental validation.


## Model Benchmark

The models were evaluated using composition-grouped validation to prevent the same alloy composition from appearing across train and test sets.

| Property | Final Model | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| Yield Strength | XGBoost + ExtraTrees Ensemble | 198.89 MPa | 285.60 MPa | **0.7138** |
| Ultimate Tensile Strength | XGBoost | 289.48 MPa | 423.54 MPa | **0.7260** |
| Elongation | XGBoost | 12.73% | 17.49% | **0.3445** |
| Hardness | Extra Trees | 78.45 HV | 108.20 HV | **0.6930** |

### Yield Strength Model Benchmark

Multiple machine-learning approaches were benchmarked for yield-strength prediction, including tree-based models and a graph-based GCN representation.

| Model | MAE (MPa) | RMSE (MPa) | R² |
|---|---:|---:|---:|
| Random Forest | 354.81 | 474.60 | 0.2095 |
| Extra Trees | 207.02 | 296.96 | 0.6905 |
| **XGBoost + ExtraTrees Ensemble** | **198.89** | **285.60** | **0.7138** |
| GNN - Original GCN | 257.79 | 347.32 | 0.5767 |

The XGBoost + ExtraTrees ensemble achieved the strongest yield-strength performance in this benchmark, while the original GCN served as the retained graph-learning benchmark.

