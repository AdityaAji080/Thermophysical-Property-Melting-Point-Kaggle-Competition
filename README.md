# Melting-Point Prediction from Molecular Structure

Predict the melting point (K) of organic molecules from their SMILES strings. This was built for APMonitor's [Melting Point Prediction Challenge](https://www.kaggle.com/competitions/melting-point) on Kaggle, which is scored by mean absolute error.

Molecules are featurised with RDKit, the competition's 2,662 training molecules are augmented with about 16,700 molecules from the open Bradley melting-point dataset, and gradient-boosted trees and MLPs are compared on a leakage-free hold-out set.

## Results

20% hold-out, stratified by data source (3,881 molecules, 532 of them from the Kaggle training set):

<!-- RESULTS_TABLE -->
| Model | MAE (K) | RMSE (K) | R² | MAE on Kaggle rows (K) |
|---|---|---|---|---|
| CatBoost (depth 8, early stopping) | **27.2** | 39.0 | 0.812 | 25.1 |
| MLP 512-256-128 (tanh) | 33.8 | 47.5 | 0.721 | 33.6 |
| MLP + MACCS keys | 33.2 | 46.0 | 0.739 | 32.6 |
| Predict the training mean | 71.1 | | | |

The *Kaggle rows* column is the closest proxy for the competition's test set. For scale, 90% of the training melting points lie between 215 K and 524 K (standard deviation 91 K).

## Features

| Block | Size | Notes |
|---|---|---|
| Morgan fingerprint bits | 1024 | radius 2, feature invariants |
| Morgan fingerprint counts | 1024 | `log1p`, standardised |
| RDKit descriptors | 32 | mass, TPSA, molar refractivity, H-bond donors/acceptors, ring counts, rotatable bonds, element fractions, salt flag; standardised |
| MACCS keys | 167 | `mlp_maccs` only |

Scaling statistics are fitted on the training split only.

## Data and leakage

`prepare_data.py` merges the Kaggle training set with the [Jean-Claude Bradley Open Melting Point Dataset](https://doi.org/10.6084/m9.figshare.1031637) (CC0). It drops rows flagged *do not use* and converts °C to K. **Molecules are matched by RDKit canonical SMILES**, and Bradley molecules that also appear in the Kaggle train or test sets are removed.

Canonicalising matters. The same molecule can be written as many different SMILES strings, so matching raw strings misses most duplicates. With raw-string matching, **492 of the 666 test molecules (74%)** would have leaked into training through the Bradley data, and the hold-out score would have been optimistic. Canonical matching removes 1,409 such Bradley rows.

## Usage

```bash
pip install -r requirements.txt
# Download train.csv and test.csv from the competition page into data/
python prepare_data.py                 # -> data/train_merged.csv
python train.py --model catboost       # --device gpu for a CUDA GPU
python train.py --model mlp
python train.py --model mlp_maccs
```

Each run writes `outputs/metrics_<model>.json` and a Kaggle submission `outputs/submission_<model>.csv`.

## Repository layout

```
features.py       # RDKit featurisation + train-only block scaling
prepare_data.py   # merge Kaggle + Bradley data with canonical-SMILES de-duplication
train.py          # CatBoost / MLP training, hold-out metrics, submission
data/             # Bradley dataset (Kaggle files go here after download)
```

## Ideas for improvement

- Cross-validation and an ensemble of CatBoost and the MLP.
- Graph neural networks (e.g. message passing on the molecular graph) instead of fixed fingerprints.
- Train on all labelled data, not just the training split, before the final submission.
