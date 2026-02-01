# Experiment 8: Stratification Analysis

## Overview

Experiment 8 investigates whether improved cross-validation stratification can reduce the high fold-to-fold variance (I²=80%) observed in previous experiments.

## Objective

1. Analyse clinical feature distributions to identify imbalanced features
2. Compare outcome-only stratification (baseline) with multi-label stratification
3. Quantify the impact on fold balance and CV stability

## Motivation

Previous experiments showed high heterogeneity:
- **I² = 80%**: 80% of variance is due to real differences between folds
- **Wide CIs**: 95% CIs span 0.14-0.29 AUC range
- **Fold 4 outlier**: Exp7 fold 4 achieved AUC 0.933 vs mean 0.762

This suggests fold composition significantly affects results, potentially due to:
- Outcome-only stratification missing feature imbalances
- Small minority classes clustering in certain folds

## Dataset

- **Patients**: 204 (after cleaning invalid outcomes)
- **Outcome distribution**: 104 failure (0), 100 success (1)
- **Cross-validation**: 5-fold stratified

## Feature Imbalance Analysis

### Severely Imbalanced (>95% majority)

| Feature | Description | Majority % | Minority n |
|---------|-------------|------------|------------|
| `ld` | Learning disability | 98.5% | 3 |
| `birth_t` | Birth trauma | 97.5% | 5 |
| `febrile` | Febrile seizure history | 96.0% | 8 |
| `ci` | Comorbidity index | 95.5% | 9 |

**Issue**: With only 3-9 minority samples across 5 folds, some folds may have zero minority samples.

### Warning (85-95% majority)

| Feature | Description | Majority % | Minority n |
|---------|-------------|------------|------------|
| `fam_hx` | Family history | 88.5% | 23 |
| `cvd` | Cardiovascular disease | 88.4% | 23 |
| `drug` | Drug abuse history | 83.7% | 32 |
| `alcohol` | Alcohol use | 81.7% | 36 |

### Acceptable (<85% majority)

| Feature | Description | Majority % | Minority n |
|---------|-------------|------------|------------|
| `focal` | Focal seizure type | 79.2% | 42 |
| `sex` | Patient sex | 62.7% | 76 |

## Data Quality Issues

### Fixed During Cleaning

1. **`psy` column**: Mixed types ('0', '1', '0.0', '1.0', '?')
   - Converted to standardised 0/1 integers
   - 1 invalid value ('?') converted to NaN

2. **`lesion` column**: Mixed types (1, 2, 3, '1.0', '2.0', '3.0', 'NOT AVAILABLE')
   - Converted to standardised integers
   - 1 invalid value ('NOT AVAILABLE') converted to NaN

3. **`outcome` column**: String values ('1', '2')
   - Converted to numeric before filtering
   - 2 invalid outcomes dropped

## Stratification Methods Compared

### Baseline: Outcome-Only

```python
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(X, y):
    ...
```

Only balances outcome (success/failure) across folds.

### Multi-Label Stratification

```python
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Stratify on: outcome, focal, sex
for train_idx, val_idx in mskf.split(X, y_multilabel):
    ...
```

Balances multiple features simultaneously using iterative stratification.

## Results

### Fold Balance Comparison

| Feature | Metric | Outcome-Only | Multi-Label | Improvement |
|---------|--------|--------------|-------------|-------------|
| focal | fold_std | 10.73% | 1.34% | **8x better** |
| focal | min-max range | 7.3%-32.5% | 19.5%-22.5% | Much tighter |
| sex | fold_std | 5.82% | 1.06% | **5x better** |
| sex | min-max range | 29.3%-43.9% | 36.6%-39.0% | Much tighter |
| outcome | fold_std | 0.55% | 0.55% | Same |

### Interpretation

Multi-label stratification dramatically reduces fold-to-fold variance:
- **Focal**: Folds now have consistent 19-22% non-focal patients (vs 7-33%)
- **Sex**: Folds now have consistent 37-39% female patients (vs 29-44%)
- **Outcome**: Remains balanced (both methods achieve this)

## Recommendations

### 1. Use Multi-Label Stratification

For future experiments, stratify on `outcome + focal + sex`:
```python
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
stratify_cols = ['outcome', 'focal', 'sex']
```

### 2. Consider Dropping Severely Imbalanced Features

Features with <10 minority samples provide minimal signal:
- `ld` (3 samples)
- `birth_t` (5 samples)
- `febrile` (8 samples)
- `ci` (9 samples)

### 3. Re-run Best Model with Improved Stratification

Compare Exp7a results with multi-label stratification to quantify:
- Change in mean AUC
- Reduction in fold std
- Reduction in I² heterogeneity

## Files

| File | Purpose |
|------|---------|
| `exp8_stratification/config.py` | Configuration |
| `exp8_stratification/data_cleaning.py` | Data cleaning utilities |
| `exp8_stratification/feature_analysis.py` | Distribution analysis |
| `exp8_stratification/stratified_cv.py` | Multi-label stratification |
| `exp8_stratification/data_pipeline.py` | Data loading (reuses exp7) |
| `exp8_stratification/training.py` | Training loop |
| `exp8_stratification/run_experiments.py` | Entry point |

## Running the Experiment

```bash
source .venv-others/bin/activate

# Feature analysis only
python -m exp8_stratification.feature_analysis

# Full experiment (baseline vs multi-label)
python -m exp8_stratification.run_experiments
```

## Dependencies

```bash
uv pip install iterative-stratification
```
