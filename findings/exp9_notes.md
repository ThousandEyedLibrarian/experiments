# Experiment 9: EEG Variance Investigation

**Date:** 02 February 2026
**Objective:** Investigate and reduce the high fold-to-fold variance observed in EEG-based experiments

---

## Problem Statement

EEG experiments showed exceptionally high variance compared to text-based experiments:

| Experiment | Modality | AUC std | Range | Issue |
|------------|----------|---------|-------|-------|
| Exp5c | Clinical+EEG | 0.113 | 0.545-0.866 | Fold 4 outlier |
| Exp2b | EEG+SMILES FuseMoE | - | F1 std 0.27-0.28 | Catastrophic failures |
| Exp1b | LLM+SMILES | 0.07-0.10 | - | Much more stable |

The exceptional fold 4 in Exp5c (0.866 AUC vs mean 0.644) proves the model CAN learn EEG patterns - the inconsistency is the problem.

---

## Root Cause Analysis

### 1. Fold Composition Issues (PRIMARY CAUSE)

Analysis revealed severe imbalance in clinical features across folds with outcome-only stratification:

| Feature | Fold Range | Correlation with AUC |
|---------|------------|---------------------|
| `focal` | 67.5% - 92.7% | r = 0.74 |
| `sex` | 56.1% - 70.7% | r = 0.72 |
| EEG padding ratio | - | r = -0.78 |

**Best vs Worst Fold Comparison (Fold 4 vs Fold 5):**
- focal: 92.7% vs 71.8% (+20.9%)
- psy: 28.2% vs 10.0% (+18.2%)
- fam_hx: 5.0% vs 15.0% (-10.0%)

### 2. Multi-label Stratification Solution

Using outcome + focal + sex for stratification:

| Feature | Outcome-only (range) | Multi-label (range) | Improvement |
|---------|---------------------|---------------------|-------------|
| `focal` | 25.2% | 3.0% | **8x better** |
| `sex` | 14.6% | 2.5% | **6x better** |

### 3. EEG Data Quality Heterogeneity

Quality metrics computed:
- Signal-to-noise ratio (SNR)
- Artifact proportion (amplitude > 500uV)
- Flatline detection
- Inter-channel correlation

These allow filtering of low-quality recordings that may cause training instability.

---

## Implemented Changes

### Training Pipeline Updates

**Multi-label stratification added to:**
- `exp5_clinical_fusion/training.py`
- `exp2_fusion/training.py`

Usage:
```python
run_cross_validation_eeg(
    eeg_model="simplecnn",
    use_multilabel_stratification=True,  # New parameter
)
```

### EEG Pipeline Enhancements

**Normalisation options** (`exp2_fusion/eeg_pipeline.py`):
- `zscore` - Global z-score per channel
- `window_zscore` - Per-window normalisation
- `robust` - IQR-based (artifact-resistant)

```python
preprocessor = EEGPreprocessor(
    normalisation="window_zscore",
    clip_std=5.0,  # Optional clipping
)
```

**Quality metrics:**
```python
preprocessor = EEGPreprocessor(compute_quality=True)
result = preprocessor.process(edf_path, return_quality=True)
windows, mask, n_channels, quality_metrics = result
```

### Alternative Window Aggregators

**New aggregators** (`exp2_fusion/models/aggregators.py`):

| Aggregator | Description | Use Case |
|------------|-------------|----------|
| `AttentionPooling` | Learnable window weights | Robust to noisy windows |
| `MaskedMaxPooling` | Max over valid windows | Conservative feature selection |
| `MeanMaxPooling` | Concat mean + max | Richer representation |
| `LSTMAggregator` | Bidirectional LSTM | Temporal dependencies |
| `MultiScaleAggregator` | Multi-resolution pooling | Hierarchical features |

### Alternative EEG Encoders

**EEGNet now available** (`exp2_fusion/models/eeg_encoders.py`):

| Encoder | Parameters | Notes |
|---------|------------|-------|
| SimpleCNN | 857K | Current baseline |
| EEGNet | 256K | 3x fewer params, EEG-specific |
| LaBraM | 1.0M | Transformer-based, high memory |

---

## Ablation Study Design

12 experiments defined in `exp9_eeg_investigation/run_experiments.py`:

1. **Baseline:** SimpleCNN + Transformer aggregator
2. **Encoder ablations:** EEGNet, frozen encoder
3. **Aggregator ablations:** Attention, MaxPool, MeanMax, LSTM
4. **Depth ablations:** 0, 1, 2, 4 transformer layers
5. **Dimension ablations:** embed_dim 64, 128, 256

Run with:
```bash
python -m exp9_eeg_investigation.run_experiments --quick  # Just baseline
python -m exp9_eeg_investigation.run_experiments          # All experiments
```

---

## HPC Run 1: Initial Encoder Ablation (12 February 2026)

### Jobs Submitted

Three encoder ablation experiments submitted to M3 HPC:

| Job | Experiment | Encoder | Status |
|-----|-----------|---------|--------|
| 1 | baseline_simplecnn_transformer | SimpleCNN | FAILED |
| 2 | encoder_eegnet | EEGNet | FAILED |
| 3 | encoder_labram | LaBraM | FAILED |

### Results

All three jobs failed with the same error:

```
iterative-stratification required for multi-label stratification.
Install with: uv pip install iterative-stratification
```

**Root cause:** The `iterative-stratification` package was not installed in the HPC `.venv-others` environment. The `exp9_eeg_investigation/run_experiments.py` script defaults to multi-label stratification (via `exp8_stratification/stratified_cv.py`), which requires `MultilabelStratifiedKFold` from the `iterstrat` package.

**Contributing factors:**
1. `check_environment.py` does not validate `iterative-stratification` as a dependency
2. `exp9` has a `--no-multilabel` flag but defaults to requiring the package
3. The error occurs at runtime when `get_multilabel_splits()` is called, not at import time

### Fix Required

1. Install `iterative-stratification` in HPC environment: `uv pip install iterative-stratification`
2. Add `iterative-stratification` to `check_environment.py` validation
3. Consider adding a graceful fallback with warning instead of hard failure

---

## Expected Results

With multi-label stratification:
- AUC std should reduce from 0.113 to ~0.03-0.04
- Fold min-max range should reduce from 0.321 to <0.15
- More reliable model comparison between architectures

---

## Encoder Inventory (Updated)

| Encoder | Parameters | Notes |
|---------|------------|-------|
| SimpleCNN | 857K | Current baseline |
| EEGNet | 256K | 3x fewer params, EEG-specific |
| LaBraM | 1.0M | Transformer-based, high memory |
| EEG2Vec | 510K | CVAE with EEGNet backbone (new) |

EEG2Vec encoder added 12 February 2026, based on arxiv 2207.08002.

---

## Files Created

| File | Purpose |
|------|---------|
| `exp9_eeg_investigation/__init__.py` | Package init |
| `exp9_eeg_investigation/config.py` | Configuration |
| `exp9_eeg_investigation/fold_analysis.py` | Fold composition analysis |
| `exp9_eeg_investigation/quality_analysis.py` | EEG quality metrics |
| `exp9_eeg_investigation/run_experiments.py` | Ablation framework |
| `exp2_fusion/models/aggregators.py` | Alternative aggregators |
| `exp2_fusion/models/eeg_encoders.py` | EEG2Vec encoder added |

---

## Next Steps

1. Install `iterative-stratification` on HPC and resubmit jobs
2. Add EEG2Vec encoder job to submission batch
3. Run fold analysis to validate stratification improvements
4. Run quality analysis to identify problem recordings
5. Re-run Exp5c with multi-label stratification
6. Execute full ablation study to identify best architecture
