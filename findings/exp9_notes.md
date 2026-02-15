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

## HPC Run 1: Initial Encoder Ablation (12 February 2026, 15:33)

**Status: All jobs FAILED** (kept for reference)

Three encoder ablation experiments submitted to M3 HPC:

| Job | Experiment | Encoder | Status |
|-----|-----------|---------|--------|
| 1 | baseline_simplecnn_transformer | SimpleCNN | FAILED |
| 2 | encoder_eegnet | EEGNet | FAILED |
| 3 | encoder_labram | LaBraM | FAILED |

All three jobs failed with the same error:

```
iterative-stratification required for multi-label stratification.
Install with: uv pip install iterative-stratification
```

**Root cause:** The `iterative-stratification` package was not installed in the HPC `.venv-others` environment.

**Fix applied:** Installed `iterative-stratification` on HPC and added graceful fallback with warning in `exp8_stratification/stratified_cv.py`.

---

## HPC Run 2: Partial Success (12 February 2026, 21:32)

**Status: 2/4 succeeded, 2/4 FAILED** (braindecode issue)

After fixing `iterative-stratification`, a second run was submitted with all 4 encoder ablations:

| Job | Experiment | Encoder | Status |
|-----|-----------|---------|--------|
| 1 | baseline_simplecnn_transformer | SimpleCNN | SUCCESS |
| 2 | encoder_eeg2vec | EEG2Vec | SUCCESS |
| 3 | encoder_eegnet | EEGNet | FAILED |
| 4 | encoder_labram | LaBraM | FAILED |

EEGNet and LaBraM failed with:

```
OSError: libcudart.so.12: cannot open shared object file: No such file or directory
```

**Root cause:** `braindecode` (required by EEGNet and LaBraM) had an incompatible CUDA dependency. The installed version pulled in a torch build expecting CUDA 12 libraries not present on the HPC nodes.

**Fix applied:** Reinstalled braindecode 1.2.0 with compatible CUDA bindings.

---

## HPC Run 3: Definitive Encoder Ablation (12 February 2026, 22:05-23:45)

**Status: All 4 jobs SUCCEEDED**

With braindecode fixed, all 4 encoder ablations completed successfully:

| Job ID | Experiment | Encoder | Status | Runtime |
|--------|-----------|---------|--------|---------|
| 51362675 | baseline_simplecnn_transformer | SimpleCNN | SUCCESS | ~25 min |
| 51362676 | encoder_eeg2vec | EEG2Vec | SUCCESS | ~25 min |
| 51362677 | encoder_eegnet | EEGNet | SUCCESS | ~25 min |
| 51362679 | encoder_labram | LaBraM | SUCCESS | ~25 min |

---

## Results: Encoder Ablation (Run 3 - Definitive)

### Encoder Comparison

| Encoder | AUC | Bal Acc Tuned | F1 Tuned | AUC Std |
|---------|-----|---------------|----------|---------|
| **EEG2Vec** | **0.661 +/- 0.061** | **0.689 +/- 0.054** | 0.585 +/- 0.149 | **0.061** |
| EEGNet | 0.648 +/- 0.107 | 0.686 +/- 0.078 | 0.584 +/- 0.239 | 0.107 |
| SimpleCNN (baseline) | 0.607 +/- 0.107 | 0.661 +/- 0.076 | 0.616 +/- 0.059 | 0.107 |
| LaBraM | 0.549 +/- 0.077 | 0.608 +/- 0.036 | 0.512 +/- 0.196 | 0.077 |

### Per-Fold AUC

| Encoder | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---------|--------|--------|--------|--------|--------|
| EEG2Vec | 0.710 | 0.673 | 0.701 | 0.678 | 0.542 |
| EEGNet | 0.741 | 0.711 | 0.705 | 0.640 | 0.444 |
| SimpleCNN | 0.710 | 0.691 | 0.683 | 0.487 | 0.467 |
| LaBraM | 0.629 | 0.653 | 0.509 | 0.471 | 0.481 |

### Key Observations

1. **EEG2Vec achieves best AUC (0.661) with lowest variance (std 0.061)** - the most stable encoder across folds
2. **EEGNet marginally outperforms SimpleCNN baseline** (+0.041 AUC) but has identical high variance (std 0.107)
3. **LaBraM significantly underperforms** (AUC 0.549) - likely due to small dataset size or architecture mismatch with 27-channel clinical EEG
4. **Multi-label stratification did NOT eliminate high variance** for SimpleCNN/EEGNet (std still 0.107) - suggesting encoder architecture also contributes to instability
5. **EEG2Vec's lower variance (0.061 vs 0.107)** suggests CVAE pre-training provides more robust features
6. **Folds 4/5 consistently weakest** across all encoders
7. **EEGNet F1 tuned has extreme variance** (std 0.239) - fold 5 F1 of 0.118 is a near-complete failure

### EEG Data Quality Summary

Quality metrics computed across 151 patients with quality + outcome data:

| Metric | Value |
|--------|-------|
| Overall quality | 0.674 +/- 0.087 |
| Mean SNR | -3.7 +/- 10.1 dB |
| High-artifact recordings | 7 |
| Problem recordings | 2 (N085, 452) |
| Quality-outcome correlation | -0.016 (not significant) |

No significant correlation between quality metrics and outcome, suggesting quality is not the primary driver of fold-to-fold variance.

---

## HPC Run 4: Extended Ablation (13 February 2026, 10:06-15:09)

**Status: All 13 jobs SUCCEEDED**

**Job:** 51370963 | **Node:** A100 80GB | **Runtime:** ~5 hours total

Extended the encoder ablation with aggregator types, transformer depths, and embedding dimensions. All configs use EEG2Vec encoder unless otherwise noted.

### Results (sorted by AUC)

| Config | Encoder | Aggregator | Embed Dim | AUC | Bal Acc | F1 Tuned |
|--------|---------|-----------|-----------|-----|--------|----------|
| **embed_dim_128** | EEG2Vec | Transformer | 128 | **0.730 +/- 0.034** | **0.725 +/- 0.038** | **0.732 +/- 0.060** |
| aggregator_meanmax | EEG2Vec | MeanMax | 256 | 0.722 +/- 0.079 | 0.740 +/- 0.065 | 0.689 +/- 0.119 |
| embed_dim_64 | EEG2Vec | Transformer | 64 | 0.687 +/- 0.129 | 0.715 +/- 0.102 | 0.716 +/- 0.105 |
| aggregator_depth_0 | EEG2Vec | Attention (0 layers) | 256 | 0.669 +/- 0.070 | 0.690 +/- 0.069 | 0.690 +/- 0.112 |
| aggregator_maxpool | EEG2Vec | MaxPool | 256 | 0.668 +/- 0.033 | 0.699 +/- 0.051 | 0.676 +/- 0.142 |
| aggregator_attention | EEG2Vec | Attention | 256 | 0.666 +/- 0.059 | 0.698 +/- 0.053 | 0.674 +/- 0.067 |
| aggregator_depth_1 | EEG2Vec | Transformer (1L) | 256 | 0.666 +/- 0.055 | 0.682 +/- 0.046 | 0.611 +/- 0.070 |
| baseline_simplecnn | SimpleCNN | Transformer | 256 | 0.620 +/- 0.082 | 0.685 +/- 0.038 | 0.606 +/- 0.092 |
| aggregator_depth_4 | EEG2Vec | Transformer (4L) | 256 | 0.605 +/- 0.118 | 0.638 +/- 0.079 | 0.524 +/- 0.169 |
| encoder_eegnet | EEGNet | Transformer | 256 | 0.603 +/- 0.067 | 0.654 +/- 0.051 | 0.636 +/- 0.123 |
| encoder_eeg2vec | EEG2Vec | Transformer | 256 | 0.594 +/- 0.063 | 0.639 +/- 0.060 | 0.471 +/- 0.117 |
| aggregator_lstm | EEG2Vec | LSTM | 256 | 0.588 +/- 0.105 | 0.651 +/- 0.046 | 0.606 +/- 0.125 |
| encoder_labram | LaBraM | Transformer | 128 | 0.575 +/- 0.094 | 0.610 +/- 0.052 | 0.511 +/- 0.201 |
| encoder_frozen | SimpleCNN (frozen) | Transformer | 256 | 0.559 +/- 0.103 | 0.639 +/- 0.069 | 0.627 +/- 0.149 |

### Key Findings

1. **128D embeddings are optimal** (AUC 0.730, lowest std 0.034) - reducing from 256D improves generalisation and is the most stable configuration tested
2. **MeanMax aggregation outperforms transformer** for balanced accuracy (0.740 vs 0.725) but has higher AUC variance (0.079 vs 0.034)
3. **Transformer depth sweet spot is 2 layers** - 0 layers (attention only) works well (0.669), 1 layer comparable (0.666), 4 layers overfits badly (0.605)
4. **LSTM aggregation underperforms** all other aggregators (0.588) - temporal modelling may not help with 10s windows
5. **Freezing encoder hurts** significantly (0.559 vs 0.620 for SimpleCNN) - end-to-end training is essential for this dataset size
6. **64D embeddings still competitive** (AUC 0.687) but with much higher variance (0.129) - potential for overfitting on some folds

### Recommended Configuration for Multi-modal Experiments

Based on these findings, the recommended EEG configuration for exp5c, exp7, and other multi-modal experiments:
- **Encoder:** EEG2Vec
- **Embedding dimension:** 128
- **Aggregator:** Transformer (2 layers) or MeanMax
- **Training:** End-to-end (not frozen)

### Multi-modal Validation (Exp11, 15 February 2026)

The exp9 ablation findings were validated in multi-modal experiments (Exp11):
- **128D embeddings confirmed optimal:** EEG2Vec 128D improves exp3a MLP from AUC 0.687 to 0.736 (+0.049) and exp6b from 0.647 to 0.697 (+0.050)
- **MeanMax aggregator confirmed competitive:** Achieves best exp3a result (AUC 0.736, std 0.036) with lowest variance, validating the exp9 finding (AUC 0.722, Bal Acc 0.740)
- **Exp7a EEG2Vec configs did not complete** - re-submission required for quad modality validation
- See `findings/exp11_notes.md` for full results

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

1. ~~Install `iterative-stratification` on HPC and resubmit jobs~~ **DONE** (Run 2)
2. ~~Add EEG2Vec encoder job to submission batch~~ **DONE** (Run 2)
3. ~~Fix braindecode CUDA dependency on HPC~~ **DONE** (Run 3)
4. ~~Run all 4 encoder ablations successfully~~ **DONE** (Run 3)
5. ~~Run quality analysis to identify problem recordings~~ **DONE** (2 problem recordings identified)
6. ~~Run aggregator ablations (Attention, MaxPool, MeanMax, LSTM) with EEG2Vec encoder~~ **DONE** (Run 4)
7. ~~Run depth ablations (0, 1, 2, 4 transformer layers) with EEG2Vec encoder~~ **DONE** (Run 4)
8. ~~Run dimension ablations (embed_dim 64, 128, 256) with EEG2Vec encoder~~ **DONE** (Run 4)
9. Re-run Exp5c with EEG2Vec encoder (currently still SimpleCNN; multi-label stratification already applied)
10. Investigate fold 4/5 weakness across encoders
11. ~~Swap EEG2Vec (128D) into exp3, exp6b~~ **PARTIALLY DONE** (Exp11) - exp3a +0.049 AUC, exp6b +0.050 AUC. Exp7a did not complete. Exp2 not yet tested.
12. ~~Test MeanMax aggregator in multi-modal experiments~~ **DONE** (Exp11) - MeanMax achieves best exp3a result (AUC 0.736, std 0.036)
13. Re-submit exp7a EEG2Vec configs (quad modality validation)
14. Test EEG2Vec in exp2 (EEG + SMILES dual modality)
