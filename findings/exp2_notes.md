# Experiment 2: EEG + SMILES Fusion

**Date:** 28 January 2026 (re-run with class weighting and threshold tuning)
**Dataset:** 151 patients with EEG recordings and SMILES embeddings

---

## Objective

Test whether combining EEG signal embeddings with drug molecular structure (SMILES) embeddings can predict ASM treatment outcomes.

---

## Architecture

### EEG Processing
- **Preprocessing:** 200 Hz, 0.1-75 Hz bandpass, 50 Hz notch filter
- **Duration:** Skip first 5 min, use next 20 min
- **Windowing:** 10s windows (max 120 windows per patient)
- **Channels:** 27 (10-20 standard montage)

### Models Tested

| Variant | EEG Encoder | SMILES Model | Fusion | Parameters |
|---------|-------------|--------------|--------|------------|
| Exp2a | SimpleCNN | ChemBERTa/SMILES-Trf | MLP | ~1.2M |
| Exp2b | SimpleCNN | ChemBERTa/SMILES-Trf | FuseMoE | ~2.8M |

### SimpleCNN Encoder
| Layer | Channels | Kernel | Pool |
|-------|----------|--------|------|
| Conv1 | 27 -> 64 | 7 | MaxPool(4) |
| Conv2 | 64 -> 128 | 5 | MaxPool(4) |
| Conv3 | 128 -> 256 | 3 | AdaptiveAvgPool(1) |

Window embeddings (256D) aggregated via 2-layer Transformer encoder.

---

## Results (5-fold CV with Threshold Tuning)

| Experiment | SMILES Model | Fusion | AUC | Bal Acc Tuned | F1 Tuned |
|------------|--------------|--------|-----|---------------|----------|
| **Exp2a** | SMILES-Trf | MLP | 0.634 +/- 0.045 | **0.699 +/- 0.047** | **0.720 +/- 0.056** |
| **Exp2a** | ChemBERTa | MLP | 0.611 +/- 0.074 | 0.672 +/- 0.045 | 0.632 +/- 0.075 |
| Exp2b | SMILES-Trf | FuseMoE | 0.576 +/- 0.095 | 0.579 +/- 0.051 | 0.537 +/- 0.272 |
| Exp2b | ChemBERTa | FuseMoE | 0.562 +/- 0.084 | 0.583 +/- 0.054 | 0.554 +/- 0.278 |

### Per-Fold AUC Values (Best Model: Exp2a SMILES-Trf MLP)

| Fold | AUC | Bal Acc Tuned |
|------|-----|---------------|
| 1 | 0.554 | 0.608 |
| 2 | 0.626 | 0.733 |
| 3 | 0.688 | 0.708 |
| 4 | 0.661 | 0.717 |
| 5 | 0.643 | 0.728 |

---

## Key Findings

1. **Best model:** Exp2a with SMILES Transformer + MLP (Bal Acc 0.699, F1 0.720)

2. **MLP significantly outperforms FuseMoE:** Balanced accuracy 0.67-0.70 vs 0.58 for FuseMoE

3. **SMILES-Trf outperforms ChemBERTa:** Consistent across both fusion methods

4. **FuseMoE unstable with EEG:** High F1 variance (0.27-0.28 std), may overfit

---

## Comparison with Original Run

Original run lacked class weighting and threshold tuning:

| Metric | Original | Re-run |
|--------|----------|--------|
| Best AUC | 0.668 (SMILES-Trf MLP) | 0.634 (SMILES-Trf MLP) |
| Class weighting | **No** | **Yes** |
| Threshold tuning | No | Yes (Youden's J) |
| Balanced Acc | Not computed | 0.58-0.70 |

Note: AUC slightly lower with class weighting as model optimises for balanced performance rather than majority class.

---

## Comparison with Exp1 (LLM + SMILES)

| Experiment | Best AUC | Best Bal Acc | Dataset |
|------------|----------|--------------|---------|
| Exp1 | 0.648 | 0.713 | 121 |
| **Exp2** | 0.634 | **0.699** | 151 |

EEG-based fusion (Exp2) achieves comparable balanced accuracy to text-based fusion (Exp1) with larger dataset.

---

## Limitations

- Only SimpleCNN encoder tested (LaBraM not available)
- FuseMoE appears unsuitable for EEG fusion at this dataset size
- High variance in some folds

---

## Technical Notes

- Class weighting: Inverse frequency (added in re-run)
- Threshold selection: Youden's J statistic (TPR - FPR)
- Training: 100 epochs, early stopping (patience 20), batch size 8
- Optimiser: AdamW, LR 1e-4, weight decay 1e-4
- Window chunking for memory efficiency (chunk size 16)
