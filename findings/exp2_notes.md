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
| **Exp2a** | SMILES-Trf | MLP | **0.634 +/- 0.045** | **0.699 +/- 0.047** | **0.720 +/- 0.056** |
| **Exp2a** | ChemBERTa | MLP | 0.611 +/- 0.074 | 0.672 +/- 0.045 | 0.632 +/- 0.075 |
| Exp2b | SMILES-Trf | FuseMoE | 0.611 +/- 0.056 | 0.621 +/- 0.049 | 0.556 +/- 0.175 |
| Exp2b | ChemBERTa | FuseMoE | 0.572 +/- 0.024 | 0.599 +/- 0.012 | 0.569 +/- 0.133 |

*Note: Exp2b rows updated 13 February 2026 with revised FuseMoE (Laplace gating, MI loss, 3-layer residual experts, temperature annealing). Previous Exp2b results: SMILES-Trf AUC 0.576, ChemBERTa AUC 0.562.*

### Exp2b Results: Exp12 Tuned HP (17 February 2026)

Applied Exp12 best hyperparameters (lr=5e-5, 4 experts, no temp decay) to exp2b.

| SMILES Model | Fusion | AUC | Bal Acc Tuned | F1 Tuned |
|--------------|--------|-----|---------------|----------|
| ChemBERTa | FuseMoE (Exp12 HP) | 0.585 +/- 0.077 | 0.603 +/- 0.056 | 0.524 +/- 0.178 |
| SMILES-Trf | FuseMoE (Exp12 HP) | 0.569 +/- 0.087 | 0.602 +/- 0.055 | 0.640 +/- 0.067 |

**Comparison with default revised FuseMoE HP:**

| SMILES Model | Default HP AUC | Exp12 HP AUC | Delta | Std Change |
|--------------|---------------|-------------|-------|------------|
| ChemBERTa | 0.572 +/- 0.024 | 0.585 +/- 0.077 | +0.013 | +0.053 |
| SMILES-Trf | 0.611 +/- 0.056 | 0.569 +/- 0.087 | -0.042 | +0.031 |

**Observations:**
- ChemBERTa gains +0.013 AUC but variance increases substantially (0.024 -> 0.077)
- SMILES-Trf regresses -0.042 AUC - temperature annealing was beneficial for this pairing
- Unlike exp1b/exp7b, exp2b shows increased variance with Exp12 HP for both configs
- EEG + SMILES may need different FuseMoE HP than text-containing experiments
- Best exp2b FuseMoE remains SMILES-Trf at 0.611 with default revised HP

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

2. **MLP still outperforms FuseMoE but gap narrowed:** AUC 0.634 vs 0.611 for SMILES-Trf (was 0.634 vs 0.576)

3. **SMILES-Trf outperforms ChemBERTa:** Consistent across both fusion methods

4. **Revised FuseMoE substantially more stable:** AUC std 0.056 (was 0.095), F1 std 0.175 (was 0.272) for SMILES-Trf

5. **Exp12 HP not beneficial for EEG + SMILES FuseMoE:** Both variance and AUC worsen for SMILES-Trf (-0.042 AUC). The EEG modality may benefit from temperature annealing's gradual specialisation of experts.

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
- FuseMoE HP tuning needs to be experiment-specific - Exp12 HP (optimised for triple modality) does not transfer to dual EEG + SMILES

---

## Technical Notes

- Class weighting: Inverse frequency (added in re-run)
- Threshold selection: Youden's J statistic (TPR - FPR)
- Training: 100 epochs, early stopping (patience 20), batch size 8
- Optimiser: AdamW, LR 1e-4, weight decay 1e-4
- Window chunking for memory efficiency (chunk size 16)
