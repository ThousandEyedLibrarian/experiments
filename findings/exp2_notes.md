# Experiment 2: EEG + SMILES Fusion

**Date:** 23 January 2026
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

## Results (5-fold CV)

| Experiment | SMILES Model | Fusion | AUC | Accuracy | F1 |
|------------|--------------|--------|-----|----------|-----|
| **Exp2a** | SMILES-Trf | MLP | **0.668 +/- 0.072** | 0.563 +/- 0.048 | 0.585 +/- 0.089 |
| Exp2a | ChemBERTa | MLP | 0.608 +/- 0.066 | 0.543 +/- 0.049 | 0.478 +/- 0.253 |
| Exp2b | SMILES-Trf | FuseMoE | 0.608 +/- 0.046 | 0.576 +/- 0.056 | **0.658 +/- 0.043** |
| Exp2b | ChemBERTa | FuseMoE | 0.554 +/- 0.105 | 0.523 +/- 0.089 | 0.501 +/- 0.272 |

### Per-Fold AUC Values (Best Model: Exp2a SMILES-Trf MLP)

| Fold | AUC |
|------|-----|
| 1 | 0.558 |
| 2 | 0.720 |
| 3 | 0.636 |
| 4 | 0.658 |
| 5 | 0.768 |

---

## Key Findings

1. **Best AUC:** Exp2a with SMILES Transformer + MLP fusion (0.668)

2. **SMILES-Trf outperforms ChemBERTa:** Consistent across both fusion methods

3. **Simple MLP beats FuseMoE for AUC:** More complex architecture may overfit on small dataset

4. **FuseMoE achieves best F1:** May produce more balanced predictions (0.658 vs 0.585)

5. **High variance across folds:** Std up to 0.105 for AUC indicates sensitivity to data splits

---

## Comparison with Exp1 (LLM + SMILES)

| Experiment | Best AUC | Best Model |
|------------|----------|------------|
| Exp1 | 0.658 | PubMedBERT + ChemBERTa + FuseMoE |
| **Exp2** | **0.668** | SimpleCNN + SMILES-Trf + MLP |

EEG + SMILES slightly outperforms LLM + SMILES, suggesting EEG contains complementary signal.

---

## Limitations

- No class weighting applied (addressed in later experiments)
- No threshold tuning (fixed 0.5 threshold)
- LaBraM encoder not tested due to memory constraints
- SimpleCNN only encoder tested

---

## Technical Notes

- Training: 100 epochs, early stopping (patience 20), batch size 8
- Optimiser: AdamW, LR 1e-4, weight decay 1e-4
- Window chunking for memory efficiency (chunk size 16)
