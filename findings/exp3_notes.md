# Experiment 3: Triple Modality Fusion (LLM + EEG + SMILES)

**Date:** 27-28 January 2026
**Dataset:** 107 patients with all three modalities available

---

## Objective

Test whether combining all three modalities (text reports, EEG signals, drug structure) improves prediction over dual-modality approaches.

---

## Architecture

### Input Embeddings

| Modality | Models | Dimension |
|----------|--------|-----------|
| Text Reports | ClinicalBERT, PubMedBERT | 768D |
| EEG Signals | SimpleCNN + Transformer | 256D |
| Drug Structure | ChemBERTa, SMILES-Trf | 768D / 256D |

### Fusion Methods

**Exp3a (MLP):** Project all modalities to 256D, concatenate (768D), MLP classifier

**Exp3b (FuseMoE):** Learnable modality tokens, self-attention, 2x MoE layers (4 experts, top-2)

---

## Results (5-fold CV with Threshold Tuning)

| Experiment | Text | SMILES | Fusion | AUC | Bal Acc Tuned | F1 Tuned |
|------------|------|--------|--------|-----|---------------|----------|
| exp3a | ClinicalBERT | ChemBERTa | MLP | 0.647 +/- 0.112 | **0.706 +/- 0.059** | 0.692 +/- 0.092 |
| exp3a | ClinicalBERT | SMILES-Trf | MLP | 0.667 +/- 0.038 | 0.689 +/- 0.040 | 0.722 +/- 0.087 |
| exp3a | PubMedBERT | ChemBERTa | MLP | **0.672 +/- 0.016** | 0.704 +/- 0.028 | 0.686 +/- 0.038 |
| exp3a | PubMedBERT | SMILES-Trf | MLP | 0.650 +/- 0.098 | 0.676 +/- 0.063 | 0.693 +/- 0.085 |
| exp3b | ClinicalBERT | ChemBERTa | FuseMoE | 0.662 +/- 0.061 | 0.696 +/- 0.052 | 0.702 +/- 0.041 |
| exp3b | ClinicalBERT | SMILES-Trf | FuseMoE | 0.648 +/- 0.078 | 0.672 +/- 0.047 | 0.654 +/- 0.082 |
| exp3b | PubMedBERT | ChemBERTa | FuseMoE | 0.618 +/- 0.023 | 0.681 +/- 0.032 | 0.661 +/- 0.102 |
| exp3b | PubMedBERT | SMILES-Trf | FuseMoE | 0.629 +/- 0.028 | 0.691 +/- 0.031 | 0.600 +/- 0.082 |

---

## Key Findings

1. **Best AUC:** PubMedBERT + ChemBERTa + MLP (0.672) - lowest variance (std 0.016)

2. **Best Balanced Accuracy:** ClinicalBERT + ChemBERTa + MLP (0.706)

3. **MLP generally outperforms FuseMoE:** Simpler fusion works better with limited data (n=107)

4. **ChemBERTa more consistent than SMILES-Trf:** Lower variance in most configurations

5. **Reduced dataset size:** Only 107 patients have all modalities (vs 151 for Exp2)

---

## Comparison Across Experiments

| Experiment | Best AUC | Dataset Size |
|------------|----------|--------------|
| Exp1 (LLM + SMILES) | 0.658 | 121 |
| Exp2 (EEG + SMILES) | 0.668 | 151 |
| **Exp3 (Triple)** | **0.672** | 107 |

Triple modality achieves highest AUC despite smallest dataset.

---

## Threshold Tuning Impact

Threshold tuning via Youden's J statistic significantly improved metrics:

| Metric | Default (0.5) | Tuned |
|--------|---------------|-------|
| F1 | ~0.53-0.70 | ~0.65-0.72 |
| Optimal thresholds | - | 0.32-0.45 |

This confirms class imbalance effects and the importance of proper threshold selection.

---

## Limitations

- Smallest dataset (107 patients) limits model complexity
- High variance in some configurations (std up to 0.112)
- Only SimpleCNN tested for EEG encoding
- No hyperparameter optimisation performed

---

## Technical Notes

- Class weighting: Inverse frequency applied
- Threshold selection: Youden's J statistic (maximises TPR - FPR)
- Training: 100 epochs, early stopping (patience 20), batch size 8
- Optimiser: AdamW, LR varies by model (1e-4 for MLP, 5e-5 for FuseMoE)
