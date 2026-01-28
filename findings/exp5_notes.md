# Experiment 5: Clinical + Single Modality Fusion

**Date:** 28 January 2026
**Dataset:** Varies by modality (see below)

---

## Objective

Test whether fusing clinical features with a single embedding modality improves upon the clinical-only baseline (Exp4a: AUC 0.664). This establishes which modalities provide complementary signal to clinical data before attempting full multimodal fusion.

---

## Experiments

| Exp | Modality | Embedding Model | Dataset Size |
|-----|----------|-----------------|--------------|
| 5a | Clinical + SMILES | ChemBERTa (768D), SMILES-Trf (256D) | 204 patients |
| 5b | Clinical + Text | ClinicalBERT (768D), PubMedBERT (768D) | 121 patients |
| 5c | Clinical + EEG | SimpleCNN + Transformer (256D) | 149 patients |

---

## Architecture

All experiments use **late fusion**:

```
Clinical (20D) --> Encoder --> 64D --|
                                      |--> Concat (128D) --> Classifier --> 2 classes
Modality (xD)  --> Encoder --> 64D --|
```

**Clinical Encoder:** Linear(20 -> 64) + ReLU + LayerNorm + Dropout(0.3)

**Modality Encoders:**
- SMILES: Linear(768/256 -> 64) + ReLU + LayerNorm + Dropout(0.3)
- Text: Linear(768 -> 64) + ReLU + LayerNorm + Dropout(0.3)
- EEG: SimpleCNN (27ch -> 256D per window) + TransformerEncoder (2 layers) + MeanPool -> 64D

**Classifier:** Linear(128 -> 64) + ReLU + LayerNorm + Dropout(0.3) + Linear(64 -> 2)

---

## Results (5-fold CV)

| Experiment | Model | AUC | Balanced Acc | F1 Tuned |
|------------|-------|-----|--------------|----------|
| **Exp5a** | ChemBERTa | **0.689 +/- 0.060** | 0.680 +/- 0.048 | 0.638 +/- 0.103 |
| **Exp5a** | SMILES-Trf | 0.687 +/- 0.041 | 0.682 +/- 0.042 | 0.674 +/- 0.063 |
| **Exp5b** | ClinicalBERT | 0.676 +/- 0.083 | **0.708 +/- 0.073** | 0.716 +/- 0.090 |
| **Exp5b** | PubMedBERT | 0.620 +/- 0.038 | 0.690 +/- 0.060 | 0.729 +/- 0.043 |
| **Exp5c** | SimpleCNN | 0.644 +/- 0.113 | 0.690 +/- 0.089 | 0.693 +/- 0.120 |

### Per-Fold AUC Values

| Fold | 5a ChemBERTa | 5a SMILES-Trf | 5b ClinicalBERT | 5b PubMedBERT | 5c SimpleCNN |
|------|--------------|---------------|-----------------|---------------|--------------|
| 1 | 0.738 | 0.662 | 0.782 | 0.647 | 0.600 |
| 2 | 0.612 | 0.633 | 0.576 | 0.632 | 0.604 |
| 3 | 0.774 | 0.740 | 0.762 | 0.643 | 0.604 |
| 4 | 0.686 | 0.671 | 0.657 | 0.545 | 0.866 |
| 5 | 0.638 | 0.730 | 0.601 | 0.629 | 0.545 |

---

## Comparison with Exp4a Baseline (AUC 0.664)

| Experiment | AUC | Delta | Interpretation |
|------------|-----|-------|----------------|
| exp5a_chemberta | 0.689 | **+0.025** | Modest improvement |
| exp5a_smilestrf | 0.687 | **+0.023** | Modest improvement |
| exp5b_clinicalbert | 0.676 | +0.012 | Marginal improvement |
| exp5b_pubmedbert | 0.620 | -0.044 | Degradation |
| exp5c_simplecnn | 0.644 | -0.020 | Slight degradation |

---

## Key Findings

1. **SMILES embeddings provide most consistent lift**: Both ChemBERTa and SMILES-Trf improve AUC by ~0.02 over clinical baseline with relatively low variance

2. **Text embeddings show mixed results**: ClinicalBERT achieves highest balanced accuracy (0.708) but has high variance; PubMedBERT underperforms baseline

3. **EEG fusion is unstable**: High variance across folds (std 0.113), with one exceptional fold (0.866) and one poor fold (0.545)

4. **Dataset size matters**: Text experiments have smallest dataset (121 patients) which may explain higher variance

5. **Balanced accuracy vs AUC trade-off**: ClinicalBERT shows best balanced accuracy despite modest AUC, suggesting better calibration at operating threshold

---

## Implications

1. **SMILES embeddings are promising**: Drug molecular structure appears to provide complementary signal to clinical features - worth including in multimodal fusion

2. **Text embeddings need larger dataset**: 121 patients insufficient for stable text fusion; consider data augmentation or transfer learning

3. **EEG requires investigation**: The high-variance fold (0.866) suggests potential signal exists but current architecture may not capture it reliably

4. **Next step - multimodal fusion**: Combine Clinical + SMILES + one other modality (text or EEG) in Experiment 6

---

## Technical Notes

- Threshold tuning uses Youden's J statistic (maximises TPR - FPR)
- Class weighting via inverse frequency for imbalanced classes
- Training: 100 epochs, early stopping (patience 15), batch size 16
- Optimiser: AdamW, LR 1e-3, weight decay 1e-4
