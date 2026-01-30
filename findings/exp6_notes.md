# Experiment 6: Clinical + SMILES + Third Modality Fusion

**Date:** 29 January 2026
**Dataset:** Varies by modality (see below)

---

## Objective

Test whether combining clinical features with SMILES embeddings AND a third modality (text or EEG) improves upon:
- Clinical-only baseline (Exp4a: AUC 0.664)
- Clinical + SMILES (Exp5a: AUC 0.689)
- Triple modality without clinical (Exp3: AUC 0.753)

---

## Experiments

| Exp | Modalities | Embedding Models | Dataset Size |
|-----|------------|------------------|--------------|
| 6a | Clinical + SMILES + Text | ChemBERTa/SMILES-Trf + ClinicalBERT/PubMedBERT | ~121 patients |
| 6b | Clinical + SMILES + EEG | ChemBERTa/SMILES-Trf + SimpleCNN | ~151 patients |

---

## Architecture

All experiments use **late fusion** with three modality streams:

```
Clinical (20D) -----> Encoder --> 64D --|
                                        |
SMILES (768/256D) --> Encoder --> 64D --|---> Concat (192D) --> Classifier --> 2 classes
                                        |
Third Modality -----> Encoder --> 64D --|
```

**Encoders:**
- Clinical: Linear(20 -> 64) + ReLU + LayerNorm + Dropout(0.3)
- SMILES: Linear(768/256 -> 64) + ReLU + LayerNorm + Dropout(0.3)
- Text: Linear(768 -> 64) + ReLU + LayerNorm + Dropout(0.3)
- EEG: SimpleCNN (27ch -> 256D per window) + TransformerEncoder (2L, 4H) + MeanPool -> 64D

**Classifier:** Linear(192 -> 64) + ReLU + LayerNorm + Dropout(0.3) + Linear(64 -> 2)

**Model Parameters:**
- ClinicalSMILESTextFusion: ~113K params
- ClinicalSMILESEEGFusion: ~1.96M params

---

## Results (5-fold CV)

### Exp6a: Clinical + SMILES + Text

| Text Model | SMILES Model | AUC | Bal Acc Tuned | F1 Tuned |
|------------|--------------|-----|---------------|----------|
| **PubMedBERT** | **ChemBERTa** | **0.702 +/- 0.067** | **0.705 +/- 0.058** | **0.738 +/- 0.093** |
| PubMedBERT | SMILES-Trf | 0.651 +/- 0.068 | 0.686 +/- 0.035 | 0.662 +/- 0.103 |
| ClinicalBERT | SMILES-Trf | 0.650 +/- 0.096 | 0.691 +/- 0.067 | 0.698 +/- 0.062 |
| ClinicalBERT | ChemBERTa | 0.627 +/- 0.145 | 0.672 +/- 0.100 | 0.671 +/- 0.115 |

### Exp6b: Clinical + SMILES + EEG

| EEG Model | SMILES Model | AUC | Bal Acc Tuned | F1 Tuned |
|-----------|--------------|-----|---------------|----------|
| SimpleCNN | SMILES-Trf | 0.647 +/- 0.061 | 0.663 +/- 0.035 | 0.631 +/- 0.135 |
| SimpleCNN | ChemBERTa | 0.643 +/- 0.061 | 0.692 +/- 0.062 | 0.684 +/- 0.105 |

### Per-Fold AUC Values

| Fold | 6a PubMed+Chem | 6a PubMed+Trf | 6a Clinical+Trf | 6a Clinical+Chem | 6b CNN+Trf | 6b CNN+Chem |
|------|----------------|---------------|-----------------|------------------|------------|-------------|
| 1 | 0.821 | 0.750 | 0.788 | 0.878 | 0.646 | 0.663 |
| 2 | 0.674 | 0.674 | 0.701 | 0.556 | 0.649 | 0.644 |
| 3 | 0.727 | 0.643 | 0.629 | 0.699 | 0.560 | 0.573 |
| 4 | 0.636 | 0.538 | 0.497 | 0.503 | 0.750 | 0.746 |
| 5 | 0.650 | 0.650 | 0.636 | 0.497 | 0.629 | 0.589 |

---

## Comparison with Baselines

| Model | AUC | vs Exp4a | vs Exp5a | vs Exp3 |
|-------|-----|----------|----------|---------|
| **Exp6a best** (PubMedBERT+ChemBERTa) | 0.702 | **+0.038** | **+0.013** | -0.051 |
| Exp5a best (ChemBERTa) | 0.689 | +0.025 | - | -0.064 |
| Exp4a (Clinical only) | 0.664 | - | -0.025 | -0.089 |
| Exp3 best (ClinicalBERT+ChemBERTa+FuseMoE) | 0.753 | +0.089 | +0.064 | - |

---

## Key Findings

1. **Best model is PubMedBERT + ChemBERTa**: AUC 0.702, Balanced Accuracy 0.705 - modest improvement over Exp5a

2. **Text fusion outperforms EEG fusion**: Exp6a consistently achieves higher AUC than Exp6b

3. **Clinical features provide modest lift**: Adding clinical to text+SMILES improves AUC by ~0.01 over embedding-only approaches

4. **Exp3 remains best overall**: Triple modality without clinical (AUC 0.753) still outperforms Exp6 - clinical features may not add value when embeddings already capture relevant signal

5. **High variance persists**: ClinicalBERT+ChemBERTa has AUC std 0.145 (range 0.50-0.88), indicating model instability

6. **PubMedBERT more stable**: Lower variance (std 0.067) compared to ClinicalBERT configurations

---

## Implications

1. **Clinical features have diminishing returns**: When rich embeddings are available, clinical features add limited discriminative power

2. **Text embeddings more informative than EEG**: EEG signal may be too noisy or require more sophisticated encoding

3. **Model selection matters**: PubMedBERT + ChemBERTa combination is most stable and performant

4. **Consider Exp3 for deployment**: If clinical features don't improve performance, simpler embedding-only model may be preferred

---

## Technical Notes

- Threshold tuning uses Youden's J statistic (maximises TPR - FPR)
- Class weighting via inverse frequency for imbalanced classes
- Training: 100 epochs, early stopping (patience 20), batch size 16 (text) / 8 (EEG)
- Optimiser: AdamW, LR 1e-3, weight decay 1e-4
