# ASM Outcome Prediction: Experimental Findings

**Date:** 23 January 2026
**Dataset:** 151 patients with EEG recordings and anti-seizure medication (ASM) outcomes

---

## Executive Summary

We evaluated multimodal fusion approaches for predicting ASM treatment outcomes. Two experiment sets were conducted:

- **Experiment 1:** Text report embeddings (LLM) + drug structure embeddings (SMILES)
- **Experiment 2:** EEG signal embeddings + drug structure embeddings (SMILES)

The best performing model achieved an **AUC of 0.668** using EEG + SMILES Transformer embeddings with MLP fusion.

---

## Experiment 1: LLM + SMILES Fusion

Combined clinical text report embeddings with molecular structure embeddings.

### Models Tested
- **Text encoders:** ClinicalBERT, PubMedBERT
- **SMILES encoders:** ChemBERTa, SMILES Transformer
- **Fusion methods:** Concatenation + MLP (1a), FuseMoE (1b)

### Results (5-fold CV)

| Experiment | Text Model | SMILES Model | AUC | Accuracy | F1 |
|------------|------------|--------------|-----|----------|-----|
| exp1b | PubMedBERT | ChemBERTa | **0.658** | 0.546 | 0.612 |
| exp1a | ClinicalBERT | ChemBERTa | 0.640 | 0.528 | 0.336 |
| exp1b | ClinicalBERT | SMILES-Trf | 0.639 | 0.570 | 0.622 |
| exp1b | PubMedBERT | SMILES-Trf | 0.652 | 0.562 | 0.557 |
| exp1a | ClinicalBERT | SMILES-Trf | 0.607 | 0.578 | 0.662 |
| exp1a | PubMedBERT | ChemBERTa | 0.569 | 0.570 | 0.630 |
| exp1b | ClinicalBERT | ChemBERTa | 0.614 | 0.562 | 0.547 |
| exp1a | PubMedBERT | SMILES-Trf | 0.506 | 0.545 | 0.629 |

### Key Observations
- FuseMoE (exp1b) generally outperformed simple concatenation (exp1a) for AUC
- ChemBERTa embeddings yielded higher AUC than SMILES Transformer in most cases
- PubMedBERT + ChemBERTa with FuseMoE achieved the best AUC (0.658)

---

## Experiment 2: EEG + SMILES Fusion

Combined EEG signal embeddings with molecular structure embeddings.

### Models Tested
- **EEG encoder:** SimpleCNN (27 channels, 10s windows)
- **SMILES encoders:** ChemBERTa, SMILES Transformer
- **Fusion methods:** Concatenation + MLP (2a), FuseMoE (2b)

### Results (5-fold CV)

| Experiment | SMILES Model | Fusion | AUC | Accuracy | F1 |
|------------|--------------|--------|-----|----------|-----|
| exp2a | SMILES-Trf | MLP | **0.668** | 0.563 | 0.585 |
| exp2a | ChemBERTa | MLP | 0.608 | 0.543 | 0.478 |
| exp2b | SMILES-Trf | FuseMoE | 0.608 | 0.576 | **0.658** |
| exp2b | ChemBERTa | FuseMoE | 0.554 | 0.523 | 0.501 |

### Key Observations
- SMILES Transformer embeddings consistently outperformed ChemBERTa for EEG fusion
- Simple MLP fusion achieved higher AUC than FuseMoE
- FuseMoE achieved better F1 score, suggesting more balanced predictions
- EEG + SMILES-Trf + MLP achieved the overall best AUC (0.668)

---

## Comparison: Exp1 vs Exp2

| Modality | Best Model | AUC | Accuracy |
|----------|------------|-----|----------|
| EEG + SMILES | SimpleCNN + SMILES-Trf + MLP | **0.668** | 0.563 |
| LLM + SMILES | PubMedBERT + ChemBERTa + FuseMoE | 0.658 | 0.546 |

EEG-based models showed marginally better discriminative performance than text-based models.

---

## Limitations

- Relatively small sample size (n=151)
- High variance across folds (std up to 0.12 for AUC)
- LaBraM EEG encoder not tested due to dependency issues
- No hyperparameter tuning performed

---

## Next Steps

1. **Experiment 3:** Combine all three modalities (LLM + EEG + SMILES)
2. Test LaBraM encoder once braindecode dependencies are resolved
3. Hyperparameter optimisation for best-performing models
4. Investigate high fold variance with stratified analysis
