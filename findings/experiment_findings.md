# ASM Outcome Prediction: Experimental Findings

**Date:** 27 January 2026
**Dataset:** 151 patients with EEG recordings and anti-seizure medication (ASM) outcomes

---

## Executive Summary

We evaluated multimodal fusion approaches for predicting ASM treatment outcomes. Three experiment sets were conducted:

- **Experiment 1:** Text report embeddings (LLM) + drug structure embeddings (SMILES)
- **Experiment 2:** EEG signal embeddings + drug structure embeddings (SMILES)
- **Experiment 3:** LLM + EEG + SMILES embeddings (triple modality)

The best performing model achieved an **AUC of 0.733** using ClinicalBERT + ChemBERTa + MLP fusion (triple modality).

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
- FuseMoE achieved better F1 score, maybe suggesting more balanced/robust predictions?
- EEG + SMILES-Trf + MLP achieved the overall best AUC (0.668)


---

## Experiment 3: LLM + EEG + SMILES Fusion

Combined all three modalities: text report embeddings, EEG signal embeddings, and molecular structure embeddings.

### Models Tested
- **Text encoders:** ClinicalBERT, PubMedBERT
- **EEG encoder:** SimpleCNN (27 channels, 10s windows)
- **SMILES encoders:** ChemBERTa, SMILES Transformer
- **Fusion methods:** Concatenation + MLP (3a), FuseMoE (3b)

### Results (5-fold CV)

| Experiment | Text Model | SMILES Model | Fusion | AUC | Accuracy | F1 |
|------------|------------|--------------|--------|-----|----------|-----|
| exp3a | ClinicalBERT | ChemBERTa | MLP | **0.733** | 0.530 | 0.224 |
| exp3a | ClinicalBERT | SMILES-Trf | MLP | 0.688 | 0.604 | 0.706 |
| exp3a | PubMedBERT | ChemBERTa | MLP | 0.675 | 0.541 | 0.698 |
| exp3b | ClinicalBERT | SMILES-Trf | FuseMoE | 0.656 | 0.603 | 0.580 |
| exp3b | PubMedBERT | ChemBERTa | FuseMoE | 0.652 | 0.540 | 0.497 |
| exp3b | PubMedBERT | SMILES-Trf | FuseMoE | 0.638 | 0.513 | 0.529 |
| exp3b | ClinicalBERT | ChemBERTa | FuseMoE | 0.628 | 0.532 | 0.552 |
| exp3a | PubMedBERT | SMILES-Trf | MLP | 0.627 | 0.585 | 0.674 |

### Key Observations

- Triple modality achieves highest AUC (0.733) across all experiments
- MLP fusion outperformed FuseMoE for exp3 (opposite pattern to exp1)
- ClinicalBERT + ChemBERTa was the best combination for AUC
- Best AUC model has poor F1 (0.224) - may be biased towards negative class
- ClinicalBERT + SMILES-Trf offers best balance of AUC (0.688) and F1 (0.706)
- Only 107 patients had all three modalities (vs 151 for dual-modality)


---

## Comparison: Exp1 vs Exp2 vs Exp3

| Modality | Best Model | AUC | Accuracy | F1 |
|----------|------------|-----|----------|-----|
| LLM + EEG + SMILES | ClinicalBERT + ChemBERTa + MLP | **0.733** | 0.530 | 0.224 |
| EEG + SMILES | SimpleCNN + SMILES-Trf + MLP | 0.668 | 0.563 | 0.585 |
| LLM + SMILES | PubMedBERT + ChemBERTa + FuseMoE | 0.658 | 0.546 | 0.612 |

Triple modality provides a clear AUC improvement (~10%) over dual-modality approaches, though at the cost of reduced sample size (107 vs 151 patients) and lower F1 in the best AUC model.

---

## Limitations

- Relatively small sample size (n=151 for dual-modality, n=107 for triple-modality)
- High variance across folds (std up to 0.10 for AUC)
- LaBraM EEG encoder not tested due to dependency issues with braindecode
- No hyperparameter tuning performed
- Best AUC model (exp3) has poor F1 - potential precision/recall trade-off to investigate


---

## Next Steps

1. Investigate F1/AUC trade-off in exp3 - consider class weighting or threshold tuning
2. Test LaBraM encoder once braindecode dependencies are resolved
3. Hyperparameter optimisation for best-performing models (esp. exp3a ClinicalBERT + ChemBERTa)
4. Investigate high fold variance with stratified analysis
