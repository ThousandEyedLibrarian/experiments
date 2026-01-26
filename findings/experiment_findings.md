# ASM Outcome Prediction: Experimental Findings

**Date:** 27 January 2026
**Dataset:** 151 patients with EEG recordings and anti-seizure medication (ASM) outcomes

---

## Executive Summary

We evaluated multimodal fusion approaches for predicting ASM treatment outcomes. Three experiment sets were conducted:

- **Experiment 1:** Text report embeddings (LLM) + drug structure embeddings (SMILES)
- **Experiment 2:** EEG signal embeddings + drug structure embeddings (SMILES)
- **Experiment 3:** LLM + EEG + SMILES embeddings (triple modality) - *results invalid, under investigation*

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

### Results (5-fold CV) - INVALID

| Experiment | Text Model | SMILES Model | Fusion | AUC | Accuracy | F1 |
|------------|------------|--------------|--------|-----|----------|-----|
| exp3a | ClinicalBERT | ChemBERTa | MLP | 0.500 | 1.000 | 0.000 |
| exp3a | ClinicalBERT | SMILES-Trf | MLP | 0.500 | 1.000 | 0.000 |
| exp3a | PubMedBERT | ChemBERTa | MLP | 0.500 | 1.000 | 0.000 |
| exp3a | PubMedBERT | SMILES-Trf | MLP | 0.500 | 1.000 | 0.000 |
| exp3b | ClinicalBERT | ChemBERTa | FuseMoE | 0.500 | 1.000 | 0.000 |
| exp3b | ClinicalBERT | SMILES-Trf | FuseMoE | 0.500 | 1.000 | 0.000 |
| exp3b | PubMedBERT | ChemBERTa | FuseMoE | 0.500 | 1.000 | 0.000 |
| exp3b | PubMedBERT | SMILES-Trf | FuseMoE | 0.500 | 1.000 | 0.000 |

### Key Observations

**These results are invalid and require investigation.** The pattern of AUC=0.5 (random chance), Accuracy=1.0, F1=0.0, with zero variance across all folds indicates a critical issue:

- Model is likely predicting a single class for all samples
- Possible causes: class imbalance handling, label loading bug, or data alignment issue
- Note: Only 107 patients had all three modalities available (vs 151 for dual-modality experiments)


---

## Comparison: Exp1 vs Exp2 vs Exp3

| Modality | Best Model | AUC | Accuracy | Status |
|----------|------------|-----|----------|--------|
| EEG + SMILES | SimpleCNN + SMILES-Trf + MLP | **0.668** | 0.563 | Valid |
| LLM + SMILES | PubMedBERT + ChemBERTa + FuseMoE | 0.658 | 0.546 | Valid |
| LLM + EEG + SMILES | All configs | 0.500 | 1.000 | Invalid |

EEG-based models showed marginally better discriminative performance than text-based models. Triple-modality results are pending investigation.

---

## Limitations

- Relatively small sample size (n=151) - have we got more data to bring in perhaps?
- High variance across folds (std up to 0.12 for AUC)
- LaBraM EEG encoder not tested due to dependency issues with braindecode
- No hyperparameter tuning performed
- **Exp3 implementation broken** - triple-modality fusion producing invalid results (likely label/data loading issue)


---

## Next Steps

1. **Debug Exp3:** Investigate triple-modality implementation - check data loading, label alignment, and class balance handling
2. Test LaBraM encoder once braindecode dependencies are resolved
3. Hyperparameter optimisation for best-performing models
4. Investigate high fold variance with stratified analysis
