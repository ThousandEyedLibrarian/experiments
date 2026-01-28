# ASM Outcome Prediction: Experimental Findings

**Date:** 27 January 2026
**Dataset:** 151 patients with EEG recordings and anti-seizure medication (ASM) outcomes

---

## Executive Summary

We evaluated multimodal fusion approaches for predicting ASM treatment outcomes. Three experiment sets were conducted:

- **Experiment 1:** Text report embeddings (LLM) + drug structure embeddings (SMILES)
- **Experiment 2:** EEG signal embeddings + drug structure embeddings (SMILES)
- **Experiment 3:** LLM + EEG + SMILES embeddings (triple modality)

The best performing model achieved a **balanced accuracy of 0.774** using ClinicalBERT + ChemBERTa + FuseMoE fusion (triple modality). Class weighting and threshold tuning (via Youden's J statistic) were applied to address class imbalance.

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
- **Class balancing:** Inverse frequency class weights in loss function
- **Threshold tuning:** Optimal threshold selected via precision-recall curve

### Results (5-fold CV)

| Experiment | Text Model | SMILES Model | Fusion | AUC | F1 | F1_tuned |
|------------|------------|--------------|--------|-----|-----|----------|
| exp3b | ClinicalBERT | SMILES-Trf | FuseMoE | 0.694 | 0.541 | **0.780** |
| exp3b | PubMedBERT | SMILES-Trf | FuseMoE | 0.618 | 0.657 | 0.772 |
| exp3a | PubMedBERT | ChemBERTa | MLP | **0.701** | 0.537 | 0.767 |
| exp3a | PubMedBERT | SMILES-Trf | MLP | 0.672 | 0.655 | 0.756 |
| exp3a | ClinicalBERT | ChemBERTa | MLP | 0.632 | 0.528 | 0.754 |
| exp3a | ClinicalBERT | SMILES-Trf | MLP | 0.683 | 0.518 | 0.750 |
| exp3b | ClinicalBERT | ChemBERTa | FuseMoE | 0.617 | 0.585 | 0.737 |
| exp3b | PubMedBERT | ChemBERTa | FuseMoE | 0.614 | 0.489 | 0.736 |

### Key Observations

- **F1_tuned is the recommended metric** - shows model potential with proper threshold selection
- Optimal thresholds ranged from 0.32-0.40 (vs default 0.5), confirming class imbalance
- FuseMoE now achieves best F1_tuned (0.780) with ClinicalBERT + SMILES-Trf
- F1_tuned variance is low (~0.02-0.05 std) vs high variance at default threshold
- Class weights improved F1 from ~0.22 to ~0.54 at default threshold
- Only 107 patients had all three modalities (vs 151 for dual-modality)

### Updated Results: Balanced Accuracy Threshold Selection

**Methodology change:** Threshold now selected by maximising balanced accuracy using Youden's J statistic (`J = TPR - FPR`) from the ROC curve, rather than maximising F1 from the precision-recall curve. This ensures equal weighting of both classes.

| Experiment | Text Model | SMILES Model | Fusion | AUC | Bal Acc Tuned | F1 Tuned |
|------------|------------|--------------|--------|-----|---------------|----------|
| exp3b | ClinicalBERT | ChemBERTa | FuseMoE | **0.753** | **0.774** | **0.801** |
| exp3b | PubMedBERT | ChemBERTa | FuseMoE | 0.688 | 0.733 | 0.732 |
| exp3b | ClinicalBERT | SMILES-Trf | FuseMoE | 0.675 | 0.725 | 0.733 |
| exp3a | ClinicalBERT | ChemBERTa | MLP | 0.687 | 0.713 | 0.654 |
| exp3a | ClinicalBERT | SMILES-Trf | MLP | 0.649 | 0.707 | 0.736 |
| exp3a | PubMedBERT | ChemBERTa | MLP | 0.625 | 0.686 | 0.630 |
| exp3b | PubMedBERT | SMILES-Trf | FuseMoE | 0.618 | 0.681 | 0.739 |
| exp3a | PubMedBERT | SMILES-Trf | MLP | 0.620 | 0.673 | 0.624 |

### Fold Deviation Statistics (5-fold CV, Jan 28 run)

| Experiment | Text Model | SMILES Model | Fusion | AUC Range | Bal Acc Range | AUC Std | Bal Acc Std |
|------------|------------|--------------|--------|-----------|---------------|---------|-------------|
| exp3a | ClinicalBERT | ChemBERTa | MLP | 0.558-0.868 | 0.658-0.818 | 0.112 | 0.059 |
| exp3a | PubMedBERT | ChemBERTa | MLP | 0.658-0.702 | 0.667-0.742 | 0.016 | 0.028 |
| exp3b | ClinicalBERT | ChemBERTa | FuseMoE | 0.598-0.769 | 0.642-0.773 | 0.061 | 0.052 |
| exp3b | PubMedBERT | ChemBERTa | FuseMoE | 0.583-0.653 | 0.650-0.742 | 0.023 | 0.032 |
| exp3a | ClinicalBERT | SMILES-Trf | MLP | 0.600-0.717 | 0.658-0.767 | 0.038 | 0.040 |
| exp3a | PubMedBERT | SMILES-Trf | MLP | 0.542-0.825 | 0.608-0.783 | 0.098 | 0.063 |
| exp3b | ClinicalBERT | SMILES-Trf | FuseMoE | 0.550-0.744 | 0.608-0.727 | 0.078 | 0.047 |
| exp3b | PubMedBERT | SMILES-Trf | FuseMoE | 0.600-0.674 | 0.667-0.750 | 0.028 | 0.031 |

### Key Observations (Updated)

- **Best model:** ClinicalBERT + ChemBERTa + FuseMoE (AUC 0.753, Balanced Accuracy 0.774)
- FuseMoE consistently outperforms MLP when using balanced accuracy threshold
- ChemBERTa now outperforms SMILES-Trf (different from F1-optimised results)
- Balanced accuracy and F1 are well-aligned for the best model
- Fold variance reasonable (std ~0.06 for balanced accuracy)


---

## Comparison: Exp1 vs Exp2 vs Exp3

| Modality | Best Model | AUC | Bal Acc Tuned | F1 Tuned |
|----------|------------|-----|---------------|----------|
| LLM + EEG + SMILES | ClinicalBERT + ChemBERTa + FuseMoE | **0.753** | **0.774** | **0.801** |
| EEG + SMILES | SimpleCNN + SMILES-Trf + MLP | 0.668 | N/A | N/A |
| LLM + SMILES | PubMedBERT + ChemBERTa + FuseMoE | 0.658 | N/A | N/A |

Triple modality with class weighting and balanced accuracy threshold tuning achieves the best performance. The combination of all three modalities provides complementary signal for predicting ASM outcomes.

---

## Limitations

- Relatively small sample size (n=151 for dual-modality, n=107 for triple-modality)
- High variance across folds (std up to 0.11 for AUC)
- LaBraM EEG encoder not tested due to dependency issues with braindecode
- No hyperparameter tuning performed
- Exp1/Exp2 not yet re-run with class weighting and threshold tuning


---

## Next Steps

1. Apply class weighting and threshold tuning to Exp1 and Exp2 for fair comparison
2. Test LaBraM encoder once braindecode dependencies are resolved
3. Hyperparameter optimisation for best-performing model (exp3b ClinicalBERT + SMILES-Trf + FuseMoE)
4. Investigate high fold variance with stratified analysis
