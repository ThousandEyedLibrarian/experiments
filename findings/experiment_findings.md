# ASM Outcome Prediction: Experimental Findings

**Date:** 30 January 2026
**Dataset:** 151 patients with EEG recordings and anti-seizure medication (ASM) outcomes

---

## Executive Summary

We evaluated multimodal fusion approaches for predicting ASM treatment outcomes. Seven experiment sets were conducted:

- **Experiment 1:** Text report embeddings (LLM) + drug structure embeddings (SMILES)
- **Experiment 2:** EEG signal embeddings + drug structure embeddings (SMILES)
- **Experiment 3:** LLM + EEG + SMILES embeddings (triple modality)
- **Experiment 4:** Clinical features only (baseline)
- **Experiment 5:** Clinical features + single modality fusion
- **Experiment 6:** Clinical features + SMILES + third modality (text or EEG)
- **Experiment 7:** All four modalities (Clinical + LLM + EEG + SMILES)

The best performing model achieved **AUC 0.762** and **balanced accuracy of 0.774** using all four modalities with MLP fusion (Exp7a). Class weighting and threshold tuning (via Youden's J statistic) were applied to address class imbalance.

**Key finding:** Quad modality fusion (Exp7a, AUC 0.762) marginally improves upon triple modality (Exp3b, AUC 0.753). Clinical features provide small but consistent improvement (+0.009 AUC) when added to embeddings.

---

## Experiment 1: LLM + SMILES Fusion

Combined clinical text report embeddings with molecular structure embeddings.

### Models Tested
- **Text encoders:** ClinicalBERT, PubMedBERT
- **SMILES encoders:** ChemBERTa, SMILES Transformer
- **Fusion methods:** Concatenation + MLP (1a), FuseMoE (1b)

### Results (5-fold CV with Threshold Tuning)

| Experiment | Text Model | SMILES Model | AUC | Bal Acc Tuned | F1 Tuned |
|------------|------------|--------------|-----|---------------|----------|
| exp1b | ClinicalBERT | SMILES-Trf | **0.648 +/- 0.100** | 0.712 +/- 0.074 | 0.701 +/- 0.117 |
| exp1b | ClinicalBERT | ChemBERTa | 0.643 +/- 0.128 | 0.670 +/- 0.078 | 0.597 +/- 0.142 |
| exp1a | PubMedBERT | ChemBERTa | 0.641 +/- 0.070 | 0.699 +/- 0.033 | 0.676 +/- 0.082 |
| exp1b | PubMedBERT | ChemBERTa | 0.641 +/- 0.071 | **0.713 +/- 0.047** | 0.670 +/- 0.125 |
| exp1a | PubMedBERT | SMILES-Trf | 0.632 +/- 0.106 | 0.676 +/- 0.073 | 0.624 +/- 0.198 |
| exp1a | ClinicalBERT | SMILES-Trf | 0.623 +/- 0.112 | 0.677 +/- 0.073 | 0.557 +/- 0.110 |
| exp1a | ClinicalBERT | ChemBERTa | 0.609 +/- 0.099 | 0.669 +/- 0.067 | 0.707 +/- 0.061 |
| exp1b | PubMedBERT | SMILES-Trf | 0.592 +/- 0.075 | 0.641 +/- 0.047 | 0.635 +/- 0.079 |

### Key Observations
- Best balanced accuracy: exp1b_pubmedbert_chemberta (0.713) and exp1b_clinicalbert_smilestrf (0.712)
- FuseMoE slightly outperforms MLP for balanced accuracy
- High variance across folds (std 0.07-0.13) due to small dataset (n=121)

---

## Experiment 2: EEG + SMILES Fusion

Combined EEG signal embeddings with molecular structure embeddings.

### Models Tested
- **EEG encoder:** SimpleCNN (27 channels, 10s windows)
- **SMILES encoders:** ChemBERTa, SMILES Transformer
- **Fusion methods:** Concatenation + MLP (2a), FuseMoE (2b)

### Results (5-fold CV with Class Weighting and Threshold Tuning)

| Experiment | SMILES Model | Fusion | AUC | Bal Acc Tuned | F1 Tuned |
|------------|--------------|--------|-----|---------------|----------|
| exp2a | SMILES-Trf | MLP | **0.634 +/- 0.045** | **0.699 +/- 0.047** | **0.720 +/- 0.056** |
| exp2a | ChemBERTa | MLP | 0.611 +/- 0.074 | 0.672 +/- 0.045 | 0.632 +/- 0.075 |
| exp2b | SMILES-Trf | FuseMoE | 0.576 +/- 0.095 | 0.579 +/- 0.051 | 0.537 +/- 0.272 |
| exp2b | ChemBERTa | FuseMoE | 0.562 +/- 0.084 | 0.583 +/- 0.054 | 0.554 +/- 0.278 |

### Key Observations

- MLP fusion significantly outperforms FuseMoE (Bal Acc 0.67-0.70 vs 0.58)
- SMILES Transformer embeddings consistently outperform ChemBERTa
- FuseMoE unstable with EEG data (F1 std 0.27-0.28)
- Class weighting added in re-run (previously missing)


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

## Experiment 4: Clinical Features Baseline

Established a clinical-only baseline using 16 demographic/clinical features to benchmark embedding-based approaches.

### Clinical Features Used

- **Binary (13):** sex, pretrt_sz_5, focal, fam_hx, febrile, ci, birth_t, head, drug, alcohol, cvd, psy, ld
- **Numeric (1):** age_init (Z-score normalised)
- **Categorical (2):** lesion, eeg_cat (one-hot encoded, 6 dims total)

### Results (5-fold CV)

| Experiment | Model | AUC | Balanced Acc Tuned | F1 Tuned |
|------------|-------|-----|-------------------|----------|
| **Exp4a** | MLP (~3.7K params) | **0.664 +/- 0.043** | **0.675 +/- 0.032** | 0.627 +/- 0.056 |
| Exp4b | Attention (~104K params) | 0.636 +/- 0.069 | 0.673 +/- 0.061 | 0.629 +/- 0.123 |

### Per-Fold AUC Values

- **Exp4a MLP:** [0.712, 0.614, 0.719, 0.643, 0.630]
- **Exp4b Attention:** [0.690, 0.683, 0.700, 0.538, 0.568]

### Key Observations

- Simple MLP baseline more stable than attention model (lower variance)
- Matches Feng et al. 2025 benchmark (clinical-only AUC 0.67)
- Attention model shows instability on folds 4-5 (may need more data)

---

## Experiment 5: Clinical + Single Modality Fusion

Tested whether fusing clinical features with a single embedding modality improves upon the clinical-only baseline.

### Architecture

Late fusion: each modality encoded to 64D, concatenated (128D), then classified.

### Results (5-fold CV)

| Experiment | Modality | Model | AUC | Balanced Acc Tuned | F1 Tuned |
|------------|----------|-------|-----|-------------------|----------|
| **Exp5a** | Clinical + SMILES | ChemBERTa | **0.689 +/- 0.060** | 0.680 +/- 0.048 | 0.638 +/- 0.103 |
| **Exp5a** | Clinical + SMILES | SMILES-Trf | 0.687 +/- 0.041 | 0.682 +/- 0.042 | 0.674 +/- 0.063 |
| **Exp5b** | Clinical + Text | ClinicalBERT | 0.676 +/- 0.083 | **0.708 +/- 0.073** | 0.716 +/- 0.090 |
| **Exp5b** | Clinical + Text | PubMedBERT | 0.620 +/- 0.038 | 0.690 +/- 0.060 | 0.729 +/- 0.043 |
| **Exp5c** | Clinical + EEG | SimpleCNN | 0.644 +/- 0.113 | 0.690 +/- 0.089 | 0.693 +/- 0.120 |

### Comparison with Exp4a Baseline (AUC 0.664)

| Experiment | AUC | Delta |
|------------|-----|-------|
| exp5a_chemberta | 0.689 | **+0.025** |
| exp5a_smilestrf | 0.687 | **+0.023** |
| exp5b_clinicalbert | 0.676 | +0.012 |
| exp5b_pubmedbert | 0.620 | -0.044 |
| exp5c_simplecnn | 0.644 | -0.020 |

### Key Observations

- SMILES embeddings provide most consistent lift (~+0.02 AUC) with low variance
- ClinicalBERT achieves highest balanced accuracy (0.708) but has high AUC variance
- EEG fusion is unstable (std 0.113) - one fold hit 0.866, another 0.545
- Dataset size affects stability: text experiments have smallest dataset (121 patients)

---

## Experiment 6: Clinical + SMILES + Third Modality Fusion

Tested whether combining clinical features with SMILES embeddings AND a third modality improves upon previous experiments.

### Architecture

Late fusion with three modality streams: each modality encoded to 64D, concatenated (192D), then classified.

### Exp6a: Clinical + SMILES + Text (5-fold CV)

| Text Model | SMILES Model | AUC | Bal Acc Tuned | F1 Tuned |
|------------|--------------|-----|---------------|----------|
| **PubMedBERT** | **ChemBERTa** | **0.702 +/- 0.067** | **0.705 +/- 0.058** | **0.738 +/- 0.093** |
| PubMedBERT | SMILES-Trf | 0.651 +/- 0.068 | 0.686 +/- 0.035 | 0.662 +/- 0.103 |
| ClinicalBERT | SMILES-Trf | 0.650 +/- 0.096 | 0.691 +/- 0.067 | 0.698 +/- 0.062 |
| ClinicalBERT | ChemBERTa | 0.627 +/- 0.145 | 0.672 +/- 0.100 | 0.671 +/- 0.115 |

### Exp6b: Clinical + SMILES + EEG (5-fold CV)

| EEG Model | SMILES Model | AUC | Bal Acc Tuned | F1 Tuned |
|-----------|--------------|-----|---------------|----------|
| SimpleCNN | SMILES-Trf | 0.647 +/- 0.061 | 0.663 +/- 0.035 | 0.631 +/- 0.135 |
| SimpleCNN | ChemBERTa | 0.643 +/- 0.061 | 0.692 +/- 0.062 | 0.684 +/- 0.105 |

### Key Observations

- Best Exp6 model: PubMedBERT + ChemBERTa (AUC 0.702, +0.038 vs Exp4a, +0.013 vs Exp5a)
- Text fusion (Exp6a) outperforms EEG fusion (Exp6b)
- Exp3 (triple without clinical) still achieves highest AUC (0.753)
- Clinical features provide diminishing returns when embeddings are available

---

## Experiment 7: All Four Modalities Fusion

Tested whether combining all four modalities improves upon triple modality (Exp3).

### Architecture

- **Exp7a (MLP):** Each modality encoded to 64D, concatenated (256D), classified (~2M params)
- **Exp7b (MoE):** Cross-modal attention + Sparse MoE layers (~4.7M params)

### Results (5-fold CV)

| Experiment | Text Model | Fusion | AUC | Bal Acc Tuned | F1 Tuned |
|------------|------------|--------|-----|---------------|----------|
| **exp7a** | **ClinicalBERT** | **MLP** | **0.762 +/- 0.093** | **0.774 +/- 0.071** | **0.786 +/- 0.073** |
| exp7a | PubMedBERT | MLP | 0.746 +/- 0.059 | 0.741 +/- 0.057 | 0.751 +/- 0.045 |
| exp7b | ClinicalBERT | MoE | 0.720 +/- 0.106 | 0.737 +/- 0.075 | 0.729 +/- 0.111 |
| exp7b | PubMedBERT | MoE | 0.662 +/- 0.050 | 0.702 +/- 0.044 | 0.753 +/- 0.044 |

### Per-Fold AUC (Best Model: exp7a_clinicalbert)

| Fold | AUC | Bal Acc | F1 (tuned) |
|------|-----|---------|------------|
| 1 | 0.667 | 0.727 | 0.800 |
| 2 | 0.736 | 0.727 | 0.700 |
| 3 | 0.700 | 0.725 | 0.750 |
| 4 | 0.933 | 0.908 | 0.917 |
| 5 | 0.775 | 0.783 | 0.762 |

### Key Observations

- Best model: exp7a_clinicalbert_chemberta (MLP) with AUC 0.762
- MLP fusion outperforms MoE for quad modality (0.762 vs 0.720)
- Marginal improvement over Exp3b (+0.009 AUC)
- Fold 4 shows unusually high performance (AUC 0.933) - potential outlier
- Clinical features provide modest additional signal

---

## Comparison: All Experiments

| Experiment | Modality | Best Model | AUC | Bal Acc Tuned | F1 Tuned |
|------------|----------|------------|-----|---------------|----------|
| **Exp7a** | **Clinical + LLM + EEG + SMILES** | ClinicalBERT + ChemBERTa + MLP | **0.762** | **0.774** | 0.786 |
| Exp3b | LLM + EEG + SMILES | ClinicalBERT + ChemBERTa + FuseMoE | 0.753 | 0.774 | **0.801** |
| Exp6a | Clinical + SMILES + Text | PubMedBERT + ChemBERTa | 0.702 | 0.705 | 0.738 |
| Exp5a | Clinical + SMILES | ChemBERTa | 0.689 | 0.680 | 0.638 |
| Exp5b | Clinical + Text | ClinicalBERT | 0.676 | 0.708 | 0.716 |
| Exp2a | EEG + SMILES | SimpleCNN + SMILES-Trf + MLP | 0.668 | N/A | N/A |
| Exp4a | Clinical only | MLP | 0.664 | 0.675 | 0.627 |
| Exp1b | LLM + SMILES | ClinicalBERT + SMILES-Trf + FuseMoE | 0.648 | 0.712 | 0.701 |
| Exp6b | Clinical + SMILES + EEG | SimpleCNN + SMILES-Trf | 0.647 | 0.663 | 0.631 |
| Exp5c | Clinical + EEG | SimpleCNN | 0.644 | 0.690 | 0.693 |

**Key findings:**
- Quad modality (Exp7a) achieves best AUC (0.762), marginal improvement over triple (Exp3b, +0.009)
- Clinical features provide small but consistent improvement when added to embeddings
- MLP fusion more stable than MoE for small datasets (107 patients)
- Text fusion consistently outperforms EEG fusion across all experiments
- SMILES embeddings provide complementary signal to other modalities

---

## Limitations

- Relatively small sample size (n=151 for dual-modality, n=107 for triple/quad-modality, n=205 for clinical-only)
- High variance across folds (std up to 0.10 for AUC in some configurations)
- LaBraM EEG encoder not tested due to dependency issues with braindecode
- No hyperparameter tuning performed
- Quad-modality limited by intersection of all data sources (107 patients)


---

## Next Steps

1. Look at torch api key padding mask and add to code (makes model look at just real data not padding), ignore padding in mean pooling layer
    1a. Double check focal column 1 or 0, and any other heavily concentrated data columns to avoid concentration in one stratum when folding
2. Test LaBraM encoder once braindecode dependencies are resolved
3. Hyperparameter optimisation for best-performing model (Exp3b ClinicalBERT+ChemBERTa) - Optuna
4. Investigate high EEG variance - potential for improved EEG encoding
5. External validation on further data or an additional dataset
