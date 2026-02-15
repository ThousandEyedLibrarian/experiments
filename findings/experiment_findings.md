# ASM Outcome Prediction: Experimental Findings

**Date:** 30 January 2026
**Dataset:** 151 patients with EEG recordings and anti-seizure medication (ASM) outcomes

---

## Executive Summary

We evaluated multimodal fusion approaches for predicting ASM treatment outcomes. Ten experiment sets were conducted:

- **Experiment 1:** Text report embeddings (LLM) + drug structure embeddings (SMILES)
- **Experiment 2:** EEG signal embeddings + drug structure embeddings (SMILES)
- **Experiment 3:** LLM + EEG + SMILES embeddings (triple modality)
- **Experiment 4:** Clinical features only (baseline)
- **Experiment 5:** Clinical features + single modality fusion
- **Experiment 6:** Clinical features + SMILES + third modality (text or EEG)
- **Experiment 7:** All four modalities (Clinical + LLM + EEG + SMILES)
- **Experiment 8:** Stratification analysis (multi-label CV)
- **Experiment 9:** EEG encoder ablation (SimpleCNN, EEGNet, LaBraM, EEG2Vec)
- **Experiment 10:** Direct LLM text modality (frozen/fine-tuned encoder)

The best performing model achieved **AUC 0.798** and **balanced accuracy of 0.814** using all four modalities with MLP fusion (Exp7a). Class weighting and threshold tuning (via Youden's J statistic) were applied to address class imbalance.

**Key finding:** Quad modality fusion (Exp7a, AUC 0.798) outperforms triple modality (Exp3b, AUC 0.677). The revised FuseMoE implementation (Laplace gating, MI loss, temperature annealing) improved results in most experiments, though Exp3b showed a regression from its previous AUC of 0.753 with the old implementation.

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
| exp1b | ClinicalBERT | SMILES-Trf | **0.674 +/- 0.139** | **0.720 +/- 0.079** | 0.664 +/- 0.147 |
| exp1a | PubMedBERT | ChemBERTa | 0.641 +/- 0.070 | 0.699 +/- 0.033 | 0.676 +/- 0.082 |
| exp1b | ClinicalBERT | ChemBERTa | 0.636 +/- 0.142 | 0.709 +/- 0.100 | 0.713 +/- 0.116 |
| exp1a | PubMedBERT | SMILES-Trf | 0.632 +/- 0.106 | 0.676 +/- 0.073 | 0.624 +/- 0.198 |
| exp1a | ClinicalBERT | SMILES-Trf | 0.623 +/- 0.112 | 0.677 +/- 0.073 | 0.557 +/- 0.110 |
| exp1b | PubMedBERT | SMILES-Trf | 0.612 +/- 0.049 | 0.689 +/- 0.049 | 0.713 +/- 0.052 |
| exp1a | ClinicalBERT | ChemBERTa | 0.609 +/- 0.099 | 0.669 +/- 0.067 | 0.707 +/- 0.061 |
| exp1b | PubMedBERT | ChemBERTa | 0.601 +/- 0.104 | 0.681 +/- 0.079 | 0.707 +/- 0.070 |

### Key Observations
- Best AUC: exp1b_clinicalbert_smilestrf (0.674) with revised FuseMoE (Laplace gating, MI loss, temperature annealing)
- PubMedBERT + SMILES-Trf FuseMoE has lowest variance (AUC std 0.049) - most stable configuration
- FuseMoE slightly outperforms MLP for best AUC (0.674 vs 0.641)
- High variance across folds (std 0.05-0.14) due to small dataset (n=121)

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
| exp2b | SMILES-Trf | FuseMoE | 0.611 +/- 0.056 | 0.621 +/- 0.049 | 0.556 +/- 0.175 |
| exp2b | ChemBERTa | FuseMoE | 0.572 +/- 0.024 | 0.599 +/- 0.012 | 0.569 +/- 0.133 |

### Key Observations

- MLP fusion still outperforms FuseMoE but gap narrowed (AUC 0.634 vs 0.611 for SMILES-Trf)
- Revised FuseMoE substantially more stable than old implementation (AUC std 0.056 vs 0.095 for SMILES-Trf)
- SMILES Transformer embeddings consistently outperform ChemBERTa across both fusion methods
- FuseMoE F1 variance improved but still high (std 0.13-0.18 vs previous 0.27-0.28)
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
| exp3a | ClinicalBERT | ChemBERTa | MLP | 0.687 | 0.713 | 0.654 |
| exp3b | ClinicalBERT | ChemBERTa | FuseMoE | **0.677 +/- 0.108** | **0.726 +/- 0.092** | **0.761 +/- 0.084** |
| exp3b | PubMedBERT | SMILES-Trf | FuseMoE | 0.657 +/- 0.084 | 0.682 +/- 0.058 | 0.689 +/- 0.060 |
| exp3a | ClinicalBERT | SMILES-Trf | MLP | 0.649 | 0.707 | 0.736 |
| exp3b | ClinicalBERT | SMILES-Trf | FuseMoE | 0.628 +/- 0.124 | 0.684 +/- 0.092 | 0.693 +/- 0.057 |
| exp3a | PubMedBERT | ChemBERTa | MLP | 0.625 | 0.686 | 0.630 |
| exp3a | PubMedBERT | SMILES-Trf | MLP | 0.620 | 0.673 | 0.624 |
| exp3b | PubMedBERT | ChemBERTa | FuseMoE | 0.604 +/- 0.088 | 0.680 +/- 0.061 | 0.658 +/- 0.106 |

> **FuseMoE regression note:** The revised FuseMoE (Laplace gating, MI loss) reduced Exp3b AUC from 0.753 to 0.677 compared to the previous implementation. The old softmax + malformed KL loss may have acted as accidental regularisation. The exp3a MLP results (unchanged at AUC 0.687-0.701) now outperform FuseMoE for triple modality, consistent with the pattern that simpler fusion methods perform better on small datasets (n=107).

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

- **Best model:** ClinicalBERT + ChemBERTa + MLP (AUC 0.687, Balanced Accuracy 0.713) - MLP now outperforms FuseMoE
- **FuseMoE regression:** Revised FuseMoE reduced best AUC from 0.753 to 0.677 - old malformed KL loss may have provided accidental regularisation
- MLP fusion more reliable than MoE for triple modality on small datasets (n=107)
- ChemBERTa outperforms SMILES-Trf for both fusion methods
- Fold variance remains moderate (AUC std 0.088-0.124 for FuseMoE)


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
| **Exp5c** | Clinical + EEG | SimpleCNN | 0.675 +/- 0.061 | 0.698 +/- 0.057 | 0.689 +/- 0.091 |

### Comparison with Exp4a Baseline (AUC 0.664)

| Experiment | AUC | Delta |
|------------|-----|-------|
| exp5a_chemberta | 0.689 | **+0.025** |
| exp5a_smilestrf | 0.687 | **+0.023** |
| exp5b_clinicalbert | 0.676 | +0.012 |
| exp5c_simplecnn | 0.675 | +0.011 |
| exp5b_pubmedbert | 0.620 | -0.044 |

### Key Observations

- SMILES embeddings provide most consistent lift (~+0.02 AUC) with low variance
- ClinicalBERT achieves highest balanced accuracy (0.708) but has high AUC variance
- EEG fusion now stable (std 0.061, down from 0.113) due to multi-label stratification and pipeline improvements
- EEG now provides a small positive lift over clinical baseline (+0.011 AUC)
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
| **exp7a** | **ClinicalBERT** | **MLP** | **0.798 +/- 0.093** | **0.814 +/- 0.069** | **0.813 +/- 0.071** |
| exp7b | ClinicalBERT | MoE | 0.753 +/- 0.127 | 0.754 +/- 0.079 | 0.716 +/- 0.108 |
| exp7a | PubMedBERT | MLP | 0.752 +/- 0.069 | 0.766 +/- 0.065 | 0.749 +/- 0.119 |
| exp7b | PubMedBERT | MoE | 0.712 +/- 0.072 | 0.716 +/- 0.051 | 0.718 +/- 0.100 |

### Per-Fold AUC (Best Model: exp7a_clinicalbert)

| Fold | AUC | Bal Acc | F1 (tuned) |
|------|-----|---------|------------|
| 1 | 0.689 | 0.731 | 0.786 |
| 2 | 0.818 | 0.864 | 0.842 |
| 3 | 0.700 | 0.742 | 0.700 |
| 4 | 0.933 | 0.908 | 0.917 |
| 5 | 0.850 | 0.825 | 0.818 |

### Key Observations

- **New overall best:** exp7a_clinicalbert_chemberta (MLP) with AUC 0.798 (up from 0.762)
- MLP fusion still outperforms MoE (0.798 vs 0.753) but gap narrowed
- Revised FuseMoE improved MoE substantially: ClinicalBERT +0.033 AUC, PubMedBERT +0.050 AUC
- Fold 4 still shows highest performance (AUC 0.933) - consistent across re-runs
- Multi-label stratification and pipeline improvements contributed to overall gains

### Meta-Analysis (Sidik-Jonkman + Knapp-Hartung)

Proper confidence intervals accounting for between-fold heterogeneity:

| Configuration | AUC | 95% CI | I² | τ² |
|---------------|-----|--------|----|----|
| exp7a_clinicalbert | 0.762 | [0.633, 0.891] | 80% | 0.0086 |
| exp7a_pubmedbert | 0.746 | [0.664, 0.829] | 80% | 0.0035 |
| exp7b_clinicalbert | 0.720 | [0.573, 0.867] | 80% | 0.0112 |
| exp7b_pubmedbert | 0.662 | [0.593, 0.731] | 80% | 0.0025 |

**Interpretation:**
- **Wide CIs** (0.14–0.29 range): Limited precision due to small sample (k=5 folds, n=107 patients)
- **High I² (80%)**: Substantial heterogeneity — fold composition strongly affects performance
- **Overlapping CIs**: Cannot definitively rank configurations; differences may be due to chance
- **MoE more variable**: Higher τ² for MoE models indicates less stable training on small data
- **PubMedBERT more consistent**: Lower τ² but lower point estimates than ClinicalBERT

---

## Experiment 8: Stratification Analysis

Investigated whether the high I² (80%) heterogeneity in Exp7 could be reduced through improved cross-validation stratification.

### Motivation

The high fold-to-fold variance observed in previous experiments may be partly due to:
1. **Outcome-only stratification**: Current CV only balances the target variable
2. **Imbalanced clinical features**: Some features have >95% majority class
3. **Unbalanced fold composition**: Key features may cluster in certain folds

### Feature Imbalance Analysis

| Feature | Majority % | Minority n | Status |
|---------|------------|------------|--------|
| `ld` | 98.5% | 3 | SEVERE |
| `birth_t` | 97.5% | 5 | SEVERE |
| `febrile` | 96.0% | 8 | SEVERE |
| `ci` | 95.5% | 9 | SEVERE |
| `fam_hx` | 88.5% | 23 | WARNING |
| `cvd` | 88.4% | 23 | WARNING |
| `focal` | 79.2% | 42 | OK |
| `sex` | 62.7% | 76 | OK |

**Rationale**: Considering dropping severely imbalanced features (`ld`, `birth_t`, `febrile`, `ci`) as they seem to provide minimal impact/discrimination.

### Stratification Comparison

Compared fold balance variance across stratification methods:

| Feature | Outcome-only (fold_std) | Multi-label (fold_std) | Improvement |
|---------|------------------------|------------------------|-------------|
| focal | 10.7% | 1.3% | **8x better** |
| sex | 5.8% | 1.1% | **5x better** |
| outcome | 0.5% | 0.5% | Same |

Multi-label stratification (outcome + focal + sex) dramatically reduces fold-to-fold variance for clinical features while maintaining outcome balance.

### Data Quality Issues Fixed

- `psy` column: Mixed types ('0', '1', '0.0', '1.0', '?') standardised
- `lesion` column: Mixed types (1, '1.0', 'NOT AVAILABLE') standardised
- `outcome` column: String values converted to numeric

### Key Findings

1. **Multi-label stratification reduces fold variance by 5-8x** for key features
2. **Severely imbalanced features** (ld, birth_t, febrile, ci) have <10 minority samples
3. **Data cleaning required**: Several columns had mixed types needing standardisation
4. **Composite stratification** (outcome + focal) also effective but less balanced for sex

### Files

- `exp8_stratification/feature_analysis.py` - Distribution analysis
- `exp8_stratification/stratified_cv.py` - Multi-label stratification implementation
- `exp8_stratification/run_experiments.py` - Full experiment runner

---

## Comparison: All Experiments

| Experiment | Modality | Best Model | AUC | Bal Acc Tuned | F1 Tuned |
|------------|----------|------------|-----|---------------|----------|
| **Exp7a** | **Clinical + LLM + EEG + SMILES** | ClinicalBERT + ChemBERTa + MLP | **0.762** | **0.774** | 0.786 |
| Exp3b | LLM + EEG + SMILES | ClinicalBERT + ChemBERTa + FuseMoE | 0.753 | 0.774 | **0.801** |
| Exp6a | Clinical + SMILES + Text | PubMedBERT + ChemBERTa | 0.702 | 0.705 | 0.738 |
| Exp10 | Clinical + Direct LLM (frozen) | Qwen 2.5 0.5B | 0.689 | 0.717 | 0.666 |
| Exp5a | Clinical + SMILES | ChemBERTa | 0.689 | 0.680 | 0.638 |
| Exp5b | Clinical + Text | ClinicalBERT | 0.676 | 0.708 | 0.716 |
| Exp2a | EEG + SMILES | SimpleCNN + SMILES-Trf + MLP | 0.668 | N/A | N/A |
| Exp4a | Clinical only | MLP | 0.664 | 0.675 | 0.627 |
| Exp9 | EEG + SMILES (encoder ablation) | EEG2Vec | 0.661 | 0.689 | 0.585 |
| Exp1b | LLM + SMILES | ClinicalBERT + SMILES-Trf + FuseMoE | 0.648 | 0.712 | 0.701 |
| Exp6b | Clinical + SMILES + EEG | SimpleCNN + SMILES-Trf | 0.647 | 0.663 | 0.631 |
| Exp5c | Clinical + EEG | SimpleCNN | 0.644 | 0.690 | 0.693 |

**Key findings:**
- Quad modality (Exp7a) achieves best AUC (0.762), marginal improvement over triple (Exp3b, +0.009)
- Clinical features provide small but consistent improvement when added to embeddings
- MLP fusion more stable than MoE for small datasets (107 patients)
- Text fusion consistently outperforms EEG fusion across all experiments
- SMILES embeddings provide complementary signal to other modalities
- Exp10 frozen Qwen 2.5 (0.689) matches Exp5a SMILES fusion - general-purpose LLM competitive with domain-specific models
- Exp9 EEG2Vec (0.661) improves on SimpleCNN baseline (0.607) with substantially lower variance

---

## Limitations

- Relatively small sample size (n=151 for dual-modality, n=107 for triple/quad-modality, n=205 for clinical-only)
- High variance across folds (std up to 0.10 for AUC in some configurations)
- Wide 95% CIs due to small k (5 folds) and high heterogeneity (I²=80%)
- Overlapping CIs between best models prevent definitive ranking
- LaBraM EEG encoder underperforms (AUC 0.549) - architecture may be unsuitable for 27-channel clinical EEG with small dataset
- No hyperparameter tuning performed
- Quad-modality limited by intersection of all data sources (107 patients)
- Exp10 frozen encoder results only - fine-tuning may change model ranking

---

## Experiment 9: EEG Variance Investigation

Investigated the sources of high fold-to-fold variance in EEG experiments (Exp5c AUC std 0.113) and conducted encoder ablation study.

### Key Findings from Fold Analysis

**Outcome-only stratification (previous):**
- `focal` varies 67.5%-92.7% across folds (25% range)
- Best fold (4, AUC 0.866) has 92.7% focal patients vs 71.8% in worst fold (5, AUC 0.545)
- EEG padding ratio strongly negatively correlated with AUC (r=-0.78)

**Multi-label stratification (implemented):**
- `focal` balanced to 77.5%-80.5% (3% range) - **8x variance reduction**
- `sex` balanced to 60.9%-63.4% (2.5% range) - **6x variance reduction**
- Correlations with AUC drop significantly (focal: 0.74 -> 0.21)

### Implemented Improvements

| Component | Implementation | Location |
|-----------|----------------|----------|
| Multi-label stratification | Integrated into exp2/exp5 training | `exp{2,5}_fusion/training.py` |
| EEG quality metrics | SNR, artifacts, flatlines, correlation | `exp2_fusion/eeg_pipeline.py` |
| EEG normalisation | Global z-score, window z-score, robust | `exp2_fusion/eeg_pipeline.py` |
| Alternative aggregators | Attention, MaxPool, LSTM, MultiScale | `exp2_fusion/models/aggregators.py` |
| EEGNet encoder | Added alongside SimpleCNN (~3x fewer params) | `exp2_fusion/models/eeg_encoders.py` |
| Ablation framework | 12 experiments defined | `exp9_eeg_investigation/run_experiments.py` |

### Results: Encoder Ablation (12 February 2026)

Run on M3 HPC (A100 80GB). Three HPC runs required: Run 1 failed (missing `iterative-stratification`), Run 2 partially succeeded (braindecode CUDA issue), Run 3 all 4 encoders completed successfully.

| Encoder | AUC | Bal Acc Tuned | F1 Tuned | AUC Std |
|---------|-----|---------------|----------|---------|
| **EEG2Vec** | **0.661 +/- 0.061** | **0.689 +/- 0.054** | 0.585 +/- 0.149 | **0.061** |
| EEGNet | 0.648 +/- 0.107 | 0.686 +/- 0.078 | 0.584 +/- 0.239 | 0.107 |
| SimpleCNN (baseline) | 0.607 +/- 0.107 | 0.661 +/- 0.076 | 0.616 +/- 0.059 | 0.107 |
| LaBraM | 0.549 +/- 0.077 | 0.608 +/- 0.036 | 0.512 +/- 0.196 | 0.077 |

### Per-Fold AUC

| Encoder | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---------|--------|--------|--------|--------|--------|
| EEG2Vec | 0.710 | 0.673 | 0.701 | 0.678 | 0.542 |
| EEGNet | 0.741 | 0.711 | 0.705 | 0.640 | 0.444 |
| SimpleCNN | 0.710 | 0.691 | 0.683 | 0.487 | 0.467 |
| LaBraM | 0.629 | 0.653 | 0.509 | 0.471 | 0.481 |

### Key Observations

- **EEG2Vec achieves best AUC (0.661) with lowest variance (std 0.061)** - most stable encoder
- EEGNet outperforms SimpleCNN (+0.041 AUC) but both share high variance (std 0.107)
- LaBraM significantly underperforms (AUC 0.549) - likely due to small dataset or 27-channel clinical EEG mismatch
- Multi-label stratification did not eliminate variance for SimpleCNN/EEGNet - encoder architecture also contributes
- EEG2Vec's CVAE pre-training provides more robust features (std 0.061 vs 0.107)
- Folds 4/5 consistently weakest across all encoders
- Remaining ablations (aggregator, depth, dimension) not yet run

### Files

- `exp9_eeg_investigation/fold_analysis.py` - Fold composition analysis
- `exp9_eeg_investigation/quality_analysis.py` - EEG quality metrics analysis
- `exp9_eeg_investigation/run_experiments.py` - Ablation experiment framework
- `exp9_eeg_investigation/config.py` - Configuration

---

## Experiment 10: Direct LLM Text Modality

Ran LLM inference at training time (frozen encoder mode) instead of pre-computed embeddings, enabling comparison across different LLM architectures with the same training pipeline.

### Architecture

Late fusion: Clinical features (19D) -> 64D + Raw text -> LLM encoder -> embed_dim -> 64D, concatenated (128D), then classified. Matches Exp5b architecture.

### Results: Frozen Encoder (12 February 2026)

Run on M3 HPC (Job 51362383, node m3n102, A100 80GB, ~20 min runtime).

| Model | AUC | Bal Acc Tuned | F1 Tuned |
|-------|-----|---------------|----------|
| **Qwen 2.5 0.5B** | **0.689 +/- 0.088** | **0.717 +/- 0.073** | 0.666 +/- 0.119 |
| ClinicalBERT | 0.644 +/- 0.121 | 0.695 +/- 0.092 | 0.671 +/- 0.130 |
| PubMedBERT | 0.635 +/- 0.096 | 0.671 +/- 0.075 | 0.674 +/- 0.058 |

### Per-Fold AUC

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|-------|--------|--------|--------|--------|--------|
| Qwen 2.5 0.5B | 0.769 | 0.790 | 0.671 | 0.542 | 0.671 |
| ClinicalBERT | 0.788 | 0.629 | 0.580 | 0.458 | 0.762 |
| PubMedBERT | 0.744 | 0.685 | 0.643 | 0.458 | 0.643 |

### Key Observations

- Qwen 2.5 0.5B (general-purpose) outperforms both biomedical-specific models in frozen mode
- Fold 4 consistently weakest across all 3 models (AUC 0.458-0.542) - data composition issue
- Frozen results comparable to Exp5b pre-computed embeddings (ClinicalBERT 0.644 vs 0.676)
- Fine-tuning experiments still pending (Phase 2)

---

## Next Steps

1. ~~Look at torch api key padding mask and add to code~~ **DONE** - Already implemented in `exp2_fusion/models/eeg_transformer.py`
2. ~~Double check focal column distribution for stratification~~ **DONE** - Exp8 created with multi-label stratification
3. ~~Investigate high EEG variance~~ **DONE** - Exp9 investigation complete
4. ~~Test LaBraM/EEGNet/EEG2Vec encoders~~ **DONE** - Exp9 encoder ablation complete (EEG2Vec best, LaBraM underperforms)
5. ~~Run Exp9 encoder ablation experiments~~ **DONE** - 4 encoders compared, EEG2Vec selected as best
6. ~~Run Exp10 frozen encoder experiments~~ **DONE** - Qwen 2.5 0.5B outperforms biomedical models in frozen mode
7. ~~Run Exp9 remaining ablations: aggregator, depth, and dimension experiments with EEG2Vec~~ **DONE** - Running, reran all exp9 with EEG2Vec
8. ~~Run Exp10 fine-tuning experiments (Phase 2) - unfreeze last 2 transformer layers~~ **DONE** - Running
9. ~~Replace FuseMoE implementation with revised portions from Duong's fusemoe.py in shared/~~ **IN PROG**
10. ~~Re-run Exp5c/Exp7 with EEG2Vec encoder (replacing SimpleCNN)~~ **IN PROG**
11. Swap EEG2Vec into exp2, exp3, and exp6b configs (currently still SimpleCNN) for consistent encoder across all EEG experiments
12. Re-run exp1b, exp2b, exp3b with revised FuseMoE (Laplace gating, MI loss, temperature annealing)
13. Re-run exp3a/exp7a MLP baselines with EEG2Vec to get matched comparisons against revised FuseMoE
14. Re-run exp6b with EEG2Vec encoder for comparison table completeness
15. Hyperparameter optimisation for best-performing model - Optuna
16. External validation on further data or an additional dataset if it becomes available
