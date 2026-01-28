# Experiment 4: Clinical Features Baseline

**Date:** 28 January 2026
**Dataset:** 205 patients with clinical data (full cohort, no embedding requirements)

---

## Objective

Establish a clinical-only baseline using the 16 demographic/clinical features available in the dataset. This baseline allows us to measure the incremental value of embedding-based modalities (LLM, EEG, SMILES) in subsequent experiments.

---

## Clinical Features (16 total)

| Category | Features | Encoding |
|----------|----------|----------|
| Binary (13) | sex, pretrt_sz_5, focal, fam_hx, febrile, ci, birth_t, head, drug, alcohol, cvd, psy, ld | 0/1 |
| Numeric (1) | age_init | Z-score normalised |
| Categorical (2) | lesion (3 levels), eeg_cat (3 levels) | One-hot (6 dims) |

**Total input dimension:** 20 (after one-hot encoding)

---

## Models Tested

### Exp4a: Clinical MLP
- Architecture: Linear(20 -> 64) -> ReLU -> LayerNorm -> Dropout(0.3) -> Linear(64 -> 32) -> ReLU -> LayerNorm -> Dropout(0.3) -> Linear(32 -> 2)
- Parameters: ~3,700
- Batch size: 16, LR: 1e-3, Weight decay: 1e-4

### Exp4b: Clinical Attention
- Architecture: Feature embeddings (20 x 64) + positional embeddings + [CLS] token -> 2-layer Transformer Encoder (4 heads) -> classifier
- Parameters: ~104,000
- Batch size: 8, LR: 5e-4, Weight decay: 1e-4

---

## Results (5-fold CV)

| Experiment | Model | AUC | Balanced Acc | F1 Tuned |
|------------|-------|-----|--------------|----------|
| **Exp4a** | MLP | **0.664 +/- 0.043** | **0.675 +/- 0.032** | 0.627 +/- 0.056 |
| **Exp4b** | Attention | 0.636 +/- 0.069 | 0.673 +/- 0.061 | 0.629 +/- 0.123 |

### Per-Fold AUC Values

| Fold | Exp4a MLP | Exp4b Attention |
|------|-----------|-----------------|
| 1 | 0.712 | 0.690 |
| 2 | 0.614 | 0.683 |
| 3 | 0.719 | 0.700 |
| 4 | 0.643 | 0.538 |
| 5 | 0.630 | 0.568 |

---

## Comparison with Experiments 1-3

| Experiment | Best Config | AUC | vs Exp4a |
|------------|-------------|-----|----------|
| **Exp4a** | Clinical MLP | **0.664** | baseline |
| Exp1 | PubMedBERT + ChemBERTa (FuseMoE) | 0.658 | -0.006 |
| Exp2 | SimpleCNN + SMILES-Trf (MLP) | 0.668 | +0.004 |
| Exp3 | PubMedBERT + ChemBERTa (MLP) | 0.672 | +0.008 |

---

## Key Findings

1. **Clinical baseline is strong**: AUC 0.664 matches Feng et al. 2025 benchmark (AUC 0.67 for clinical-only)

2. **Simple MLP preferred over attention**: Lower variance (std 0.043 vs 0.069), more consistent across folds

3. **Embeddings provide minimal lift**: All embedding-based experiments (Exp 1-3) perform within +/- 0.01 AUC of clinical baseline

4. **Attention model shows instability**: Two poor folds (0.538, 0.568) suggest overfitting with limited data

---

## Implications

The comparable performance of clinical features alone versus complex embedding pipelines suggests:

1. Clinical features contain substantial predictive signal for ASM outcomes
2. Current embedding approaches may not be capturing information beyond what clinical features provide
3. Future experiments should focus on clinical + embedding fusion to test for complementary signal

---

## Reference

Feng et al. 2025. "Integrative Deep Learning of Genomic and Clinical Data for Predicting Treatment Response in Newly Diagnosed Epilepsy." Neurology.
- Clinical-only baseline: AUC 0.67 (development cohort, n=286)
- Best multimodal (clinical + genomic): AUC 0.74
- Used same 16 clinical factors as our dataset
