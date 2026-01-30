# Experiment 7: All Four Modalities Fusion

## Overview

Experiment 7 combines all four available modalities to predict ASM treatment outcomes:
- **Clinical features** (20D): Demographics, medical history
- **Text embeddings** (768D): EEG report embeddings from ClinicalBERT/PubMedBERT
- **EEG signals**: Raw EEG windows processed through CNN + Transformer
- **SMILES embeddings** (768D): Drug molecular structure from ChemBERTa

## Objective

Test whether adding clinical features to the triple modality fusion (Exp3) improves prediction performance.

## Dataset

- **Patients**: 107 unique patients (same as Exp3)
- **Class distribution**: Imbalanced (failure vs success)
- **Cross-validation**: 5-fold stratified

## Architectures

### Exp7a: Late Fusion MLP (~2M params)

```
Clinical (20D) ───> Encoder -> 64D ─┐
                                     │
Text (768D) ──────> Encoder -> 64D ─┼─> Concat (256D) -> Classifier -> 2
                                     │
EEG (windows) ────> CNN+Trf -> 64D ─┤
                                     │
SMILES (768D) ────> Encoder -> 64D ─┘
```

### Exp7b: FuseMoE (~4.7M params)

```
Clinical (20D) ───> Projection -> 256D ─┐
                                         │
Text (768D) ──────> Projection -> 256D ─┼─> Cross-Attention -> MoE -> Classifier
                                         │
EEG (windows) ────> CNN+Trf -> 256D ────┤
                                         │
SMILES (768D) ────> Projection -> 256D ─┘
```

## Results

| Experiment | Text Model | Fusion | AUC | Std | Bal Acc | Std |
|------------|------------|--------|-----|-----|---------|-----|
| **exp7a_clinicalbert** | ClinicalBERT | MLP | **0.762** | 0.093 | **0.774** | 0.071 |
| exp7a_pubmedbert | PubMedBERT | MLP | 0.746 | 0.059 | 0.741 | 0.057 |
| exp7b_clinicalbert | ClinicalBERT | MoE | 0.720 | 0.106 | 0.737 | 0.075 |
| exp7b_pubmedbert | PubMedBERT | MoE | 0.662 | 0.050 | 0.702 | 0.044 |

### Per-Fold AUC (Best Model: exp7a_clinicalbert)

| Fold | AUC | Bal Acc | F1 (tuned) |
|------|-----|---------|------------|
| 1 | 0.667 | 0.727 | 0.800 |
| 2 | 0.736 | 0.727 | 0.700 |
| 3 | 0.700 | 0.725 | 0.750 |
| 4 | 0.933 | 0.908 | 0.917 |
| 5 | 0.775 | 0.783 | 0.762 |

## Comparison with Baselines

| Experiment | Description | AUC | Delta |
|------------|-------------|-----|-------|
| **Exp7a** | Clinical + Text + EEG + SMILES (MLP) | **0.762** | - |
| Exp3b | Text + EEG + SMILES (FuseMoE) | 0.753 | +0.009 |
| Exp6a | Clinical + Text + SMILES | 0.702 | +0.060 |
| Exp4a | Clinical only | 0.664 | +0.098 |

## Key Findings

1. **Best configuration**: exp7a_clinicalbert_chemberta (MLP fusion)
   - AUC: 0.762 +/- 0.093
   - Balanced Accuracy: 0.774 +/- 0.071

2. **MLP outperforms MoE** for quad modality fusion
   - MLP: 0.762 vs MoE: 0.720 (with ClinicalBERT)
   - Simpler architecture more stable for small dataset

3. **Marginal improvement over Exp3b** (+0.009 AUC)
   - Clinical features provide slight additional signal
   - Most information already captured by embeddings

4. **ClinicalBERT > PubMedBERT** for quad modality
   - Consistent with Exp3 findings
   - Clinical domain-specific pretraining beneficial

5. **High fold variance**
   - Fold 4 shows unusually high AUC (0.933)
   - Likely due to favourable class split in small dataset

## Interpretation

The marginal improvement (+0.009 AUC) from adding clinical features suggests that:
- The embedding modalities (text, EEG, SMILES) already capture most predictive signal
- Clinical features provide some complementary information
- The small dataset size (107 patients) limits the benefit of additional modalities

## Training Configuration

| Parameter | MLP (7a) | MoE (7b) |
|-----------|----------|----------|
| Learning rate | 1e-3 | 5e-4 |
| Batch size | 8 | 8 |
| Epochs | 100 | 100 |
| Early stopping | 20 | 20 |
| Weight decay | 1e-4 | 1e-4 |

## Files

- `exp7_all_modalities/run_experiments.py` - Entry point
- `exp7_all_modalities/models.py` - QuadFusionMLP, QuadFusionMoE
- `exp7_all_modalities/data_pipeline.py` - QuadModalityDataset
- `exp7_all_modalities/training.py` - Training loop with CV
- `outputs/exp7_results/results_20260130_152752.json` - Results
