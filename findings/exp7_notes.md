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
| **exp7a_clinicalbert** | ClinicalBERT | MLP | **0.798** | 0.093 | **0.814** | 0.069 |
| exp7b_clinicalbert | ClinicalBERT | MoE | 0.753 | 0.127 | 0.754 | 0.079 |
| exp7a_pubmedbert | PubMedBERT | MLP | 0.752 | 0.069 | 0.766 | 0.065 |
| exp7b_pubmedbert | PubMedBERT | MoE | 0.712 | 0.072 | 0.716 | 0.051 |

*Updated 13 February 2026 with pipeline improvements (multi-label stratification, code review fixes) for exp7a, and revised FuseMoE (Laplace gating, MI loss, temperature annealing) for exp7b. Previous best: exp7a ClinicalBERT AUC 0.762.*

### Per-Fold AUC (Best Model: exp7a_clinicalbert)

| Fold | AUC | Bal Acc | F1 (tuned) |
|------|-----|---------|------------|
| 1 | 0.689 | 0.731 | 0.786 |
| 2 | 0.818 | 0.864 | 0.842 |
| 3 | 0.700 | 0.742 | 0.700 |
| 4 | 0.933 | 0.908 | 0.917 |
| 5 | 0.850 | 0.825 | 0.818 |

## Comparison with Baselines

| Experiment | Description | AUC | Delta |
|------------|-------------|-----|-------|
| **Exp7a** | Clinical + Text + EEG + SMILES (MLP) | **0.798** | - |
| Exp3b | Text + EEG + SMILES (FuseMoE, revised) | 0.677 | +0.121 |
| Exp6a | Clinical + Text + SMILES | 0.702 | +0.096 |
| Exp4a | Clinical only | 0.664 | +0.134 |

## Key Findings

1. **New overall best**: exp7a_clinicalbert_chemberta (MLP fusion)
   - AUC: 0.798 +/- 0.093 (up from 0.762)
   - Balanced Accuracy: 0.814 +/- 0.069

2. **MLP still outperforms MoE** but gap narrowed with revised FuseMoE
   - MLP: 0.798 vs MoE: 0.753 (with ClinicalBERT)
   - Previous: MLP 0.762 vs MoE 0.720

3. **Revised FuseMoE improved MoE substantially**
   - ClinicalBERT MoE: 0.720 -> 0.753 (+0.033 AUC)
   - PubMedBERT MoE: 0.662 -> 0.712 (+0.050 AUC)

4. **ClinicalBERT > PubMedBERT** for quad modality
   - Consistent with Exp3 findings
   - Clinical domain-specific pretraining beneficial

5. **Fold 4 consistently strongest** (AUC 0.933) across re-runs

### EEG2Vec Upgrade (Exp11, 15 February 2026)

Exp11 replaced SimpleCNN with EEG2Vec encoder (128D embeddings) for the exp7a MLP architecture. See `findings/exp11_notes.md` for full details.

| Text Model | SMILES Model | Aggregator | AUC | Bal Acc Tuned | F1 Tuned |
|------------|--------------|-----------|-----|---------------|----------|
| **ClinicalBERT** | **ChemBERTa** | **Transformer** | **0.791 +/- 0.081** | 0.776 +/- 0.052 | 0.794 +/- 0.061 |
| PubMedBERT | ChemBERTa | MeanMax | 0.781 +/- 0.106 | **0.810 +/- 0.091** | **0.822 +/- 0.085** |

EEG2Vec does not improve quad modality AUC (0.791 vs 0.798 SimpleCNN), unlike the +0.049 improvement seen for triple modality (exp3a). Clinical features may already compensate for SimpleCNN's weaker EEG encoding.

## Interpretation

The substantial improvement over Exp3b (+0.121 AUC) confirms that clinical features provide meaningful additional signal when combined with all embedding modalities. Pipeline improvements (multi-label stratification, code review fixes) contributed to the overall gains.

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
