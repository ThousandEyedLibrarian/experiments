# Experiment 7: All Four Modalities Fusion

## Overview

Experiment 7 combines all four available modalities to predict ASM treatment outcomes:
- **Clinical features** (19D): Demographics, medical history
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
Clinical (19D) ───> Encoder -> 64D ─┐
                                     │
Text (768D) ──────> Encoder -> 64D ─┼─> Concat (256D) -> Classifier -> 2
                                     │
EEG (windows) ────> CNN+Trf -> 64D ─┤
                                     │
SMILES (768D) ────> Encoder -> 64D ─┘
```

### Exp7b: FuseMoE (~4.7M params)

```
Clinical (19D) ───> Projection -> 256D ─┐
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
| exp7b_pubmedbert | PubMedBERT | MoE (Exp12 HP) | 0.738 | 0.084 | 0.737 | 0.062 |

*Updated 17 February 2026. exp7a: pipeline improvements (multi-label stratification, code review fixes). exp7b ClinicalBERT: revised FuseMoE (Laplace gating, MI loss, temperature annealing). exp7b PubMedBERT: Exp12 tuned HP (lr=5e-5, 4 experts, no temp decay). Previous: exp7a ClinicalBERT AUC 0.762, exp7b PubMedBERT AUC 0.712.*

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

6. **Exp12 HP benefits PubMedBERT FuseMoE (+0.026) but not ClinicalBERT (-0.007)**
   - Consistent with the pattern seen across all FuseMoE experiments
   - PubMedBERT FuseMoE now at 0.738, narrowing gap with ClinicalBERT MoE (0.753)
   - ClinicalBERT variance reduction (0.127 -> 0.098) may be valuable for deployment stability

### EEG2Vec Upgrade (Exp11, 15 February 2026)

Exp11 replaced SimpleCNN with EEG2Vec encoder (128D embeddings) for the exp7a MLP architecture. See `findings/exp11_notes.md` for full details.

| Text Model | SMILES Model | Aggregator | AUC | Bal Acc Tuned | F1 Tuned |
|------------|--------------|-----------|-----|---------------|----------|
| **ClinicalBERT** | **ChemBERTa** | **Transformer** | **0.791 +/- 0.081** | 0.776 +/- 0.052 | 0.794 +/- 0.061 |
| PubMedBERT | ChemBERTa | MeanMax | 0.781 +/- 0.106 | **0.810 +/- 0.091** | **0.822 +/- 0.085** |

EEG2Vec does not improve quad modality AUC (0.791 vs 0.798 SimpleCNN), unlike the +0.049 improvement seen for triple modality (exp3a). Clinical features may already compensate for SimpleCNN's weaker EEG encoding.

### Exp7b Results: Exp12 Tuned HP (17 February 2026)

Applied Exp12 best hyperparameters (lr=5e-5, 4 experts, no temp decay) to exp7b FuseMoE.

| Text Model | Fusion | AUC | Std | Bal Acc Tuned | Std | F1 Tuned | Std |
|------------|--------|-----|-----|---------------|-----|----------|-----|
| ClinicalBERT | MoE (Exp12 HP) | 0.746 | 0.098 | 0.779 | 0.078 | 0.788 | 0.077 |
| **PubMedBERT** | **MoE (Exp12 HP)** | **0.738** | **0.084** | **0.737** | **0.062** | **0.721** | **0.073** |

**Per-Fold AUC (Exp12 HP):**

| Fold | ClinicalBERT AUC | ClinicalBERT Bal Acc | PubMedBERT AUC | PubMedBERT Bal Acc |
|------|-------------------|---------------------|----------------|-------------------|
| 1 | 0.636 | 0.693 | 0.614 | 0.659 |
| 2 | 0.711 | 0.727 | 0.678 | 0.682 |
| 3 | 0.683 | 0.742 | 0.750 | 0.733 |
| 4 | 0.917 | 0.908 | 0.842 | 0.825 |
| 5 | 0.783 | 0.825 | 0.808 | 0.783 |

**Comparison with default revised FuseMoE HP:**

| Text Model | Default HP AUC | Exp12 HP AUC | Delta | Std Change |
|------------|---------------|-------------|-------|------------|
| ClinicalBERT | 0.753 +/- 0.127 | 0.746 +/- 0.098 | -0.007 | -0.029 |
| PubMedBERT | 0.712 +/- 0.072 | 0.738 +/- 0.084 | **+0.026** | +0.012 |

**Observations:**
- PubMedBERT FuseMoE improves substantially (+0.026 AUC) - moves from 8th to 7th in global comparison table
- ClinicalBERT FuseMoE marginally declines in AUC (-0.007) but variance reduces meaningfully (0.127 -> 0.098)
- PubMedBERT FuseMoE (0.738) now surpasses Exp9 EEG2Vec ablation (0.730) in ranking
- ClinicalBERT MLP (0.798) still clearly outperforms both FuseMoE variants
- Best exp7b configuration remains ClinicalBERT at 0.753 (default HP) for peak AUC
- For PubMedBERT, Exp12 HP is unambiguously better (0.738 vs 0.712)

## Interpretation

The substantial improvement over Exp3b (+0.121 AUC) confirms that clinical features provide meaningful additional signal when combined with all embedding modalities. Pipeline improvements (multi-label stratification, code review fixes) contributed to the overall gains.

## Training Configuration

| Parameter | MLP (7a) | MoE (7b, default) | MoE (7b, Exp12 HP) |
|-----------|----------|----------|----------|
| Learning rate | 1e-3 | 5e-4 | 5e-5 |
| Batch size | 8 | 8 | 8 |
| Epochs | 100 | 100 | 100 |
| Early stopping | 20 | 20 | 20 |
| Weight decay | 1e-4 | 1e-4 | 1e-4 |
| Num experts | - | 4 | 4 |
| Temp decay | - | 0.9995 | None |

## Files

- `exp7_all_modalities/run_experiments.py` - Entry point
- `exp7_all_modalities/models.py` - QuadFusionMLP, QuadFusionMoE
- `exp7_all_modalities/data_pipeline.py` - QuadModalityDataset
- `exp7_all_modalities/training.py` - Training loop with CV
- `outputs/exp7_results/results_20260130_152752.json` - Results
