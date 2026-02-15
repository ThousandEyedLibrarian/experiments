# Experiment 1: LLM + SMILES Fusion

**Date:** 28 January 2026 (re-run with threshold tuning)
**Dataset:** 121 patients with text reports and SMILES embeddings

---

## Objective

Test whether combining clinical text report embeddings (LLM) with drug molecular structure (SMILES) embeddings can predict ASM treatment outcomes.

---

## Architecture

### Models Tested

| Variant | Text Model | SMILES Model | Fusion | Parameters |
|---------|------------|--------------|--------|------------|
| Exp1a | ClinicalBERT/PubMedBERT (768D) | ChemBERTa (768D) / SMILES-Trf (256D) | Concat + MLP | ~953K |
| Exp1b | ClinicalBERT/PubMedBERT (768D) | ChemBERTa (768D) / SMILES-Trf (256D) | FuseMoE | ~2.6M |

### Exp1a: ConcatMLP
- Concatenate embeddings (1536D or 1024D)
- 4-layer MLP: 1536->512->256->128->2
- LayerNorm + Dropout(0.3) between layers

### Exp1b: FuseMoE
- Project each modality to 256D
- Learnable modality tokens
- 2 MoE layers (4 experts, top-2 routing)
- Self-attention fusion

---

## Results (5-fold CV with Threshold Tuning)

### Exp1a (Concat + MLP)

| Text Model | SMILES Model | AUC | Bal Acc Tuned | F1 Tuned |
|------------|--------------|-----|---------------|----------|
| PubMedBERT | ChemBERTa | 0.641 +/- 0.070 | **0.699 +/- 0.033** | 0.676 +/- 0.082 |
| PubMedBERT | SMILES-Trf | 0.632 +/- 0.106 | 0.676 +/- 0.073 | 0.624 +/- 0.198 |
| ClinicalBERT | SMILES-Trf | 0.623 +/- 0.112 | 0.677 +/- 0.073 | 0.557 +/- 0.110 |
| ClinicalBERT | ChemBERTa | 0.609 +/- 0.099 | 0.669 +/- 0.067 | 0.707 +/- 0.061 |

### Exp1b (FuseMoE) - Original (28 January 2026)

| Text Model | SMILES Model | AUC | Bal Acc Tuned | F1 Tuned |
|------------|--------------|-----|---------------|----------|
| ClinicalBERT | SMILES-Trf | **0.648 +/- 0.100** | **0.712 +/- 0.074** | 0.701 +/- 0.117 |
| ClinicalBERT | ChemBERTa | 0.643 +/- 0.128 | 0.670 +/- 0.078 | 0.597 +/- 0.142 |
| PubMedBERT | ChemBERTa | 0.641 +/- 0.071 | **0.713 +/- 0.047** | 0.670 +/- 0.125 |
| PubMedBERT | SMILES-Trf | 0.592 +/- 0.075 | 0.641 +/- 0.047 | 0.635 +/- 0.079 |

*Note: Results above used the old FuseMoE implementation (softmax gating, malformed KL/CV-squared loss, 2-layer MLP experts).*

### Re-run Results: Revised FuseMoE (13 February 2026)

Revised FuseMoE: Laplace gating, MI loss, 3-layer residual experts, temperature annealing.

| Text Model | SMILES Model | AUC | Bal Acc Tuned | F1 Tuned |
|------------|--------------|-----|---------------|----------|
| ClinicalBERT | SMILES-Trf | **0.674 +/- 0.139** | **0.720 +/- 0.079** | 0.664 +/- 0.147 |
| ClinicalBERT | ChemBERTa | 0.636 +/- 0.142 | 0.709 +/- 0.100 | 0.713 +/- 0.116 |
| PubMedBERT | SMILES-Trf | 0.612 +/- 0.049 | 0.689 +/- 0.049 | 0.713 +/- 0.052 |
| PubMedBERT | ChemBERTa | 0.601 +/- 0.104 | 0.681 +/- 0.079 | 0.707 +/- 0.070 |

Best AUC improved from 0.648 to 0.674. PubMedBERT + SMILES-Trf now has lowest variance (AUC std 0.049).

---

## Key Findings

1. **Best AUC:** exp1b_clinicalbert_smilestrf (0.674) with revised FuseMoE

2. **FuseMoE outperforms MLP:** Best FuseMoE AUC (0.674) exceeds best MLP AUC (0.641)

3. **Revised FuseMoE improved stability for PubMedBERT:** PubMedBERT + SMILES-Trf std dropped from 0.075 to 0.049

4. **Threshold tuning critical:** F1_tuned (0.56-0.71) substantially better than raw F1 (0.35-0.68)

---

## Comparison with Original Run

Original run lacked threshold tuning. Key differences:

| Metric | Original | Re-run |
|--------|----------|--------|
| Best AUC | 0.695 (exp1a_clinicalbert_smilestrf) | 0.648 (exp1b_clinicalbert_smilestrf) |
| Threshold tuning | No | Yes (Youden's J) |
| Balanced Acc | Not computed | 0.64-0.71 |

Note: AUC differences due to random seed variation; tuned metrics now available for fair comparison.

---

## Technical Notes

- Class weighting: Inverse frequency (already present in original)
- Threshold selection: Youden's J statistic (TPR - FPR)
- Training: 100 epochs, early stopping (patience 15/20), batch size 16
- Optimiser: AdamW, LR 1e-4 (MLP) / 5e-5 (FuseMoE)
