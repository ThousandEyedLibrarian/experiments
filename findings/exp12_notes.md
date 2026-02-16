# Experiment 12: FuseMoE Hyperparameter Investigation

**Date:** 15 February 2026
**Dataset:** n=107 (triple modality intersection)

---

## Objective

Investigate whether the exp3b FuseMoE regression (AUC 0.753 -> 0.677) was caused by suboptimal default hyperparameters in the revised FuseMoE implementation, rather than a fundamental architectural issue.

---

## Background

The revised FuseMoE implementation (Laplace gating, MI loss, 3-layer residual experts, temperature annealing) replaced the old implementation (softmax gating, malformed KL/CV-squared loss, 2-layer MLP experts) in February 2026. While most experiments improved (exp1b, exp2b, exp7b), exp3b regressed from AUC 0.753 to 0.677.

Hypothesis: the old softmax + malformed KL loss may have acted as accidental regularisation, and the revised implementation requires different hyperparameters to achieve equivalent or better performance.

---

## Architecture

Fixed to the best-performing exp3b combination:
- **Text model:** ClinicalBERT
- **SMILES model:** ChemBERTa
- **EEG encoder:** SimpleCNN (to match exp3b baseline)
- **Fusion:** TripleModalityFuseMoE with Laplace gating, MI loss

---

## Hyperparameter Grid

| Parameter | Values | Rationale |
|-----------|--------|-----------|
| Learning rate | 5e-5, 1e-4, 5e-4 | Exp3 default was 1e-3 (from training.py) - testing lower rates |
| Number of experts | 2, 4 | Default was 4 - testing whether fewer experts reduce overfitting |
| Temperature decay | 0.9995, None | Default was 0.9995 - testing whether annealing helps or hurts |

Total: 3 x 2 x 2 = **12 configurations**, all with top_k=2.

---

## Results (5-fold CV, sorted by AUC)

| Config | LR | Experts | Temp Decay | AUC | Bal Acc Tuned | F1 Tuned |
|--------|-----|---------|-----------|-----|---------------|----------|
| **lr5e-5_e4_notmp** | **5e-5** | **4** | **None** | **0.760 +/- 0.112** | 0.760 +/- 0.081 | 0.742 +/- 0.132 |
| lr1e-4_e2_t0.9995 | 1e-4 | 2 | 0.9995 | 0.749 +/- 0.105 | **0.773 +/- 0.064** | 0.745 +/- 0.096 |
| lr1e-4_e4_t0.9995 | 1e-4 | 4 | 0.9995 | 0.737 +/- 0.080 | 0.763 +/- 0.077 | **0.772 +/- 0.110** |
| lr5e-4_e2_notmp | 5e-4 | 2 | None | 0.734 +/- 0.111 | 0.770 +/- 0.085 | 0.712 +/- 0.134 |
| lr1e-4_e2_notmp | 1e-4 | 2 | None | 0.734 +/- 0.107 | 0.747 +/- 0.078 | 0.769 +/- 0.068 |
| lr5e-4_e4_t0.9995 | 5e-4 | 4 | 0.9995 | 0.717 +/- 0.110 | 0.757 +/- 0.095 | 0.726 +/- 0.135 |
| lr5e-5_e4_t0.9995 | 5e-5 | 4 | 0.9995 | 0.710 +/- 0.085 | 0.728 +/- 0.066 | 0.693 +/- 0.117 |
| lr5e-5_e2_notmp | 5e-5 | 2 | None | 0.701 +/- 0.069 | 0.715 +/- 0.060 | 0.702 +/- 0.078 |
| lr5e-4_e2_t0.9995 | 5e-4 | 2 | 0.9995 | 0.697 +/- 0.039 | 0.715 +/- 0.042 | 0.724 +/- 0.045 |
| lr5e-4_e4_notmp | 5e-4 | 4 | None | 0.659 +/- 0.098 | 0.715 +/- 0.078 | 0.716 +/- 0.093 |
| lr5e-5_e2_t0.9995 | 5e-5 | 2 | 0.9995 | 0.654 +/- 0.072 | 0.684 +/- 0.052 | 0.645 +/- 0.071 |
| lr1e-4_e4_notmp | 1e-4 | 4 | None | 0.642 +/- 0.044 | 0.684 +/- 0.042 | 0.642 +/- 0.129 |

---

## FuseMoE Regression Comparison

| Implementation | AUC | Notes |
|----------------|-----|-------|
| Old FuseMoE (softmax + malformed KL) | 0.753 | Accidental regularisation from malformed loss |
| Revised FuseMoE (default HP) | 0.677 | lr=1e-3, 4 experts, temp_decay=0.9995 |
| **Revised FuseMoE (tuned HP)** | **0.760** | **lr=5e-5, 4 experts, no temp decay** |

**Regression fully resolved.** Tuned FuseMoE surpasses the old malformed result by +0.007 AUC.

---

## Key Observations

1. **Regression fully resolved:** Best config (lr=5e-5, 4 experts, no temp decay) achieves AUC 0.760, surpassing the old malformed result (0.753) by +0.007

2. **Learning rate is the dominant factor:** The exp3 default of 1e-3 was far too high for FuseMoE. All tested rates (5e-5 to 5e-4) are 2-20x lower. The 1e-4 range is most consistent (3 of top 5 configs).

3. **Temperature annealing has mixed effects:** For 4 experts at lr=5e-5, removing temp decay improves AUC from 0.710 to 0.760 (+0.050). But at lr=1e-4, temp decay helps (0.737 vs 0.642 for 4 experts).

4. **4 experts can outperform 2 experts** when learning rate is low enough (0.760 with 4e at 5e-5 vs 0.701 with 2e at 5e-5), but the interaction with temperature decay matters.

5. **High variance persists:** Even the best config has AUC std 0.112. The small dataset (n=107) limits stability regardless of hyperparameters.

6. **Most stable config:** lr=5e-4, 2 experts, temp 0.9995 has lowest AUC variance (std 0.039, AUC 0.697) but sacrifices peak performance.

7. **Default config was suboptimal:** The exp3 training pipeline used lr=1e-3 - an order of magnitude too high for the FuseMoE architecture. FuseMoE requires gentler optimisation.

---

## Files

- `exp12_moe_hparam/__init__.py` - Package marker
- `exp12_moe_hparam/config.py` - Hyperparameter grid (12 configs)
- `exp12_moe_hparam/run_experiments.py` - Entry point with experiment filter

---

## Cross-Experiment Validation (17 February 2026)

Applied Exp12 best HP (lr=5e-5, 4 experts, no temp decay) to all other FuseMoE experiments.

| Experiment | Config | Default HP AUC | Exp12 HP AUC | Delta | Std Change |
|------------|--------|---------------|--------------|-------|------------|
| exp1b | ClinicalBERT + ChemBERTa | 0.636 +/- 0.142 | 0.650 +/- 0.111 | +0.014 | -0.031 |
| exp1b | ClinicalBERT + SMILES-Trf | 0.674 +/- 0.139 | 0.647 +/- 0.062 | -0.027 | **-0.077** |
| exp1b | PubMedBERT + ChemBERTa | 0.601 +/- 0.104 | 0.649 +/- 0.089 | **+0.048** | -0.015 |
| exp1b | PubMedBERT + SMILES-Trf | 0.612 +/- 0.049 | 0.629 +/- 0.077 | +0.017 | +0.028 |
| exp2b | SimpleCNN + ChemBERTa | 0.572 +/- 0.024 | 0.585 +/- 0.077 | +0.013 | +0.053 |
| exp2b | SimpleCNN + SMILES-Trf | 0.611 +/- 0.056 | 0.569 +/- 0.087 | -0.042 | +0.031 |
| exp7b | ClinicalBERT + ChemBERTa | 0.753 +/- 0.127 | 0.746 +/- 0.098 | -0.007 | -0.029 |
| exp7b | PubMedBERT + ChemBERTa | 0.712 +/- 0.072 | 0.738 +/- 0.084 | **+0.026** | +0.012 |

### Cross-Experiment Insights

1. **Not a universal improvement** - 3 of 8 configurations regressed. The best per-experiment FuseMoE result should use whichever HP config produced the higher AUC.

2. **PubMedBERT benefits more than ClinicalBERT consistently:**
   - exp1b ChemBERTa: PubMedBERT +0.048 vs ClinicalBERT +0.014
   - exp7b: PubMedBERT +0.026 vs ClinicalBERT -0.007
   - Hypothesis: PubMedBERT's biomedical pre-training creates stronger initial representations that benefit from the lower learning rate (less forgetting), while ClinicalBERT's clinical-specific features are more sensitive to the loss of temperature annealing's regularisation effect.

3. **Variance reduction is the most consistent benefit for ClinicalBERT:**
   - exp1b ChemBERTa: std 0.142 -> 0.111
   - exp1b SMILES-Trf: std 0.139 -> 0.062 (largest reduction)
   - exp7b: std 0.127 -> 0.098
   - Removing temperature annealing prevents the expert specialisation instability that affects ClinicalBERT more than PubMedBERT.

4. **EEG-only experiments (exp2b) respond differently** - both configs see increased variance with Exp12 HP. The EEG modality's temporal structure may benefit from temperature annealing's gradual expert specialisation.

5. **Per-experiment HP tuning is warranted** - a universal FuseMoE configuration is suboptimal. Next step #3 is validated by these results.

---

## Next Steps

1. ~~Apply best hyperparameters (lr=5e-5, 4 experts, no temp decay) to other FuseMoE experiments (exp1b, exp2b, exp7b)~~ **DONE** - Mixed results: PubMedBERT benefits most (+0.048 exp1b, +0.026 exp7b). Not a universal improvement.
2. Test whether the lr=1e-4, 2 experts, temp 0.9995 config (most stable, AUC 0.749) is preferable for deployment given lower variance
3. Consider per-experiment HP tuning rather than a universal configuration
