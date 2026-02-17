# Experiment 14: Optuna HP Tuning for Top 3 Models

**Date:** 17 February 2026
**Dataset:** n=107 (quad modality intersection)

---

## Objective

Systematic hyperparameter tuning using Optuna's TPE sampler to improve the top 3 models beyond their baseline AUCs. The top 3 models were selected from the cross-experiment comparison table: Exp7a QuadFusionMLP (AUC 0.798), Exp11 QuadMLPv2 with EEG2Vec (AUC 0.791), and Exp12 TripleFuseMoE (AUC 0.760).

---

## Background

All prior hyperparameters were either manually set (default training pipeline values) or grid-searched (Exp12's 12-configuration grid). Optuna enables Bayesian optimisation via Tree-structured Parzen Estimators (TPE), which is more sample-efficient than grid search, and supports early stopping via median pruning to discard unpromising trials.

---

## Architecture

- **Sampler:** TPE (Tree-structured Parzen Estimator), seed=42
- **Pruner:** MedianPruner (n_startup_trials=10, n_warmup_steps=2)
- **Direction:** Maximise AUC-ROC
- **Cross-validation:** 5-fold stratified, shuffle=True, random_state=42
- **Training fixed:** 100 epochs max, patience=20, grad_clip_norm=1.0
- **Trial budget:** 100 per model (interrupted early - see Trial Statistics)
- **Storage:** SQLite (`outputs/exp14_results/optuna_studies.db`), resumable

---

## Search Spaces

### Exp7a QuadFusionMLP (Clinical + LLM + EEG + SMILES)

Fixed: ClinicalBERT text, ChemBERTa SMILES, EEG2Vec encoder.

| Parameter | Type | Range | Log-scale |
|-----------|------|-------|-----------|
| learning_rate | float | [5e-4, 5e-3] | Yes |
| weight_decay | float | [1e-5, 1e-3] | Yes |
| dropout | float | [0.1, 0.5] | No |
| hidden_dim | categorical | {32, 64, 128} | - |
| batch_size | categorical | {4, 8, 16} | - |

### Exp11 QuadMLPv2 with EEG2Vec (Clinical + LLM + EEG + SMILES)

Fixed: ClinicalBERT text, ChemBERTa SMILES, EEG2Vec encoder type.

| Parameter | Type | Range | Log-scale |
|-----------|------|-------|-----------|
| learning_rate | float | [5e-4, 5e-3] | Yes |
| weight_decay | float | [1e-5, 1e-3] | Yes |
| dropout | float | [0.1, 0.5] | No |
| hidden_dim | categorical | {32, 64, 128} | - |
| batch_size | categorical | {4, 8, 16} | - |
| aggregator_type | categorical | {transformer, meanmax} | - |
| eeg_embed_dim | categorical | {64, 128, 256} | - |

### Exp12 TripleFuseMoE (LLM + EEG + SMILES)

Fixed: ClinicalBERT text, ChemBERTa SMILES, SimpleCNN EEG, hidden_dim=256, num_heads=4, batch_size=8.

| Parameter | Type | Range | Log-scale |
|-----------|------|-------|-----------|
| learning_rate | float | [1e-5, 5e-4] | Yes |
| weight_decay | float | [1e-5, 1e-3] | Yes |
| dropout | float | [0.05, 0.3] | No |
| num_experts | categorical | {2, 4, 6} | - |
| top_k | categorical | {1, 2} | - |
| aux_loss_weight | float | [0.01, 0.5] | Yes |
| temp_decay | categorical | {None, 0.999, 0.9995, 0.9999} | - |

---

## Trial Statistics

All three studies were interrupted before reaching the 100-trial budget (each has 1 stale RUNNING trial from the interruption).

| Study | Completed | Pruned | Total | Target |
|-------|-----------|--------|-------|--------|
| Exp7a QuadFusionMLP | 17 | 10 | 28 | 100 |
| Exp11 QuadMLPv2 | 16 | 15 | 32 | 100 |
| Exp12 TripleFuseMoE | 40 | 5 | 46 | 100 |
| **Total** | **73** | **30** | **106** | **300** |

Pruning rate: 29% overall (10/28 for Exp7a, 15/32 for Exp11, 5/46 for Exp12).

---

## Results

### Exp7a QuadFusionMLP - Best: Trial #24, AUC 0.831

| Parameter | Best Value | Baseline |
|-----------|-----------|----------|
| learning_rate | 5.29e-4 | 1e-3 |
| weight_decay | 2.73e-5 | 1e-4 |
| dropout | 0.277 | 0.3 |
| hidden_dim | 64 | 64 |
| batch_size | 8 | 8 |

Completed trial distribution: mean=0.776, std=0.031, min=0.725, max=0.831.

The biggest change from baseline was halving the learning rate (5.29e-4 vs 1e-3) and reducing weight decay by ~4x (2.73e-5 vs 1e-4). Hidden dim and batch size stayed at baseline values.

### Exp11 QuadMLPv2 (EEG2Vec) - Best: Trial #16, AUC 0.822

| Parameter | Best Value | Baseline |
|-----------|-----------|----------|
| learning_rate | 7.38e-4 | 1e-3 |
| weight_decay | 2.57e-4 | 1e-4 |
| dropout | 0.341 | 0.3 |
| hidden_dim | 32 | 64 |
| batch_size | 8 | 8 |
| aggregator_type | transformer | transformer |
| eeg_embed_dim | 64 | 128 |

Completed trial distribution: mean=0.775, std=0.033, min=0.690, max=0.822.

Smaller hidden_dim (32 vs 64) and smaller eeg_embed_dim (64 vs 128) suggest the baseline was slightly over-parameterised. Aggregator stayed as transformer.

### Exp12 TripleFuseMoE - Best: Trial #10, AUC 0.749

| Parameter | Best Value | Baseline |
|-----------|-----------|----------|
| learning_rate | 1.04e-4 | 5e-5 |
| weight_decay | 6.22e-5 | 1e-4 |
| dropout | 0.052 | 0.1 |
| num_experts | 6 | 4 |
| top_k | 1 | 2 |
| aux_loss_weight | 0.032 | 0.1 |
| temp_decay | None | None |

Completed trial distribution: mean=0.681, std=0.028, min=0.631, max=0.749.

FuseMoE's best tuned result (0.749) is worse than its Exp12 grid-search baseline (0.760). With 40 completed trials (the most of any study), this is reasonably well-explored. The model prefers more experts (6 vs 4), fewer active experts (top_k=1 vs 2), and lower aux loss weight - all suggesting a sparser, less regularised configuration, but this still cannot match the baseline AUC.

---

## Baseline Comparison

| Model | Baseline AUC | Tuned AUC | Delta | Trials (complete/pruned) |
|-------|-------------|-----------|-------|--------------------------|
| **Exp7a QuadFusionMLP** | 0.798 | **0.831** | **+0.033** | 17/10 (28 total) |
| Exp11 QuadMLPv2 (EEG2Vec) | 0.791 | **0.822** | **+0.031** | 16/15 (32 total) |
| Exp12 TripleFuseMoE | 0.760 | 0.749 | -0.011 | 40/5 (46 total) |

**New best overall result:** Exp7a QuadFusionMLP with Optuna-tuned HP achieves AUC 0.831, up from 0.798 (+0.033).

---

## Key Observations

1. **New overall best result:** Exp7a MLP achieves AUC 0.831, surpassing the previous best of 0.798 - the first model to exceed 0.8 AUC.

2. **Lower learning rates benefit MLP models:** Both Exp7a and Exp11 improved by roughly halving the learning rate (~5e-4 to 7e-4 vs the 1e-3 baseline). This mirrors the Exp12 grid search finding that 1e-3 was too high for FuseMoE.

3. **Weight decay reduction helps Exp7a:** Reducing weight decay by ~4x (2.73e-5 vs 1e-4) alongside the lower LR suggests the baseline was over-regularised for the MLP architecture.

4. **Smaller dimensions sufficient for Exp11:** Hidden dim 32 (vs 64) and EEG embed dim 64 (vs 128) improved Exp11, suggesting the baseline was slightly over-parameterised. The transformer aggregator remained optimal.

5. **FuseMoE already near-optimal from Exp12 grid search:** With 40 completed trials, Optuna could not improve upon the Exp12 best configuration (AUC 0.760 vs 0.749). The Exp12 grid search was more targeted and already found a strong configuration.

6. **MedianPruner is effective:** 29% of trials were pruned overall, saving computational time. Exp11 had the highest pruning rate (47%), likely due to its larger search space (7 parameters vs 5 for Exp7a).

7. **Studies interrupted before target:** All three studies ran only 28-46% of the 100-trial budget. Exp7a and Exp11 may have further room for improvement with additional trials, though the FuseMoE study appears to have converged.

8. **Single best-trial caveat:** These AUC values are from the best single trial (mean across 5 folds within that trial), not confirmed reruns. Rerunning the best configurations with full metrics is recommended.

---

## Files

- `exp14_optuna_tuning/__init__.py` - Package marker
- `exp14_optuna_tuning/__main__.py` - Entry point (`python -m exp14_optuna_tuning`)
- `exp14_optuna_tuning/config.py` - Search spaces, paths, baselines
- `exp14_optuna_tuning/objectives.py` - Three objective functions
- `exp14_optuna_tuning/run_tuning.py` - CLI runner with argparse
- `exp14_optuna_tuning/analyse_results.py` - Results analysis and rerun script
- `outputs/exp14_results/optuna_studies.db` - SQLite database with trial results

---

## Next Steps

1. Rerun best Exp7a and Exp11 configurations with full metrics (balanced accuracy, F1, per-fold breakdowns) to confirm improvements
2. Resume studies to complete remaining trials (particularly Exp7a at 17/100 and Exp11 at 16/100)
3. Parameter importance analysis (Optuna `get_param_importances()`) to identify which hyperparameters matter most
4. Final model selection based on confirmed rerun results
