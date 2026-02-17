# Experiment 14: Optuna HP Tuning for Top 3 Models

**Date:** 18 February 2026 (updated)
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

After HPC resume jobs (Job 51546904 for Exp7a, Job 51546909 for Exp11), studies are effectively complete. Exp7a was cancelled at 99/100 trials due to 8h time limit (2 stale RUNNING trials). Exp11 reached 100/100 trials. Exp12 was not resumed (already well-explored at 46 trials).

| Study | Completed | Pruned | Total | Target |
|-------|-----------|--------|-------|--------|
| Exp7a QuadFusionMLP | 26 | 71 | 99 | 100 |
| Exp11 QuadMLPv2 | 27 | 72 | 100 | 100 |
| Exp12 TripleFuseMoE | 40 | 5 | 46 | 100 |
| **Total** | **93** | **148** | **245** | **300** |

Pruning rate: 60% overall (71/99 for Exp7a, 72/100 for Exp11, 5/46 for Exp12).

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

**Updated best after resume (Job 51546909): Trial #81, AUC 0.825**

| Parameter | Trial #81 Value | Trial #16 Value | Baseline |
|-----------|----------------|----------------|----------|
| learning_rate | 3.73e-3 | 7.38e-4 | 1e-3 |
| weight_decay | 7.82e-4 | 2.57e-4 | 1e-4 |
| dropout | 0.483 | 0.341 | 0.3 |
| hidden_dim | 32 | 32 | 64 |
| batch_size | 4 | 8 | 8 |
| aggregator_type | meanmax | transformer | transformer |
| eeg_embed_dim | 64 | 64 | 128 |

Trial #81 switches from transformer to meanmax aggregation and uses a much higher learning rate (3.73e-3 vs 7.38e-4) with smaller batch size (4 vs 8). The meanmax aggregator finding aligns with Exp9 extended ablation results where MeanMax was a strong alternative (Bal Acc 0.740). This trial has not yet been confirmed with a rerun.

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

| Model | Baseline AUC | Best Trial AUC | Delta | Trials (complete/pruned) |
|-------|-------------|---------------|-------|--------------------------|
| **Exp7a QuadFusionMLP** | 0.798 | **0.831** (Trial #24) | **+0.033** | 26/71 (99 total) |
| Exp11 QuadMLPv2 (EEG2Vec) | 0.791 | **0.825** (Trial #81) | **+0.034** | 27/72 (100 total) |
| Exp11 QuadMLPv2 (EEG2Vec) | 0.791 | 0.822 (Trial #16) | +0.031 | (from initial run) |
| Exp12 TripleFuseMoE | 0.760 | 0.749 (Trial #10) | -0.011 | 40/5 (46 total) |

**Important:** These are single best-trial AUCs (mean across 5 CV folds within one trial). See "Confirmed Results (Rerun)" below for actual rerun performance.

---

## Confirmed Results (Rerun)

Best configurations from the initial Optuna run were rerun with full metrics (HPC Job 51546896). **None of the Optuna-tuned configurations reproduced their trial AUC on rerun.** The gap between trial AUC and rerun AUC is 0.03-0.06, caused by training variance (different weight initialisations, data shuffling order).

### Rerun Summary

| Model | Optuna Best | Rerun AUC | Rerun Bal Acc | Rerun F1 Tuned | vs Baseline |
|-------|------------|-----------|---------------|----------------|-------------|
| Exp7a QuadFusionMLP | 0.831 | 0.770 +/- 0.053 | 0.760 +/- 0.045 | 0.759 +/- 0.049 | -0.028 vs 0.798 |
| Exp11 QuadMLPv2 (EEG2Vec) | 0.822 | 0.789 +/- 0.088 | 0.768 +/- 0.062 | 0.777 +/- 0.047 | -0.002 vs 0.791 |
| Exp12 TripleFuseMoE | 0.749 | 0.697 +/- 0.110 | 0.720 +/- 0.094 | 0.698 +/- 0.131 | -0.063 vs 0.760 |

Note: Exp11 rerun used Trial #16 params (transformer aggregator). Trial #81 (meanmax, AUC 0.825) has not yet been rerun.

### Per-Fold AUC (Rerun)

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|-------|--------|--------|--------|--------|--------|
| Exp7a QuadFusionMLP | 0.674 | 0.793 | 0.767 | 0.783 | 0.833 |
| Exp11 QuadMLPv2 | 0.674 | 0.744 | 0.792 | 0.942 | 0.792 |
| Exp12 TripleFuseMoE | 0.606 | 0.810 | 0.567 | 0.842 | 0.658 |

### Rerun Key Findings

- **Optuna trial AUCs are optimistic and do not reproduce.** The variance gap between trial AUC and rerun AUC ranges from 0.033 (Exp11) to 0.061 (Exp7a). This is expected - each trial's AUC is a single sample from the distribution of possible training runs.
- **Exp7a rerun (0.770) is worse than its baseline (0.798).** The Optuna-tuned HP do not reliably improve upon the original configuration.
- **Exp11 rerun (0.789) essentially matches its baseline (0.791).** Delta is only -0.002, within noise.
- **Exp12 rerun (0.697) is substantially worse than its baseline (0.760).** FuseMoE shows the highest variance (std 0.110) and least reliable results.
- **The original Exp7a baseline (AUC 0.798) remains the best confirmed result.**
- Exp12 FuseMoE has extreme fold variance: Fold 3 AUC 0.567 vs Fold 4 AUC 0.842 (range 0.275).

---

## Parameter Importance (fANOVA)

Hyperparameter importance scores computed using Optuna's fANOVA analysis.

### Exp7a QuadFusionMLP

| Parameter | Importance | |
|-----------|-----------|---|
| learning_rate | 0.676 | ################ |
| dropout | 0.160 | #### |
| hidden_dim | 0.102 | ## |
| batch_size | 0.040 | # |
| weight_decay | 0.022 | |

Learning rate dominates (68%), consistent with the finding that halving LR from 1e-3 to ~5e-4 was the key change.

### Exp11 QuadMLPv2 (EEG2Vec)

| Parameter | Importance | |
|-----------|-----------|---|
| aggregator_type | 0.519 | ############# |
| batch_size | 0.219 | ###### |
| learning_rate | 0.088 | ## |
| weight_decay | 0.072 | ## |
| eeg_embed_dim | 0.054 | # |
| hidden_dim | 0.026 | # |
| dropout | 0.023 | |

Aggregator type (transformer vs meanmax) is by far the most important parameter (52%). This aligns with Trial #81 switching to meanmax and achieving the new best (AUC 0.825). Batch size is unexpectedly the second most important parameter (22%).

### Exp12 TripleFuseMoE

| Parameter | Importance | |
|-----------|-----------|---|
| weight_decay | 0.293 | ####### |
| learning_rate | 0.234 | ###### |
| dropout | 0.206 | ##### |
| temp_decay | 0.109 | ### |
| aux_loss_weight | 0.061 | ## |
| num_experts | 0.054 | # |
| top_k | 0.044 | # |

FuseMoE importance is more evenly distributed - no single parameter dominates. The top three (weight_decay, learning_rate, dropout) account for 73% and are all continuous regularisation parameters, suggesting FuseMoE performance is highly sensitive to the regularisation-capacity tradeoff.

---

## Key Observations

1. **Optuna trial AUCs are optimistic and do not reproduce.** Rerunning the best Optuna configurations yields AUCs 0.03-0.06 lower than the trial values. Exp7a rerun (0.770) is worse than its baseline (0.798), Exp11 rerun (0.789) matches its baseline (0.791), and Exp12 rerun (0.697) is substantially worse than its baseline (0.760). The original Exp7a baseline (AUC 0.798) remains the best confirmed result.

2. **Lower learning rates benefit MLP models:** Both Exp7a and Exp11 best trials used lower learning rates (~5e-4 to 7e-4 vs the 1e-3 baseline). However, this did not translate to reproducible improvements on rerun.

3. **Weight decay reduction helps Exp7a:** Reducing weight decay by ~4x (2.73e-5 vs 1e-4) alongside the lower LR was found by Optuna, but the combined effect is not reliably reproduced.

4. **Aggregator type is the most important parameter for Exp11:** fANOVA shows aggregator_type accounts for 52% of variance. Trial #81 (AUC 0.825) switches from transformer to meanmax, consistent with Exp9 extended ablation findings. This trial has not yet been rerun.

5. **FuseMoE already near-optimal from Exp12 grid search:** With 40 completed trials, Optuna could not improve upon the Exp12 best configuration (AUC 0.760 vs 0.749). The rerun (0.697) confirms FuseMoE is the least reliable architecture.

6. **MedianPruner is highly effective:** 60% of trials were pruned overall (148/245). Exp7a and Exp11 had pruning rates of 72%, indicating the TPE sampler explored many configurations that were quickly identified as unpromising.

7. **Studies effectively complete:** Exp7a reached 99/100 trials (cancelled at 8h time limit), Exp11 completed 100/100 trials. Best trials were found early (Trial #24 for Exp7a, Trial #81 for Exp11), suggesting diminishing returns from additional trials.

8. **Training variance is the dominant source of uncertainty.** The gap between Optuna trial AUC and rerun AUC (0.033-0.061) is larger than the Optuna improvement over baseline (0.031-0.033). This confirms that on small datasets (n=107), single-trial optimisation cannot overcome inherent training variance.

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

1. ~~Rerun best Exp7a and Exp11 configurations with full metrics~~ **DONE** - Reruns complete (Job 51546896). Optuna AUCs not reproduced; baseline Exp7a (0.798) remains best confirmed result.
2. ~~Resume studies to complete remaining trials~~ **DONE** - Exp11 100/100 complete (Job 51546909). Exp7a 99/100, cancelled at 8h time limit (Job 51546904).
3. ~~Parameter importance analysis (fANOVA)~~ **DONE** - learning_rate most important for Exp7a (0.676), aggregator_type most important for Exp11 (0.519), weight_decay most important for Exp12 (0.293).
4. Rerun Exp11 Trial #81 parameters (meanmax aggregator, lr=3.73e-3, batch_size=4) - new best from resumed study (AUC 0.825), not yet confirmed.
5. Final model selection based on confirmed rerun results.
