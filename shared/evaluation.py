"""Shared evaluation utilities for all experiments.

This module provides high-level wrappers for cross-validation evaluation
with proper statistical inference (DeLong confidence intervals, meta-analysis).
"""

import numpy as np

from .stats_util import (
    compute_classification_metrics,
    delong_ci,
    choose_threshold_max_ba,
    meta_analysis_sj_robust,
)


def evaluate_fold(y_true, y_prob, tune_threshold=True):
    """Evaluate a single fold with DeLong CI and optional threshold tuning.

    Args:
        y_true: True binary labels (0/1).
        y_prob: Predicted probabilities for the positive class.
        tune_threshold: If True, use Youden's J to find optimal threshold.

    Returns:
        dict with:
            - auc, auc_var, auc_ci_low, auc_ci_high: AUC with DeLong variance
            - threshold: Decision threshold used
            - accuracy, f1, precision, recall, balanced_accuracy: Metrics at threshold
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # DeLong CI for AUC
    auc, ci_low, ci_high, std, var = delong_ci(y_true, y_prob)

    # Optimal threshold via Youden's J or default 0.5
    threshold = choose_threshold_max_ba(y_true, y_prob) if tune_threshold else 0.5

    # Classification metrics at threshold
    metrics = compute_classification_metrics(y_true, y_prob, threshold)

    return {
        "auc": auc,
        "auc_var": var,
        "auc_ci_low": ci_low,
        "auc_ci_high": ci_high,
        "threshold": threshold,
        **metrics,
    }


def aggregate_folds(fold_results):
    """Aggregate fold results using Sidik-Jonkman meta-analysis.

    Uses the Sidik-Jonkman tau^2 estimator with Knapp-Hartung adjustment
    for robust confidence intervals that account for between-fold heterogeneity.

    Args:
        fold_results: List of dicts from evaluate_fold().

    Returns:
        dict with:
            - auc_pooled, auc_pooled_se: Pooled AUC estimate and SE
            - auc_ci_low, auc_ci_high: 95% CI from meta-analysis
            - auc_tau2, auc_I2: Heterogeneity statistics
            - {metric}_mean, {metric}_std: Simple aggregation for other metrics
    """
    # Extract per-fold AUC and variance for meta-analysis
    aucs = [f["auc"] for f in fold_results]
    variances = [f["auc_var"] for f in fold_results]

    # Random-effects meta-analysis with Knapp-Hartung adjustment
    meta = meta_analysis_sj_robust(aucs, variances)

    aggregated = {
        "auc_pooled": meta["pooled_effect"],
        "auc_pooled_se": meta["pooled_se"],
        "auc_ci_low": meta["ci_low"],
        "auc_ci_high": meta["ci_high"],
        "auc_tau2": meta["tau2"],
        "auc_I2": meta["I2"],
        "auc_Q": meta["Q"],
        "k": meta["k"],
    }

    # Simple mean/std for other metrics
    other_metrics = ["accuracy", "f1", "precision", "recall", "balanced_accuracy"]
    for m in other_metrics:
        values = [f[m] for f in fold_results]
        aggregated[f"{m}_mean"] = float(np.mean(values))
        aggregated[f"{m}_std"] = float(np.std(values))

    # Threshold statistics
    thresholds = [f["threshold"] for f in fold_results]
    aggregated["threshold_mean"] = float(np.mean(thresholds))
    aggregated["threshold_std"] = float(np.std(thresholds))

    return aggregated


def format_auc_result(aggregated):
    """Format aggregated AUC result for reporting.

    Args:
        aggregated: Dict from aggregate_folds().

    Returns:
        Formatted string like "0.762 [0.651, 0.873], I2=45.2%"
    """
    return (
        f"{aggregated['auc_pooled']:.3f} "
        f"[{aggregated['auc_ci_low']:.3f}, {aggregated['auc_ci_high']:.3f}], "
        f"I2={aggregated['auc_I2']:.1%}"
    )
