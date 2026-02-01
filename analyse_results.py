#!/usr/bin/env python3
"""Post-hoc analysis of experiment results with meta-analysis.

This script re-analyses saved experiment results using proper statistical
methods (Sidik-Jonkman meta-analysis with Knapp-Hartung adjustment) without
modifying original experiments.

Usage:
    python analyse_results.py <results.json>

Example:
    python analyse_results.py outputs/exp7_results/results_20260130_152752.json
"""

import json
import sys
from pathlib import Path

import numpy as np

from shared.stats_util import meta_analysis_sj_robust


def analyse_experiment(results_path):
    """Analyse saved experiment results with meta-analysis.

    Args:
        results_path: Path to experiment results JSON file.
    """
    with open(results_path) as f:
        results = json.load(f)

    print(f"\nAnalysing: {Path(results_path).name}")
    print("=" * 60)

    for config_name, data in results.items():
        fold_aucs = data["fold_metrics"]["auc"]
        k = len(fold_aucs)

        # Approximate variance from fold variability
        # Note: This is a conservative approximation. For precise variance,
        # DeLong's method should be applied to original predictions.
        fold_var = np.var(fold_aucs, ddof=1)
        approx_var = fold_var / k
        variances = [approx_var] * k

        # Meta-analysis
        meta = meta_analysis_sj_robust(fold_aucs, variances)

        # Original statistics
        orig_mean = data["summary"]["auc"]["mean"]
        orig_std = data["summary"]["auc"]["std"]

        print(f"\n{config_name}:")
        print(f"  Original:      {orig_mean:.3f} +/- {orig_std:.3f}")
        print(
            f"  Meta-analysis: {meta['pooled_effect']:.3f} "
            f"[{meta['ci_low']:.3f}, {meta['ci_high']:.3f}]"
        )
        print(f"  Heterogeneity: I2={meta['I2']:.1%}, tau2={meta['tau2']:.4f}")

        # Additional metrics if available
        if "balanced_acc_tuned" in data["fold_metrics"]:
            ba_mean = data["summary"]["balanced_acc_tuned"]["mean"]
            ba_std = data["summary"]["balanced_acc_tuned"]["std"]
            print(f"  Balanced Acc:  {ba_mean:.3f} +/- {ba_std:.3f}")


def main():
    """Entry point."""
    if len(sys.argv) < 2:
        print("Usage: python analyse_results.py <results.json>")
        print("\nExample:")
        print("  python analyse_results.py outputs/exp7_results/results_20260130_152752.json")
        sys.exit(1)

    results_path = sys.argv[1]
    if not Path(results_path).exists():
        print(f"Error: File not found: {results_path}")
        sys.exit(1)

    analyse_experiment(results_path)


if __name__ == "__main__":
    main()
