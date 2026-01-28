"""Run Experiment 4 configurations with cross-validation.

Usage:
    python -m exp4_baseline.run_experiments
    python -m exp4_baseline.run_experiments --model mlp
    python -m exp4_baseline.run_experiments --model attention
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .config import EXPERIMENTS, RESULTS_DIR
from .training import run_cross_validation

logger = logging.getLogger("exp4")


def compute_summary(values: List[float]) -> Dict[str, float]:
    """Compute summary statistics for a list of values.

    Args:
        values: List of metric values from each fold.

    Returns:
        Dictionary with mean, std, min, max.
    """
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def run_all_experiments(
    experiments: Optional[List[Dict]] = None,
    device: torch.device = None,
) -> Dict[str, Dict]:
    """Run all experiments and collect results.

    Args:
        experiments: List of experiment configs. Defaults to EXPERIMENTS.
        device: Device to use.

    Returns:
        Dictionary mapping experiment names to results.
    """
    if experiments is None:
        experiments = EXPERIMENTS

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_results = {}

    for exp in experiments:
        exp_name = exp["name"]
        model_type = exp["model"]

        logger.info(f"\n{'='*60}")
        logger.info(f"Running experiment: {exp_name}")
        logger.info(f"{'='*60}")

        fold_metrics = run_cross_validation(model_type=model_type, device=device)

        # Compute summary statistics
        summary = {}
        for metric_name, values in fold_metrics.items():
            summary[metric_name] = compute_summary(values)

        all_results[exp_name] = {
            "config": exp,
            "fold_metrics": fold_metrics,
            "summary": summary,
        }

    return all_results


def print_results_table(all_results: Dict[str, Dict]):
    """Print results in formatted table.

    Args:
        all_results: Dictionary of experiment results.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 4 RESULTS SUMMARY")
    print("=" * 80)

    # Header
    print(f"{'Experiment':<25} {'AUC':>12} {'Bal Acc':>12} {'F1':>12} {'F1 Tuned':>12}")
    print("-" * 80)

    for exp_name, results in all_results.items():
        summary = results["summary"]
        auc = f"{summary['auc']['mean']:.4f} +/- {summary['auc']['std']:.4f}"
        bal_acc = f"{summary['balanced_acc_tuned']['mean']:.4f} +/- {summary['balanced_acc_tuned']['std']:.4f}"
        f1 = f"{summary['f1']['mean']:.4f} +/- {summary['f1']['std']:.4f}"
        f1_tuned = f"{summary['f1_tuned']['mean']:.4f} +/- {summary['f1_tuned']['std']:.4f}"

        print(f"{exp_name:<25} {auc:>12} {bal_acc:>12} {f1:>12} {f1_tuned:>12}")

    print("=" * 80)

    # Print per-fold details
    print("\nPer-fold AUC values:")
    for exp_name, results in all_results.items():
        aucs = results["fold_metrics"]["auc"]
        fold_str = ", ".join([f"{v:.4f}" for v in aucs])
        print(f"  {exp_name}: [{fold_str}]")


def save_results(all_results: Dict[str, Dict], output_path: Path):
    """Save results to JSON.

    Args:
        all_results: Dictionary of experiment results.
        output_path: Path to save results.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types for JSON serialisation
    def convert_to_serialisable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serialisable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serialisable(v) for v in obj]
        return obj

    serialisable_results = convert_to_serialisable(all_results)

    with open(output_path, "w") as f:
        json.dump(serialisable_results, f, indent=2)

    logger.info(f"Results saved to {output_path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run Experiment 4: Clinical features baseline"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["mlp", "attention", "all"],
        default="all",
        help="Model type to run (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: outputs/exp4_results/results_TIMESTAMP.json)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: auto-detect)",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Filter experiments if specific model requested
    if args.model == "all":
        experiments = EXPERIMENTS
    else:
        experiments = [exp for exp in EXPERIMENTS if exp["model"] == args.model]

    # Run experiments
    all_results = run_all_experiments(experiments=experiments, device=device)

    # Print results
    print_results_table(all_results)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = RESULTS_DIR / f"results_{timestamp}.json"

    save_results(all_results, output_path)


if __name__ == "__main__":
    main()
