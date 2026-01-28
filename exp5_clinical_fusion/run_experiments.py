"""Run Experiment 5 configurations with cross-validation.

Usage:
    python -m exp5_clinical_fusion.run_experiments
    python -m exp5_clinical_fusion.run_experiments --exp 5a
    python -m exp5_clinical_fusion.run_experiments --exp 5b
    python -m exp5_clinical_fusion.run_experiments --exp 5c
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
from .training import (
    run_cross_validation_eeg,
    run_cross_validation_smiles,
    run_cross_validation_text,
)

logger = logging.getLogger("exp5")


def compute_summary(values: List[float]) -> Dict[str, float]:
    """Compute summary statistics for a list of values."""
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def run_experiment(
    exp_config: Dict,
    device: torch.device,
) -> Dict:
    """Run a single experiment configuration.

    Args:
        exp_config: Experiment configuration dict.
        device: Device to use.

    Returns:
        Results dict with fold_metrics and summary.
    """
    exp_name = exp_config["name"]
    modality = exp_config["modality"]

    logger.info(f"\n{'='*60}")
    logger.info(f"Running experiment: {exp_name}")
    logger.info(f"{'='*60}")

    if modality == "smiles":
        fold_metrics = run_cross_validation_smiles(
            smiles_model=exp_config["smiles_model"],
            device=device,
        )
    elif modality == "text":
        fold_metrics = run_cross_validation_text(
            text_model=exp_config["text_model"],
            device=device,
        )
    elif modality == "eeg":
        fold_metrics = run_cross_validation_eeg(
            eeg_model=exp_config["eeg_model"],
            device=device,
        )
    else:
        raise ValueError(f"Unknown modality: {modality}")

    # Compute summary statistics
    summary = {}
    for metric_name, values in fold_metrics.items():
        summary[metric_name] = compute_summary(values)

    return {
        "config": exp_config,
        "fold_metrics": fold_metrics,
        "summary": summary,
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

    for exp_config in experiments:
        exp_name = exp_config["name"]
        results = run_experiment(exp_config, device)
        all_results[exp_name] = results

    return all_results


def print_results_table(all_results: Dict[str, Dict]):
    """Print results in formatted table."""
    print("\n" + "=" * 90)
    print("EXPERIMENT 5 RESULTS SUMMARY")
    print("=" * 90)

    # Header
    print(f"{'Experiment':<25} {'AUC':>18} {'Bal Acc':>18} {'F1 Tuned':>18}")
    print("-" * 90)

    for exp_name, results in all_results.items():
        summary = results["summary"]
        auc = f"{summary['auc']['mean']:.4f} +/- {summary['auc']['std']:.4f}"
        bal_acc = f"{summary['balanced_acc_tuned']['mean']:.4f} +/- {summary['balanced_acc_tuned']['std']:.4f}"
        f1_tuned = f"{summary['f1_tuned']['mean']:.4f} +/- {summary['f1_tuned']['std']:.4f}"

        print(f"{exp_name:<25} {auc:>18} {bal_acc:>18} {f1_tuned:>18}")

    print("=" * 90)

    # Comparison with Exp4 baseline
    print("\nComparison with Exp4a baseline (AUC 0.664):")
    for exp_name, results in all_results.items():
        auc_mean = results["summary"]["auc"]["mean"]
        diff = auc_mean - 0.664
        sign = "+" if diff >= 0 else ""
        print(f"  {exp_name}: {auc_mean:.4f} ({sign}{diff:.4f})")

    # Print per-fold details
    print("\nPer-fold AUC values:")
    for exp_name, results in all_results.items():
        aucs = results["fold_metrics"]["auc"]
        fold_str = ", ".join([f"{v:.4f}" for v in aucs])
        print(f"  {exp_name}: [{fold_str}]")


def save_results(all_results: Dict[str, Dict], output_path: Path):
    """Save results to JSON."""
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
        description="Run Experiment 5: Clinical + Single Modality Fusion"
    )
    parser.add_argument(
        "--exp",
        type=str,
        choices=["5a", "5b", "5c", "all"],
        default="all",
        help="Experiment to run (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: outputs/exp5_results/results_TIMESTAMP.json)",
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

    # Filter experiments
    if args.exp == "all":
        experiments = EXPERIMENTS
    elif args.exp == "5a":
        experiments = [exp for exp in EXPERIMENTS if exp["modality"] == "smiles"]
    elif args.exp == "5b":
        experiments = [exp for exp in EXPERIMENTS if exp["modality"] == "text"]
    elif args.exp == "5c":
        experiments = [exp for exp in EXPERIMENTS if exp["modality"] == "eeg"]
    else:
        experiments = EXPERIMENTS

    logger.info(f"Running {len(experiments)} experiment(s)")

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
