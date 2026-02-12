"""Run Experiment 10 configurations with cross-validation.

Usage:
    python -m exp10_direct_llm.run_experiments
    python -m exp10_direct_llm.run_experiments --experiment exp10_pubmedbert_frozen
    python -m exp10_direct_llm.run_experiments --finetune
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .config import EXPERIMENTS, FINETUNE_CONFIG, FINETUNE_EXPERIMENTS, RESULTS_DIR
from .training import run_cross_validation

logger = logging.getLogger("exp10")


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
        exp_config: Experiment configuration dict with keys:
            name, llm_model, freeze, and optionally unfreeze_layers.
        device: Device to use.

    Returns:
        Results dict with fold_metrics and summary.
    """
    exp_name = exp_config["name"]
    freeze = exp_config["freeze"]
    unfreeze_layers = exp_config.get(
        "unfreeze_layers", FINETUNE_CONFIG["unfreeze_layers"] if not freeze else 0
    )

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Running experiment: {exp_name}")
    logger.info(f"{'=' * 60}")

    fold_metrics = run_cross_validation(
        llm_model=exp_config["llm_model"],
        freeze=freeze,
        unfreeze_layers=unfreeze_layers,
        device=device,
    )

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
    print("EXPERIMENT 10 RESULTS SUMMARY")
    print("=" * 90)

    # Header
    print(f"{'Experiment':<30} {'AUC':>18} {'Bal Acc':>18} {'F1 Tuned':>18}")
    print("-" * 90)

    for exp_name, results in all_results.items():
        summary = results["summary"]
        auc = (
            f"{summary['auc']['mean']:.4f} +/- "
            f"{summary['auc']['std']:.4f}"
        )
        bal_acc = (
            f"{summary['balanced_acc_tuned']['mean']:.4f} +/- "
            f"{summary['balanced_acc_tuned']['std']:.4f}"
        )
        f1_tuned = (
            f"{summary['f1_tuned']['mean']:.4f} +/- "
            f"{summary['f1_tuned']['std']:.4f}"
        )
        print(f"{exp_name:<30} {auc:>18} {bal_acc:>18} {f1_tuned:>18}")

    print("=" * 90)

    # Per-fold details
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
        description="Run Experiment 10: Direct LLM Text Modality"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Run a specific experiment by name (default: run all frozen)",
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Run fine-tuning experiments instead of frozen",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: outputs/exp10_results/results_TIMESTAMP.json)",
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

    # Select experiments
    if args.finetune:
        experiments = FINETUNE_EXPERIMENTS
    elif args.experiment:
        all_exps = EXPERIMENTS + FINETUNE_EXPERIMENTS
        experiments = [e for e in all_exps if e["name"] == args.experiment]
        if not experiments:
            available = [e["name"] for e in all_exps]
            raise ValueError(
                f"Unknown experiment: {args.experiment}. "
                f"Available: {available}"
            )
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
