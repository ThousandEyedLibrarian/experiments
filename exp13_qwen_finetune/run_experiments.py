"""Run Experiment 13: Qwen 2.5 0.5B Fine-tuning.

Tests fine-tuning Qwen 2.5 0.5B with 1, 2, and 4 unfrozen transformer layers.
Uses differential learning rates (encoder_lr < head_lr) to prevent catastrophic
forgetting. Frozen Qwen baseline: AUC 0.689.

Usage:
    python -m exp13_qwen_finetune.run_experiments                  # All 3 configs
    python -m exp13_qwen_finetune.run_experiments --experiments exp13_qwen_finetune_2layer
    python -m exp13_qwen_finetune.run_experiments --dry-run
"""

import argparse
import json
import logging
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from .config import (
    CLINICAL_DIM,
    CV_CONFIG,
    FINETUNE_CONFIGS,
    QWEN_CONFIG,
    RESULTS_DIR,
    TRAINING_CONFIG,
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from exp10_direct_llm.training import run_cross_validation

logger = logging.getLogger("exp13")


def main():
    parser = argparse.ArgumentParser(description="Experiment 13: Qwen 2.5 Fine-tuning")
    parser.add_argument("--experiments", nargs="+", help="Only run specific experiment names")
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--output", type=str, help="Override output path")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    configs = FINETUNE_CONFIGS
    if args.experiments:
        configs = [c for c in configs if c["name"] in args.experiments]

    logger.info(f"Running {len(configs)} Qwen fine-tuning configurations")

    if args.dry_run:
        for c in configs:
            logger.info(
                f"  {c['name']}: unfreeze={c['unfreeze_layers']} layers, "
                f"batch_size={c['batch_size']}, encoder_lr={c['encoder_lr']}, head_lr={c['head_lr']}"
            )
        logger.info("Dry run complete.")
        return

    # Temporarily override exp10 FINETUNE_CONFIG for each run
    from exp10_direct_llm import config as exp10_config

    all_results = {}
    for ft_config in configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {ft_config['name']}")
        logger.info(f"  unfreeze_layers={ft_config['unfreeze_layers']}, "
                     f"batch_size={ft_config['batch_size']}, "
                     f"encoder_lr={ft_config['encoder_lr']}, head_lr={ft_config['head_lr']}")
        logger.info(f"{'='*60}")

        # Override exp10 FINETUNE_CONFIG
        original_ft_config = exp10_config.FINETUNE_CONFIG.copy()
        exp10_config.FINETUNE_CONFIG.update({
            "batch_size": ft_config["batch_size"],
            "encoder_lr": ft_config["encoder_lr"],
            "head_lr": ft_config["head_lr"],
            "unfreeze_layers": ft_config["unfreeze_layers"],
            "epochs": TRAINING_CONFIG["epochs"],
            "patience": TRAINING_CONFIG["patience"],
            "dropout": TRAINING_CONFIG["dropout"],
            "num_classes": TRAINING_CONFIG["num_classes"],
            "weight_decay": TRAINING_CONFIG["weight_decay"],
        })

        try:
            fold_metrics = run_cross_validation(
                llm_model="qwen",
                freeze=False,
                unfreeze_layers=ft_config["unfreeze_layers"],
                device=device,
                use_multilabel_stratification=True,
            )

            summary = {}
            for key, values in fold_metrics.items():
                summary[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }

            all_results[ft_config["name"]] = {
                "config": ft_config,
                "qwen_config": QWEN_CONFIG,
                "fold_metrics": {k: [float(v) for v in vals] for k, vals in fold_metrics.items()},
                "summary": summary,
            }

            logger.info(
                f"  Result: AUC={summary['auc']['mean']:.4f} +/- {summary['auc']['std']:.4f}"
            )

        except Exception as e:
            logger.error(f"  FAILED: {e}")
            all_results[ft_config["name"]] = {
                "config": ft_config,
                "error": str(e),
            }

        finally:
            # Restore original config
            exp10_config.FINETUNE_CONFIG.update(original_ft_config)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or str(RESULTS_DIR / f"results_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    # Print summary table
    print(f"\n{'Config':<40} {'Layers':>7} {'AUC':>14} {'Bal Acc':>14}")
    print("-" * 80)
    for name, result in sorted(all_results.items(), key=lambda x: x[1].get("summary", {}).get("auc", {}).get("mean", 0), reverse=True):
        if "error" in result:
            print(f"{name:<40} {'FAILED':>7} {result['error'][:30]}")
        else:
            s = result["summary"]
            print(
                f"{name:<40} {result['config']['unfreeze_layers']:>7} "
                f"{s['auc']['mean']:.3f}+/-{s['auc']['std']:.3f}  "
                f"{s['balanced_acc_tuned']['mean']:.3f}+/-{s['balanced_acc_tuned']['std']:.3f}"
            )

    print(f"\nBaseline (Qwen frozen): AUC 0.689")
    print(f"Baseline (ClinicalBERT fine-tuned): AUC 0.691")


if __name__ == "__main__":
    main()
