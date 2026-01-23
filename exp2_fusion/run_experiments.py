"""Run Experiment 2: EEG + SMILES fusion experiments."""

import argparse
import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

import torch

from .config import EXPERIMENTS, OUTPUTS_DIR, SMILES_EMBED_DIMS
from .utils.logging_utils import setup_logging, log_environment_info, log_exception

# Set up module logger (will be configured in main)
logger = logging.getLogger("exp2")


def check_environment() -> bool:
    """Check if required dependencies are available."""
    logger.info("Checking environment...")

    # Check EEG encoder availability
    from .models.eeg_encoders import is_labram_available, get_labram_import_error

    if is_labram_available():
        logger.info("LaBraM encoder: available")
    else:
        logger.warning(f"LaBraM encoder: NOT available - {get_labram_import_error()}")
        logger.warning("Use --eeg-encoder simplecnn to avoid this issue")

    # Check CUDA
    if torch.cuda.is_available():
        logger.info(f"CUDA: available ({torch.cuda.get_device_name(0)})")
    else:
        logger.warning("CUDA: not available (will use CPU)")

    # Check data imports work
    try:
        from .data_pipeline import prepare_data
        from .training import run_cross_validation
        logger.info("Core modules: importable")
        return True
    except ImportError as e:
        logger.error(f"Core module import failed: {e}")
        return False


def run_all_experiments(
    eeg_encoder: str = "labram",
    smiles_model: str = None,
    fusion_type: str = None,
    dry_run: bool = False,
    output_dir: Path = OUTPUTS_DIR / "exp2_results",
):
    """Run all configured experiments.

    Args:
        eeg_encoder: EEG encoder type to use ('labram', 'simplecnn').
        smiles_model: SMILES model filter (None = all).
        fusion_type: Fusion type filter (None = all).
        dry_run: If True, just print what would run.
        output_dir: Directory for results.
    """
    # Import here to allow --check-env to work even if these fail
    from .data_pipeline import prepare_data
    from .training import run_cross_validation

    # Filter experiments
    experiments = []
    for exp in EXPERIMENTS:
        if smiles_model and exp["smiles_model"] != smiles_model:
            continue
        if fusion_type and exp["fusion"] != fusion_type:
            continue
        experiments.append({**exp, "eeg_model": eeg_encoder})

    if dry_run:
        logger.info("DRY RUN - Would run the following experiments:")
        for exp in experiments:
            logger.info(f"  - {exp['eeg_model']} + {exp['smiles_model']} ({exp['fusion']})")
        return

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Running {len(experiments)} experiments")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    # Run experiments
    for i, exp in enumerate(experiments):
        exp_name = f"{exp['eeg_model']} + {exp['smiles_model']} ({exp['fusion']})"
        logger.info(f"[{i+1}/{len(experiments)}] Starting: {exp_name}")

        try:
            # Load data for this SMILES model
            smiles_model_name = exp["smiles_model"]
            logger.info(f"Loading data for SMILES model: {smiles_model_name}")
            eeg_data, smiles_embeddings, smiles_indices, df = prepare_data(
                smiles_model=smiles_model_name,
                cache_eeg=True,
            )
            logger.info(f"Loaded {len(df)} patients with EEG data")

            # Run cross-validation
            logger.info("Starting cross-validation...")
            results = run_cross_validation(
                eeg_data=eeg_data,
                smiles_embeddings=smiles_embeddings,
                smiles_indices=smiles_indices,
                df=df,
                fusion_type=exp["fusion"],
                eeg_encoder_type=exp["eeg_model"],
                smiles_model=smiles_model_name,
                device=device,
                verbose=True,
            )

            results["timestamp"] = datetime.now().isoformat()
            all_results.append(results)

            # Save individual result
            result_file = output_dir / f"{results['experiment']}.json"
            with open(result_file, "w") as f:
                json.dump(results, f, indent=2)

            logger.info(
                f"[{i+1}/{len(experiments)}] Complete: "
                f"Acc={results['accuracy']['mean']:.3f}, "
                f"AUC={results['auc']['mean']:.3f}, "
                f"F1={results['f1']['mean']:.3f}"
            )

        except Exception as e:
            log_exception(logger, e, f"Experiment {exp_name} failed")
            logger.error(f"Skipping experiment {exp_name} due to error")
            continue

    if not all_results:
        logger.error("No experiments completed successfully!")
        return

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_experiments": len(all_results),
        "experiments": all_results,
    }
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Results saved to {output_dir}")

    # Print summary table
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Experiment':<40} {'Acc':<10} {'AUC':<10} {'F1':<10}")
    logger.info("-" * 70)
    for r in sorted(all_results, key=lambda x: -x["auc"]["mean"]):
        name = r["experiment"]
        acc = f"{r['accuracy']['mean']:.3f}"
        auc = f"{r['auc']['mean']:.3f}"
        f1 = f"{r['f1']['mean']:.3f}"
        logger.info(f"{name:<40} {acc:<10} {auc:<10} {f1:<10}")


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 2: EEG + SMILES fusion")
    parser.add_argument("--eeg-encoder", type=str, default="simplecnn",
                        choices=["labram", "simplecnn"],
                        help="EEG encoder type (default: simplecnn)")
    parser.add_argument("--smiles-model", type=str, default=None,
                        choices=["chemberta", "smilestrf"],
                        help="SMILES model to use (default: all)")
    parser.add_argument("--fusion", type=str, default=None,
                        choices=["mlp", "fusemoe"],
                        help="Fusion type (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print experiments without running")
    parser.add_argument("--check-env", action="store_true",
                        help="Check environment and exit")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level (default: INFO)")

    args = parser.parse_args()

    # Set up logging
    log_level = getattr(logging, args.log_level)
    script_dir = Path(__file__).parent.parent  # experiments/
    log_dir = script_dir / "logs"

    global logger
    logger = setup_logging("exp2", log_dir=str(log_dir), level=log_level)

    # Log environment info
    log_environment_info(logger)

    # Check environment only
    if args.check_env:
        success = check_environment()
        if success:
            logger.info("Environment check passed!")
            sys.exit(0)
        else:
            logger.error("Environment check failed!")
            sys.exit(1)

    # Run experiments with exception handling
    try:
        run_all_experiments(
            eeg_encoder=args.eeg_encoder,
            smiles_model=args.smiles_model,
            fusion_type=args.fusion,
            dry_run=args.dry_run,
        )
        logger.info("Experiment 2 completed successfully")
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        log_exception(logger, e, "Fatal error in Experiment 2")
        sys.exit(1)


if __name__ == "__main__":
    main()
