"""Run Experiment 14: Optuna hyperparameter tuning for top 3 models.

Usage:
    python -m exp14_optuna_tuning.run_tuning                       # All 3 models
    python -m exp14_optuna_tuning.run_tuning --model exp7a         # Single model
    python -m exp14_optuna_tuning.run_tuning --model exp11         # Single model
    python -m exp14_optuna_tuning.run_tuning --model exp12         # Single model
    python -m exp14_optuna_tuning.run_tuning --n-trials 50         # Override trial count
    python -m exp14_optuna_tuning.run_tuning --dry-run             # Test setup only
"""

import argparse
import json
import logging
from datetime import datetime

import optuna

from .config import (
    BASELINES,
    CLI_MODEL_MAP,
    N_TRIALS,
    RESULTS_DIR,
    SEARCH_SPACES,
    STUDY_DB_PATH,
)
from .objectives import (
    objective_exp7a_mlp,
    objective_exp11_quadmlpv2,
    objective_exp12_fusemoe,
)

logger = logging.getLogger(__name__)

OBJECTIVE_MAP = {
    "exp7a_mlp": objective_exp7a_mlp,
    "exp11_quadmlpv2": objective_exp11_quadmlpv2,
    "exp12_fusemoe": objective_exp12_fusemoe,
}

MODEL_DISPLAY_NAMES = {
    "exp7a_mlp": "Exp7a QuadFusionMLP",
    "exp11_quadmlpv2": "Exp11 QuadMLPv2 (EEG2Vec)",
    "exp12_fusemoe": "Exp12 TripleFuseMoE",
}


def create_study(model_name: str) -> optuna.Study:
    """Create or load an Optuna study for the given model.

    Uses SQLite storage for persistence and resumability.
    TPE sampler with seed for reproducibility.
    MedianPruner to kill unpromising trials early.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=f"exp14_{model_name}",
        storage=f"sqlite:///{STUDY_DB_PATH}",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=2,
        ),
        direction="maximize",
        load_if_exists=True,
    )

    n_existing = len(study.trials)
    if n_existing > 0:
        logger.info(
            f"Resuming study '{study.study_name}' with {n_existing} existing trials"
        )

    return study


def run_study(model_name: str, n_trials: int) -> optuna.Study:
    """Run Optuna optimisation for a single model."""
    study = create_study(model_name)
    objective_fn = OBJECTIVE_MAP[model_name]
    display_name = MODEL_DISPLAY_NAMES[model_name]

    logger.info(f"Starting {display_name} tuning ({n_trials} trials)")
    logger.info(f"  Search space: {list(SEARCH_SPACES[model_name].keys())}")
    logger.info(f"  Baseline AUC: {BASELINES[model_name]['auc_mean']}")

    study.optimize(
        objective_fn,
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # Log results
    best = study.best_trial
    logger.info(f"\n{display_name} - Best trial #{best.number}:")
    logger.info(f"  AUC: {best.value:.4f}")
    logger.info(f"  Params: {best.params}")

    baseline_auc = BASELINES[model_name]["auc_mean"]
    delta = best.value - baseline_auc
    logger.info(f"  Delta vs baseline: {delta:+.4f}")

    return study


def save_results(study: optuna.Study, model_name: str):
    """Save best trial results as JSON in project format."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    best = study.best_trial
    baseline = BASELINES[model_name]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

    results = {
        "experiment": f"exp14_{model_name}",
        "model": MODEL_DISPLAY_NAMES[model_name],
        "best_params": best.params,
        "best_auc": best.value,
        "baseline_auc": baseline["auc_mean"],
        "baseline_config": baseline["config"],
        "delta": best.value - baseline["auc_mean"],
        "n_trials_complete": n_complete,
        "n_trials_pruned": n_pruned,
        "best_trial_number": best.number,
        "study_name": study.study_name,
        "timestamp": timestamp,
    }

    results_path = RESULTS_DIR / f"exp14_best_{model_name}_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    return results_path


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 14: Optuna HP tuning for top 3 models"
    )
    parser.add_argument(
        "--model",
        choices=["exp7a", "exp11", "exp12", "all"],
        default="all",
        help="Which model to tune (default: all)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=N_TRIALS,
        help=f"Number of Optuna trials per model (default: {N_TRIALS})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test setup only - create studies without running trials",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Reduce Optuna verbosity
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Determine which models to tune
    if args.model == "all":
        model_names = list(OBJECTIVE_MAP.keys())
    else:
        model_names = [CLI_MODEL_MAP[args.model]]

    logger.info(f"Models to tune: {model_names}")
    logger.info(f"Trials per model: {args.n_trials}")

    if args.dry_run:
        logger.info("Dry run - creating studies only")
        for model_name in model_names:
            study = create_study(model_name)
            logger.info(
                f"  {MODEL_DISPLAY_NAMES[model_name]}: study='{study.study_name}', "
                f"existing_trials={len(study.trials)}"
            )
        logger.info("Dry run complete - all studies created successfully")
        return

    # Run optimisation
    for model_name in model_names:
        study = run_study(model_name, args.n_trials)
        save_results(study, model_name)

    # Print summary comparison
    print("\n" + "=" * 70)
    print("Experiment 14: Optuna HP Tuning Summary")
    print("=" * 70)
    print(f"{'Model':<30} {'Baseline':>10} {'Tuned':>10} {'Delta':>10}")
    print("-" * 70)

    for model_name in model_names:
        study = create_study(model_name)
        if len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) > 0:
            best_auc = study.best_value
            baseline_auc = BASELINES[model_name]["auc_mean"]
            delta = best_auc - baseline_auc
            print(
                f"{MODEL_DISPLAY_NAMES[model_name]:<30} "
                f"{baseline_auc:>10.4f} {best_auc:>10.4f} {delta:>+10.4f}"
            )
        else:
            print(f"{MODEL_DISPLAY_NAMES[model_name]:<30} {'(no trials)':>30}")

    print("=" * 70)


if __name__ == "__main__":
    main()
