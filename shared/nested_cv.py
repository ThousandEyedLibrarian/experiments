"""Nested cross-validation framework for unbiased model evaluation.

Implements nested (double) cross-validation where:
- Outer loop: provides unbiased performance estimates
- Inner loop: selects best hyperparameters

Reference: de Jong et al. 2021 (Brain) used 10x10 nested CV.
Reference: scikit-learn nested CV example.

This module provides both a general-purpose nested CV runner and
utilities for defining hyperparameter grids for our PyTorch models.
"""

import itertools
import json
import logging
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    ITERSTRAT_AVAILABLE = True
except ImportError:
    ITERSTRAT_AVAILABLE = False

logger = logging.getLogger(__name__)


def expand_param_grid(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Expand a parameter grid into all combinations.

    Args:
        param_grid: Dictionary mapping parameter names to lists of values.
            Example: {"lr": [1e-3, 1e-4], "dropout": [0.1, 0.3]}

    Returns:
        List of parameter dictionaries, one per combination.
            Example: [{"lr": 1e-3, "dropout": 0.1}, {"lr": 1e-3, "dropout": 0.3}, ...]
    """
    if not param_grid:
        return [{}]

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    return [dict(zip(keys, combo)) for combo in combinations]


def nested_cv(
    train_and_evaluate_fn: Callable,
    df: pd.DataFrame,
    param_grid: Dict[str, List[Any]],
    outer_splits: int = 5,
    inner_splits: int = 5,
    stratify_col: str = "outcome",
    random_state: int = 42,
    results_dir: Optional[Path] = None,
    experiment_name: str = "nested_cv",
    **kwargs,
) -> Dict[str, Any]:
    """Run nested cross-validation for unbiased performance estimation.

    The outer loop provides unbiased test performance. The inner loop
    selects the best hyperparameters for each outer fold.

    Args:
        train_and_evaluate_fn: Function with signature:
            (df, train_idx, val_idx, params, **kwargs) -> dict
            Must return a dict containing at least 'auc' (float).
            Additional metrics are passed through.
        df: Full DataFrame with all features and outcome column.
        param_grid: Hyperparameter grid to search.
            Example: {"learning_rate": [1e-3, 1e-4], "dropout": [0.1, 0.3]}
        outer_splits: Number of outer CV folds.
        inner_splits: Number of inner CV folds for hyperparameter selection.
        stratify_col: Column to stratify on for fold splits.
        random_state: Random seed for reproducibility.
        results_dir: Directory to save intermediate results (optional).
        experiment_name: Name for logging and file naming.
        **kwargs: Additional arguments passed to train_and_evaluate_fn.

    Returns:
        Dictionary with:
            - outer_fold_results: list of per-fold result dicts
            - best_params_per_fold: list of selected params per outer fold
            - auc_mean, auc_std: aggregate outer fold AUC statistics
            - all_inner_results: full inner CV results for analysis
    """
    param_combinations = expand_param_grid(param_grid)
    n_combos = len(param_combinations)

    logger.info(
        f"Nested CV: {outer_splits} outer x {inner_splits} inner folds, "
        f"{n_combos} hyperparameter combinations"
    )
    logger.info(f"Total model trains: {outer_splits * inner_splits * n_combos + outer_splits}")

    y = df[stratify_col].values
    outer_cv = StratifiedKFold(
        n_splits=outer_splits, shuffle=True, random_state=random_state
    )

    outer_fold_results = []
    best_params_per_fold = []
    all_inner_results = []

    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(
        outer_cv.split(np.arange(len(df)), y)
    ):
        logger.info(
            f"Outer fold {outer_fold + 1}/{outer_splits}: "
            f"train={len(outer_train_idx)}, test={len(outer_test_idx)}"
        )

        # Inner loop: hyperparameter selection on outer training set
        inner_df = df.iloc[outer_train_idx].reset_index(drop=True)
        inner_y = inner_df[stratify_col].values

        inner_cv = StratifiedKFold(
            n_splits=inner_splits, shuffle=True, random_state=random_state
        )

        # Evaluate each hyperparameter combination
        combo_scores = []
        inner_fold_details = []

        for combo_idx, params in enumerate(param_combinations):
            inner_aucs = []

            for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(
                inner_cv.split(np.arange(len(inner_df)), inner_y)
            ):
                try:
                    result = train_and_evaluate_fn(
                        inner_df, inner_train_idx, inner_val_idx, params, **kwargs
                    )
                    inner_aucs.append(result.get("auc", 0.0))
                except Exception as e:
                    logger.warning(
                        f"  Inner fold {inner_fold + 1} failed for params {params}: {e}"
                    )
                    inner_aucs.append(0.0)

            mean_auc = float(np.mean(inner_aucs))
            combo_scores.append(mean_auc)
            inner_fold_details.append({
                "params": params,
                "inner_aucs": inner_aucs,
                "mean_auc": mean_auc,
            })

            logger.info(
                f"  Combo {combo_idx + 1}/{n_combos}: "
                f"AUC={mean_auc:.4f} +/- {np.std(inner_aucs):.4f} | {params}"
            )

        all_inner_results.append(inner_fold_details)

        # Select best hyperparameters
        best_combo_idx = int(np.argmax(combo_scores))
        best_params = param_combinations[best_combo_idx]
        best_params_per_fold.append(best_params)

        logger.info(
            f"  Best params (AUC={combo_scores[best_combo_idx]:.4f}): {best_params}"
        )

        # Outer loop: retrain with best params, evaluate on held-out test set
        try:
            outer_result = train_and_evaluate_fn(
                df, outer_train_idx, outer_test_idx, best_params, **kwargs
            )
            outer_result["outer_fold"] = outer_fold
            outer_result["best_params"] = best_params
            outer_fold_results.append(outer_result)

            logger.info(
                f"  Outer fold {outer_fold + 1} test AUC: "
                f"{outer_result.get('auc', 0.0):.4f}"
            )
        except Exception as e:
            logger.error(f"  Outer fold {outer_fold + 1} evaluation failed: {e}")
            outer_fold_results.append({
                "outer_fold": outer_fold,
                "best_params": best_params,
                "auc": 0.0,
                "error": str(e),
            })

    # Aggregate outer fold results
    outer_aucs = [r.get("auc", 0.0) for r in outer_fold_results]
    results = {
        "experiment_name": experiment_name,
        "outer_splits": outer_splits,
        "inner_splits": inner_splits,
        "n_param_combinations": n_combos,
        "param_grid": {k: [str(v) for v in vals] for k, vals in param_grid.items()},
        "outer_fold_results": outer_fold_results,
        "best_params_per_fold": best_params_per_fold,
        "all_inner_results": all_inner_results,
        "auc_mean": float(np.mean(outer_aucs)),
        "auc_std": float(np.std(outer_aucs)),
        "auc_per_fold": outer_aucs,
    }

    # Aggregate other metrics if available
    metric_keys = [k for k in outer_fold_results[0].keys()
                   if k not in ("outer_fold", "best_params", "error")]
    for key in metric_keys:
        values = [r.get(key, 0.0) for r in outer_fold_results if key in r]
        if values and isinstance(values[0], (int, float)):
            results[f"{key}_mean"] = float(np.mean(values))
            results[f"{key}_std"] = float(np.std(values))

    logger.info(
        f"Nested CV complete: AUC={results['auc_mean']:.4f} "
        f"+/- {results['auc_std']:.4f}"
    )

    # Save results if directory provided
    if results_dir is not None:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = results_dir / f"{experiment_name}_nested_cv_{timestamp}.json"

        # Convert non-serialisable types
        serialisable = json.loads(
            json.dumps(results, default=lambda o: str(o))
        )
        with open(results_path, "w") as f:
            json.dump(serialisable, f, indent=2)
        logger.info(f"Results saved to {results_path}")

    return results


def compare_nested_vs_standard(
    nested_results: Dict[str, Any],
    standard_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare nested CV results against standard (non-nested) CV results.

    This helps quantify the optimistic bias in standard CV when
    hyperparameters are tuned on the same data used for evaluation.

    Args:
        nested_results: Output from nested_cv().
        standard_results: Dict with 'auc_mean' and 'auc_std' from standard CV.

    Returns:
        Comparison dictionary with bias estimates.
    """
    nested_auc = nested_results["auc_mean"]
    standard_auc = standard_results.get("auc_mean", 0.0)

    return {
        "nested_auc": nested_auc,
        "standard_auc": standard_auc,
        "optimistic_bias": standard_auc - nested_auc,
        "nested_std": nested_results["auc_std"],
        "standard_std": standard_results.get("auc_std", 0.0),
        "param_stability": _assess_param_stability(
            nested_results.get("best_params_per_fold", [])
        ),
    }


def _assess_param_stability(params_list: List[Dict]) -> Dict[str, Any]:
    """Assess how stable hyperparameter selection is across outer folds.

    If the same parameters are selected in most folds, the model is
    robust to the particular train/test split.

    Args:
        params_list: List of best param dicts from each outer fold.

    Returns:
        Stability assessment dictionary.
    """
    if not params_list:
        return {"stable": True, "unique_configs": 0}

    # Convert each param dict to a hashable tuple for comparison
    param_tuples = [tuple(sorted(p.items())) for p in params_list]
    unique_configs = len(set(param_tuples))

    return {
        "n_folds": len(params_list),
        "unique_configs": unique_configs,
        "stable": unique_configs <= 2,
        "most_common": dict(max(set(param_tuples), key=param_tuples.count)),
    }
