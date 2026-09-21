"""Outer and inner cross-validation splits shared by every experiment.

Outer splitters (``outer_splits``):
  legacy      StratifiedKFold on outcome. What exp4/6/7/11/15/16/17 used, kept
              bit-for-bit so archived predictions can be reproduced.
  multilabel  Iterative stratification on outcome, focal and sex (the Methods'
              description; exp2/5/9 already used it). The clean-rerun default.
  joint       StratifiedKFold on a composite key (exp18: outcome x cohort).

``inner_val_split`` carves an early-stopping set out of an outer training fold
so the outer test fold is never used for epoch selection, LR scheduling or the
decision threshold. See docs/analysis_plan_clean_rerun_exp18.md.
"""
from __future__ import annotations

import argparse
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

SPLITTERS = ("legacy", "multilabel", "joint")
MULTILABEL_COLS = ["outcome", "focal", "sex"]
DEFAULT_SEED = 42
DEFAULT_N_SPLITS = 5
CLEAN_INNER_FRAC = 0.2

# Repeated-CV seed (analysis plan deviation, 2026-09-21). A runner sets it once
# from --cv-seed; while set it replaces the seed of every outer and inner split,
# the default determinism seed and the filename suffix, so no call site can
# silently keep 42. None (the default) leaves every original seed untouched.
_REPEAT_SEED: int | None = None


def set_repeat_seed(seed: int | None) -> None:
    global _REPEAT_SEED
    _REPEAT_SEED = seed


def current_seed(default: int = DEFAULT_SEED) -> int:
    """The active repeated-CV seed, or ``default`` when none is set."""
    return default if _REPEAT_SEED is None else _REPEAT_SEED


def joint_key(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    """Composite stratification key, e.g. outcome x cohort -> '1|HEP'."""
    parts = [df[c].astype(str) for c in cols]
    key = parts[0]
    for part in parts[1:]:
        key = key + "|" + part
    return key.to_numpy()


def outer_splits(
    df: pd.DataFrame,
    mode: str = "multilabel",
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = DEFAULT_SEED,
    key_cols: Sequence[str] | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the outer CV folds as (train_idx, test_idx) positional arrays.

    ``df`` must be indexed positionally the same way the caller indexes its
    datasets (callers pass ``df.iloc[idx]``-style positions). An active
    repeated-CV seed (``set_repeat_seed``) replaces ``seed``.
    """
    seed = current_seed(seed)
    if mode == "legacy":
        # Identical construction to the original run_cross_validation loops:
        # StratifiedKFold(5, shuffle=True, 42).split(np.zeros(n), outcome).
        y = df["outcome"].to_numpy()
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(skf.split(np.zeros(len(y)), y))
    if mode == "multilabel":
        from exp8_stratification.stratified_cv import get_multilabel_splits

        # get_multilabel_splits skips absent columns with only a warning, which
        # silently degrades to outcome-only stratification (the EEG cohort
        # frames lacked focal/sex); refuse instead.
        missing = [c for c in MULTILABEL_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"multilabel splitter needs columns {missing}; add them to the frame")

        return list(get_multilabel_splits(
            df, stratify_cols=list(MULTILABEL_COLS), n_splits=n_splits,
            shuffle=True, random_state=seed,
        ))
    if mode == "joint":
        key = joint_key(df, key_cols or ["outcome", "cohort"])
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(skf.split(np.zeros(len(key)), key))
    raise ValueError(f"unknown splitter {mode!r}; expected one of {SPLITTERS}")


def inner_val_split(
    strat_labels: np.ndarray,
    train_idx: np.ndarray,
    frac: float = CLEAN_INNER_FRAC,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Split an outer training fold into (fit_idx, es_idx).

    ``strat_labels`` is indexed by the same positions as ``train_idx`` (the
    full-cohort label or joint-key array). Both returned arrays are sorted
    subsets of ``train_idx`` and are disjoint.
    """
    train_idx = np.asarray(train_idx)
    if not 0.0 < frac < 1.0:
        raise ValueError(f"inner-val fraction must be in (0, 1), got {frac}")
    fit_idx, es_idx = train_test_split(
        train_idx,
        test_size=frac,
        stratify=np.asarray(strat_labels)[train_idx],
        random_state=seed,
    )
    return np.sort(fit_idx), np.sort(es_idx)


def add_cv_args(parser: argparse.ArgumentParser, default_splitter: str = "legacy") -> None:
    """Add the shared --splitter / --inner-val flags to an experiment's CLI.

    The defaults reproduce each experiment's original behaviour; the clean
    rerun passes ``--splitter multilabel --inner-val 0.2``.
    """
    parser.add_argument(
        "--splitter", choices=["legacy", "multilabel"], default=default_splitter,
        help="Outer CV splitter. 'legacy' = this experiment's original splitter.",
    )
    parser.add_argument(
        "--inner-val", type=float, default=0.0, dest="inner_val",
        help="Fraction of each outer training fold held out for early stopping "
             "and the threshold (0 = legacy: select on the outer fold).",
    )
    parser.add_argument(
        "--cv-seed", type=int, default=None, dest="cv_seed",
        help="Repeated-CV seed: outer split seed s, inner split seed s + fold, "
             "determinism seed s (default: the experiment's original seed, 42).",
    )


def cv_suffix(splitter: str, inner_val: float, cv_seed: int | None = None) -> str:
    """Filename suffix for a CV protocol: empty for the legacy protocol (so its
    files keep the archived names), otherwise e.g. ``_sp-multilabel_iv20``,
    plus ``_s<seed>`` for an explicit repeated-CV seed."""
    if cv_seed is None:
        cv_seed = _REPEAT_SEED
    seed = "" if cv_seed is None else f"_s{cv_seed}"
    if splitter == "legacy" and inner_val == 0:
        return seed
    return f"_sp-{splitter}_iv{int(round(inner_val * 100))}{seed}"


def base_seed(cv_seed: int | None, default: int = DEFAULT_SEED) -> int:
    """The seed for outer splits, inner splits (+ fold) and determinism."""
    return default if cv_seed is None else cv_seed


def fold_indices(
    strat_labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold: int,
    inner_val: float,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Index sets for one outer fold: (fit_idx, es_idx, clean_test_idx).

    Legacy (``inner_val == 0``) returns ``(train_idx, test_idx, None)``: fit on
    the whole outer training fold and early-stop on the outer test fold, i.e.
    the original behaviour, so callers keep their old code path. Clean runs
    return a stratified inner split of ``train_idx`` (seed ``seed + fold``) plus
    the untouched outer test fold, which is scored once after early stopping.
    """
    if inner_val == 0:
        return np.asarray(train_idx), np.asarray(test_idx), None
    fit_idx, es_idx = inner_val_split(strat_labels, train_idx, inner_val, current_seed(seed) + fold)
    return fit_idx, es_idx, np.asarray(test_idx)


def youden_threshold(y_true, y_prob) -> float:
    """Threshold maximising Youden's J (equivalently balanced accuracy)."""
    from sklearn.metrics import roc_curve

    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return float(thresholds[int(np.argmax(tpr - fpr))])


def rethreshold(metrics: dict, threshold: float) -> dict:
    """Recompute the threshold-dependent metrics at a threshold chosen elsewhere.

    ``metrics`` is an experiment's evaluate() dict for the outer test fold (it
    must carry ``y_true``/``y_prob``). Its own ``optimal_threshold`` was tuned
    on that test fold; this replaces it with the inner early-stopping set's
    threshold and recomputes ``balanced_acc_tuned`` and ``f1_tuned``.
    Threshold-free metrics (AUC, argmax accuracy/F1) are left untouched.
    """
    from sklearn.metrics import balanced_accuracy_score, f1_score

    y_true = np.asarray(metrics["y_true"])
    y_pred = (np.asarray(metrics["y_prob"]) >= threshold).astype(int)
    out = dict(metrics)
    out["balanced_acc_tuned"] = (
        float(balanced_accuracy_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.5
    )
    out["f1_tuned"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["optimal_threshold"] = float(threshold)
    out["threshold_source"] = "inner_val"
    return out
