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
    datasets (callers pass ``df.iloc[idx]``-style positions).
    """
    if mode == "legacy":
        # Identical construction to the original run_cross_validation loops:
        # StratifiedKFold(5, shuffle=True, 42).split(np.zeros(n), outcome).
        y = df["outcome"].to_numpy()
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(skf.split(np.zeros(len(y)), y))
    if mode == "multilabel":
        from exp8_stratification.stratified_cv import get_multilabel_splits

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


def cv_suffix(splitter: str, inner_val: float) -> str:
    """Filename suffix for a CV protocol: empty for the legacy protocol (so its
    files keep the archived names), otherwise e.g. ``_sp-multilabel_iv20``."""
    if splitter == "legacy" and inner_val == 0:
        return ""
    return f"_sp-{splitter}_iv{int(round(inner_val * 100))}"
