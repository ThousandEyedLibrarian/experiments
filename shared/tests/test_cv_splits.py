"""Tests for the shared outer/inner CV splitters (shared/cv_splits.py)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from shared.cv_splits import (
    current_seed, cv_suffix, fold_indices, inner_val_split, joint_key, outer_splits, set_repeat_seed,
)


def _cohort(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "outcome": rng.integers(0, 2, n),
        "focal": rng.integers(0, 2, n),
        "sex": rng.integers(0, 2, n),
        "cohort": np.where(rng.random(n) < 0.3, "MEL", "HEP"),
    })


def _assert_partition(splits, n):
    tests = [set(te) for _, te in splits]
    assert sum(len(t) for t in tests) == n
    assert set().union(*tests) == set(range(n))
    for tr, te in splits:
        assert not set(tr) & set(te)
        assert len(tr) + len(te) == n


def test_legacy_matches_original_construction():
    df = _cohort()
    want = list(StratifiedKFold(5, shuffle=True, random_state=42).split(
        np.zeros(len(df)), df["outcome"].values))
    got = outer_splits(df, mode="legacy")
    for (wt, we), (gt, ge) in zip(want, got):
        assert np.array_equal(wt, gt) and np.array_equal(we, ge)


@pytest.mark.parametrize("mode", ["legacy", "multilabel", "joint"])
def test_outer_splits_partition_the_cohort(mode):
    df = _cohort()
    splits = outer_splits(df, mode=mode)
    assert len(splits) == 5
    _assert_partition(splits, len(df))


def test_joint_splitter_balances_each_outcome_cohort_cell():
    df = _cohort(n=500)
    key = joint_key(df, ["outcome", "cohort"])
    cells, totals = np.unique(key, return_counts=True)
    for _, te in outer_splits(df, mode="joint"):
        for cell, total in zip(cells, totals):
            # StratifiedKFold keeps every cell within one patient of total/5.
            assert abs((key[te] == cell).sum() - total / 5) <= 1


def test_inner_val_split_is_a_disjoint_stratified_subset():
    df = _cohort(n=300)
    y = df["outcome"].to_numpy()
    for fold, (tr, te) in enumerate(outer_splits(df, mode="multilabel")):
        fit, es = inner_val_split(y, tr, frac=0.2, seed=42 + fold)
        assert not set(fit) & set(es)
        assert set(fit) | set(es) == set(tr)
        assert not (set(fit) | set(es)) & set(te)
        assert abs(len(es) - 0.2 * len(tr)) <= 1
        assert abs(y[es].mean() - y[tr].mean()) < 0.05


def test_inner_val_split_rejects_degenerate_fraction():
    with pytest.raises(ValueError):
        inner_val_split(np.array([0, 1] * 10), np.arange(20), frac=0.0)


def test_cv_suffix():
    assert cv_suffix("legacy", 0.0) == ""
    assert cv_suffix("multilabel", 0.2) == "_sp-multilabel_iv20"
    assert cv_suffix("legacy", 0.2) == "_sp-legacy_iv20"
    assert cv_suffix("multilabel", 0.0) == "_sp-multilabel_iv0"
    assert cv_suffix("multilabel", 0.2, 43) == "_sp-multilabel_iv20_s43"
    assert cv_suffix("legacy", 0.0, 42) == "_s42"


def test_repeat_seed_overrides_split_seeds_and_suffix():
    df = _cohort()
    try:
        set_repeat_seed(43)
        want = list(StratifiedKFold(5, shuffle=True, random_state=43).split(
            np.zeros(len(df)), df["outcome"].values))
        got = outer_splits(df, mode="legacy", seed=42)  # explicit 42 is overridden
        assert all(np.array_equal(w[1], g[1]) for w, g in zip(want, got))
        assert cv_suffix("multilabel", 0.2) == "_sp-multilabel_iv20_s43"
        assert current_seed() == 43
        y = df["outcome"].to_numpy()
        tr, te = got[0]
        fit43, _, _ = fold_indices(y, tr, te, 0, 0.2)
        set_repeat_seed(None)
        fit42, _, _ = fold_indices(y, tr, te, 0, 0.2)
        assert not np.array_equal(fit42, fit43)
    finally:
        set_repeat_seed(None)
    assert current_seed() == 42 and cv_suffix("legacy", 0) == ""


def test_multilabel_refuses_missing_columns():
    df = _cohort().drop(columns=["sex"])
    with pytest.raises(ValueError, match="sex"):
        outer_splits(df, mode="multilabel")
