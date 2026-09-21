"""Tests for exp18's metric and test helpers (exp18_mixed_cohort/analyse.py)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from exp18_mixed_cohort.analyse import holm, nadeau_bengio, stratified_auc


def test_stratified_auc_ignores_cross_cohort_pairs():
    # Within each cohort the score ranks perfectly, but cohort B's scores sit
    # far below cohort A's, so the pooled AUC is < 1 while the stratified AUC is 1.
    df = pd.DataFrame({
        "cohort": ["A"] * 4 + ["B"] * 4,
        "y_true": [0, 0, 1, 1, 0, 0, 1, 1],
        "y_prob": [0.8, 0.85, 0.9, 0.95, 0.1, 0.15, 0.2, 0.25],
    })
    assert stratified_auc(df) == pytest.approx(1.0)
    assert roc_auc_score(df["y_true"], df["y_prob"]) < 1.0


def test_stratified_auc_is_pair_weighted_mean_of_within_cohort_aucs():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"cohort": np.repeat(["A", "B"], [60, 140]),
                       "y_true": rng.integers(0, 2, 200), "y_prob": rng.random(200)})
    w, a = [], []
    for _, g in df.groupby("cohort"):
        w.append(g["y_true"].sum() * (1 - g["y_true"]).sum())
        a.append(roc_auc_score(g["y_true"], g["y_prob"]))
    assert stratified_auc(df) == pytest.approx(np.average(a, weights=w))


def test_nadeau_bengio_inflates_variance_relative_to_naive_t():
    d = np.array([0.05, 0.02, 0.08, -0.01, 0.04, 0.06, 0.03, 0.01, 0.07, 0.02])
    _, t_nb, p_nb = nadeau_bengio(d, test_train_ratio=0.25)
    t_naive = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    assert abs(t_nb) < abs(t_naive) and 0 < p_nb < 1


def test_holm_matches_hand_computation():
    assert holm([0.01, 0.04]) == pytest.approx([0.02, 0.04])
    assert holm([0.04, 0.01]) == pytest.approx([0.04, 0.02])
    assert holm([0.6, 0.7]) == pytest.approx([1.0, 1.0])  # 2 x 0.6 caps at 1
