import numpy as np
from scipy.stats import norm, bootstrap
from scipy import stats
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
)


def compute_classification_metrics(y_true, y_pred, threshold=0.5):
    """Compute classification metrics.

    Args:
        y_true: True binary labels.
        y_pred: Predicted probabilities.
        threshold: Decision threshold for binary classification.

    Returns:
        Dict of metric names to values.
    """
    y_pred_binary = (np.asarray(y_pred) >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred_binary),
        "f1": f1_score(y_true, y_pred_binary, zero_division=0),
        "precision": precision_score(y_true, y_pred_binary, zero_division=0),
        "recall": recall_score(y_true, y_pred_binary, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred_binary),
    }

def compute_midrank(x):
    """Computes midranks for a 1D numpy array"""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0

    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j

    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    pos_scores = predictions_sorted_transposed[:, :m]
    neg_scores = predictions_sorted_transposed[:, m:]

    tx = np.array([compute_midrank(x) for x in pos_scores])
    ty = np.array([compute_midrank(x) for x in neg_scores])
    tz = np.array([compute_midrank(x) for x in predictions_sorted_transposed])

    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2) / (m * n)

    v01 = (tz[:, :m] - tx) / n
    v10 = (tz[:, m:] - ty) / m

    sx = np.cov(v01)
    sy = np.cov(v10)

    # Ensure outputs are 2D matrices
    sx = np.atleast_2d(sx)
    sy = np.atleast_2d(sy)

    s = sx / m + sy / n
    return aucs, s


def delong_ci(y_true, y_scores, alpha=0.95):
    """
    Computes AUC, variance, standard deviation, and confidence interval using DeLong's method.

    Args:
        y_true: array-like of shape (n_samples,) — true binary labels (0 and 1)
        y_scores: array-like of shape (n_samples,) — predicted scores
        alpha: float — confidence level (e.g., 0.95 for 95% CI)

    Returns:
        auc: float — AUC
        lower: float — lower bound of confidence interval
        upper: float — upper bound of confidence interval
        std: float — standard deviation of AUC
        var: float — variance of AUC
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    order = np.argsort(-y_scores)
    y_true = y_true[order]
    y_scores = y_scores[order]

    pos_count = np.sum(y_true == 1)
    aucs, auc_cov = fastDeLong(y_scores[np.newaxis, :], pos_count)
    auc = aucs[0]
    var = auc_cov[0, 0]
    std = np.sqrt(var)

    z = norm.ppf(1 - (1 - alpha) / 2)
    lower = auc - z * std
    upper = auc + z * std

    return auc, max(0.0, lower), min(1.0, upper), std, var


def choose_threshold_max_ba(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    specs = 1 - fpr
    ba_scores = (specs + tpr) / 2
    best_threshold = thresholds[np.argmax(ba_scores)]
    return best_threshold


def bootstrap_ci(data, metric_fn, n_resamples=1000, confidence_level=0.95, random_state=None, threshold=0.5):
    """
    Compute bias-corrected (BCa) bootstrap confidence intervals for metrics.

    Parameters
    ----------
    data : array-like or tuple of array-like
        The input data for bootstrapping. Can be a single array or tuple of arrays.
    metric_fn : callable
        Function that takes data (same shape as input) and returns a dict of metrics.
    n_resamples : int, optional (default=1000)
        Number of bootstrap resamples.
    confidence_level : float, optional (default=0.95)
        Confidence level for the interval.
    random_state : int or np.random.Generator, optional
        Random seed or generator for reproducibility.

    Returns
    -------
    results : dict
        Dictionary with keys formatted as:
        '{metric_name}_estimate', '{metric_name}_ci_low', '{metric_name}_ci_high'.
    """
    # Compute base metrics on the full dataset
    base_metrics = metric_fn(*data) if isinstance(data, tuple) else metric_fn(data)
    base_metrics["pauc"] = roc_auc_score(*data, max_fpr=0.5)
    results = {}

    for metric_name, base_value in base_metrics.items():
        # Define wrapper returning a single metric
        # Use default argument to capture current metric_name (avoid closure bug)
        def stat_fn(*args, _name=metric_name, _threshold=threshold):
            if _name == "pauc":
                return roc_auc_score(*args, max_fpr=0.5)
            else:
                m = metric_fn(*args, threshold=_threshold)
                return m[_name]

        # Perform BCa bootstrap
        res = bootstrap(
            data,
            stat_fn,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            vectorized=False,
            paired=True,
            method='bca',
            random_state=random_state,
        )

        # Store flattened outputs
        results[f"{metric_name}_estimate"] = base_value
        results[f"{metric_name}_ci_low"] = res.confidence_interval.low
        results[f"{metric_name}_ci_high"] = res.confidence_interval.high
        results[f"{metric_name}_var"] = res.standard_error ** 2
        results[f"{metric_name}_std_err"] = res.standard_error
    return results


def meta_analysis_sj_robust(yi, vi, alpha=0.05):
    """
    Random-effects meta-analysis with Sidik–Jonkman estimator and robust variance.

    Parameters
    ----------
    yi : array-like
        Effect estimates from individual studies.
    vi : array-like
        Variances of the effect estimates.
    alpha : float, optional
        Significance level for CI (default = 0.05 for 95% CI).

    Returns
    -------
    results : dict
        {
            'pooled_effect': float,
            'pooled_se': float,
            'ci_low': float,
            'ci_high': float,
            'tau2': float,
            'Q': float,
            'I2': float,
            'H2': float,
            'k': int
        }
    """
    yi = np.asarray(yi)
    vi = np.asarray(vi)
    k = len(yi)

    # --- Step 1: Fixed-effect mean and Cochran's Q
    wi = 1 / vi
    mu_fixed = np.sum(wi * yi) / np.sum(wi)
    Q = np.sum(wi * (yi - mu_fixed)**2)
    
    # --- Step 2: Sidik–Jonkman τ² estimator
    mean_v = np.mean(vi)
    term = np.sum((yi - np.mean(yi))**2) / (k - 1)
    tau2 = max(0, term - mean_v)  # ensure non-negative τ²

    # --- Step 3: Random-effects weighted mean
    wi_star = 1 / (vi + tau2)
    mu_hat = np.sum(wi_star * yi) / np.sum(wi_star)
    se_mu = np.sqrt(1 / np.sum(wi_star))

    # --- Step 4: t-based confidence interval (Knapp–Hartung style)
    # KH scaling factor
    df = k - 1
    Q_e = np.sum(wi_star * (yi - mu_hat) ** 2)
    q = Q_e / df
    
    # Optional but common: truncate to avoid overly-narrow CI when q < 1
    q = max(q, 1.0)
    
    # KH-adjusted SE
    kh_se = np.sqrt(q) * se_mu
    
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    ci_low = mu_hat - tcrit * kh_se
    ci_high = mu_hat + tcrit * kh_se
    
    # --- Step 6: Heterogeneity statistics
    I2 = max(0, (Q - (k - 1)) / Q) if Q > (k - 1) else 0
    H2 = 1 / (1 - I2) if I2 < 1 else np.inf

    return {
        'pooled_effect': mu_hat,
        'pooled_se': kh_se,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'tau2': tau2,
        'Q': Q,
        'I2': I2,
        'H2': H2,
        'k': k
    }
