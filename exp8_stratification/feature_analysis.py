"""Feature distribution analysis for stratification planning.

Analyses clinical feature distributions and identifies imbalance issues.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import (
    CLINICAL_CONFIG,
    SEVERELY_IMBALANCED_FEATURES,
    STRATIFICATION_FEATURES,
)
from .data_cleaning import load_and_clean_data

logger = logging.getLogger(__name__)


def analyse_binary_feature(series: pd.Series) -> Dict:
    """Analyse a binary feature's distribution.

    Args:
        series: Binary feature column.

    Returns:
        Dict with distribution statistics.
    """
    counts = series.value_counts(dropna=False)
    total = len(series)
    n_missing = series.isna().sum()

    if n_missing == total:
        return {
            "majority_pct": 0,
            "minority_n": 0,
            "n_missing": n_missing,
            "distribution": {},
            "status": "EMPTY",
        }

    valid_counts = series.dropna().value_counts()
    if len(valid_counts) == 0:
        return {
            "majority_pct": 0,
            "minority_n": 0,
            "n_missing": n_missing,
            "distribution": {},
            "status": "EMPTY",
        }

    majority_pct = valid_counts.max() / valid_counts.sum() * 100
    minority_n = valid_counts.min() if len(valid_counts) > 1 else 0

    if majority_pct > 95:
        status = "SEVERE"
    elif majority_pct > 85:
        status = "WARNING"
    else:
        status = "OK"

    return {
        "majority_pct": majority_pct,
        "minority_n": minority_n,
        "n_missing": n_missing,
        "distribution": dict(valid_counts),
        "status": status,
    }


def analyse_categorical_feature(series: pd.Series) -> Dict:
    """Analyse a categorical feature's distribution.

    Args:
        series: Categorical feature column.

    Returns:
        Dict with distribution statistics.
    """
    counts = series.value_counts(dropna=False)
    n_missing = series.isna().sum()
    valid_counts = series.dropna().value_counts()

    if len(valid_counts) == 0:
        return {
            "n_categories": 0,
            "min_category_n": 0,
            "n_missing": n_missing,
            "distribution": {},
            "status": "EMPTY",
        }

    min_n = valid_counts.min()
    status = "SEVERE" if min_n < 5 else "WARNING" if min_n < 10 else "OK"

    return {
        "n_categories": len(valid_counts),
        "min_category_n": min_n,
        "n_missing": n_missing,
        "distribution": dict(valid_counts),
        "status": status,
    }


def analyse_all_features(df: pd.DataFrame) -> Dict[str, Dict]:
    """Analyse all clinical features.

    Args:
        df: Cleaned dataframe.

    Returns:
        Dict mapping feature names to their analysis.
    """
    results = {}

    # Binary features
    for col in CLINICAL_CONFIG["binary_features"]:
        if col in df.columns:
            results[col] = analyse_binary_feature(df[col])
            results[col]["type"] = "binary"

    # Categorical features
    for col in CLINICAL_CONFIG["categorical_features"]:
        if col in df.columns:
            results[col] = analyse_categorical_feature(df[col])
            results[col]["type"] = "categorical"

    # Outcome
    if "outcome" in df.columns:
        results["outcome"] = analyse_binary_feature(df["outcome"])
        results["outcome"]["type"] = "target"

    return results


def check_stratification_feasibility(
    df: pd.DataFrame,
    stratify_cols: List[str],
    n_splits: int = 5,
) -> Tuple[bool, List[str]]:
    """Check if multi-label stratification is feasible.

    Args:
        df: Dataframe with features.
        stratify_cols: Columns to stratify on.
        n_splits: Number of CV splits.

    Returns:
        Tuple of (is_feasible, warnings).
    """
    warnings = []

    for col in stratify_cols:
        if col not in df.columns:
            warnings.append(f"Column '{col}' not found in dataframe")
            continue

        counts = df[col].value_counts(dropna=False)
        min_count = counts.min()

        if min_count < n_splits:
            warnings.append(
                f"Column '{col}' has category with only {min_count} samples "
                f"(need at least {n_splits} for {n_splits}-fold CV)"
            )

    is_feasible = len(warnings) == 0
    return is_feasible, warnings


def generate_analysis_report(feature_analysis: Dict[str, Dict]) -> str:
    """Generate a human-readable feature analysis report.

    Args:
        feature_analysis: Results from analyse_all_features().

    Returns:
        Formatted report string.
    """
    lines = [
        "=" * 60,
        "FEATURE DISTRIBUTION ANALYSIS",
        "=" * 60,
        "",
        "BINARY FEATURES (sorted by imbalance):",
        "-" * 60,
    ]

    # Sort binary features by imbalance
    binary_features = [
        (name, info) for name, info in feature_analysis.items()
        if info.get("type") == "binary"
    ]
    binary_features.sort(key=lambda x: x[1].get("majority_pct", 0), reverse=True)

    for name, info in binary_features:
        status = info.get("status", "?")
        majority_pct = info.get("majority_pct", 0)
        minority_n = info.get("minority_n", 0)
        dist = info.get("distribution", {})

        status_marker = {
            "SEVERE": "[!]",
            "WARNING": "[~]",
            "OK": "[+]",
        }.get(status, "[?]")

        lines.append(
            f"{status_marker} {name:15} {majority_pct:5.1f}% majority, "
            f"minority n={minority_n:3d}, dist={dist}"
        )

    lines.extend([
        "",
        "CATEGORICAL FEATURES:",
        "-" * 60,
    ])

    for name, info in feature_analysis.items():
        if info.get("type") == "categorical":
            status = info.get("status", "?")
            n_cats = info.get("n_categories", 0)
            min_n = info.get("min_category_n", 0)
            dist = info.get("distribution", {})

            status_marker = "[!]" if status == "SEVERE" else "[~]" if status == "WARNING" else "[+]"
            lines.append(
                f"{status_marker} {name:15} {n_cats} categories, "
                f"min n={min_n}, dist={dist}"
            )

    lines.extend([
        "",
        "OUTCOME DISTRIBUTION:",
        "-" * 60,
    ])

    if "outcome" in feature_analysis:
        info = feature_analysis["outcome"]
        dist = info.get("distribution", {})
        lines.append(f"    Distribution: {dist}")

    lines.extend([
        "",
        "LEGEND:",
        "  [!] SEVERE: >95% majority or <5 in smallest category",
        "  [~] WARNING: >85% majority or <10 in smallest category",
        "  [+] OK: Reasonably balanced",
        "",
        "=" * 60,
    ])

    return "\n".join(lines)


def recommend_stratification_strategy(feature_analysis: Dict[str, Dict]) -> Dict:
    """Recommend a stratification strategy based on feature analysis.

    Args:
        feature_analysis: Results from analyse_all_features().

    Returns:
        Dict with recommendations.
    """
    # Find features suitable for stratification
    suitable_features = []
    unsuitable_features = []

    for name, info in feature_analysis.items():
        if info.get("type") not in ["binary", "target"]:
            continue

        minority_n = info.get("minority_n", 0)
        if minority_n >= 10:  # At least 2 per fold for 5-fold CV
            suitable_features.append(name)
        elif minority_n >= 5:
            suitable_features.append(name)  # Marginal but usable
        else:
            unsuitable_features.append(name)

    return {
        "recommended_features": ["outcome"] + [
            f for f in STRATIFICATION_FEATURES if f in suitable_features
        ],
        "suitable_features": suitable_features,
        "unsuitable_features": unsuitable_features,
        "severely_imbalanced": [
            name for name, info in feature_analysis.items()
            if info.get("status") == "SEVERE" and info.get("type") == "binary"
        ],
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Load and clean data
    df, _ = load_and_clean_data()

    # Analyse features
    analysis = analyse_all_features(df)
    print(generate_analysis_report(analysis))

    # Check stratification feasibility
    print("\nSTRATIFICATION FEASIBILITY CHECK:")
    print("-" * 60)

    stratify_cols = ["outcome", "focal", "sex"]
    is_feasible, warnings = check_stratification_feasibility(df, stratify_cols)

    if is_feasible:
        print(f"Multi-label stratification on {stratify_cols} is FEASIBLE")
    else:
        print(f"Multi-label stratification has issues:")
        for w in warnings:
            print(f"  - {w}")

    # Recommendations
    print("\nRECOMMENDATIONS:")
    print("-" * 60)
    recs = recommend_stratification_strategy(analysis)
    print(f"Recommended stratification features: {recs['recommended_features']}")
    print(f"Severely imbalanced (consider dropping): {recs['severely_imbalanced']}")
