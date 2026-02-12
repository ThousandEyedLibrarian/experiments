"""Multi-label stratified cross-validation utilities.

Provides stratification methods that balance multiple features across folds.
"""

import logging
from typing import Generator, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    ITERSTRAT_AVAILABLE = True
except ImportError:
    ITERSTRAT_AVAILABLE = False
    logging.warning(
        "iterative-stratification not installed. "
        "Install with: uv pip install iterative-stratification"
    )

from .config import CV_CONFIG, STRATIFICATION_FEATURES
from exp4_baseline.config import AGE_BINS


def bin_age(age_series: pd.Series, bins: list = None) -> pd.Series:
    """Bin age into groups following Hakeem et al. 2022.

    Args:
        age_series: Series of age values.
        bins: Bin edges. Defaults to [0, 18, 29, 46, inf].

    Returns:
        Series of integer bin labels (0, 1, 2, 3).
    """
    if bins is None:
        bins = AGE_BINS
    return pd.cut(age_series, bins=bins, labels=False, right=False)


def get_outcome_only_splits(
    df: pd.DataFrame,
    n_splits: int = None,
    shuffle: bool = None,
    random_state: int = None,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Generate CV splits stratified on outcome only (baseline).

    Args:
        df: Dataframe with 'outcome' column.
        n_splits: Number of folds.
        shuffle: Whether to shuffle.
        random_state: Random seed.

    Yields:
        Tuple of (train_indices, val_indices).
    """
    n_splits = n_splits or CV_CONFIG["n_splits"]
    shuffle = shuffle if shuffle is not None else CV_CONFIG["shuffle"]
    random_state = random_state or CV_CONFIG["random_state"]

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )

    y = df["outcome"].values
    X = np.arange(len(df))

    for train_idx, val_idx in skf.split(X, y):
        yield train_idx, val_idx


def get_multilabel_splits(
    df: pd.DataFrame,
    stratify_cols: List[str] = None,
    n_splits: int = None,
    shuffle: bool = None,
    random_state: int = None,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Generate CV splits stratified on multiple features.

    Uses iterative stratification to balance multiple binary/categorical
    features across folds.

    Args:
        df: Dataframe with stratification columns.
        stratify_cols: Columns to stratify on. Defaults to outcome + config features.
        n_splits: Number of folds.
        shuffle: Whether to shuffle.
        random_state: Random seed.

    Yields:
        Tuple of (train_indices, val_indices).
    """
    if not ITERSTRAT_AVAILABLE:
        raise ImportError(
            "iterative-stratification required for multi-label stratification. "
            "Install with: uv pip install iterative-stratification"
        )

    n_splits = n_splits or CV_CONFIG["n_splits"]
    shuffle = shuffle if shuffle is not None else CV_CONFIG["shuffle"]
    random_state = random_state or CV_CONFIG["random_state"]

    # Default stratification columns
    if stratify_cols is None:
        stratify_cols = ["outcome"] + STRATIFICATION_FEATURES

    # Compute derived columns if needed
    df = df.copy()
    if "age_group" in stratify_cols and "age_group" not in df.columns:
        if "age_init" in df.columns:
            df["age_group"] = bin_age(df["age_init"].fillna(df["age_init"].median()))
        else:
            logging.warning("'age_init' not found, cannot compute age_group")
            stratify_cols = [c for c in stratify_cols if c != "age_group"]

    # Build multi-label matrix
    # Each column becomes a one-hot encoding for multi-label stratification
    label_matrix = []

    for col in stratify_cols:
        if col not in df.columns:
            logging.warning(f"Column '{col}' not found, skipping")
            continue

        series = df[col].fillna(-1)  # Handle NaN as separate category

        # One-hot encode
        unique_vals = sorted(series.unique())
        for val in unique_vals:
            label_matrix.append((series == val).astype(int).values)

    if not label_matrix:
        raise ValueError("No valid stratification columns found")

    y_multilabel = np.column_stack(label_matrix)
    X = np.arange(len(df))

    mskf = MultilabelStratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )

    for train_idx, val_idx in mskf.split(X, y_multilabel):
        yield train_idx, val_idx


def get_composite_key_splits(
    df: pd.DataFrame,
    key_cols: List[str] = None,
    n_splits: int = None,
    shuffle: bool = None,
    random_state: int = None,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Generate CV splits using composite stratification key.

    Creates a single stratification key by concatenating multiple columns.
    Simpler than multi-label but may have issues with rare combinations.

    Args:
        df: Dataframe with key columns.
        key_cols: Columns to combine into key. Defaults to outcome + focal.
        n_splits: Number of folds.
        shuffle: Whether to shuffle.
        random_state: Random seed.

    Yields:
        Tuple of (train_indices, val_indices).
    """
    n_splits = n_splits or CV_CONFIG["n_splits"]
    shuffle = shuffle if shuffle is not None else CV_CONFIG["shuffle"]
    random_state = random_state or CV_CONFIG["random_state"]

    if key_cols is None:
        key_cols = ["outcome", "focal"]

    # Create composite key
    key_parts = []
    for col in key_cols:
        if col in df.columns:
            key_parts.append(df[col].fillna(-1).astype(str))

    if not key_parts:
        raise ValueError("No valid key columns found")

    composite_key = key_parts[0]
    for part in key_parts[1:]:
        composite_key = composite_key + "_" + part

    # Check for rare combinations
    key_counts = composite_key.value_counts()
    rare_keys = key_counts[key_counts < n_splits]
    if len(rare_keys) > 0:
        logging.warning(
            f"Composite key has {len(rare_keys)} combinations with <{n_splits} samples. "
            "Some folds may be missing these combinations."
        )

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )

    X = np.arange(len(df))

    for train_idx, val_idx in skf.split(X, composite_key):
        yield train_idx, val_idx


def analyse_fold_balance(
    df: pd.DataFrame,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    check_cols: List[str] = None,
) -> pd.DataFrame:
    """Analyse feature balance across CV folds.

    Args:
        df: Original dataframe.
        splits: List of (train_idx, val_idx) tuples.
        check_cols: Columns to check balance for.

    Returns:
        DataFrame with per-fold distribution statistics.
    """
    if check_cols is None:
        check_cols = ["outcome", "focal", "sex"]

    results = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        fold_data = {"fold": fold_idx, "n_train": len(train_idx), "n_val": len(val_idx)}

        for col in check_cols:
            if col not in df.columns:
                continue

            # Validation set distribution
            val_series = df.iloc[val_idx][col]
            val_counts = val_series.value_counts(normalize=True)

            for val, pct in val_counts.items():
                fold_data[f"{col}_{val}_pct"] = pct * 100

            # Count of minority class in val
            if len(val_counts) > 1:
                fold_data[f"{col}_minority_n"] = val_series.value_counts().min()

        results.append(fold_data)

    return pd.DataFrame(results)


def compare_stratification_methods(df: pd.DataFrame) -> pd.DataFrame:
    """Compare different stratification methods.

    Args:
        df: Dataframe with features.

    Returns:
        DataFrame comparing fold balance across methods.
    """
    methods = {
        "outcome_only": list(get_outcome_only_splits(df)),
    }

    if ITERSTRAT_AVAILABLE:
        methods["multilabel"] = list(get_multilabel_splits(df))

    methods["composite_key"] = list(get_composite_key_splits(df))

    comparison = []
    check_cols = ["outcome", "focal", "sex"]

    for method_name, splits in methods.items():
        balance_df = analyse_fold_balance(df, splits, check_cols)

        # Compute variance across folds for each feature
        for col in check_cols:
            pct_cols = [c for c in balance_df.columns if c.startswith(f"{col}_") and c.endswith("_pct")]
            for pct_col in pct_cols:
                if pct_col in balance_df.columns:
                    std = balance_df[pct_col].std()
                    comparison.append({
                        "method": method_name,
                        "feature": pct_col,
                        "fold_std": std,
                        "fold_min": balance_df[pct_col].min(),
                        "fold_max": balance_df[pct_col].max(),
                    })

    return pd.DataFrame(comparison)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from .data_cleaning import load_and_clean_data

    # Load data
    df, _ = load_and_clean_data()

    print("=" * 60)
    print("STRATIFICATION METHOD COMPARISON")
    print("=" * 60)

    # Compare methods
    comparison = compare_stratification_methods(df)
    print("\nFold balance variance by method:")
    print(comparison.to_string(index=False))

    # Detailed fold analysis for multi-label
    if ITERSTRAT_AVAILABLE:
        print("\n" + "=" * 60)
        print("MULTI-LABEL STRATIFICATION FOLD DETAILS")
        print("=" * 60)

        splits = list(get_multilabel_splits(df))
        balance = analyse_fold_balance(df, splits)
        print(balance.to_string(index=False))
