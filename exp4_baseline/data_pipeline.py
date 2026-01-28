"""Data pipeline for Experiment 4: Clinical features baseline."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import (
    CLINICAL_CONFIG,
    CSV_PATH,
    OUTCOME_MAPPING,
)

logger = logging.getLogger("exp4")


def clean_psy_column(df: pd.DataFrame) -> pd.DataFrame:
    """Clean 'psy' column which has mixed types ('0', '1', '0.0', '1.0', '?').

    Args:
        df: DataFrame with 'psy' column.

    Returns:
        DataFrame with cleaned 'psy' column (float with NaN for invalid).
    """
    df = df.copy()

    def parse_psy(val):
        if pd.isna(val):
            return np.nan
        val_str = str(val).strip()
        if val_str in ("0", "0.0"):
            return 0.0
        elif val_str in ("1", "1.0"):
            return 1.0
        else:
            return np.nan

    df["psy"] = df["psy"].apply(parse_psy)
    return df


def clean_lesion_column(df: pd.DataFrame) -> pd.DataFrame:
    """Clean 'lesion' column which has mixed types and 'NOT AVAILABLE'.

    Args:
        df: DataFrame with 'lesion' column.

    Returns:
        DataFrame with cleaned 'lesion' column (float with NaN for invalid).
    """
    df = df.copy()

    def parse_lesion(val):
        if pd.isna(val):
            return np.nan
        val_str = str(val).strip().upper()
        if val_str in ("NOT AVAILABLE", "NA", "N/A", ""):
            return np.nan
        try:
            num = float(val_str)
            if num in (1.0, 2.0, 3.0):
                return num
            return np.nan
        except ValueError:
            return np.nan

    df["lesion"] = df["lesion"].apply(parse_lesion)
    return df


def clean_outcome_column(df: pd.DataFrame) -> pd.DataFrame:
    """Clean 'outcome' column and filter to valid outcomes.

    Args:
        df: DataFrame with 'outcome' column.

    Returns:
        DataFrame filtered to valid outcomes with mapped values (1->0, 2->1).
    """
    df = df.copy()

    # Convert to numeric, coercing errors to NaN
    df["outcome"] = pd.to_numeric(df["outcome"], errors="coerce")

    # Filter to valid outcomes (1 or 2)
    df = df[df["outcome"].isin([1, 2])].copy()

    # Map outcomes: 1=failure->0, 2=success->1
    df["outcome"] = df["outcome"].map(OUTCOME_MAPPING).astype(int)

    return df


def load_clinical_data(filepath: Path = CSV_PATH) -> pd.DataFrame:
    """Load and preprocess the clinical data.

    Args:
        filepath: Path to CSV file.

    Returns:
        Cleaned DataFrame with valid outcomes.
    """
    df = pd.read_csv(filepath)

    # Clean columns with mixed types
    df = clean_psy_column(df)
    df = clean_lesion_column(df)
    df = clean_outcome_column(df)

    df = df.reset_index(drop=True)
    logger.info(f"Loaded {len(df)} patients with valid outcomes")

    return df


class ClinicalFeaturePreprocessor:
    """Preprocess clinical features with proper train/val splitting.

    This class must be fit on training data only to prevent data leakage.
    It handles:
    - Mode imputation for binary/categorical features
    - Z-score standardisation for numeric features
    - One-hot encoding for categorical features
    """

    def __init__(self):
        self.numeric_features = CLINICAL_CONFIG["numeric_features"]
        self.binary_features = CLINICAL_CONFIG["binary_features"]
        self.categorical_features = CLINICAL_CONFIG["categorical_features"]

        # Fitted parameters (computed on training set only)
        self.numeric_mean: Optional[Dict[str, float]] = None
        self.numeric_std: Optional[Dict[str, float]] = None
        self.binary_modes: Optional[Dict[str, float]] = None
        self.categorical_modes: Optional[Dict[str, float]] = None
        self.categorical_categories: Optional[Dict[str, List[float]]] = None

        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "ClinicalFeaturePreprocessor":
        """Fit preprocessor on training data only.

        Args:
            df: Training DataFrame.

        Returns:
            Self for chaining.
        """
        # Compute statistics for numeric features
        self.numeric_mean = {}
        self.numeric_std = {}
        for col in self.numeric_features:
            self.numeric_mean[col] = df[col].mean()
            self.numeric_std[col] = df[col].std()
            # Avoid division by zero
            if self.numeric_std[col] == 0:
                self.numeric_std[col] = 1.0

        # Compute modes for binary features
        self.binary_modes = {}
        for col in self.binary_features:
            # Convert to numeric if needed
            col_data = pd.to_numeric(df[col], errors="coerce")
            mode_val = col_data.mode()
            self.binary_modes[col] = mode_val.iloc[0] if len(mode_val) > 0 else 0.0

        # Compute modes and categories for categorical features
        self.categorical_modes = {}
        self.categorical_categories = {}
        for col in self.categorical_features:
            col_data = pd.to_numeric(df[col], errors="coerce")
            mode_val = col_data.mode()
            self.categorical_modes[col] = mode_val.iloc[0] if len(mode_val) > 0 else 1.0
            # Define fixed categories (1, 2, 3) for both lesion and eeg_cat
            self.categorical_categories[col] = [1.0, 2.0, 3.0]

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted parameters.

        Args:
            df: DataFrame to transform.

        Returns:
            NumPy array of shape (n_samples, input_dim).
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fit before transform")

        features = []

        # Process binary features (13 features)
        for col in self.binary_features:
            col_data = pd.to_numeric(df[col], errors="coerce")
            # Impute missing with mode
            col_data = col_data.fillna(self.binary_modes[col])
            features.append(col_data.values.reshape(-1, 1))

        # Process numeric features (1 feature: age_init)
        for col in self.numeric_features:
            col_data = df[col].copy()
            # Impute missing with mean
            col_data = col_data.fillna(self.numeric_mean[col])
            # Standardise
            col_data = (col_data - self.numeric_mean[col]) / self.numeric_std[col]
            features.append(col_data.values.reshape(-1, 1))

        # Process categorical features with one-hot encoding (6 features total)
        for col in self.categorical_features:
            col_data = pd.to_numeric(df[col], errors="coerce")
            # Impute missing with mode
            col_data = col_data.fillna(self.categorical_modes[col])

            # One-hot encode
            categories = self.categorical_categories[col]
            for cat in categories:
                one_hot = (col_data == cat).astype(float).values.reshape(-1, 1)
                features.append(one_hot)

        # Concatenate all features
        feature_matrix = np.hstack(features).astype(np.float32)

        return feature_matrix

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            df: DataFrame to fit and transform.

        Returns:
            NumPy array of shape (n_samples, input_dim).
        """
        return self.fit(df).transform(df)


class ClinicalDataset(Dataset):
    """PyTorch Dataset for clinical features."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """Initialise dataset.

        Args:
            features: Feature array of shape (n_samples, input_dim).
            labels: Label array of shape (n_samples,).
        """
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def create_datasets(
    df: pd.DataFrame,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
) -> Tuple[ClinicalDataset, ClinicalDataset, ClinicalFeaturePreprocessor]:
    """Create train and validation datasets from fold indices.

    CRITICAL: Preprocessor is fit on training data only to prevent data leakage.

    Args:
        df: Full DataFrame with clinical features.
        train_indices: Indices for training set.
        val_indices: Indices for validation set.

    Returns:
        Tuple of (train_dataset, val_dataset, fitted_preprocessor).
    """
    # Split data
    train_df = df.iloc[train_indices].copy()
    val_df = df.iloc[val_indices].copy()

    # Create and fit preprocessor on training data only
    preprocessor = ClinicalFeaturePreprocessor()
    train_features = preprocessor.fit_transform(train_df)

    # Transform validation data using training statistics
    val_features = preprocessor.transform(val_df)

    # Get labels
    train_labels = train_df["outcome"].values
    val_labels = val_df["outcome"].values

    # Create datasets
    train_dataset = ClinicalDataset(train_features, train_labels)
    val_dataset = ClinicalDataset(val_features, val_labels)

    return train_dataset, val_dataset, preprocessor


def test_data_pipeline():
    """Test the clinical data pipeline."""
    logging.basicConfig(level=logging.INFO)
    print("Testing clinical data pipeline...")

    # Load data
    df = load_clinical_data()

    print(f"\nDataset summary:")
    print(f"  Total patients: {len(df)}")
    print(f"  Outcome distribution: {df['outcome'].value_counts().to_dict()}")

    # Test preprocessing
    preprocessor = ClinicalFeaturePreprocessor()
    features = preprocessor.fit_transform(df)

    print(f"\nFeature matrix shape: {features.shape}")
    print(f"  Expected: ({len(df)}, {CLINICAL_CONFIG['input_dim']})")

    # Check for NaN
    nan_count = np.isnan(features).sum()
    print(f"  NaN values: {nan_count}")

    # Test dataset creation
    n = len(df)
    indices = np.arange(n)
    train_indices = indices[: int(0.8 * n)]
    val_indices = indices[int(0.8 * n) :]

    train_ds, val_ds, _ = create_datasets(df, train_indices, val_indices)

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_ds)}")
    print(f"  Val: {len(val_ds)}")

    # Test getting a sample
    features, label = train_ds[0]
    print(f"\nSample:")
    print(f"  Features shape: {features.shape}")
    print(f"  Label: {label}")


if __name__ == "__main__":
    test_data_pipeline()
