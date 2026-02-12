"""Data pipeline for Experiment 10: Direct LLM Text Modality.

Loads raw EEG report text and clinical features. Text is
pre-tokenised per LLM model so the DataLoader can collate
tensors normally, while the LLM forward pass runs at training time.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import (
    CLINICAL_CONFIG,
    CSV_PATH,
    ERROR_PATTERNS,
    MIN_REPORT_LENGTH,
    OUTCOME_MAPPING,
)

# Import clinical preprocessing from exp4
sys.path.insert(0, str(Path(__file__).parent.parent))
from exp4_baseline.data_pipeline import (
    ClinicalFeaturePreprocessor,
    clean_lesion_column,
    clean_outcome_column,
    clean_psy_column,
)

logger = logging.getLogger("exp10")


# ============================================================================
# Data Loading
# ============================================================================

def load_csv_for_llm(
    filepath: Path = CSV_PATH,
    filter_outcome: bool = True,
) -> pd.DataFrame:
    """Load CSV with text-specific filtering.

    Filters for patients with valid, non-error EEG reports
    of sufficient length.

    Args:
        filepath: Path to CSV file.
        filter_outcome: Whether to filter and map outcomes.

    Returns:
        Cleaned DataFrame with raw eeg_report text.
    """
    df = pd.read_csv(filepath)

    # Clinical column cleaning
    df = clean_psy_column(df)
    df = clean_lesion_column(df)

    # Filter for patients with valid EEG reports
    df = df[df["eeg_report"].notna()].copy()
    df = df[df["eeg_report"].str.strip() != ""].copy()

    # Filter out reports that are too short
    df["report_length"] = df["eeg_report"].str.len()
    df = df[df["report_length"] >= MIN_REPORT_LENGTH].copy()

    # Remove error patterns
    for pattern in ERROR_PATTERNS:
        df = df[~df["eeg_report"].str.contains(pattern, na=False)]

    # Convert outcome to numeric
    df["outcome"] = pd.to_numeric(df["outcome"], errors="coerce")

    if filter_outcome:
        df = clean_outcome_column(df)

    df = df.reset_index(drop=True)
    logger.info(f"Loaded {len(df)} patients with valid EEG reports")
    return df


# ============================================================================
# Dataset
# ============================================================================

class ClinicalLLMDataset(Dataset):
    """Dataset combining clinical features and pre-tokenised text.

    Text is tokenised once at dataset creation using the target
    LLM's tokeniser. The LLM forward pass runs during training.
    """

    def __init__(
        self,
        clinical_features: np.ndarray,
        input_ids: torch.Tensor,
        attention_masks: torch.Tensor,
        labels: np.ndarray,
    ):
        """Initialise dataset.

        Args:
            clinical_features: Clinical feature array (n_samples, 19).
            input_ids: Token IDs (n_samples, seq_len).
            attention_masks: Attention masks (n_samples, seq_len).
            labels: Label array (n_samples,).
        """
        self.clinical_features = torch.from_numpy(clinical_features).float()
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a sample.

        Returns:
            Tuple of (clinical, input_ids, attention_mask, label).
        """
        return (
            self.clinical_features[idx],
            self.input_ids[idx],
            self.attention_masks[idx],
            self.labels[idx],
        )


# ============================================================================
# Data Preparation
# ============================================================================

def prepare_data(
    llm_model_name: str,
    tokeniser,
    max_length: int = 512,
) -> Tuple[pd.DataFrame, torch.Tensor, torch.Tensor]:
    """Load data and pre-tokenise text for a specific LLM.

    Args:
        llm_model_name: LLM model key (for logging).
        tokeniser: HuggingFace tokeniser instance.
        max_length: Maximum token sequence length.

    Returns:
        Tuple of (df, input_ids, attention_masks).
    """
    logger.info(f"Preparing data for LLM: {llm_model_name}")

    df = load_csv_for_llm(filter_outcome=True)

    # Pre-tokenise all texts
    texts = df["eeg_report"].tolist()
    logger.info(f"Tokenising {len(texts)} reports (max_length={max_length})")

    tokens = tokeniser(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    logger.info(f"Token shape: {tokens['input_ids'].shape}")
    return df, tokens["input_ids"], tokens["attention_mask"]


def create_datasets(
    df: pd.DataFrame,
    input_ids: torch.Tensor,
    attention_masks: torch.Tensor,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
) -> Tuple[ClinicalLLMDataset, ClinicalLLMDataset, ClinicalFeaturePreprocessor]:
    """Create train/val datasets for a single CV fold.

    Args:
        df: DataFrame with clinical data and outcomes.
        input_ids: Pre-tokenised input IDs (n_samples, seq_len).
        attention_masks: Pre-tokenised attention masks (n_samples, seq_len).
        train_indices: Training fold indices.
        val_indices: Validation fold indices.

    Returns:
        Tuple of (train_dataset, val_dataset, preprocessor).
    """
    # Split data
    train_df = df.iloc[train_indices].copy()
    val_df = df.iloc[val_indices].copy()

    # Fit clinical preprocessor on training data only
    preprocessor = ClinicalFeaturePreprocessor()
    train_clinical = preprocessor.fit_transform(train_df)
    val_clinical = preprocessor.transform(val_df)

    # Split tokenised text
    train_ids = input_ids[train_indices]
    train_masks = attention_masks[train_indices]
    val_ids = input_ids[val_indices]
    val_masks = attention_masks[val_indices]

    # Labels
    train_labels = train_df["outcome"].values
    val_labels = val_df["outcome"].values

    train_dataset = ClinicalLLMDataset(
        clinical_features=train_clinical,
        input_ids=train_ids,
        attention_masks=train_masks,
        labels=train_labels,
    )

    val_dataset = ClinicalLLMDataset(
        clinical_features=val_clinical,
        input_ids=val_ids,
        attention_masks=val_masks,
        labels=val_labels,
    )

    return train_dataset, val_dataset, preprocessor
