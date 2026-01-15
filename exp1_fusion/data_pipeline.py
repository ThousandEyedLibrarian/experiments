"""Data pipeline for Experiment 1: LLM + SMILES fusion."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional

from .config import (
    CSV_PATH, TEXT_EMBEDDINGS, SMILES_EMBEDDINGS, ASM_NAMES_FILE,
    ASM_NAME_MAPPING, OUTCOME_MAPPING, TEXT_DIM, SMILES_DIMS
)


def load_asm_drug_names(filepath: str = ASM_NAMES_FILE) -> List[str]:
    """Load ordered drug names from file."""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f.readlines()]


def load_csv_data(filepath: str = CSV_PATH, filter_outcome: bool = True) -> pd.DataFrame:
    """
    Load and preprocess the CSV data.

    Applies the same filtering as the original embedding generation script:
    1. Remove NaN/empty EEG reports
    2. Filter reports < 20 characters
    3. Remove error patterns
    4. (Optionally) Filter for valid outcomes

    Args:
        filepath: Path to CSV file
        filter_outcome: Whether to filter for valid outcomes (1 or 2)
    """
    df = pd.read_csv(filepath)

    # Filter for patients with valid EEG reports
    df = df[df['eeg_report'].notna()].copy()
    df = df[df['eeg_report'].str.strip() != ''].copy()

    # Filter out reports that are too short (matches original script)
    MIN_REPORT_LENGTH = 20
    df['report_length'] = df['eeg_report'].str.len()
    df = df[df['report_length'] >= MIN_REPORT_LENGTH].copy()

    # Remove error patterns (matches original script)
    error_patterns = ['Err:', 'Exceed time window', '#N/A', 'No EEG data']
    for pattern in error_patterns:
        df = df[~df['eeg_report'].str.contains(pattern, na=False)]

    # Convert outcome to numeric (handles string values)
    df['outcome'] = pd.to_numeric(df['outcome'], errors='coerce')

    if filter_outcome:
        # Filter for valid outcomes (1 or 2)
        df = df[df['outcome'].isin([1, 2])].copy()
        # Convert outcome to int
        df['outcome'] = df['outcome'].astype(int)

    # Reset index
    df = df.reset_index(drop=True)

    return df


def align_embeddings_with_csv(
    df: pd.DataFrame,
    text_embeddings: np.ndarray,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Align text embeddings with CSV data.

    Returns filtered text embeddings, ASM labels, and outcome labels.

    Note: Assumes text_embeddings were generated from the same filtered CSV
    in the same order.
    """
    # Get ASM labels and outcomes
    asm_labels = df['ASM'].tolist()
    outcomes = df['outcome'].values

    # Map outcomes: 1->0 (failure), 2->1 (success)
    outcomes = np.array([OUTCOME_MAPPING.get(o, o) for o in outcomes])

    # Verify alignment
    if len(text_embeddings) != len(df):
        raise ValueError(
            f"Text embeddings ({len(text_embeddings)}) don't match "
            f"CSV rows ({len(df)}). Regenerate embeddings with same filtering."
        )

    return text_embeddings, asm_labels, outcomes


class ASMFusionDataset(Dataset):
    """
    PyTorch Dataset for ASM outcome prediction with text + SMILES fusion.

    Aligns patient-level text embeddings with per-drug SMILES embeddings
    based on each patient's prescribed ASM.
    """

    def __init__(
        self,
        text_embeddings: np.ndarray,
        asm_labels: List[str],
        outcomes: np.ndarray,
        smiles_embeddings: np.ndarray,
        drug_names: List[str],
        asm_mapping: Dict[str, str] = ASM_NAME_MAPPING,
    ):
        """
        Args:
            text_embeddings: (n_samples, text_dim) text/EEG report embeddings
            asm_labels: List of ASM names from CSV (length n_samples)
            outcomes: (n_samples,) binary outcome labels (0=failure, 1=success)
            smiles_embeddings: (n_drugs, smiles_dim) SMILES embeddings
            drug_names: Ordered list of drug names matching smiles_embeddings
            asm_mapping: Dict mapping CSV ASM names to drug_names keys
        """
        self.text_emb = torch.FloatTensor(text_embeddings)
        self.outcomes = torch.LongTensor(outcomes)

        # Create drug name to index mapping
        drug_to_idx = {name: i for i, name in enumerate(drug_names)}

        # Pre-compute aligned SMILES embeddings for each patient
        n_samples = len(asm_labels)
        smiles_dim = smiles_embeddings.shape[1]
        self.smiles_emb = torch.zeros(n_samples, smiles_dim)

        # Track unknown ASMs
        unknown_asms = set()

        for i, asm in enumerate(asm_labels):
            # Normalize ASM name
            normalized = asm_mapping.get(asm.strip(), asm.strip())

            if normalized in drug_to_idx:
                self.smiles_emb[i] = torch.FloatTensor(
                    smiles_embeddings[drug_to_idx[normalized]]
                )
            else:
                # Use mean embedding for unknown drugs
                self.smiles_emb[i] = torch.FloatTensor(
                    smiles_embeddings.mean(axis=0)
                )
                unknown_asms.add(asm)

        if unknown_asms:
            print(f"Warning: Unknown ASMs (using mean embedding): {unknown_asms}")

    def __len__(self) -> int:
        return len(self.outcomes)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'text_emb': self.text_emb[idx],
            'smiles_emb': self.smiles_emb[idx],
            'label': self.outcomes[idx],
        }


def create_dataloaders(
    text_model: str,
    smiles_model: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int = 16,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for a specific model combination.

    Args:
        text_model: 'clinicalbert' or 'pubmedbert'
        smiles_model: 'chemberta' or 'smilestrf'
        train_idx: Training sample indices
        val_idx: Validation sample indices
        batch_size: Batch size for dataloaders

    Returns:
        train_loader, val_loader
    """
    # Load data
    df = load_csv_data()
    text_embeddings = np.load(TEXT_EMBEDDINGS[text_model])
    smiles_embeddings = np.load(SMILES_EMBEDDINGS[smiles_model])
    drug_names = load_asm_drug_names()

    # Align with CSV
    text_emb, asm_labels, outcomes = align_embeddings_with_csv(df, text_embeddings)

    # Create datasets for train/val splits
    train_dataset = ASMFusionDataset(
        text_embeddings=text_emb[train_idx],
        asm_labels=[asm_labels[i] for i in train_idx],
        outcomes=outcomes[train_idx],
        smiles_embeddings=smiles_embeddings,
        drug_names=drug_names,
    )

    val_dataset = ASMFusionDataset(
        text_embeddings=text_emb[val_idx],
        asm_labels=[asm_labels[i] for i in val_idx],
        outcomes=outcomes[val_idx],
        smiles_embeddings=smiles_embeddings,
        drug_names=drug_names,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader


def get_full_dataset(
    text_model: str,
    smiles_model: str,
) -> Tuple[ASMFusionDataset, np.ndarray]:
    """
    Get the full dataset and outcomes for cross-validation splitting.

    Returns:
        dataset: Full ASMFusionDataset
        outcomes: Array of outcome labels for stratified splitting
    """
    # Load data WITHOUT outcome filtering first (to match embedding count)
    df_all = load_csv_data(filter_outcome=False)
    text_embeddings = np.load(TEXT_EMBEDDINGS[text_model])
    smiles_embeddings = np.load(SMILES_EMBEDDINGS[smiles_model])
    drug_names = load_asm_drug_names()

    # Verify embedding count matches
    if len(text_embeddings) != len(df_all):
        raise ValueError(
            f"Text embeddings ({len(text_embeddings)}) don't match "
            f"CSV rows ({len(df_all)}). Embeddings may have been generated "
            f"with different filtering."
        )

    # Now filter for valid outcomes (1 or 2)
    valid_mask = df_all['outcome'].isin([1, 2])
    df = df_all[valid_mask].copy()
    df['outcome'] = df['outcome'].astype(int)

    # Get valid indices for filtering embeddings
    valid_indices = np.where(valid_mask.values)[0]
    text_emb = text_embeddings[valid_indices]

    # Get ASM labels and outcomes
    asm_labels = df['ASM'].tolist()
    outcomes = df['outcome'].values

    # Map outcomes: 1->0 (failure), 2->1 (success)
    outcomes = np.array([OUTCOME_MAPPING.get(o, o) for o in outcomes])

    # Create full dataset
    dataset = ASMFusionDataset(
        text_embeddings=text_emb,
        asm_labels=asm_labels,
        outcomes=outcomes,
        smiles_embeddings=smiles_embeddings,
        drug_names=drug_names,
    )

    return dataset, outcomes


if __name__ == '__main__':
    # Test the data pipeline
    print("Testing data pipeline...")

    # Load CSV
    df = load_csv_data()
    print(f"Loaded CSV: {len(df)} samples")
    print(f"Outcome distribution: {df['outcome'].value_counts().to_dict()}")
    print(f"ASM distribution: {df['ASM'].value_counts().to_dict()}")

    # Test dataset creation
    for text_model in ['clinicalbert', 'pubmedbert']:
        for smiles_model in ['chemberta', 'smilestrf']:
            print(f"\nTesting {text_model} + {smiles_model}...")
            dataset, outcomes = get_full_dataset(text_model, smiles_model)
            print(f"  Dataset size: {len(dataset)}")
            sample = dataset[0]
            print(f"  Text embedding shape: {sample['text_emb'].shape}")
            print(f"  SMILES embedding shape: {sample['smiles_emb'].shape}")
            print(f"  Label: {sample['label']}")

    print("\nData pipeline test complete!")
