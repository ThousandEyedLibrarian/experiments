"""Data pipeline for Experiment 3: LLM + EEG + SMILES triple fusion."""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .config import (
    ASM_NAME_MAPPING,
    ASM_NAMES_FILE,
    CSV_PATH,
    EEG_DIR,
    OUTCOME_MAPPING,
    OUTPUTS_DIR,
    SMILES_EMBEDDINGS,
    TEXT_EMBEDDINGS,
)

# Import EEG processing from exp2
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from exp2_fusion.eeg_pipeline import EEGPreprocessor, get_valid_patient_eeg_pairs

logger = logging.getLogger("exp3")


class TripleModalityDataset(Dataset):
    """Dataset combining text embeddings, EEG windows, and SMILES embeddings."""

    def __init__(
        self,
        patient_ids: List[str],
        text_embeddings: Dict[str, np.ndarray],
        eeg_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        smiles_embeddings: np.ndarray,
        smiles_indices: Dict[str, int],
        labels: Dict[str, int],
        asm_drugs: Dict[str, str],
        max_channels: int = 27,
    ):
        """Initialize dataset.

        Args:
            patient_ids: List of patient IDs to include.
            text_embeddings: Dict mapping patient ID to text embedding.
            eeg_data: Dict mapping patient ID to (windows, padding_mask).
            smiles_embeddings: SMILES embeddings array [n_drugs, embed_dim].
            smiles_indices: Dict mapping ASM name to embedding index.
            labels: Dict mapping patient ID to outcome label.
            asm_drugs: Dict mapping patient ID to ASM drug name.
            max_channels: Max number of EEG channels.
        """
        self.patient_ids = patient_ids
        self.text_embeddings = text_embeddings
        self.eeg_data = eeg_data
        self.smiles_embeddings = smiles_embeddings
        self.smiles_indices = smiles_indices
        self.labels = labels
        self.asm_drugs = asm_drugs
        self.max_channels = max_channels

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single sample.

        Returns:
            Tuple of:
            - text_emb: (text_dim,)
            - eeg_windows: (num_windows, max_channels, n_times)
            - padding_mask: (num_windows,) boolean, True for padded
            - smiles_emb: (smiles_dim,)
            - label: scalar
        """
        pid = self.patient_ids[idx]

        # Get text embedding
        text_emb = torch.from_numpy(self.text_embeddings[pid]).float()

        # Get EEG data
        windows, padding_mask = self.eeg_data[pid]
        n_windows, n_channels, n_times = windows.shape

        # Pad channels if needed
        if n_channels < self.max_channels:
            padded = np.zeros((n_windows, self.max_channels, n_times), dtype=np.float32)
            padded[:, :n_channels, :] = windows
            windows = padded

        eeg_windows = torch.from_numpy(windows).float()
        padding_mask = torch.from_numpy(padding_mask).bool()

        # Get SMILES embedding
        asm = self.asm_drugs[pid]
        asm_full = ASM_NAME_MAPPING.get(asm.strip(), asm.strip())
        smiles_idx = self.smiles_indices.get(asm_full, 0)
        smiles_emb = torch.from_numpy(self.smiles_embeddings[smiles_idx]).float()

        # Get label
        label = torch.tensor(self.labels[pid], dtype=torch.long)

        return text_emb, eeg_windows, padding_mask, smiles_emb, label


def load_asm_drug_names(filepath: Path = ASM_NAMES_FILE) -> List[str]:
    """Load ordered drug names from file."""
    with open(filepath, "r") as f:
        return [line.strip() for line in f.readlines()]


def load_csv_data(filepath: Path = CSV_PATH, filter_outcome: bool = True) -> pd.DataFrame:
    """Load and preprocess the CSV data."""
    df = pd.read_csv(filepath)

    # Filter for patients with valid EEG reports
    df = df[df["eeg_report"].notna()].copy()
    df = df[df["eeg_report"].str.strip() != ""].copy()

    # Filter out reports that are too short
    MIN_REPORT_LENGTH = 20
    df["report_length"] = df["eeg_report"].str.len()
    df = df[df["report_length"] >= MIN_REPORT_LENGTH].copy()

    # Remove error patterns
    error_patterns = ["Err:", "Exceed time window", "#N/A", "No EEG data"]
    for pattern in error_patterns:
        df = df[~df["eeg_report"].str.contains(pattern, na=False)]

    # Convert outcome to numeric
    df["outcome"] = pd.to_numeric(df["outcome"], errors="coerce")

    if filter_outcome:
        df = df[df["outcome"].isin([1, 2])].copy()
        df["outcome"] = df["outcome"].astype(int)

    df = df.reset_index(drop=True)
    return df


def load_text_embeddings(
    text_model: str,
    df: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """Load text embeddings and align with patient IDs.

    Args:
        text_model: 'clinicalbert' or 'pubmedbert'
        df: Filtered DataFrame with patient info.

    Returns:
        Dict mapping patient ID to text embedding.
    """
    # Load embeddings (generated from same filtered CSV)
    emb_path = TEXT_EMBEDDINGS[text_model]
    all_embeddings = np.load(emb_path)

    # Load full CSV without outcome filtering to match embedding order
    df_all = load_csv_data(filter_outcome=False)

    if len(all_embeddings) != len(df_all):
        raise ValueError(
            f"Text embeddings ({len(all_embeddings)}) don't match "
            f"CSV rows ({len(df_all)}). Regenerate embeddings."
        )

    # Create pid -> embedding mapping
    pid_to_emb = {}
    for idx, row in df_all.iterrows():
        pid = str(row["pid"])
        pid_to_emb[pid] = all_embeddings[idx]

    # Filter to only valid patients from filtered df
    text_embeddings = {}
    for _, row in df.iterrows():
        pid = str(row["pid"])
        if pid in pid_to_emb:
            text_embeddings[pid] = pid_to_emb[pid]

    return text_embeddings


def load_smiles_embeddings(smiles_model: str) -> Tuple[np.ndarray, Dict[str, int]]:
    """Load SMILES embeddings and create index mapping.

    Args:
        smiles_model: 'chemberta' or 'smilestrf'

    Returns:
        Tuple of (embeddings array, drug name -> index mapping).
    """
    emb_path = SMILES_EMBEDDINGS[smiles_model]
    embeddings = np.load(emb_path)
    drug_names = load_asm_drug_names()
    index_map = {name: i for i, name in enumerate(drug_names)}
    return embeddings, index_map


def preprocess_all_eeg(
    df: pd.DataFrame,
    cache_path: Optional[Path] = None,
    force_reprocess: bool = False,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Preprocess all EEG files and optionally cache results."""
    # Try to load from cache
    if cache_path and cache_path.exists() and not force_reprocess:
        logger.info(f"Loading cached EEG data from {cache_path}")
        with open(cache_path, "rb") as f:
            cached_data = pickle.load(f)
        logger.info(f"Loaded {len(cached_data)} patients from cache")
        return cached_data

    logger.info(f"Preprocessing EEG data for {len(df)} patients...")
    preprocessor = EEGPreprocessor()
    eeg_data = {}
    skipped = 0

    for idx, row in df.iterrows():
        pid = str(row["pid"])
        eeg_path = Path(row["eeg_path"])

        try:
            result = preprocessor.process(eeg_path)
            if result is None:
                skipped += 1
                continue

            windows, padding_mask, n_channels = result
            eeg_data[pid] = (windows, padding_mask)

            if len(eeg_data) % 20 == 0:
                logger.info(f"  Processed {len(eeg_data)} / {len(df)} patients...")

        except Exception as e:
            logger.warning(f"Error processing {pid}: {e}")
            skipped += 1
            continue

    logger.info(f"EEG preprocessing complete: {len(eeg_data)} processed, {skipped} skipped")

    # Cache results
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(eeg_data, f)
        logger.info(f"Cached EEG data to {cache_path}")

    return eeg_data


def get_max_channels(eeg_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> int:
    """Get maximum number of channels across all EEG data."""
    return max(data[0].shape[1] for data in eeg_data.values())


def prepare_data(
    text_model: str = "clinicalbert",
    smiles_model: str = "chemberta",
    cache_eeg: bool = True,
    force_reprocess: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[np.ndarray, np.ndarray]], np.ndarray, Dict[str, int], pd.DataFrame]:
    """Prepare all data for training.

    Args:
        text_model: 'clinicalbert' or 'pubmedbert'
        smiles_model: 'chemberta' or 'smilestrf'
        cache_eeg: Whether to cache preprocessed EEG data.
        force_reprocess: Force reprocess EEG even if cache exists.

    Returns:
        Tuple of (text_embeddings, eeg_data, smiles_embeddings, smiles_indices, df).
    """
    logger.info(f"Preparing data: text={text_model}, smiles={smiles_model}")

    # Get valid patient-EEG pairs
    df = get_valid_patient_eeg_pairs()
    logger.info(f"Found {len(df)} patients with valid EEG files and outcomes")

    # Load SMILES embeddings
    smiles_embeddings, smiles_indices = load_smiles_embeddings(smiles_model)
    logger.info(f"Loaded SMILES embeddings: shape={smiles_embeddings.shape}")

    # Load text embeddings
    text_embeddings = load_text_embeddings(text_model, df)
    logger.info(f"Loaded text embeddings for {len(text_embeddings)} patients")

    # Preprocess EEG data
    cache_path = OUTPUTS_DIR / "eeg_cache" / "processed_eeg.pkl" if cache_eeg else None
    eeg_data = preprocess_all_eeg(df, cache_path, force_reprocess)

    # Find intersection of all three modalities
    common_pids = set(text_embeddings.keys()) & set(eeg_data.keys())
    logger.info(f"Patients with all three modalities: {len(common_pids)}")

    # Filter df to common patients
    df = df[df["pid"].astype(str).isin(common_pids)].copy()

    # Filter embeddings to common patients
    text_embeddings = {pid: emb for pid, emb in text_embeddings.items() if pid in common_pids}
    eeg_data = {pid: data for pid, data in eeg_data.items() if pid in common_pids}

    # Log class distribution
    outcome_counts = df["outcome"].value_counts()
    logger.info(f"Outcome distribution: {dict(outcome_counts)}")

    return text_embeddings, eeg_data, smiles_embeddings, smiles_indices, df


def create_datasets(
    text_embeddings: Dict[str, np.ndarray],
    eeg_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    smiles_embeddings: np.ndarray,
    smiles_indices: Dict[str, int],
    df: pd.DataFrame,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    max_channels: int = None,
) -> Tuple[TripleModalityDataset, TripleModalityDataset]:
    """Create train and validation datasets from fold indices."""
    # Build lookup dicts
    labels = {str(row["pid"]): OUTCOME_MAPPING.get(int(row["outcome"]), int(row["outcome"]))
              for _, row in df.iterrows()}
    asm_drugs = {str(row["pid"]): row["ASM"] for _, row in df.iterrows()}

    # Get patient IDs for each split
    pids = df["pid"].astype(str).values
    train_pids = [pids[i] for i in train_indices]
    val_pids = [pids[i] for i in val_indices]

    # Compute max channels if not provided
    if max_channels is None:
        max_channels = get_max_channels(eeg_data)

    # Create datasets
    train_dataset = TripleModalityDataset(
        patient_ids=train_pids,
        text_embeddings=text_embeddings,
        eeg_data=eeg_data,
        smiles_embeddings=smiles_embeddings,
        smiles_indices=smiles_indices,
        labels=labels,
        asm_drugs=asm_drugs,
        max_channels=max_channels,
    )

    val_dataset = TripleModalityDataset(
        patient_ids=val_pids,
        text_embeddings=text_embeddings,
        eeg_data=eeg_data,
        smiles_embeddings=smiles_embeddings,
        smiles_indices=smiles_indices,
        labels=labels,
        asm_drugs=asm_drugs,
        max_channels=max_channels,
    )

    return train_dataset, val_dataset


def test_data_pipeline():
    """Test the data pipeline."""
    logging.basicConfig(level=logging.INFO)
    print("Testing triple modality data pipeline...")

    # Prepare data
    text_emb, eeg_data, smiles_emb, smiles_idx, df = prepare_data(
        text_model="clinicalbert",
        smiles_model="chemberta",
        cache_eeg=True,
    )

    print(f"\nDataset summary:")
    print(f"  Patients with all modalities: {len(df)}")
    print(f"  Text embeddings: {len(text_emb)}, dim={next(iter(text_emb.values())).shape}")
    print(f"  EEG data: {len(eeg_data)}")
    print(f"  SMILES embeddings: {smiles_emb.shape}")

    # Test dataset creation
    n = len(df)
    indices = np.arange(n)
    train_indices = indices[: int(0.8 * n)]
    val_indices = indices[int(0.8 * n) :]

    train_ds, val_ds = create_datasets(
        text_emb, eeg_data, smiles_emb, smiles_idx, df,
        train_indices, val_indices,
    )

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_ds)}")
    print(f"  Val: {len(val_ds)}")

    # Test getting a sample
    text, eeg_windows, padding_mask, smiles, label = train_ds[0]
    print(f"\nSample shapes:")
    print(f"  Text embedding: {text.shape}")
    print(f"  EEG windows: {eeg_windows.shape}")
    print(f"  Padding mask: {padding_mask.shape}, valid: {(~padding_mask).sum()}")
    print(f"  SMILES embedding: {smiles.shape}")
    print(f"  Label: {label}")


if __name__ == "__main__":
    test_data_pipeline()
