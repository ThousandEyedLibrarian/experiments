"""EEG data loading, preprocessing, and windowing pipeline."""

import logging
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np
import pandas as pd

from .config import (
    CSV_PATH,
    EEG_CONFIG,
    EEG_DIR,
    MAX_WINDOWS,
)

# Suppress MNE verbose output
mne.set_log_level("ERROR")
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = logging.getLogger("exp2")


def extract_patient_id(filename: str) -> Optional[str]:
    """Extract patient ID from EEG filename.

    Handles various naming conventions:
    - 083_7085712_15-2-2019.edf -> 083
    - 093,EEG,07022018.edf -> 093
    - 1002,EEG,7565706.edf -> 1002
    - 138_15-3-2018.edf -> 138
    """
    basename = Path(filename).stem

    # Try comma-separated format first (most common)
    if "," in basename:
        pid = basename.split(",")[0]
        return pid.strip()

    # Try underscore-separated format
    if "_" in basename:
        pid = basename.split("_")[0]
        return pid.strip()

    return None


def build_patient_eeg_map(eeg_dir: Path = EEG_DIR) -> Dict[str, Path]:
    """Build mapping from patient ID to EEG file path.

    Returns:
        Dictionary mapping patient ID (string) to EEG file path.
    """
    patient_map = {}

    for edf_file in eeg_dir.glob("*.edf"):
        pid = extract_patient_id(edf_file.name)
        if pid:
            # If multiple files per patient, keep the first one
            if pid not in patient_map:
                patient_map[pid] = edf_file

    return patient_map


def load_edf(filepath: Path, target_sr: int = 200) -> Tuple[np.ndarray, float, List[str]]:
    """Load EDF file and return raw data.

    Args:
        filepath: Path to .edf file.
        target_sr: Target sample rate for resampling.

    Returns:
        Tuple of (data array [channels x samples], sample rate, channel names).
    """
    # Try different encodings for annotation channels
    encodings = ["utf-8", "latin1", "iso-8859-1"]
    raw = None
    successful_encoding = None

    for encoding in encodings:
        try:
            raw = mne.io.read_raw_edf(
                filepath,
                preload=True,
                verbose=False,
                encoding=encoding,
            )
            successful_encoding = encoding
            break
        except Exception as e:
            logger.debug(f"Failed to load {filepath.name} with encoding {encoding}: {e}")
            continue

    if raw is None:
        logger.error(f"Could not load EDF file with any encoding: {filepath}")
        raise ValueError(f"Could not load EDF file: {filepath}")

    if successful_encoding != "utf-8":
        logger.debug(f"Loaded {filepath.name} with fallback encoding: {successful_encoding}")

    # Pick only EEG channels (exclude ECG, EOG, EMG, etc.)
    try:
        raw.pick_types(eeg=True, exclude=[])
    except Exception as e:
        logger.debug(f"Could not filter EEG channels for {filepath.name}: {e}, keeping all channels")

    original_sr = raw.info["sfreq"]

    # Resample if needed
    if original_sr != target_sr:
        logger.debug(f"Resampling {filepath.name} from {original_sr}Hz to {target_sr}Hz")
        raw.resample(target_sr, verbose=False)

    data = raw.get_data()  # Shape: (n_channels, n_samples)
    ch_names = raw.ch_names

    return data, target_sr, ch_names


def apply_filters(
    data: np.ndarray,
    sr: float,
    lowcut: float = 0.1,
    highcut: float = 75.0,
    notch_freq: float = 50.0,
) -> np.ndarray:
    """Apply bandpass and notch filters to EEG data.

    Args:
        data: EEG data array [channels x samples].
        sr: Sample rate.
        lowcut: Low frequency cutoff for bandpass.
        highcut: High frequency cutoff for bandpass.
        notch_freq: Frequency for notch filter (power line noise).

    Returns:
        Filtered data array.
    """
    n_channels = data.shape[0]

    # Create MNE RawArray for filtering
    info = mne.create_info(
        ch_names=[f"EEG{i}" for i in range(n_channels)],
        sfreq=sr,
        ch_types="eeg",
    )
    raw = mne.io.RawArray(data, info, verbose=False)

    # Apply bandpass filter
    raw.filter(lowcut, highcut, verbose=False)

    # Apply notch filter for power line noise
    raw.notch_filter(notch_freq, verbose=False)

    return raw.get_data()


def extract_time_window(
    data: np.ndarray,
    sr: float,
    skip_start_sec: float = 300,
    use_duration_sec: float = 1200,
    min_duration_sec: float = 600,
) -> Optional[np.ndarray]:
    """Extract the relevant time window from EEG data.

    Per OBJECTIVES.md:
    - Skip first 5 minutes
    - Use up to 20 minutes of data
    - Reject EEGs shorter than 10 minutes total

    Args:
        data: EEG data [channels x samples].
        sr: Sample rate.
        skip_start_sec: Seconds to skip at start (5 min = 300s).
        use_duration_sec: Max duration to use (20 min = 1200s).
        min_duration_sec: Minimum total duration required (10 min = 600s).

    Returns:
        Extracted data or None if too short.
    """
    n_samples = data.shape[1]
    total_duration = n_samples / sr

    # Reject if total duration < minimum
    if total_duration < min_duration_sec:
        return None

    # Calculate start and end samples
    start_sample = int(skip_start_sec * sr)

    # If the EEG is shorter than skip_start + some data, adjust
    if start_sample >= n_samples:
        # Not enough data after skipping
        return None

    # Calculate end sample
    end_sample = int(start_sample + use_duration_sec * sr)
    end_sample = min(end_sample, n_samples)

    return data[:, start_sample:end_sample]


def create_windows(
    data: np.ndarray,
    sr: float,
    window_sec: float = 10,
    max_windows: int = MAX_WINDOWS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split EEG data into fixed-size windows with padding.

    Args:
        data: EEG data [channels x samples].
        sr: Sample rate.
        window_sec: Window duration in seconds.
        max_windows: Maximum number of windows.

    Returns:
        Tuple of:
        - windows: Array of shape [max_windows, channels, samples_per_window]
        - padding_mask: Boolean array [max_windows], True for padded windows
    """
    n_channels, n_samples = data.shape
    samples_per_window = int(window_sec * sr)

    # Calculate actual number of windows
    n_windows = n_samples // samples_per_window
    n_windows = min(n_windows, max_windows)

    # Initialize output arrays
    windows = np.zeros((max_windows, n_channels, samples_per_window), dtype=np.float32)
    padding_mask = np.ones(max_windows, dtype=bool)  # True = padded

    # Fill in actual windows
    for i in range(n_windows):
        start = i * samples_per_window
        end = start + samples_per_window
        windows[i] = data[:, start:end]
        padding_mask[i] = False  # Not padded

    return windows, padding_mask


class EEGPreprocessor:
    """EEG preprocessing pipeline for loading, filtering, and windowing."""

    def __init__(
        self,
        target_sr: int = EEG_CONFIG["target_sr"],
        min_duration_sec: float = EEG_CONFIG["min_duration_sec"],
        skip_start_sec: float = EEG_CONFIG["skip_start_sec"],
        use_duration_sec: float = EEG_CONFIG["use_duration_sec"],
        window_sec: float = EEG_CONFIG["window_sec"],
        lowcut: float = EEG_CONFIG["lowcut"],
        highcut: float = EEG_CONFIG["highcut"],
        notch_freq: float = EEG_CONFIG["notch_freq"],
    ):
        self.target_sr = target_sr
        self.min_duration_sec = min_duration_sec
        self.skip_start_sec = skip_start_sec
        self.use_duration_sec = use_duration_sec
        self.window_sec = window_sec
        self.lowcut = lowcut
        self.highcut = highcut
        self.notch_freq = notch_freq

        self.samples_per_window = int(window_sec * target_sr)
        self.max_windows = int(use_duration_sec / window_sec)

    def process(self, edf_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
        """Process a single EEG file.

        Args:
            edf_path: Path to .edf file.

        Returns:
            Tuple of (windows, padding_mask, n_channels) or None if invalid.
            - windows: [max_windows, n_channels, samples_per_window]
            - padding_mask: [max_windows] boolean, True for padded
            - n_channels: Number of EEG channels
        """
        logger.debug(f"Processing EDF: {edf_path.name}")

        # Load EDF
        data, sr, ch_names = load_edf(edf_path, self.target_sr)
        n_channels = data.shape[0]
        total_duration_sec = data.shape[1] / sr
        logger.debug(f"  Loaded: {n_channels} channels, {total_duration_sec:.1f}s duration")

        # Apply filters
        data = apply_filters(
            data, sr,
            lowcut=self.lowcut,
            highcut=self.highcut,
            notch_freq=self.notch_freq,
        )
        logger.debug(f"  Applied filters: {self.lowcut}-{self.highcut}Hz bandpass, {self.notch_freq}Hz notch")

        # Extract time window
        data = extract_time_window(
            data, sr,
            skip_start_sec=self.skip_start_sec,
            use_duration_sec=self.use_duration_sec,
            min_duration_sec=self.min_duration_sec,
        )

        if data is None:
            logger.debug(f"  Rejected: duration {total_duration_sec:.1f}s < minimum {self.min_duration_sec}s")
            return None

        extracted_duration = data.shape[1] / sr
        logger.debug(f"  Extracted: {extracted_duration:.1f}s after skipping first {self.skip_start_sec}s")

        # Create windows
        windows, padding_mask = create_windows(
            data, sr,
            window_sec=self.window_sec,
            max_windows=self.max_windows,
        )

        n_valid_windows = (~padding_mask).sum()
        logger.debug(f"  Windows: {n_valid_windows}/{len(padding_mask)} valid ({self.window_sec}s each)")

        return windows, padding_mask, n_channels


def get_valid_patient_eeg_pairs(
    csv_path: Path = CSV_PATH,
    eeg_dir: Path = EEG_DIR,
) -> pd.DataFrame:
    """Get dataframe of patients with valid EEG files and outcomes.

    Returns:
        DataFrame with columns: pid, outcome, eeg_path, ASM
    """
    # Load CSV
    df = pd.read_csv(csv_path)

    # Filter for valid outcomes
    df["outcome"] = pd.to_numeric(df["outcome"], errors="coerce")
    df = df[df["outcome"].isin([1, 2])].copy()

    # Map outcomes: 1 (failure) -> 0, 2 (success) -> 1
    df["outcome"] = df["outcome"].map({1: 0, 2: 1})

    # Build EEG map
    eeg_map = build_patient_eeg_map(eeg_dir)

    # Add EEG paths
    df["eeg_path"] = df["pid"].astype(str).map(eeg_map)

    # Filter for patients with EEG files
    df = df[df["eeg_path"].notna()].copy()

    # Select relevant columns
    result = df[["pid", "outcome", "eeg_path", "ASM"]].copy()
    result["eeg_path"] = result["eeg_path"].astype(str)

    return result


def test_pipeline():
    """Test the EEG pipeline on a sample file."""
    print("Testing EEG pipeline...")

    # Get patient-EEG pairs
    df = get_valid_patient_eeg_pairs()
    print(f"Found {len(df)} patients with EEG and valid outcomes")

    if len(df) == 0:
        print("No valid patients found!")
        return

    # Test on first patient
    sample = df.iloc[0]
    print(f"\nTesting on patient {sample['pid']}")
    print(f"EEG file: {sample['eeg_path']}")
    print(f"Outcome: {sample['outcome']}")
    print(f"ASM: {sample['ASM']}")

    # Process EEG
    preprocessor = EEGPreprocessor()
    result = preprocessor.process(Path(sample["eeg_path"]))

    if result is None:
        print("EEG too short, skipping")
        return

    windows, padding_mask, n_channels = result
    n_valid_windows = (~padding_mask).sum()

    print(f"\nResults:")
    print(f"  Channels: {n_channels}")
    print(f"  Window shape: {windows.shape}")
    print(f"  Valid windows: {n_valid_windows} / {len(padding_mask)}")
    print(f"  Samples per window: {windows.shape[2]}")
    print(f"  Duration per window: {windows.shape[2] / preprocessor.target_sr}s")


if __name__ == "__main__":
    test_pipeline()
