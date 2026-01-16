"""Configuration for Experiment 2: EEG + SMILES fusion."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR.parent / "asm_data"
EEG_DIR = DATA_DIR / "Alfred" / "EEG"
CSV_PATH = DATA_DIR / "alfred_1st_regimen.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"

# SMILES embeddings from Experiment 1
CHEMBERTA_EMBEDDINGS = OUTPUTS_DIR / "chemberta_asm_embeddings.npy"
SMILESTRF_EMBEDDINGS = OUTPUTS_DIR / "smilestrf_asm_embeddings.npy"

# EEG processing parameters
EEG_CONFIG = {
    "target_sr": 200,           # Sample rate for LaBraM
    "min_duration_sec": 600,    # 10 minutes minimum
    "skip_start_sec": 300,      # Skip first 5 minutes
    "use_duration_sec": 1200,   # Use up to 20 minutes (after skip)
    "window_sec": 10,           # 10-second windows
    "lowcut": 0.1,              # Bandpass low cutoff
    "highcut": 75.0,            # Bandpass high cutoff
    "notch_freq": 50.0,         # Power line noise frequency
}

# Derived values
MAX_WINDOWS = int(EEG_CONFIG["use_duration_sec"] / EEG_CONFIG["window_sec"])  # 120

# Training configuration
TRAIN_CONFIG = {
    "batch_size": 4,  # Reduced for memory efficiency with LaBraM
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "epochs": 100,
    "patience": 20,
    "n_folds": 5,
    "seed": 42,
}

# Encoder-specific batch sizes
BATCH_SIZE_BY_ENCODER = {
    "simplecnn": 8,
    "labram": 1,  # LaBraM is very memory-intensive
}

# Encoder-specific window chunk sizes for memory efficiency
CHUNK_SIZE_BY_ENCODER = {
    "simplecnn": 32,
    "labram": 4,  # Process fewer windows at once for LaBraM
}

# Model configuration
MODEL_CONFIG = {
    "hidden_dim": 256,
    "num_experts": 4,
    "top_k": 2,
    "num_heads": 4,
    "num_layers": 2,
    "dropout": 0.1,
    "aux_loss_weight": 0.1,
    "num_classes": 2,
}

# EEG embedding dimensions (will be populated after model loading)
EEG_EMBED_DIMS = {
    "labram": 128,  # Reduced for memory efficiency
    "simplecnn": 256,
    "eegformer": 256,
    "eeg2vec": 256,
}

# SMILES embedding dimensions
SMILES_EMBED_DIMS = {
    "chemberta": 768,
    "smilestrf": 256,
}

# Experiment matrix
EXPERIMENTS = [
    {"eeg_model": "labram", "smiles_model": "chemberta", "fusion": "mlp"},
    {"eeg_model": "labram", "smiles_model": "chemberta", "fusion": "fusemoe"},
    {"eeg_model": "labram", "smiles_model": "smilestrf", "fusion": "mlp"},
    {"eeg_model": "labram", "smiles_model": "smilestrf", "fusion": "fusemoe"},
]

# ASM name mapping (same as exp1)
ASM_NAME_MAP = {
    "LEV": "Levetiracetam",
    "VPA": "Valproate",
    "CBZ": "Carbamazepine",
    "PHT": "Phenytoin",
    "LTG": "Lamotrigine",
    "TPM": "Topiramate",
    "OXC": "Oxcarbazepine",
    "ZNS": "Zonisamide",
    "LCM": "Lacosamide",
    "PER": "Perampanel",
    "BRV": "Brivaracetam",
    "ESL": "Eslicarbazepine",
    "PB": "Phenobarbital",
    "CLB": "Clobazam",
    "CZP": "Clonazepam",
    "GBP": "Gabapentin",
    "PGB": "Pregabalin",
    "ETX": "Ethosuximide",
    "RFM": "Rufinamide",
    "VGB": "Vigabatrin",
}
