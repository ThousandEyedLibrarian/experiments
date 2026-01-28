"""Configuration for Experiment 5: Clinical + Single Modality Fusion."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR.parent / "asm_data"
EEG_DIR = DATA_DIR / "Alfred" / "EEG"
CSV_PATH = DATA_DIR / "alfred_1st_regimen.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "exp5_results"
EEG_CACHE_PATH = OUTPUTS_DIR / "eeg_cache" / "processed_eeg.pkl"

# Clinical feature configuration (from exp4)
CLINICAL_CONFIG = {
    "numeric_features": ["age_init"],
    "binary_features": [
        "sex",
        "pretrt_sz_5",
        "focal",
        "fam_hx",
        "febrile",
        "ci",
        "birth_t",
        "head",
        "drug",
        "alcohol",
        "cvd",
        "psy",
        "ld",
    ],
    "categorical_features": ["lesion", "eeg_cat"],
    "input_dim": 20,  # 13 binary + 1 numeric + 6 one-hot
}

# Text embeddings (from exp1/exp3)
TEXT_EMBEDDINGS = {
    "clinicalbert": OUTPUTS_DIR / "bert_alfred_1stregimen_eeg_embeddings.npy",
    "pubmedbert": OUTPUTS_DIR / "pubmedBert_alfred_1stregimen_eeg_embeddings.npy",
}

# SMILES embeddings (from exp1/exp3)
SMILES_EMBEDDINGS = {
    "chemberta": OUTPUTS_DIR / "chemberta_asm_embeddings.npy",
    "smilestrf": OUTPUTS_DIR / "smilestrf_asm_embeddings.npy",
}

ASM_NAMES_FILE = OUTPUTS_DIR / "asm_drug_names.txt"

# Embedding dimensions
CLINICAL_DIM = 20
TEXT_DIM = 768
EEG_DIM = 256  # SimpleCNN output after aggregation
SMILES_DIMS = {
    "chemberta": 768,
    "smilestrf": 256,
}

# EEG processing parameters (from exp2/exp3)
EEG_CONFIG = {
    "target_sr": 200,
    "min_duration_sec": 600,
    "skip_start_sec": 300,
    "use_duration_sec": 1200,
    "window_sec": 10,
    "lowcut": 0.1,
    "highcut": 75.0,
    "notch_freq": 50.0,
}

MAX_WINDOWS = int(EEG_CONFIG["use_duration_sec"] / EEG_CONFIG["window_sec"])  # 120

# EEG encoder config (reuse SimpleCNN from exp2)
EEG_ENCODER_CONFIG = {
    "encoder_type": "simplecnn",
    "n_channels": 27,
    "n_times": 2000,  # 10s @ 200Hz
    "embed_dim": 256,
    "num_heads": 4,
    "num_layers": 2,
    "max_windows": MAX_WINDOWS,
    "window_chunk_size": 32,
}

# Training configuration for all Exp5 variants
TRAINING_CONFIG = {
    "batch_size": 16,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 100,
    "patience": 20,
    "dropout": 0.3,
    "num_classes": 2,
}

# Cross-validation config (same as exp4)
CV_CONFIG = {
    "n_splits": 5,
    "shuffle": True,
    "random_state": 42,
}

# ASM name mapping (from exp3)
ASM_NAME_MAPPING = {
    "LEV": "Levetiracetam",
    "VPA": "Valproic_acid",
    "LTG": "Lamotrigine",
    "CBZ": "Carbamazepine",
    "cBZ": "Carbamazepine",
    "PTN": "Phenytoin",
    "TPM": "Topiramate",
    "OXC": "Oxcarbazepine",
    "LCM": "Lacosamide",
    "BRV": "Brivaracetam",
    "PER": "Perampanel",
    "ZNS": "Zonisamide",
    "GBP": "Gabapentin",
    "PGB": "Pregabalin",
    "CLB": "Clobazam",
    "CZP": "Clonazepam",
}

# Outcome mapping: 1=failure->0, 2=success->1
OUTCOME_MAPPING = {1: 0, 2: 1}

# Experiment definitions
EXPERIMENTS = [
    # Exp5a: Clinical + SMILES
    {"name": "exp5a_chemberta", "modality": "smiles", "smiles_model": "chemberta"},
    {"name": "exp5a_smilestrf", "modality": "smiles", "smiles_model": "smilestrf"},
    # Exp5b: Clinical + LLM
    {"name": "exp5b_clinicalbert", "modality": "text", "text_model": "clinicalbert"},
    {"name": "exp5b_pubmedbert", "modality": "text", "text_model": "pubmedbert"},
    # Exp5c: Clinical + EEG
    {"name": "exp5c_simplecnn", "modality": "eeg", "eeg_model": "simplecnn"},
]
