"""Configuration for Experiment 7: All Four Modalities Fusion."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR.parent / "asm_data"
CSV_PATH = DATA_DIR / "alfred_1st_regimen.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "exp7_results"
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
    "input_dim": 19,  # 13 binary + 4 age bins + 2 binary categorical
}

# Text embeddings (from exp1/exp3)
TEXT_EMBEDDINGS = {
    "clinicalbert": OUTPUTS_DIR / "bert_alfred_1stregimen_eeg_embeddings.npy",
    "pubmedbert": OUTPUTS_DIR / "pubmedBert_alfred_1stregimen_eeg_embeddings.npy",
}

# SMILES embeddings (from exp1/exp3)
SMILES_EMBEDDINGS = {
    "chemberta": OUTPUTS_DIR / "chemberta_asm_embeddings.npy",
}

ASM_NAMES_FILE = OUTPUTS_DIR / "asm_drug_names.txt"

# Embedding dimensions
CLINICAL_DIM = 19
TEXT_DIM = 768
EEG_DIM = 256  # SimpleCNN output after aggregation
SMILES_DIM = 768  # ChemBERTa

# EEG processing parameters
EEG_CONFIG = {
    "target_sr": 200,
    "min_duration_sec": 600,
    "skip_start_sec": 300,
    "use_duration_sec": 1200,
    "window_sec": 10,
}

MAX_WINDOWS = int(EEG_CONFIG["use_duration_sec"] / EEG_CONFIG["window_sec"])  # 120

# EEG encoder config
EEG_ENCODER_CONFIG = {
    "encoder_type": "eeg2vec",
    "n_channels": 27,
    "n_times": 2000,  # 10s @ 200Hz
    "embed_dim": 256,
    "num_heads": 4,
    "num_layers": 2,
    "max_windows": MAX_WINDOWS,
    "window_chunk_size": 32,
}

# MLP training configuration
MLP_CONFIG = {
    "batch_size": 8,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 100,
    "patience": 20,
    "dropout": 0.3,
    "hidden_dim": 64,
    "num_classes": 2,
}

# FuseMoE training configuration
MOE_CONFIG = {
    "batch_size": 8,
    "learning_rate": 5e-5,
    "weight_decay": 1e-4,
    "epochs": 100,
    "patience": 20,
    "dropout": 0.1,
    "hidden_dim": 256,
    "num_classes": 2,
    "num_experts": 4,
    "top_k": 2,
    "num_heads": 4,
    "num_moe_layers": 2,
    "aux_loss_weight": 0.1,
}

# Cross-validation config
CV_CONFIG = {
    "n_splits": 5,
    "shuffle": True,
    "random_state": 42,
}

# ASM name mapping
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
    # Exp7a: MLP fusion
    {
        "name": "exp7a_clinicalbert_chemberta",
        "fusion": "mlp",
        "text_model": "clinicalbert",
        "smiles_model": "chemberta",
    },
    {
        "name": "exp7a_pubmedbert_chemberta",
        "fusion": "mlp",
        "text_model": "pubmedbert",
        "smiles_model": "chemberta",
    },
    # Exp7b: FuseMoE
    {
        "name": "exp7b_clinicalbert_chemberta",
        "fusion": "moe",
        "text_model": "clinicalbert",
        "smiles_model": "chemberta",
    },
    {
        "name": "exp7b_pubmedbert_chemberta",
        "fusion": "moe",
        "text_model": "pubmedbert",
        "smiles_model": "chemberta",
    },
]
