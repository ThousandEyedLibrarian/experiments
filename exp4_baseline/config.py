"""Configuration for Experiment 4: Clinical features baseline."""

from pathlib import Path

# Paths (following exp3 pattern)
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR.parent / "asm_data"
CSV_PATH = DATA_DIR / "alfred_1st_regimen.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "exp4_results"

# Clinical feature configuration
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
    # Final dimension: 13 binary + 1 numeric + 3 (lesion one-hot) + 3 (eeg_cat one-hot) = 20
    "input_dim": 20,
}

# Experiment 4a: Simple MLP
CONFIG_4A = {
    "batch_size": 16,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 100,
    "patience": 20,
    "hidden_dims": [64, 32],
    "dropout": 0.3,
    "num_classes": 2,
}

# Experiment 4b: MLP with Self-Attention (following Feng et al. 2025)
CONFIG_4B = {
    "batch_size": 16,
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,
    "epochs": 100,
    "patience": 20,
    "hidden_dim": 64,
    "num_heads": 4,
    "num_layers": 2,
    "dropout": 0.2,
    "num_classes": 2,
}

# Cross-validation configuration (matching exp1-3)
CV_CONFIG = {
    "n_splits": 5,
    "shuffle": True,
    "random_state": 42,
}

# Outcome mapping: 1=failure->0, 2=success->1 (matching exp3)
OUTCOME_MAPPING = {1: 0, 2: 1}

# Experiments to run
EXPERIMENTS = [
    {"name": "exp4a_mlp", "model": "mlp"},
    {"name": "exp4b_attention", "model": "attention"},
]
