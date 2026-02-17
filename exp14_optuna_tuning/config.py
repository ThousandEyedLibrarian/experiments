"""Configuration for Experiment 14: Optuna Hyperparameter Tuning.

Tunes hyperparameters for the top 3 ASM prediction models using
Optuna TPE sampler with MedianPruner and 5-fold stratified CV.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "exp14_results"
STUDY_DB_PATH = RESULTS_DIR / "optuna_studies.db"

# Trial budget (configurable via CLI)
N_TRIALS = 100

# Cross-validation (matches all prior experiments)
CV_CONFIG = {
    "n_splits": 5,
    "shuffle": True,
    "random_state": 42,
}

# Training constants (not tuned)
TRAINING_FIXED = {
    "epochs": 100,
    "patience": 20,
    "grad_clip_norm": 1.0,
}

# Search spaces per model
# Format: "param_name": ("type", *args)
#   float: ("float", low, high, log)
#   categorical: ("categorical", [choices])
SEARCH_SPACES = {
    "exp7a_mlp": {
        "learning_rate": ("float", 5e-4, 5e-3, True),
        "weight_decay": ("float", 1e-5, 1e-3, True),
        "dropout": ("float", 0.1, 0.5, False),
        "hidden_dim": ("categorical", [32, 64, 128]),
        "batch_size": ("categorical", [4, 8, 16]),
    },
    "exp11_quadmlpv2": {
        "learning_rate": ("float", 5e-4, 5e-3, True),
        "weight_decay": ("float", 1e-5, 1e-3, True),
        "dropout": ("float", 0.1, 0.5, False),
        "hidden_dim": ("categorical", [32, 64, 128]),
        "batch_size": ("categorical", [4, 8, 16]),
        "aggregator_type": ("categorical", ["transformer", "meanmax"]),
        "eeg_embed_dim": ("categorical", [64, 128, 256]),
    },
    "exp12_fusemoe": {
        "learning_rate": ("float", 1e-5, 5e-4, True),
        "weight_decay": ("float", 1e-5, 1e-3, True),
        "dropout": ("float", 0.05, 0.3, False),
        "num_experts": ("categorical", [2, 4, 6]),
        "top_k": ("categorical", [1, 2]),
        "aux_loss_weight": ("float", 0.01, 0.5, True),
        "temp_decay": ("categorical", [None, 0.999, 0.9995, 0.9999]),
    },
}

# Baseline results for comparison
BASELINES = {
    "exp7a_mlp": {
        "auc_mean": 0.798,
        "auc_std": 0.093,
        "config": "lr=1e-3, wd=1e-4, do=0.3, hd=64, bs=8",
    },
    "exp11_quadmlpv2": {
        "auc_mean": 0.791,
        "auc_std": 0.081,
        "config": "lr=1e-3, wd=1e-4, do=0.3, hd=64, bs=8, agg=transformer, eeg=128",
    },
    "exp12_fusemoe": {
        "auc_mean": 0.760,
        "auc_std": 0.112,
        "config": "lr=5e-5, wd=1e-4, do=0.1, e=4, k=2, aux=0.1, temp=None",
    },
}

# Short names for CLI
MODEL_NAMES = ["exp7a_mlp", "exp11_quadmlpv2", "exp12_fusemoe"]
CLI_MODEL_MAP = {
    "exp7a": "exp7a_mlp",
    "exp11": "exp11_quadmlpv2",
    "exp12": "exp12_fusemoe",
}
