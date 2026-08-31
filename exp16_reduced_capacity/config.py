"""Configuration for Experiment 16: reduced-capacity quad-modal fusion.

Reuses exp7's data pipeline and paths; only the model capacity and the
experiment list differ. Training hyperparameters (batch size, LR, epochs,
patience) and the CV contract are identical to exp7 so the reduced-capacity
runs are directly comparable to the Exp7a headline.
"""

# Base training config (identical to exp7 MLP_CONFIG except hidden_dim, which
# each VARIANT overrides).
MLP_CONFIG = {
    "batch_size": 8,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 100,
    "patience": 20,
    "dropout": 0.3,
    "num_classes": 2,
}

# Cross-validation config (identical to exp7 -> same folds).
CV_CONFIG = {
    "n_splits": 5,
    "shuffle": True,
    "random_state": 42,
}

# Reduced-capacity variants. hidden_dim is the per-modality projection width and
# classifier-head width (the "lower than 64 dims" the reviewer asked about);
# eeg_embed_dim + aggregator shrink the EEG branch, which dominates the ~2M count
# in the headline model. exp14's Optuna search already preferred hidden_dim 32 /
# eeg_embed_dim 64, so "small" is the tuned-scale point and "tiny" pushes lower.
VARIANTS = [
    {
        "name": "exp16_small",
        "hidden_dim": 32,
        "eeg_embed_dim": 64,
        "aggregator_type": "meanmax",
        "text_model": "clinicalbert",
        "smiles_model": "chemberta",
    },
    {
        "name": "exp16_tiny",
        "hidden_dim": 16,
        "eeg_embed_dim": 64,
        "aggregator_type": "meanmax",
        "text_model": "clinicalbert",
        "smiles_model": "chemberta",
    },
]

# Re-used single sources of truth.
from shared.cohort import ASM_NAME_MAPPING  # noqa: E402,F401
from shared.cohort import OUTCOME_MAPPING  # noqa: E402,F401
