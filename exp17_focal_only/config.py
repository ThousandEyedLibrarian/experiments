"""Configuration for Experiment 17: focal-only quad-modal fusion.

Reuses exp7's model, training and data functions; the only change is a
focal-epilepsy cohort filter in data_pipeline. CV contract is identical to
exp7 so folds are formed the same way (on the focal subset).
"""

CV_CONFIG = {
    "n_splits": 5,
    "shuffle": True,
    "random_state": 42,
}

# Single focal-only quad configuration (standard-capacity Exp7a architecture).
EXPERIMENTS = [
    {
        "name": "exp17_focal",
        "fusion": "mlp",
        "text_model": "clinicalbert",
        "smiles_model": "chemberta",
    },
]

from shared.cohort import ASM_NAME_MAPPING  # noqa: E402,F401
from shared.cohort import OUTCOME_MAPPING  # noqa: E402,F401
