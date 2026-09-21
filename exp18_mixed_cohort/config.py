"""Configuration for Experiment 18: mixed-cohort (Melbourne + HEP1) training.

Frozen design: docs/analysis_plan_clean_rerun_exp18.md, section 6. The six
configurations are the HEP1 Table 3 rows on the cohort-portable pipeline
(ChemBERTa drug, mean-pooled ClinicalBERT text, 19-channel EEG2Vec), trained
with the same loops and hyperparameters as the external-validation scripts
(shared/portable_models.py).
"""

from shared.hep_cohort import EXPERIMENTS_ROOT

CONFIGS_NON_EEG = ("Exp4a", "Exp5a", "Exp5b")
CONFIGS_EEG = ("Exp5c", "Exp6b", "Exp7a")
CONFIGS = CONFIGS_NON_EEG + CONFIGS_EEG

# Which cached modalities each configuration consumes besides clinical features.
MODALITIES = {
    "Exp4a": (),
    "Exp5a": ("smiles",),
    "Exp5b": ("text",),
    "Exp5c": ("eeg",),
    "Exp6b": ("smiles", "eeg"),
    "Exp7a": ("text", "smiles", "eeg"),
}

# Portable model per configuration (shared/portable_models.py). Exp6b uses the
# pre-specified EEG2Vec encoder rather than the published SimpleCNN one.
PORTABLE_MODEL = {cfg: cfg for cfg in CONFIGS}
PORTABLE_MODEL["Exp6b"] = "Exp6b_eeg2vec"

SEEDS = {cfg: (42, 43, 44, 45, 46) for cfg in CONFIGS_NON_EEG}
SEEDS.update({cfg: (42, 43, 44) for cfg in CONFIGS_EEG})

# Every arm scores every outer test patient in both cohorts.
ARMS = ("mixed", "mel_only", "hep_only")
ARM_COHORTS = {"mixed": ("MEL", "HEP"), "mel_only": ("MEL",), "hep_only": ("HEP",)}

# Size-matched mixed training (Exp4a only): per test cohort, 10 draws.
SIZEMATCH_CONFIGS = ("Exp4a",)
SIZEMATCH_DRAWS = 10

N_SPLITS = 5
INNER_FRAC = 0.2
STRATIFY = ["outcome", "cohort"]

# Sensitivity analysis: drop every HEP1 patient recruited at the Royal
# Melbourne Hospital, whatever the duplicate audit finds.
RMH_PREFIX = "RMH"
SENSITIVITY_CONFIGS = ("Exp4a", "Exp5a", "Exp5b")

OUT_DIR = EXPERIMENTS_ROOT / "outputs" / "exp18_mixed_cohort"
EEG_CACHE_DIR = EXPERIMENTS_ROOT / "outputs" / "eeg_cache"
MEL_EEG_CACHE = EEG_CACHE_DIR / "processed_eeg_std19_alfred.pkl"
HEP_EEG_CACHE = EEG_CACHE_DIR / "processed_eeg_std19_hep.pkl"
