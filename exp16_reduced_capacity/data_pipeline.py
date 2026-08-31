"""Data pipeline for Experiment 16.

Identical cohort and modalities to exp7 (Clinical + Text + EEG + SMILES), so we
re-use exp7's data functions verbatim. Keeping this thin re-export means the
107-patient quad cohort, the dedupe/leakage discipline, and the SMILES/text/EEG
loading all stay a single source of truth.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from exp7_all_modalities.data_pipeline import (  # noqa: E402,F401
    create_quad_modality_datasets,
    prepare_quad_modality_data,
)
