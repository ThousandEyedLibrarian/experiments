"""Data pipeline for Experiment 17: focal-only quad-modal fusion.

Wraps exp7's quad data preparation and applies a focal-epilepsy filter to the
deduped cohort BEFORE the CV split. The `focal` column is coded 1.0 = focal,
0.0 = generalised in the Alfred cohort (a small number of blanks are dropped by
the numeric filter rather than mode-imputed, so the focal subset is
conservative). create_quad_modality_datasets is re-used unchanged.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from exp7_all_modalities.data_pipeline import (  # noqa: E402,F401
    create_quad_modality_datasets,
    prepare_quad_modality_data,
)

logger = logging.getLogger("exp17")


def prepare_focal_quad_data(
    text_model: str = "clinicalbert",
    smiles_model: str = "chemberta",
) -> Tuple[pd.DataFrame, "object", Dict[str, int], Dict, Dict]:
    """exp7's quad cohort restricted to focal-epilepsy patients."""
    df, smiles_embeddings, smiles_indices, text_embeddings, eeg_data = prepare_quad_modality_data(
        text_model, smiles_model,
    )
    n_all = len(df)
    focal_mask = pd.to_numeric(df["focal"], errors="coerce") == 1
    df = df[focal_mask].reset_index(drop=True)
    logger.info(f"Focal filter: {len(df)}/{n_all} patients are focal epilepsy")
    return df, smiles_embeddings, smiles_indices, text_embeddings, eeg_data
