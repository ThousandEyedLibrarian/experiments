"""Experiment 4: Clinical features baseline for ASM outcome prediction.

This experiment establishes a baseline using only demographic and clinical features,
without any embedding-based modalities (EEG, LLM text, SMILES).

Experiments:
- exp4a: Simple MLP classifier
- exp4b: MLP with self-attention over features (following Feng et al. 2025)

Reference:
- Feng et al. 2025, "Integrative Deep Learning of Genomic and Clinical Data
  for Predicting Treatment Response in Newly Diagnosed Epilepsy", Neurology.
"""

from .config import CONFIG_4A, CONFIG_4B, CV_CONFIG, CLINICAL_CONFIG
from .data_pipeline import ClinicalDataset, ClinicalFeaturePreprocessor, load_clinical_data
from .models import ClinicalMLP, ClinicalAttentionMLP
from .training import run_cross_validation, train_fold

__all__ = [
    "CONFIG_4A",
    "CONFIG_4B",
    "CV_CONFIG",
    "CLINICAL_CONFIG",
    "ClinicalDataset",
    "ClinicalFeaturePreprocessor",
    "load_clinical_data",
    "ClinicalMLP",
    "ClinicalAttentionMLP",
    "run_cross_validation",
    "train_fold",
]
