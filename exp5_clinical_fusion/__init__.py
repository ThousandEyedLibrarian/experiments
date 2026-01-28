"""Experiment 5: Clinical + Single Modality Fusion.

Tests whether adding individual embedding modalities to clinical features
improves over the clinical-only baseline (Exp4a AUC 0.664).

Experiments:
- Exp5a: Clinical + SMILES (ChemBERTa, SMILES-Trf)
- Exp5b: Clinical + LLM (ClinicalBERT, PubMedBERT)
- Exp5c: Clinical + EEG (SimpleCNN)
"""

from .config import (
    CLINICAL_CONFIG,
    CV_CONFIG,
    EXPERIMENTS,
    TRAINING_CONFIG,
)

__all__ = [
    "CLINICAL_CONFIG",
    "CV_CONFIG",
    "EXPERIMENTS",
    "TRAINING_CONFIG",
]
