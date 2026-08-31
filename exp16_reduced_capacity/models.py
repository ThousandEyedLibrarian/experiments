"""Models for Experiment 16: reduced-capacity quad-modal fusion.

Uses exp11's QuadMLPv2, which is exp7a's QuadFusionMLP with the EEG-branch
width and aggregator exposed as parameters. The forward signature
(clinical, text, eeg, mask, smiles) is identical to exp7's QuadFusionMLP, so
exp7's training/eval helpers work unchanged.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

from .config import MLP_CONFIG

sys.path.insert(0, str(Path(__file__).parent.parent))
from exp11_eeg_upgrade.models import QuadMLPv2  # noqa: E402


def get_model(variant: dict, device: torch.device = None) -> nn.Module:
    """Create a reduced-capacity QuadMLPv2 from a VARIANT dict."""
    model = QuadMLPv2(
        hidden_dim=variant["hidden_dim"],
        num_classes=MLP_CONFIG["num_classes"],
        dropout=MLP_CONFIG["dropout"],
        eeg_encoder_type="eeg2vec",
        eeg_embed_dim=variant["eeg_embed_dim"],
        aggregator_type=variant["aggregator_type"],
    )
    if device is not None:
        model = model.to(device)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
