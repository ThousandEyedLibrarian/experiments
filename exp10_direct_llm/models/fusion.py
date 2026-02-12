"""Fusion models for Experiment 10: Clinical + Direct LLM Text.

Late fusion architecture: each modality encoded separately,
concatenated and classified. The LLM encoder runs at training
time to enable end-to-end fine-tuning.
"""

from typing import Optional

import torch
import torch.nn as nn

from ..config import CLINICAL_DIM, LLM_MODELS, TRAINING_CONFIG
from .llm_encoder import LLMEncoder, get_llm_encoder


class ModalityEncoder(nn.Module):
    """Generic MLP encoder for a single modality."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class FusionClassifier(nn.Module):
    """MLP classifier for fused representations."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class ClinicalLLMFusion(nn.Module):
    """Fusion model for Clinical + Direct LLM Text (Experiment 10).

    Architecture:
        Clinical (19D) -> ModalityEncoder -> 64D
        Text (raw) -> LLMEncoder -> embed_dim -> ModalityEncoder -> 64D
        Concatenate -> 128D -> FusionClassifier -> 2 classes
    """

    def __init__(
        self,
        llm_encoder: LLMEncoder,
        clinical_dim: int = CLINICAL_DIM,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.llm_encoder = llm_encoder
        self.clinical_encoder = ModalityEncoder(clinical_dim, hidden_dim, dropout)
        self.text_encoder = ModalityEncoder(llm_encoder.embed_dim, hidden_dim, dropout)
        self.classifier = FusionClassifier(
            hidden_dim * 2, hidden_dim, num_classes, dropout
        )

    def forward(
        self,
        clinical: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            clinical: Clinical features (batch, 19).
            input_ids: Token IDs from LLM tokeniser (batch, seq_len).
            attention_mask: Attention mask (batch, seq_len).

        Returns:
            Logits (batch, num_classes).
        """
        # Extract text embeddings via LLM
        text_emb = self.llm_encoder(input_ids, attention_mask)

        # Encode each modality
        clinical_feat = self.clinical_encoder(clinical)
        text_feat = self.text_encoder(text_emb)

        # Late fusion
        fused = torch.cat([clinical_feat, text_feat], dim=1)
        return self.classifier(fused)


def get_model(
    llm_model: str = "pubmedbert",
    freeze: bool = True,
    unfreeze_layers: int = 0,
    device: torch.device = None,
) -> ClinicalLLMFusion:
    """Create a ClinicalLLMFusion model.

    Args:
        llm_model: Key from LLM_MODELS config.
        freeze: Whether to freeze LLM encoder weights.
        unfreeze_layers: Number of layers to unfreeze if not frozen.
        device: Device to place model on.

    Returns:
        Initialised ClinicalLLMFusion model.
    """
    dropout = TRAINING_CONFIG["dropout"]
    num_classes = TRAINING_CONFIG["num_classes"]

    llm_enc = get_llm_encoder(
        llm_model=llm_model,
        freeze=freeze,
        unfreeze_layers=unfreeze_layers,
    )

    model = ClinicalLLMFusion(
        llm_encoder=llm_enc,
        clinical_dim=CLINICAL_DIM,
        hidden_dim=64,
        num_classes=num_classes,
        dropout=dropout,
    )

    if device is not None:
        model = model.to(device)

    return model


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
