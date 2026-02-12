"""LLM encoder wrapper for direct text embedding at training time.

Wraps HuggingFace transformer models to extract text embeddings
from raw EEG report text. Supports frozen and fine-tunable modes.
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from ..config import LLM_MODELS, MAX_TOKEN_LENGTH

logger = logging.getLogger("exp10")


class LLMEncoder(nn.Module):
    """HuggingFace LLM wrapper for text embedding extraction.

    Loads a pre-trained transformer model and extracts [CLS] token
    embeddings from input text. Supports freezing the encoder for
    feature extraction or unfreezing for end-to-end fine-tuning.
    """

    def __init__(
        self,
        model_name: str = "NeuML/pubmedbert-base-embeddings",
        embed_dim: int = 768,
        freeze: bool = True,
        unfreeze_layers: int = 0,
        max_length: int = MAX_TOKEN_LENGTH,
        pooling: str = "cls",
    ):
        """Initialise LLM encoder.

        Args:
            model_name: HuggingFace model identifier.
            embed_dim: Output embedding dimension.
            freeze: Whether to freeze all encoder parameters.
            unfreeze_layers: If freeze=False, unfreeze only the last N
                transformer layers (0 = unfreeze all).
            max_length: Maximum token sequence length.
            pooling: Embedding extraction strategy. 'cls' extracts the
                first token (for BERT-family). 'last_token' extracts
                the last non-padded token (for decoder-only models).
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.max_length = max_length
        self.freeze = freeze
        self.pooling = pooling

        logger.info(f"Loading LLM: {model_name} (freeze={freeze})")
        self.tokeniser = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Handle models without a pad token (e.g. Qwen, GPT-style)
        if self.tokeniser.pad_token is None:
            self.tokeniser.pad_token = self.tokeniser.eos_token
            self.model.config.pad_token_id = self.tokeniser.eos_token_id

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        elif unfreeze_layers > 0:
            # Freeze everything first
            for param in self.model.parameters():
                param.requires_grad = False
            # Unfreeze last N encoder layers
            encoder_layers = self._get_encoder_layers()
            if encoder_layers is not None:
                for layer in encoder_layers[-unfreeze_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                logger.info(
                    f"Unfroze last {unfreeze_layers} of "
                    f"{len(encoder_layers)} encoder layers"
                )

        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"LLM parameters: {n_trainable:,} trainable / {n_total:,} total")

    def _get_encoder_layers(self) -> Optional[nn.ModuleList]:
        """Get the transformer encoder layers for selective unfreezing."""
        # Try common attribute paths
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layer"):
            return self.model.encoder.layer
        if hasattr(self.model, "layers"):
            return self.model.layers
        if hasattr(self.model, "h"):
            return self.model.h  # GPT-style
        logger.warning("Could not find encoder layers for selective unfreezing")
        return None

    def tokenise(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenise a batch of texts.

        Args:
            texts: List of raw text strings.

        Returns:
            Dictionary of tokenised tensors (input_ids, attention_mask).
        """
        return self.tokeniser(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from tokenised inputs.

        Args:
            input_ids: Token IDs (batch, seq_len).
            attention_mask: Attention mask (batch, seq_len).

        Returns:
            Text embeddings (batch, embed_dim).
        """
        if self.freeze:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        if self.pooling == "last_token":
            # For decoder-only models: extract last non-padded token
            # which has attended to all preceding tokens via causal mask
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(
                input_ids.size(0), device=input_ids.device
            )
            embedding = outputs.last_hidden_state[batch_indices, seq_lengths, :]
        else:
            # For BERT-family models: extract [CLS] token (first token)
            embedding = outputs.last_hidden_state[:, 0, :]

        return embedding


def get_llm_encoder(
    llm_model: str = "pubmedbert",
    freeze: bool = True,
    unfreeze_layers: int = 0,
) -> LLMEncoder:
    """Factory function for LLM encoders.

    Args:
        llm_model: Model key from LLM_MODELS config.
        freeze: Whether to freeze encoder weights.
        unfreeze_layers: Number of layers to unfreeze if not frozen.

    Returns:
        Initialised LLMEncoder.
    """
    if llm_model not in LLM_MODELS:
        raise ValueError(
            f"Unknown LLM model: {llm_model}. "
            f"Available: {list(LLM_MODELS.keys())}"
        )

    config = LLM_MODELS[llm_model]
    return LLMEncoder(
        model_name=config["model_name"],
        embed_dim=config["embed_dim"],
        freeze=freeze,
        unfreeze_layers=unfreeze_layers,
        pooling=config.get("pooling", "cls"),
    )
