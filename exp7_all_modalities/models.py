"""Models for Experiment 7: All Four Modalities Fusion.

Two architectures:
- QuadFusionMLP: Late fusion with MLP classifier
- QuadFusionMoE: Cross-modal attention + Mixture of Experts
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    CLINICAL_DIM,
    EEG_ENCODER_CONFIG,
    MLP_CONFIG,
    MOE_CONFIG,
    SMILES_DIM,
    TEXT_DIM,
)

# Import EEG components from exp2
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from exp2_fusion.models.eeg_encoders import SimpleCNNEncoder
from exp2_fusion.models.eeg_transformer import EEGWindowTransformer


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


class QuadFusionMLP(nn.Module):
    """Late fusion MLP model for all 4 modalities (Experiment 7a).

    Architecture:
        Clinical (20D) -> Encoder -> 64D
        Text (768D) -> Encoder -> 64D
        EEG (windows) -> CNN+Trf -> 64D
        SMILES (768D) -> Encoder -> 64D
        Concatenate -> 256D -> Classifier -> 2 classes
    """

    def __init__(
        self,
        clinical_dim: int = CLINICAL_DIM,
        text_dim: int = TEXT_DIM,
        smiles_dim: int = SMILES_DIM,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.3,
        n_channels: int = 27,
        n_times: int = 2000,
        max_windows: int = 120,
        window_chunk_size: int = 32,
    ):
        super().__init__()

        self.window_chunk_size = window_chunk_size

        # Modality encoders
        self.clinical_encoder = ModalityEncoder(clinical_dim, hidden_dim, dropout)
        self.text_encoder = ModalityEncoder(text_dim, hidden_dim, dropout)
        self.smiles_encoder = ModalityEncoder(smiles_dim, hidden_dim, dropout)

        # EEG window encoder (SimpleCNN)
        self.window_encoder = SimpleCNNEncoder(
            n_channels=n_channels,
            n_times=n_times,
            emb_size=256,
            dropout=dropout,
        )

        # EEG window aggregator
        self.aggregator = EEGWindowTransformer(
            embed_dim=256,
            output_dim=hidden_dim,
            num_heads=4,
            num_layers=2,
            dropout=dropout,
            max_windows=max_windows,
        )

        # Classifier (4 modalities * hidden_dim)
        self.classifier = FusionClassifier(hidden_dim * 4, hidden_dim, num_classes, dropout)

    def encode_eeg_windows(
        self,
        windows: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode EEG windows with chunking for memory efficiency."""
        batch_size, num_windows, n_channels, n_times = windows.shape

        # Encode windows in chunks
        all_embeddings = []
        for i in range(0, num_windows, self.window_chunk_size):
            chunk = windows[:, i:i + self.window_chunk_size]
            chunk_size = chunk.shape[1]

            # Flatten for encoding
            chunk_flat = chunk.reshape(batch_size * chunk_size, n_channels, n_times)

            # Encode
            chunk_emb = self.window_encoder(chunk_flat)

            # Reshape back
            chunk_emb = chunk_emb.reshape(batch_size, chunk_size, -1)
            all_embeddings.append(chunk_emb)

        # Concatenate all chunks
        window_embeddings = torch.cat(all_embeddings, dim=1)

        # Aggregate
        eeg_feat = self.aggregator(window_embeddings, padding_mask)

        return eeg_feat

    def forward(
        self,
        clinical: torch.Tensor,
        text: torch.Tensor,
        eeg_windows: torch.Tensor,
        padding_mask: torch.Tensor,
        smiles: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            clinical: Clinical features (batch, 20).
            text: Text embedding (batch, 768).
            eeg_windows: EEG windows (batch, num_windows, channels, time).
            padding_mask: Padding mask (batch, num_windows).
            smiles: SMILES embedding (batch, 768).

        Returns:
            Logits (batch, num_classes).
        """
        clinical_feat = self.clinical_encoder(clinical)
        text_feat = self.text_encoder(text)
        eeg_feat = self.encode_eeg_windows(eeg_windows, padding_mask)
        smiles_feat = self.smiles_encoder(smiles)

        fused = torch.cat([clinical_feat, text_feat, eeg_feat, smiles_feat], dim=1)
        return self.classifier(fused)


class Expert(nn.Module):
    """Single expert network for Mixture of Experts."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SparseMoELayer(nn.Module):
    """Sparse Mixture of Experts layer with top-k gating."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k

        # Gate network
        self.gate = nn.Linear(input_dim, num_experts)

        # Expert networks
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim, dropout)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with sparse gating."""
        orig_shape = x.shape
        if len(orig_shape) == 3:
            batch, seq, dim = x.shape
            x = x.view(batch * seq, dim)
        else:
            batch = x.shape[0]

        # Compute gate scores
        gate_scores = self.gate(x)
        gate_probs = F.softmax(gate_scores, dim=-1)

        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Compute expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)

        # Gather selected expert outputs
        top_k_indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.shape[-1])
        selected_outputs = torch.gather(expert_outputs, 1, top_k_indices_expanded)

        # Weight and sum
        output = (selected_outputs * top_k_probs.unsqueeze(-1)).sum(dim=1)

        # Reshape if needed
        if len(orig_shape) == 3:
            output = output.view(batch, seq, -1)

        # Compute load balancing loss
        expert_usage = gate_probs.mean(dim=0)
        uniform = torch.ones_like(expert_usage) / self.num_experts
        aux_loss = (expert_usage * uniform.log() - expert_usage.log() * uniform).sum()

        return output, aux_loss


class QuadFusionMoE(nn.Module):
    """FuseMoE model for all 4 modalities (Experiment 7b).

    Architecture:
        Each modality -> Projection(dim->256) + Learnable modality token
        Cross-modal self-attention across 4 modality tokens
        2 sparse MoE layers
        Mean pool -> classifier
    """

    def __init__(
        self,
        clinical_dim: int = CLINICAL_DIM,
        text_dim: int = TEXT_DIM,
        smiles_dim: int = SMILES_DIM,
        hidden_dim: int = 256,
        num_classes: int = 2,
        num_experts: int = 4,
        top_k: int = 2,
        num_heads: int = 4,
        num_moe_layers: int = 2,
        dropout: float = 0.1,
        aux_loss_weight: float = 0.1,
        n_channels: int = 27,
        n_times: int = 2000,
        eeg_embed_dim: int = 256,
        num_eeg_layers: int = 2,
        max_windows: int = 120,
        window_chunk_size: int = 32,
    ):
        super().__init__()

        self.aux_loss_weight = aux_loss_weight
        self.window_chunk_size = window_chunk_size

        # Modality projections
        self.clinical_proj = nn.Linear(clinical_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.smiles_proj = nn.Linear(smiles_dim, hidden_dim)

        # EEG window encoder
        self.window_encoder = SimpleCNNEncoder(
            n_channels=n_channels,
            n_times=n_times,
            emb_size=eeg_embed_dim,
            dropout=dropout,
        )

        # Window aggregation transformer
        self.aggregator = EEGWindowTransformer(
            embed_dim=eeg_embed_dim,
            output_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_eeg_layers,
            dropout=dropout,
            max_windows=max_windows,
        )

        # Learnable modality tokens
        self.clinical_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.text_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.eeg_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.smiles_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Self-attention for cross-modal interaction
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # MoE fusion layers
        self.moe_layers = nn.ModuleList([
            SparseMoELayer(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim * 2,
                output_dim=hidden_dim,
                num_experts=num_experts,
                top_k=top_k,
                dropout=dropout,
            )
            for _ in range(num_moe_layers)
        ])
        self.moe_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_moe_layers)
        ])

        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode_windows_chunked(
        self,
        windows: torch.Tensor,
        chunk_size: int = 32,
    ) -> torch.Tensor:
        """Encode windows in chunks to save memory."""
        n_total = windows.shape[0]
        embeddings = []

        for i in range(0, n_total, chunk_size):
            chunk = windows[i : i + chunk_size]
            with torch.no_grad() if not self.training else torch.enable_grad():
                emb = self.window_encoder(chunk)
            embeddings.append(emb)

        return torch.cat(embeddings, dim=0)

    def forward(
        self,
        clinical: torch.Tensor,
        text: torch.Tensor,
        eeg_windows: torch.Tensor,
        padding_mask: torch.Tensor,
        smiles: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            clinical: Clinical features (batch, 20).
            text: Text embedding (batch, 768).
            eeg_windows: EEG windows (batch, num_windows, channels, time).
            padding_mask: Padding mask (batch, num_windows).
            smiles: SMILES embedding (batch, 768).

        Returns:
            Tuple of (logits, aux_loss)
        """
        batch_size, num_windows, n_channels, n_times = eeg_windows.shape

        # Project clinical
        clinical_proj = self.clinical_proj(clinical)  # (batch, hidden_dim)

        # Project text
        text_proj = self.text_proj(text)  # (batch, hidden_dim)

        # Encode EEG windows
        windows_flat = eeg_windows.reshape(batch_size * num_windows, n_channels, n_times)
        window_embeddings = self.encode_windows_chunked(windows_flat, self.window_chunk_size)
        embed_dim = window_embeddings.shape[-1]
        window_embeddings = window_embeddings.reshape(batch_size, num_windows, embed_dim)

        # Aggregate windows
        eeg_emb = self.aggregator(window_embeddings, padding_mask)  # (batch, hidden_dim)

        # Project SMILES
        smiles_proj = self.smiles_proj(smiles)  # (batch, hidden_dim)

        # Add modality tokens to embeddings
        clinical_with_token = clinical_proj.unsqueeze(1) + self.clinical_token.expand(batch_size, -1, -1)
        text_with_token = text_proj.unsqueeze(1) + self.text_token.expand(batch_size, -1, -1)
        eeg_with_token = eeg_emb.unsqueeze(1) + self.eeg_token.expand(batch_size, -1, -1)
        smiles_with_token = smiles_proj.unsqueeze(1) + self.smiles_token.expand(batch_size, -1, -1)

        # Concatenate modality tokens: (batch, 4, hidden_dim)
        modality_tokens = torch.cat([
            clinical_with_token, text_with_token, eeg_with_token, smiles_with_token
        ], dim=1)

        # Cross-modal self-attention
        attn_out, _ = self.cross_attention(modality_tokens, modality_tokens, modality_tokens)
        modality_tokens = self.attn_norm(modality_tokens + attn_out)

        # MoE layers with residual connections
        total_aux_loss = 0.0
        for moe_layer, moe_norm in zip(self.moe_layers, self.moe_norms):
            moe_out, aux_loss = moe_layer(modality_tokens)
            modality_tokens = moe_norm(modality_tokens + moe_out)
            total_aux_loss = total_aux_loss + aux_loss

        # Mean pool across modality tokens
        fused = modality_tokens.mean(dim=1)  # (batch, hidden_dim)

        # Classify
        logits = self.classifier(fused)

        return logits, total_aux_loss * self.aux_loss_weight


def get_model(
    fusion: str,
    text_model: str = None,
    smiles_model: str = None,
    device: torch.device = None,
) -> nn.Module:
    """Create model based on experiment configuration.

    Args:
        fusion: 'mlp' or 'moe'
        text_model: 'clinicalbert' or 'pubmedbert' (unused, for consistency)
        smiles_model: 'chemberta' (unused, for consistency)
        device: Device to place model on.

    Returns:
        Initialised model.
    """
    if fusion == "mlp":
        config = MLP_CONFIG
        model = QuadFusionMLP(
            clinical_dim=CLINICAL_DIM,
            text_dim=TEXT_DIM,
            smiles_dim=SMILES_DIM,
            hidden_dim=config["hidden_dim"],
            num_classes=config["num_classes"],
            dropout=config["dropout"],
            n_channels=EEG_ENCODER_CONFIG["n_channels"],
            n_times=EEG_ENCODER_CONFIG["n_times"],
            max_windows=EEG_ENCODER_CONFIG["max_windows"],
            window_chunk_size=EEG_ENCODER_CONFIG["window_chunk_size"],
        )
    elif fusion == "moe":
        config = MOE_CONFIG
        model = QuadFusionMoE(
            clinical_dim=CLINICAL_DIM,
            text_dim=TEXT_DIM,
            smiles_dim=SMILES_DIM,
            hidden_dim=config["hidden_dim"],
            num_classes=config["num_classes"],
            num_experts=config["num_experts"],
            top_k=config["top_k"],
            num_heads=config["num_heads"],
            num_moe_layers=config["num_moe_layers"],
            dropout=config["dropout"],
            aux_loss_weight=config["aux_loss_weight"],
            n_channels=EEG_ENCODER_CONFIG["n_channels"],
            n_times=EEG_ENCODER_CONFIG["n_times"],
            eeg_embed_dim=EEG_ENCODER_CONFIG["embed_dim"],
            num_eeg_layers=EEG_ENCODER_CONFIG["num_layers"],
            max_windows=EEG_ENCODER_CONFIG["max_windows"],
            window_chunk_size=EEG_ENCODER_CONFIG["window_chunk_size"],
        )
    else:
        raise ValueError(f"Unknown fusion type: {fusion}")

    if device is not None:
        model = model.to(device)

    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_models():
    """Test model forward passes."""
    print("Testing Exp7 models...")

    device = torch.device("cpu")
    batch_size = 2

    # Create dummy inputs
    clinical = torch.randn(batch_size, CLINICAL_DIM)
    text = torch.randn(batch_size, TEXT_DIM)
    eeg_windows = torch.randn(batch_size, 10, 27, 2000)  # 10 windows for quick test
    padding_mask = torch.zeros(batch_size, 10, dtype=torch.bool)
    smiles = torch.randn(batch_size, SMILES_DIM)

    # Test QuadFusionMLP
    print("\nTesting QuadFusionMLP:")
    model = get_model("mlp", device=device)
    print(f"  Parameters: {count_parameters(model):,}")
    output = model(clinical, text, eeg_windows, padding_mask, smiles)
    print(f"  Output shape: {output.shape}")

    # Test QuadFusionMoE
    print("\nTesting QuadFusionMoE:")
    model = get_model("moe", device=device)
    print(f"  Parameters: {count_parameters(model):,}")
    output, aux_loss = model(clinical, text, eeg_windows, padding_mask, smiles)
    print(f"  Output shape: {output.shape}")
    print(f"  Aux loss: {aux_loss.item():.4f}")

    print("\nAll model tests passed!")


if __name__ == "__main__":
    test_models()
