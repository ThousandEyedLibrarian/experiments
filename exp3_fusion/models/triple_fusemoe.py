"""Triple modality FuseMoE model (Experiment 3b)."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import EEG components from exp2
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from exp2_fusion.models.eeg_encoders import get_eeg_encoder
from exp2_fusion.models.eeg_transformer import EEGWindowTransformer


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
        """Forward pass with sparse gating.

        Args:
            x: Input tensor of shape (batch, seq, input_dim) or (batch, input_dim)

        Returns:
            Tuple of (output, aux_loss)
        """
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


class TripleModalityFuseMoE(nn.Module):
    """FuseMoE model for Text + EEG + SMILES triple modality fusion.

    Architecture:
    - Each modality -> Projection(dim->256) + Learnable modality token
    - Self-attention across 3 modality tokens
    - 2 sparse MoE layers
    - Mean pool -> classifier
    """

    def __init__(
        self,
        text_dim: int = 768,
        smiles_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 2,
        num_experts: int = 4,
        top_k: int = 2,
        num_heads: int = 4,
        num_moe_layers: int = 2,
        dropout: float = 0.1,
        aux_loss_weight: float = 0.1,
        # EEG encoder config
        eeg_encoder_type: str = "simplecnn",
        n_eeg_channels: int = 27,
        n_eeg_times: int = 2000,
        eeg_embed_dim: int = 256,
        num_eeg_layers: int = 2,
        max_windows: int = 120,
        window_chunk_size: int = 32,
    ):
        super().__init__()

        self.aux_loss_weight = aux_loss_weight
        self.window_chunk_size = window_chunk_size

        # Text projection
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # EEG window encoder
        self.window_encoder = get_eeg_encoder(
            encoder_type=eeg_encoder_type,
            n_channels=n_eeg_channels,
            n_times=n_eeg_times,
            emb_size=eeg_embed_dim,
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

        # SMILES projection
        self.smiles_proj = nn.Linear(smiles_dim, hidden_dim)

        # Learnable modality tokens
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
        text_emb: torch.Tensor,
        eeg_windows: torch.Tensor,
        padding_mask: torch.Tensor,
        smiles_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            text_emb: (batch, text_dim)
            eeg_windows: (batch, num_windows, channels, time)
            padding_mask: (batch, num_windows) boolean, True for padded
            smiles_emb: (batch, smiles_dim)

        Returns:
            Tuple of (logits, aux_loss)
        """
        batch_size, num_windows, n_channels, n_times = eeg_windows.shape

        # Project text
        text_proj = self.text_proj(text_emb)  # (batch, hidden_dim)

        # Encode EEG windows
        windows_flat = eeg_windows.view(batch_size * num_windows, n_channels, n_times)
        window_embeddings = self.encode_windows_chunked(windows_flat, self.window_chunk_size)
        embed_dim = window_embeddings.shape[-1]
        window_embeddings = window_embeddings.view(batch_size, num_windows, embed_dim)

        # Aggregate windows
        eeg_emb = self.aggregator(window_embeddings, padding_mask)  # (batch, hidden_dim)

        # Project SMILES
        smiles_proj = self.smiles_proj(smiles_emb)  # (batch, hidden_dim)

        # Add modality tokens to embeddings
        text_with_token = text_proj.unsqueeze(1) + self.text_token.expand(batch_size, -1, -1)
        eeg_with_token = eeg_emb.unsqueeze(1) + self.eeg_token.expand(batch_size, -1, -1)
        smiles_with_token = smiles_proj.unsqueeze(1) + self.smiles_token.expand(batch_size, -1, -1)

        # Concatenate modality tokens: (batch, 3, hidden_dim)
        modality_tokens = torch.cat([text_with_token, eeg_with_token, smiles_with_token], dim=1)

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


def test_triple_fusemoe():
    """Test TripleModalityFuseMoE model."""
    print("Testing TripleModalityFuseMoE...")

    batch_size = 2
    text_dim = 768
    smiles_dim = 768
    num_windows = 120
    n_channels = 27
    n_times = 2000

    # Create inputs
    text = torch.randn(batch_size, text_dim)
    eeg = torch.randn(batch_size, num_windows, n_channels, n_times)
    mask = torch.zeros(batch_size, num_windows, dtype=torch.bool)
    mask[:, 90:] = True  # Last 30 windows are padding
    smiles = torch.randn(batch_size, smiles_dim)

    # Create model
    model = TripleModalityFuseMoE(
        text_dim=text_dim,
        smiles_dim=smiles_dim,
        hidden_dim=256,
        num_classes=2,
        eeg_encoder_type="simplecnn",
        n_eeg_channels=n_channels,
    )

    # Forward pass
    logits, aux_loss = model(text, eeg, mask, smiles)
    print(f"Output shape: {logits.shape}")
    print(f"Aux loss: {aux_loss.item():.4f}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")


if __name__ == "__main__":
    test_triple_fusemoe()
