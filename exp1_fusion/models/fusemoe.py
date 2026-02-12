"""FuseMoE for Experiment 1b: Mixture-of-Experts multimodal fusion.

Based on: "FuseMoE: Mixture-of-Experts Transformers for Fleximodal Fusion"
(arXiv:2402.03226)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.fuse_moe import FuseMoE


class MoEFusionLayer(nn.Module):
    """Single MoE-based cross-modal fusion layer."""

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Self-attention for each modality
        self.text_self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.smiles_self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Cross-modal MoE (shared reference implementation)
        self.fuse_moe = FuseMoE(
            strategy="permodality",
            input_dims=hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_experts=num_experts,
            k=top_k,
            num_modalities=2,
        )

        # Layer norms
        self.norm1_text = nn.LayerNorm(hidden_dim)
        self.norm1_smiles = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        text_h: torch.Tensor,
        smiles_h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through fusion layer.

        Args:
            text_h: (batch, 1, hidden_dim) text representation
            smiles_h: (batch, 1, hidden_dim) SMILES representation

        Returns:
            text_h: Updated text representation
            smiles_h: Updated SMILES representation
            aux_loss: Load balancing loss
        """
        # Self-attention
        text_attn, _ = self.text_self_attn(text_h, text_h, text_h)
        text_h = self.norm1_text(text_h + text_attn)

        smiles_attn, _ = self.smiles_self_attn(smiles_h, smiles_h, smiles_h)
        smiles_h = self.norm1_smiles(smiles_h + smiles_attn)

        # Cross-modal MoE fusion (per-modality routing)
        text_flat = text_h.squeeze(1)      # (B, H)
        smiles_flat = smiles_h.squeeze(1)  # (B, H)
        fused_out, aux_loss = self.fuse_moe(text_flat, smiles_flat)
        fused_out = self.norm2(fused_out)
        # Add fused output as residual to both modalities
        text_h = text_h + fused_out.unsqueeze(1)
        smiles_h = smiles_h + fused_out.unsqueeze(1)

        # FFN
        text_h = self.norm3(text_h + self.ffn(text_h))
        smiles_h = self.norm3(smiles_h + self.ffn(smiles_h))

        return text_h, smiles_h, aux_loss


class SimplifiedFuseMoE(nn.Module):
    """
    Simplified FuseMoE for text + SMILES bimodal fusion.

    Architecture:
        1. Project each modality to common dimension
        2. Add learnable modality tokens
        3. Apply MoE fusion layers
        4. Concatenate and classify
    """

    def __init__(
        self,
        text_dim: int = 768,
        smiles_dim: int = 768,
        hidden_dim: int = 256,
        num_experts: int = 4,
        top_k: int = 2,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_classes: int = 2,
    ):
        """
        Args:
            text_dim: Dimension of input text embeddings
            smiles_dim: Dimension of input SMILES embeddings
            hidden_dim: Common hidden dimension for fusion
            num_experts: Number of experts in MoE layers
            top_k: Number of experts to route to
            num_heads: Number of attention heads
            num_layers: Number of fusion layers
            dropout: Dropout rate
            num_classes: Number of output classes
        """
        super().__init__()

        self.hidden_dim = hidden_dim

        # Modal-specific projections to common dimension
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.smiles_proj = nn.Sequential(
            nn.Linear(smiles_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Learnable modality tokens
        self.text_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.smiles_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # MoE fusion layers
        self.fusion_layers = nn.ModuleList([
            MoEFusionLayer(
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                top_k=top_k,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def update_temperature(self, global_step: int):
        """Update FuseMoE temperature for annealing across all fusion layers."""
        for layer in self.fusion_layers:
            layer.fuse_moe.update_temperature(global_step)

    def forward(
        self,
        text_emb: torch.Tensor,
        smiles_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            text_emb: (batch, text_dim) text embeddings
            smiles_emb: (batch, smiles_dim) SMILES embeddings

        Returns:
            logits: (batch, num_classes) classification logits
            aux_loss: Auxiliary load balancing loss (scalar)
        """
        batch_size = text_emb.size(0)

        # Project to common dimension: (B, D) -> (B, 1, H)
        text_h = self.text_proj(text_emb).unsqueeze(1)
        smiles_h = self.smiles_proj(smiles_emb).unsqueeze(1)

        # Add modality tokens
        text_tokens = self.text_token.expand(batch_size, -1, -1)
        smiles_tokens = self.smiles_token.expand(batch_size, -1, -1)

        text_h = text_h + text_tokens
        smiles_h = smiles_h + smiles_tokens

        # Apply MoE fusion layers
        total_aux_loss = 0.0
        for layer in self.fusion_layers:
            text_h, smiles_h, layer_aux = layer(text_h, smiles_h)
            total_aux_loss = total_aux_loss + layer_aux

        # Pool and concatenate
        text_out = text_h.squeeze(1)  # (B, H)
        smiles_out = smiles_h.squeeze(1)  # (B, H)

        fused = torch.cat([text_out, smiles_out], dim=-1)  # (B, 2H)
        logits = self.classifier(fused)

        return logits, total_aux_loss

    def get_fused_representation(
        self,
        text_emb: torch.Tensor,
        smiles_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Get the fused representation before classification."""
        batch_size = text_emb.size(0)

        text_h = self.text_proj(text_emb).unsqueeze(1)
        smiles_h = self.smiles_proj(smiles_emb).unsqueeze(1)

        text_tokens = self.text_token.expand(batch_size, -1, -1)
        smiles_tokens = self.smiles_token.expand(batch_size, -1, -1)

        text_h = text_h + text_tokens
        smiles_h = smiles_h + smiles_tokens

        for layer in self.fusion_layers:
            text_h, smiles_h, _ = layer(text_h, smiles_h)

        text_out = text_h.squeeze(1)
        smiles_out = smiles_h.squeeze(1)

        return torch.cat([text_out, smiles_out], dim=-1)


if __name__ == '__main__':
    # Test the model
    print("Testing SimplifiedFuseMoE...")

    # Test with different dimension combinations
    test_configs = [
        (768, 768),   # clinicalbert + chemberta
        (768, 256),   # clinicalbert + smilestrf
    ]

    batch_size = 4

    for text_dim, smiles_dim in test_configs:
        print(f"\nTesting text_dim={text_dim}, smiles_dim={smiles_dim}")

        model = SimplifiedFuseMoE(
            text_dim=text_dim,
            smiles_dim=smiles_dim,
            hidden_dim=256,
            num_experts=4,
            top_k=2,
            num_heads=4,
            num_layers=2,
            dropout=0.1,
            num_classes=2,
        )

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {n_params:,}")

        # Test forward pass
        text_emb = torch.randn(batch_size, text_dim)
        smiles_emb = torch.randn(batch_size, smiles_dim)

        model.eval()
        with torch.no_grad():
            logits, aux_loss = model(text_emb, smiles_emb)
            print(f"  Output shape: {logits.shape}")
            print(f"  Aux loss: {aux_loss.item():.4f}")

            # Test softmax
            probs = torch.softmax(logits, dim=-1)
            print(f"  Probabilities sum: {probs.sum(dim=-1)}")

        # Test training mode (with gradient)
        model.train()
        logits, aux_loss = model(text_emb, smiles_emb)
        loss = F.cross_entropy(logits, torch.randint(0, 2, (batch_size,)))
        total_loss = loss + 0.1 * aux_loss
        total_loss.backward()
        print(f"  Training loss backward: OK")

    print("\nSimplifiedFuseMoE test complete!")
