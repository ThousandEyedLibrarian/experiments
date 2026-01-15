"""FuseMoE for Experiment 1b: Mixture-of-Experts multimodal fusion.

Based on: "FuseMoE: Mixture-of-Experts Transformers for Fleximodal Fusion"
(arXiv:2402.03226)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SparseGatedMoE(nn.Module):
    """
    Sparse gated mixture of experts layer.

    Uses top-k routing to select a subset of experts for each input,
    enabling efficient computation while maintaining model capacity.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        noise_std: float = 0.1,
    ):
        """
        Args:
            input_dim: Input/output dimension
            hidden_dim: Hidden dimension within each expert
            num_experts: Number of expert networks
            top_k: Number of experts to route each input to
            noise_std: Standard deviation of noise for load balancing
        """
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std

        # Expert networks (simple 2-layer MLPs)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )
            for _ in range(num_experts)
        ])

        # Gating network
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with sparse routing.

        Args:
            x: (batch, seq_len, dim) or (batch, dim) input tensor

        Returns:
            output: Same shape as input
            aux_loss: Load balancing loss (scalar)
        """
        # Handle both 2D and 3D inputs
        orig_shape = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, D)

        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)  # (B*S, D)

        # Compute gating logits
        gate_logits = self.gate(x_flat)  # (B*S, E)

        # Add noise during training for load balancing
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(gate_logits) * self.noise_std
            gate_logits = gate_logits + noise

        # Top-k gating
        top_k_logits, top_k_indices = gate_logits.topk(self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)  # (B*S, K)

        # Compute all expert outputs
        expert_outputs = torch.stack(
            [expert(x_flat) for expert in self.experts], dim=1
        )  # (B*S, E, D)

        # Gather top-k expert outputs
        top_k_expert_outputs = torch.gather(
            expert_outputs, 1,
            top_k_indices.unsqueeze(-1).expand(-1, -1, dim)
        )  # (B*S, K, D)

        # Weighted combination
        output = (top_k_gates.unsqueeze(-1) * top_k_expert_outputs).sum(dim=1)
        output = output.view(batch_size, seq_len, dim)

        # Restore original shape if needed
        if len(orig_shape) == 2:
            output = output.squeeze(1)

        # Compute auxiliary load balancing loss
        aux_loss = self._load_balance_loss(gate_logits)

        return output, aux_loss

    def _load_balance_loss(self, gate_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing loss to encourage even expert utilization.

        Uses coefficient of variation (CV^2) as the loss.
        """
        gates = F.softmax(gate_logits, dim=-1)
        importance = gates.sum(dim=0)  # Sum over batch
        cv = importance.std() / (importance.mean() + 1e-8)
        return cv ** 2


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

        # Cross-modal MoE
        self.moe = SparseGatedMoE(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim * 2,
            num_experts=num_experts,
            top_k=top_k,
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

        # Cross-modal MoE fusion
        combined = torch.cat([text_h, smiles_h], dim=1)  # (B, 2, H)
        fused, aux_loss = self.moe(combined)
        fused = self.norm2(combined + fused)

        # Split back
        text_h = fused[:, :1, :]
        smiles_h = fused[:, 1:, :]

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
