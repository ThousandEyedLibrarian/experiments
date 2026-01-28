"""Models for Experiment 4: Clinical features baseline."""

from typing import List

import torch
import torch.nn as nn

from .config import CLINICAL_CONFIG


class ClinicalMLP(nn.Module):
    """Simple feedforward MLP for clinical feature classification (Experiment 4a).

    Architecture:
        Input (20D) -> Linear(64) -> ReLU -> LayerNorm -> Dropout
                    -> Linear(32) -> ReLU -> LayerNorm -> Dropout
                    -> Linear(2) -> output
    """

    def __init__(
        self,
        input_dim: int = CLINICAL_CONFIG["input_dim"],
        hidden_dims: List[int] = None,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        """Initialise the MLP.

        Args:
            input_dim: Input feature dimension.
            hidden_dims: List of hidden layer dimensions.
            num_classes: Number of output classes.
            dropout: Dropout probability.
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        """Initialise weights using Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, input_dim).

        Returns:
            Logits of shape (batch, num_classes).
        """
        features = self.feature_extractor(x)
        return self.classifier(features)


class FeatureSelfAttention(nn.Module):
    """Self-attention over clinical features.

    Treats each feature as a token, learning feature-feature relationships.
    This follows the Feng et al. 2025 approach of using self-attention
    within modalities.
    """

    def __init__(
        self,
        num_features: int = CLINICAL_CONFIG["input_dim"],
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        """Initialise the self-attention module.

        Args:
            num_features: Number of input features.
            hidden_dim: Dimension of hidden representations.
            num_heads: Number of attention heads.
            num_layers: Number of transformer encoder layers.
            dropout: Dropout probability.
        """
        super().__init__()

        self.num_features = num_features
        self.hidden_dim = hidden_dim

        # Project each scalar feature to hidden_dim
        self.feature_embed = nn.Linear(1, hidden_dim)

        # Learnable positional embeddings for each feature
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_features, hidden_dim) * 0.02
        )

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for better training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Layer norm after aggregation
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, num_features).

        Returns:
            Aggregated representation of shape (batch, hidden_dim).
        """
        batch_size = x.shape[0]

        # Expand features: (batch, num_features, 1)
        x = x.unsqueeze(-1)

        # Embed each feature: (batch, num_features, hidden_dim)
        x = self.feature_embed(x)

        # Add positional embeddings
        x = x + self.pos_embed

        # Prepend CLS token: (batch, 1 + num_features, hidden_dim)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Apply transformer
        x = self.transformer(x)

        # Return normalised CLS token output
        return self.norm(x[:, 0])


class ClinicalAttentionMLP(nn.Module):
    """Clinical classifier with self-attention over features (Experiment 4b).

    This model treats each clinical feature as a token and uses self-attention
    to learn feature-feature interactions, following the approach described
    in Feng et al. 2025.
    """

    def __init__(
        self,
        num_features: int = CLINICAL_CONFIG["input_dim"],
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.2,
    ):
        """Initialise the attention-based classifier.

        Args:
            num_features: Number of input features.
            hidden_dim: Dimension of hidden representations.
            num_heads: Number of attention heads.
            num_layers: Number of transformer encoder layers.
            num_classes: Number of output classes.
            dropout: Dropout probability.
        """
        super().__init__()

        self.attention = FeatureSelfAttention(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialise weights."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, num_features).

        Returns:
            Logits of shape (batch, num_classes).
        """
        features = self.attention(x)
        return self.classifier(features)


def get_model(model_type: str, device: torch.device) -> nn.Module:
    """Create model based on type.

    Args:
        model_type: Either 'mlp' or 'attention'.
        device: Device to place model on.

    Returns:
        Initialised model.
    """
    from .config import CONFIG_4A, CONFIG_4B

    if model_type == "mlp":
        config = CONFIG_4A
        model = ClinicalMLP(
            input_dim=CLINICAL_CONFIG["input_dim"],
            hidden_dims=config["hidden_dims"],
            num_classes=config["num_classes"],
            dropout=config["dropout"],
        )
    elif model_type == "attention":
        config = CONFIG_4B
        model = ClinicalAttentionMLP(
            num_features=CLINICAL_CONFIG["input_dim"],
            hidden_dim=config["hidden_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            num_classes=config["num_classes"],
            dropout=config["dropout"],
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model.to(device)


def test_models():
    """Test model forward passes."""
    print("Testing models...")

    device = torch.device("cpu")
    batch_size = 4
    input_dim = CLINICAL_CONFIG["input_dim"]

    # Create dummy input
    x = torch.randn(batch_size, input_dim)

    # Test MLP
    print("\nTesting ClinicalMLP:")
    mlp = get_model("mlp", device)
    n_params = sum(p.numel() for p in mlp.parameters())
    print(f"  Parameters: {n_params:,}")
    output = mlp(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")

    # Test Attention model
    print("\nTesting ClinicalAttentionMLP:")
    attn = get_model("attention", device)
    n_params = sum(p.numel() for p in attn.parameters())
    print(f"  Parameters: {n_params:,}")
    output = attn(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_models()
