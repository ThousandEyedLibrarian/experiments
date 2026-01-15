"""MLP Classifier for Experiment 1a: Concatenation + MLP fusion."""

import torch
import torch.nn as nn
from typing import List


class ConcatMLPClassifier(nn.Module):
    """
    Simple baseline: concatenate text and SMILES embeddings, classify with MLP.

    Architecture:
        Input: concat(text_emb, smiles_emb) -> dim = text_dim + smiles_dim
        Hidden layers with ReLU, BatchNorm, and Dropout
        Output: Linear(hidden, num_classes)
    """

    def __init__(
        self,
        text_dim: int = 768,
        smiles_dim: int = 768,
        hidden_dims: List[int] = [512, 256, 128],
        dropout_rates: List[float] = [0.3, 0.3, 0.2],
        num_classes: int = 2,
    ):
        """
        Args:
            text_dim: Dimension of text embeddings
            smiles_dim: Dimension of SMILES embeddings
            hidden_dims: List of hidden layer dimensions
            dropout_rates: Dropout rate for each hidden layer
            num_classes: Number of output classes (2 for binary classification)
        """
        super().__init__()

        self.text_dim = text_dim
        self.smiles_dim = smiles_dim
        input_dim = text_dim + smiles_dim

        # Build hidden layers
        # Use LayerNorm instead of BatchNorm to handle batch_size=1
        layers = []
        prev_dim = input_dim

        for i, (hidden_dim, dropout) in enumerate(zip(hidden_dims, dropout_rates)):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        text_emb: torch.Tensor,
        smiles_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            text_emb: (batch, text_dim) text embeddings
            smiles_emb: (batch, smiles_dim) SMILES embeddings

        Returns:
            (batch, num_classes) logits
        """
        # Concatenate embeddings
        x = torch.cat([text_emb, smiles_emb], dim=-1)

        # Extract features and classify
        features = self.feature_extractor(x)
        logits = self.classifier(features)

        return logits

    def get_features(
        self,
        text_emb: torch.Tensor,
        smiles_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Get intermediate features before classifier."""
        x = torch.cat([text_emb, smiles_emb], dim=-1)
        return self.feature_extractor(x)


if __name__ == '__main__':
    # Test the model
    print("Testing ConcatMLPClassifier...")

    # Test with different dimension combinations
    test_configs = [
        (768, 768),   # clinicalbert + chemberta
        (768, 256),   # clinicalbert + smilestrf
    ]

    batch_size = 4

    for text_dim, smiles_dim in test_configs:
        print(f"\nTesting text_dim={text_dim}, smiles_dim={smiles_dim}")

        model = ConcatMLPClassifier(
            text_dim=text_dim,
            smiles_dim=smiles_dim,
            hidden_dims=[512, 256, 128],
            dropout_rates=[0.3, 0.3, 0.2],
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
            logits = model(text_emb, smiles_emb)
            print(f"  Output shape: {logits.shape}")

            # Test softmax
            probs = torch.softmax(logits, dim=-1)
            print(f"  Probabilities sum: {probs.sum(dim=-1)}")

    print("\nConcatMLPClassifier test complete!")
