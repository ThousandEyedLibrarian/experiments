"""Alternative window aggregation methods for EEG embeddings.

These aggregators can be used instead of the default EEGWindowTransformer
to test whether different aggregation strategies reduce variance.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """Learnable attention pooling for window aggregation.

    Learns to weight windows based on their content rather than using
    uniform mean pooling. May be more robust to noisy windows.
    """

    def __init__(
        self,
        embed_dim: int,
        output_dim: Optional[int] = None,
        num_heads: int = 1,
        dropout: float = 0.1,
    ):
        """Initialise attention pooling.

        Args:
            embed_dim: Dimension of input window embeddings.
            output_dim: Dimension of output (defaults to embed_dim).
            num_heads: Number of attention heads for multi-head attention.
            dropout: Dropout probability.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.output_dim = output_dim or embed_dim

        if num_heads == 1:
            # Simple single-head attention
            self.attention = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.Tanh(),
                nn.Linear(embed_dim // 2, 1),
            )
        else:
            # Multi-head attention with learnable query
            self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
            self.attention = nn.MultiheadAttention(
                embed_dim, num_heads, dropout=dropout, batch_first=True
            )

        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

        if self.output_dim != embed_dim:
            self.projection = nn.Linear(embed_dim, self.output_dim)
        else:
            self.projection = nn.Identity()

    def forward(
        self,
        window_embeddings: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Aggregate windows using learned attention weights.

        Args:
            window_embeddings: (batch, num_windows, embed_dim).
            padding_mask: (batch, num_windows), True = invalid.

        Returns:
            Aggregated embedding (batch, output_dim).
        """
        batch_size = window_embeddings.size(0)

        if self.num_heads == 1:
            # Single-head attention pooling
            attn_scores = self.attention(window_embeddings).squeeze(-1)  # (batch, num_windows)

            if padding_mask is not None:
                attn_scores = attn_scores.masked_fill(padding_mask, float('-inf'))

            attn_weights = F.softmax(attn_scores, dim=1)  # (batch, num_windows)
            attn_weights = self.dropout(attn_weights)

            # Weighted sum
            x = (window_embeddings * attn_weights.unsqueeze(-1)).sum(dim=1)  # (batch, embed_dim)

        else:
            # Multi-head attention with learnable query
            query = self.query.expand(batch_size, -1, -1)  # (batch, 1, embed_dim)

            x, _ = self.attention(
                query, window_embeddings, window_embeddings,
                key_padding_mask=padding_mask,
            )
            x = x.squeeze(1)  # (batch, embed_dim)

        x = self.norm(x)
        x = self.projection(x)

        return x


class MaskedMaxPooling(nn.Module):
    """Max pooling over valid windows.

    More robust to noisy windows than mean pooling as it selects
    the most confident/activated features.
    """

    def __init__(
        self,
        embed_dim: int,
        output_dim: Optional[int] = None,
    ):
        """Initialise max pooling aggregator.

        Args:
            embed_dim: Dimension of input window embeddings.
            output_dim: Dimension of output (defaults to embed_dim).
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.output_dim = output_dim or embed_dim

        self.norm = nn.LayerNorm(embed_dim)

        if self.output_dim != embed_dim:
            self.projection = nn.Linear(embed_dim, self.output_dim)
        else:
            self.projection = nn.Identity()

    def forward(
        self,
        window_embeddings: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Aggregate windows using max pooling.

        Args:
            window_embeddings: (batch, num_windows, embed_dim).
            padding_mask: (batch, num_windows), True = invalid.

        Returns:
            Aggregated embedding (batch, output_dim).
        """
        if padding_mask is not None:
            # Set padded positions to -inf so they're not selected
            mask = padding_mask.unsqueeze(-1).expand_as(window_embeddings)
            window_embeddings = window_embeddings.masked_fill(mask, float('-inf'))

        # Max pool over windows
        x = window_embeddings.max(dim=1)[0]  # (batch, embed_dim)

        x = self.norm(x)
        x = self.projection(x)

        return x


class MeanMaxPooling(nn.Module):
    """Combined mean and max pooling.

    Concatenates mean and max pooled features for a richer representation.
    """

    def __init__(
        self,
        embed_dim: int,
        output_dim: Optional[int] = None,
    ):
        """Initialise mean-max pooling aggregator.

        Args:
            embed_dim: Dimension of input window embeddings.
            output_dim: Dimension of output (defaults to embed_dim).
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.output_dim = output_dim or embed_dim

        self.norm = nn.LayerNorm(embed_dim * 2)
        self.projection = nn.Linear(embed_dim * 2, self.output_dim)

    def forward(
        self,
        window_embeddings: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Aggregate using both mean and max pooling.

        Args:
            window_embeddings: (batch, num_windows, embed_dim).
            padding_mask: (batch, num_windows), True = invalid.

        Returns:
            Aggregated embedding (batch, output_dim).
        """
        # Mean pooling over valid windows
        if padding_mask is not None:
            valid_mask = ~padding_mask
            valid_mask_expanded = valid_mask.unsqueeze(-1)
            mean_pooled = (window_embeddings * valid_mask_expanded).sum(dim=1) / valid_mask_expanded.sum(dim=1).clamp(min=1)

            # Max pooling with masking
            masked_embeddings = window_embeddings.masked_fill(padding_mask.unsqueeze(-1), float('-inf'))
            max_pooled = masked_embeddings.max(dim=1)[0]
        else:
            mean_pooled = window_embeddings.mean(dim=1)
            max_pooled = window_embeddings.max(dim=1)[0]

        # Concatenate
        x = torch.cat([mean_pooled, max_pooled], dim=-1)  # (batch, embed_dim * 2)

        x = self.norm(x)
        x = self.projection(x)

        return x


class LSTMAggregator(nn.Module):
    """Bidirectional LSTM for sequential window aggregation.

    Captures temporal dependencies between windows that the transformer
    may miss due to attention's lack of inductive bias for order.
    """

    def __init__(
        self,
        embed_dim: int,
        output_dim: Optional[int] = None,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        """Initialise LSTM aggregator.

        Args:
            embed_dim: Dimension of input window embeddings.
            output_dim: Dimension of output (defaults to embed_dim).
            hidden_dim: LSTM hidden dimension.
            num_layers: Number of LSTM layers.
            dropout: Dropout probability.
            bidirectional: Whether to use bidirectional LSTM.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.output_dim = output_dim or embed_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.norm = nn.LayerNorm(lstm_output_dim)
        self.projection = nn.Linear(lstm_output_dim, self.output_dim)

    def forward(
        self,
        window_embeddings: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Aggregate windows using LSTM.

        Args:
            window_embeddings: (batch, num_windows, embed_dim).
            padding_mask: (batch, num_windows), True = invalid.

        Returns:
            Aggregated embedding (batch, output_dim).
        """
        batch_size, num_windows, embed_dim = window_embeddings.shape

        # Pack sequences for efficient processing
        if padding_mask is not None:
            # Compute sequence lengths
            lengths = (~padding_mask).sum(dim=1).cpu()  # (batch,)
            lengths = lengths.clamp(min=1)  # Ensure at least 1

            # Pack padded sequence
            packed = nn.utils.rnn.pack_padded_sequence(
                window_embeddings, lengths, batch_first=True, enforce_sorted=False
            )
            output, (hidden, _) = self.lstm(packed)
        else:
            output, (hidden, _) = self.lstm(window_embeddings)

        # Use final hidden states
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)  # (batch, hidden*2)
        else:
            hidden = hidden[-1]  # (batch, hidden)

        x = self.norm(hidden)
        x = self.projection(x)

        return x


class MultiScaleAggregator(nn.Module):
    """Multi-scale temporal aggregation.

    Aggregates at different temporal scales (e.g., 30s, 1min, 5min)
    and combines them for a hierarchical representation.
    """

    def __init__(
        self,
        embed_dim: int,
        output_dim: Optional[int] = None,
        scales: tuple = (3, 6, 12),  # Number of windows per scale
        dropout: float = 0.1,
    ):
        """Initialise multi-scale aggregator.

        Args:
            embed_dim: Dimension of input window embeddings.
            output_dim: Dimension of output (defaults to embed_dim).
            scales: Number of windows to group at each scale.
            dropout: Dropout probability.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.output_dim = output_dim or embed_dim
        self.scales = scales

        # Projection for each scale
        self.scale_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for _ in scales
        ])

        # Combine scales
        self.combine = nn.Sequential(
            nn.Linear(embed_dim * len(scales), embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        if self.output_dim != embed_dim:
            self.projection = nn.Linear(embed_dim, self.output_dim)
        else:
            self.projection = nn.Identity()

    def forward(
        self,
        window_embeddings: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Aggregate at multiple temporal scales.

        Args:
            window_embeddings: (batch, num_windows, embed_dim).
            padding_mask: (batch, num_windows), True = invalid.

        Returns:
            Aggregated embedding (batch, output_dim).
        """
        batch_size, num_windows, embed_dim = window_embeddings.shape

        scale_outputs = []

        for scale_size, scale_proj in zip(self.scales, self.scale_projections):
            # Number of groups at this scale
            n_groups = num_windows // scale_size

            if n_groups == 0:
                # Not enough windows for this scale, use global mean
                if padding_mask is not None:
                    valid_mask = ~padding_mask
                    valid_mask_expanded = valid_mask.unsqueeze(-1)
                    scale_embed = (window_embeddings * valid_mask_expanded).sum(dim=1) / valid_mask_expanded.sum(dim=1).clamp(min=1)
                else:
                    scale_embed = window_embeddings.mean(dim=1)
            else:
                # Reshape into groups
                truncated = window_embeddings[:, :n_groups * scale_size, :]
                grouped = truncated.view(batch_size, n_groups, scale_size, embed_dim)

                # Mean within each group
                group_means = grouped.mean(dim=2)  # (batch, n_groups, embed_dim)

                # Mean across groups
                if padding_mask is not None:
                    # Create group-level mask
                    truncated_mask = padding_mask[:, :n_groups * scale_size]
                    grouped_mask = truncated_mask.view(batch_size, n_groups, scale_size)
                    group_valid = ~grouped_mask.all(dim=2)  # Group valid if any window valid
                    group_valid_expanded = group_valid.unsqueeze(-1)
                    scale_embed = (group_means * group_valid_expanded).sum(dim=1) / group_valid_expanded.sum(dim=1).clamp(min=1)
                else:
                    scale_embed = group_means.mean(dim=1)

            # Project
            scale_embed = scale_proj(scale_embed)
            scale_outputs.append(scale_embed)

        # Concatenate and combine
        x = torch.cat(scale_outputs, dim=-1)  # (batch, embed_dim * n_scales)
        x = self.combine(x)
        x = self.projection(x)

        return x


def get_aggregator(
    aggregator_type: str,
    embed_dim: int,
    output_dim: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    """Factory function to create aggregator by name.

    Args:
        aggregator_type: One of 'transformer', 'attention', 'maxpool', 'meanmax',
                        'lstm', 'multiscale'.
        embed_dim: Dimension of input window embeddings.
        output_dim: Dimension of output.
        **kwargs: Additional arguments for specific aggregator.

    Returns:
        Aggregator module.
    """
    if aggregator_type == "attention":
        return AttentionPooling(embed_dim, output_dim, **kwargs)
    elif aggregator_type == "maxpool":
        return MaskedMaxPooling(embed_dim, output_dim)
    elif aggregator_type == "meanmax":
        return MeanMaxPooling(embed_dim, output_dim)
    elif aggregator_type == "lstm":
        return LSTMAggregator(embed_dim, output_dim, **kwargs)
    elif aggregator_type == "multiscale":
        return MultiScaleAggregator(embed_dim, output_dim, **kwargs)
    else:
        raise ValueError(f"Unknown aggregator type: {aggregator_type}")
