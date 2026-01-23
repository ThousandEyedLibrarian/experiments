"""EEG encoder wrappers for extracting embeddings from EEG windows."""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

# Import guard for braindecode (may fail due to CUDA library issues)
LABRAM_AVAILABLE = False
LABRAM_IMPORT_ERROR = None

try:
    from braindecode.models import Labram
    LABRAM_AVAILABLE = True
except ImportError as e:
    LABRAM_IMPORT_ERROR = str(e)
except Exception as e:
    # Catch other errors like OSError for missing CUDA libraries
    LABRAM_IMPORT_ERROR = f"{type(e).__name__}: {e}"

logger = logging.getLogger(__name__)


class LaBraMEncoder(nn.Module):
    """Wrapper for LaBraM model to extract EEG embeddings.

    Takes individual 10-second windows and produces embeddings.
    """

    def __init__(
        self,
        n_channels: int = 27,
        n_times: int = 2000,  # 10s at 200Hz
        sfreq: float = 200,
        emb_size: int = 128,  # Reduced for memory
        n_layers: int = 2,  # Reduced for memory efficiency
        patch_size: int = 200,
        att_num_heads: int = 4,  # Reduced for memory
        dropout: float = 0.1,
    ):
        """Initialize LaBraM encoder.

        Args:
            n_channels: Number of EEG channels.
            n_times: Number of time samples per window.
            sfreq: Sampling frequency.
            emb_size: Embedding dimension.
            n_layers: Number of transformer layers.
            patch_size: Size of temporal patches.
            att_num_heads: Number of attention heads.
            dropout: Dropout probability.

        Raises:
            ImportError: If braindecode is not available.
        """
        if not LABRAM_AVAILABLE:
            error_msg = (
                f"LaBraM encoder requires braindecode, but it failed to import.\n"
                f"Error: {LABRAM_IMPORT_ERROR}\n"
                f"Try: pip install braindecode\n"
                f"Or use --eeg-encoder simplecnn instead."
            )
            logger.error(error_msg)
            raise ImportError(error_msg)

        super().__init__()

        self.n_channels = n_channels
        self.n_times = n_times
        self.emb_size = emb_size

        # Create LaBraM model
        self.model = Labram(
            n_chans=n_channels,
            n_times=n_times,
            sfreq=sfreq,
            n_outputs=emb_size,  # Use as embedding dim
            emb_size=emb_size,
            n_layers=n_layers,
            patch_size=patch_size,
            att_num_heads=att_num_heads,
            drop_prob=dropout,
            neural_tokenizer=True,
        )

        # Remove the final classification layer
        # We'll use fc_norm output as embedding
        self.model.final_layer = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from EEG windows.

        Args:
            x: EEG windows of shape (batch, channels, time).

        Returns:
            Embeddings of shape (batch, emb_size).
        """
        return self.model(x)


class SimpleCNNEncoder(nn.Module):
    """Simple CNN-based EEG encoder as a baseline/fallback.

    Uses 1D convolutions along time axis to extract features.
    """

    def __init__(
        self,
        n_channels: int = 27,
        n_times: int = 2000,
        emb_size: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_times = n_times
        self.emb_size = emb_size

        # Temporal convolutions
        self.conv_layers = nn.Sequential(
            # Layer 1: Extract basic temporal features
            nn.Conv1d(n_channels, 64, kernel_size=25, stride=5, padding=12),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 2
            nn.Conv1d(64, 128, kernel_size=15, stride=3, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 3
            nn.Conv1d(128, 256, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 4
            nn.Conv1d(256, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
        )

        # Final projection
        self.projection = nn.Linear(256, emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from EEG windows.

        Args:
            x: EEG windows of shape (batch, channels, time).

        Returns:
            Embeddings of shape (batch, emb_size).
        """
        # Conv layers
        x = self.conv_layers(x)  # (batch, 256, 1)
        x = x.squeeze(-1)  # (batch, 256)

        # Project to embedding
        x = self.projection(x)  # (batch, emb_size)

        return x


def get_eeg_encoder(
    encoder_type: str = "labram",
    n_channels: int = 27,
    n_times: int = 2000,
    emb_size: int = 200,
    **kwargs,
) -> nn.Module:
    """Factory function to get EEG encoder by type.

    Args:
        encoder_type: Type of encoder ('labram', 'simplecnn').
        n_channels: Number of EEG channels.
        n_times: Number of time samples per window.
        emb_size: Embedding dimension.
        **kwargs: Additional arguments for specific encoders.

    Returns:
        EEG encoder module.

    Raises:
        ValueError: If encoder_type is unknown.
        ImportError: If encoder dependencies are not available.
    """
    logger.info(f"Creating EEG encoder: {encoder_type}")

    if encoder_type == "labram":
        if not LABRAM_AVAILABLE:
            logger.error(f"LaBraM requested but braindecode not available: {LABRAM_IMPORT_ERROR}")
            raise ImportError(
                f"LaBraM encoder requires braindecode.\n"
                f"Import error: {LABRAM_IMPORT_ERROR}\n"
                f"Use --eeg-encoder simplecnn as an alternative."
            )
        return LaBraMEncoder(
            n_channels=n_channels,
            n_times=n_times,
            emb_size=emb_size,
            **kwargs,
        )
    elif encoder_type == "simplecnn":
        return SimpleCNNEncoder(
            n_channels=n_channels,
            n_times=n_times,
            emb_size=emb_size,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}. Available: labram, simplecnn")


def is_labram_available() -> bool:
    """Check if LaBraM encoder is available."""
    return LABRAM_AVAILABLE


def get_labram_import_error() -> Optional[str]:
    """Get the LaBraM import error message if it failed to import."""
    return LABRAM_IMPORT_ERROR


def test_encoders():
    """Test EEG encoder implementations."""
    print("Testing EEG encoders...")
    print(f"LaBraM available: {LABRAM_AVAILABLE}")
    if not LABRAM_AVAILABLE:
        print(f"LaBraM import error: {LABRAM_IMPORT_ERROR}")

    n_channels = 27
    n_times = 2000
    batch_size = 4

    x = torch.randn(batch_size, n_channels, n_times)

    # Test LaBraM encoder (if available)
    if LABRAM_AVAILABLE:
        print("\nTesting LaBraM encoder:")
        try:
            labram = LaBraMEncoder(n_channels=n_channels, n_times=n_times, emb_size=200)
            out = labram(x)
            print(f"  Input shape: {x.shape}")
            print(f"  Output shape: {out.shape}")
        except Exception as e:
            print(f"  LaBraM test failed: {e}")
    else:
        print("\nSkipping LaBraM encoder test (not available)")

    # Test SimpleCNN encoder
    print("\nTesting SimpleCNN encoder:")
    cnn = SimpleCNNEncoder(n_channels=n_channels, n_times=n_times, emb_size=256)
    out = cnn(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")

    print("\nEncoder tests complete.")


if __name__ == "__main__":
    test_encoders()
