"""Ablation study framework for EEG variance investigation.

This script provides ablation experiments to understand which components
contribute most to the high fold-to-fold variance in EEG experiments.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent directory for imports
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from exp2_fusion.config import EEG_CONFIG, MODEL_CONFIG, TRAIN_CONFIG
from exp2_fusion.data_pipeline import prepare_data, create_datasets, get_max_channels
from exp2_fusion.models.eeg_encoders import get_eeg_encoder, SimpleCNNEncoder
from exp2_fusion.models.eeg_transformer import EEGWindowTransformer
from exp2_fusion.models.aggregators import get_aggregator
from exp2_fusion.training import train_epoch, evaluate
from exp8_stratification.stratified_cv import get_multilabel_splits, get_outcome_only_splits
from .config import RESULTS_DIR, CV_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AblationModel(nn.Module):
    """Flexible model for ablation studies.

    Allows different combinations of encoders, aggregators, and parameters.
    """

    def __init__(
        self,
        encoder_type: str = "simplecnn",
        aggregator_type: str = "transformer",
        n_channels: int = 27,
        n_times: int = 2000,
        embed_dim: int = 256,
        output_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        smiles_dim: int = 768,
        freeze_encoder: bool = False,
        max_windows: int = 120,
        window_chunk_size: int = 32,
    ):
        """Initialise ablation model.

        Args:
            encoder_type: Type of window encoder ('simplecnn', 'eegnet').
            aggregator_type: Type of aggregator ('transformer', 'attention', 'maxpool', 'lstm').
            n_channels: Number of EEG channels.
            n_times: Number of time samples per window.
            embed_dim: Window embedding dimension.
            output_dim: Output dimension for aggregator.
            num_heads: Number of attention heads.
            num_layers: Number of transformer layers.
            smiles_dim: SMILES embedding dimension.
            freeze_encoder: Whether to freeze the window encoder.
            max_windows: Maximum number of windows.
            window_chunk_size: Chunk size for window processing.
        """
        super().__init__()

        self.freeze_encoder = freeze_encoder
        self.window_chunk_size = window_chunk_size

        # Window encoder
        self.encoder = get_eeg_encoder(
            encoder_type=encoder_type,
            n_channels=n_channels,
            n_times=n_times,
            emb_size=embed_dim,
        )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Aggregator
        if aggregator_type == "transformer":
            self.aggregator = EEGWindowTransformer(
                embed_dim=embed_dim,
                output_dim=output_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                max_windows=max_windows,
            )
        else:
            self.aggregator = get_aggregator(
                aggregator_type=aggregator_type,
                embed_dim=embed_dim,
                output_dim=output_dim,
            )

        # SMILES projection
        self.smiles_proj = nn.Sequential(
            nn.Linear(smiles_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(output_dim, 2),
        )

    def forward(
        self,
        eeg_windows: torch.Tensor,
        padding_mask: torch.Tensor,
        smiles_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            eeg_windows: (batch, num_windows, channels, time)
            padding_mask: (batch, num_windows)
            smiles_emb: (batch, smiles_dim)

        Returns:
            Logits (batch, 2)
        """
        batch_size, num_windows, n_channels, n_times = eeg_windows.shape

        # Encode windows in chunks
        window_embeddings = []
        for i in range(0, num_windows, self.window_chunk_size):
            chunk = eeg_windows[:, i:i+self.window_chunk_size]
            chunk = chunk.reshape(-1, n_channels, n_times)

            if self.freeze_encoder:
                with torch.no_grad():
                    chunk_emb = self.encoder(chunk)
            else:
                chunk_emb = self.encoder(chunk)

            chunk_emb = chunk_emb.reshape(batch_size, -1, chunk_emb.size(-1))
            window_embeddings.append(chunk_emb)

        window_embeddings = torch.cat(window_embeddings, dim=1)

        # Aggregate
        eeg_emb = self.aggregator(window_embeddings, padding_mask)

        # SMILES
        smiles_emb = self.smiles_proj(smiles_emb)

        # Fuse and classify
        fused = torch.cat([eeg_emb, smiles_emb], dim=-1)
        logits = self.classifier(fused)

        return logits


def run_ablation_experiment(
    ablation_config: Dict,
    eeg_data: Dict,
    smiles_embeddings: np.ndarray,
    smiles_indices: Dict,
    df,
    device: torch.device,
    use_multilabel_stratification: bool = True,
) -> Dict:
    """Run a single ablation experiment.

    Args:
        ablation_config: Configuration for this ablation.
        eeg_data: EEG data dictionary.
        smiles_embeddings: SMILES embeddings.
        smiles_indices: SMILES index mapping.
        df: DataFrame with labels.
        device: Device to use.
        use_multilabel_stratification: Whether to use multi-label stratification.

    Returns:
        Results dictionary.
    """
    logger.info(f"Running ablation: {ablation_config['name']}")

    # Get splits
    if use_multilabel_stratification:
        splits = list(get_multilabel_splits(
            df,
            stratify_cols=["outcome", "focal", "sex"],
            n_splits=CV_CONFIG["n_splits"],
        ))
    else:
        splits = list(get_outcome_only_splits(df, n_splits=CV_CONFIG["n_splits"]))

    n_channels = get_max_channels(eeg_data)
    smiles_dim = smiles_embeddings.shape[1]

    fold_metrics = {"auc": [], "balanced_acc_tuned": [], "f1_tuned": []}

    for fold, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"  Fold {fold + 1}/{len(splits)}")

        # Create datasets
        train_ds, val_ds = create_datasets(
            eeg_data, smiles_embeddings, smiles_indices, df,
            train_idx, val_idx,
            max_channels=n_channels,
        )

        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

        # Create model
        model = AblationModel(
            encoder_type=ablation_config.get("encoder_type", "simplecnn"),
            aggregator_type=ablation_config.get("aggregator_type", "transformer"),
            n_channels=n_channels,
            embed_dim=ablation_config.get("embed_dim", 256),
            output_dim=ablation_config.get("output_dim", 256),
            num_heads=ablation_config.get("num_heads", 4),
            num_layers=ablation_config.get("num_layers", 2),
            smiles_dim=smiles_dim,
            freeze_encoder=ablation_config.get("freeze_encoder", False),
            max_windows=ablation_config.get("max_windows", 120),
        ).to(device)

        # Training setup
        train_labels = [train_ds[i][3].item() for i in range(len(train_ds))]
        class_counts = np.bincount(train_labels)
        class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32).to(device)
        class_weights = class_weights / class_weights.sum()

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-3,
            weight_decay=1e-4,
        )

        # Train
        best_auc = 0.0
        best_metrics = {}
        patience_counter = 0

        for epoch in range(100):
            train_epoch(model, train_loader, optimizer, criterion, device, is_moe=False)
            _, metrics = evaluate(model, val_loader, criterion, device, is_moe=False)

            if metrics["auc"] > best_auc:
                best_auc = metrics["auc"]
                best_metrics = metrics.copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= 20:
                break

        for key in fold_metrics:
            if key in best_metrics:
                fold_metrics[key].append(best_metrics[key])

        logger.info(f"    Fold {fold + 1}: AUC={best_metrics.get('auc', 0):.4f}")

    # Aggregate results
    results = {
        "name": ablation_config["name"],
        "config": ablation_config,
        "n_folds": len(splits),
    }

    for key, values in fold_metrics.items():
        if values:
            results[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "per_fold": values,
            }

    logger.info(f"  Complete: AUC={results['auc']['mean']:.4f} +/- {results['auc']['std']:.4f}")

    return results


def define_ablation_experiments() -> List[Dict]:
    """Define the ablation experiments to run."""
    experiments = []

    # Baseline
    experiments.append({
        "name": "baseline_simplecnn_transformer",
        "encoder_type": "simplecnn",
        "aggregator_type": "transformer",
        "embed_dim": 256,
        "num_layers": 2,
        "freeze_encoder": False,
    })

    # Encoder ablations
    experiments.append({
        "name": "encoder_eegnet",
        "encoder_type": "eegnet",
        "aggregator_type": "transformer",
        "embed_dim": 256,
        "num_layers": 2,
    })

    experiments.append({
        "name": "encoder_frozen",
        "encoder_type": "simplecnn",
        "aggregator_type": "transformer",
        "embed_dim": 256,
        "num_layers": 2,
        "freeze_encoder": True,
    })

    # Aggregator ablations
    for agg_type in ["attention", "maxpool", "meanmax", "lstm"]:
        experiments.append({
            "name": f"aggregator_{agg_type}",
            "encoder_type": "simplecnn",
            "aggregator_type": agg_type,
            "embed_dim": 256,
        })

    # Transformer depth ablations
    for n_layers in [0, 1, 4]:
        if n_layers == 0:
            # No transformer, just mean pooling
            experiments.append({
                "name": "aggregator_depth_0",
                "encoder_type": "simplecnn",
                "aggregator_type": "attention",  # Use attention for simple pooling
                "embed_dim": 256,
                "num_layers": 0,
            })
        else:
            experiments.append({
                "name": f"aggregator_depth_{n_layers}",
                "encoder_type": "simplecnn",
                "aggregator_type": "transformer",
                "embed_dim": 256,
                "num_layers": n_layers,
            })

    # Embedding dimension ablations
    for dim in [64, 128]:
        experiments.append({
            "name": f"embed_dim_{dim}",
            "encoder_type": "simplecnn",
            "aggregator_type": "transformer",
            "embed_dim": dim,
            "output_dim": dim,
            "num_layers": 2,
        })

    return experiments


def run_all_ablations(
    smiles_model: str = "chemberta",
    use_multilabel: bool = True,
    experiments: List[Dict] = None,
) -> List[Dict]:
    """Run all ablation experiments.

    Args:
        smiles_model: SMILES model to use.
        use_multilabel: Whether to use multi-label stratification.
        experiments: List of experiments to run (defaults to all).

    Returns:
        List of results dictionaries.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Prepare data
    logger.info("Preparing data...")
    eeg_data, smiles_embeddings, smiles_indices, df = prepare_data(
        smiles_model=smiles_model,
        cache_eeg=True,
    )
    logger.info(f"Loaded {len(df)} patients")

    # Define experiments
    if experiments is None:
        experiments = define_ablation_experiments()

    logger.info(f"Running {len(experiments)} ablation experiments")

    # Run experiments
    all_results = []
    for exp_config in experiments:
        try:
            results = run_ablation_experiment(
                exp_config,
                eeg_data, smiles_embeddings, smiles_indices, df,
                device,
                use_multilabel_stratification=use_multilabel,
            )
            all_results.append(results)
        except Exception as e:
            logger.error(f"Experiment {exp_config['name']} failed: {e}")
            all_results.append({
                "name": exp_config["name"],
                "error": str(e),
            })

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = RESULTS_DIR / f"ablation_results_{timestamp}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Results saved to {results_path}")

    return all_results


def print_ablation_summary(results: List[Dict]):
    """Print summary table of ablation results."""
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS")
    print("=" * 80)

    print(f"\n{'Experiment':<35} {'AUC':<20} {'Bal Acc':<20}")
    print("-" * 80)

    for r in results:
        name = r.get("name", "unknown")
        if "error" in r:
            print(f"{name:<35} ERROR: {r['error'][:40]}")
            continue

        auc = r.get("auc", {})
        bal_acc = r.get("balanced_acc_tuned", {})

        auc_str = f"{auc.get('mean', 0):.3f} +/- {auc.get('std', 0):.3f}" if auc else "N/A"
        bal_str = f"{bal_acc.get('mean', 0):.3f} +/- {bal_acc.get('std', 0):.3f}" if bal_acc else "N/A"

        print(f"{name:<35} {auc_str:<20} {bal_str:<20}")

    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG Ablation Study")
    parser.add_argument("--smiles-model", default="chemberta", help="SMILES model")
    parser.add_argument("--no-multilabel", action="store_true", help="Disable multi-label stratification")
    parser.add_argument("--quick", action="store_true", help="Run only baseline experiment")
    args = parser.parse_args()

    if args.quick:
        experiments = [define_ablation_experiments()[0]]  # Just baseline
    else:
        experiments = None  # All experiments

    results = run_all_ablations(
        smiles_model=args.smiles_model,
        use_multilabel=not args.no_multilabel,
        experiments=experiments,
    )

    print_ablation_summary(results)
