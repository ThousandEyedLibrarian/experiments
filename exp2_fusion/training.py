"""Training utilities for Experiment 2: EEG + SMILES fusion."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from .config import BATCH_SIZE_BY_ENCODER, CHUNK_SIZE_BY_ENCODER, MODEL_CONFIG, TRAIN_CONFIG
from .data_pipeline import EEGSMILESDataset, create_datasets, get_max_channels, prepare_data
from .models.fusion import get_fusion_model

logger = logging.getLogger("exp2")


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    is_moe: bool = False,
) -> float:
    """Train for one epoch.

    Args:
        model: Model to train.
        dataloader: Training data loader.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Device to use.
        is_moe: Whether model returns auxiliary loss.

    Returns:
        Average training loss.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        eeg_windows, padding_mask, smiles_emb, labels = batch
        eeg_windows = eeg_windows.to(device)
        padding_mask = padding_mask.to(device)
        smiles_emb = smiles_emb.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if is_moe:
            logits, aux_loss = model(eeg_windows, padding_mask, smiles_emb)
            loss = criterion(logits, labels) + aux_loss
        else:
            logits = model(eeg_windows, padding_mask, smiles_emb)
            loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    is_moe: bool = False,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model.

    Args:
        model: Model to evaluate.
        dataloader: Validation data loader.
        criterion: Loss function.
        device: Device to use.
        is_moe: Whether model returns auxiliary loss.

    Returns:
        Tuple of (loss, metrics dict).
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            eeg_windows, padding_mask, smiles_emb, labels = batch
            eeg_windows = eeg_windows.to(device)
            padding_mask = padding_mask.to(device)
            smiles_emb = smiles_emb.to(device)
            labels = labels.to(device)

            if is_moe:
                logits, aux_loss = model(eeg_windows, padding_mask, smiles_emb)
                loss = criterion(logits, labels) + aux_loss
            else:
                logits = model(eeg_windows, padding_mask, smiles_emb)
                loss = criterion(logits, labels)

            total_loss += loss.item()
            n_batches += 1

            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }

    # Compute AUC and threshold-tuned metrics if we have both classes
    if len(np.unique(all_labels)) > 1:
        metrics["auc"] = roc_auc_score(all_labels, all_probs)

        # Threshold tuning using Youden's J statistic
        fpr, tpr, thresholds_roc = roc_curve(all_labels, all_probs)
        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
        optimal_threshold = thresholds_roc[best_idx]
        tuned_preds = (all_probs >= optimal_threshold).astype(int)
        metrics["balanced_acc_tuned"] = balanced_accuracy_score(all_labels, tuned_preds)
        metrics["f1_tuned"] = f1_score(all_labels, tuned_preds, zero_division=0)
        metrics["optimal_threshold"] = optimal_threshold
    else:
        metrics["auc"] = 0.5
        metrics["balanced_acc_tuned"] = 0.5
        metrics["f1_tuned"] = 0.0
        metrics["optimal_threshold"] = 0.5

    return total_loss / n_batches, metrics


def train_fold(
    train_dataset: EEGSMILESDataset,
    val_dataset: EEGSMILESDataset,
    fusion_type: str,
    eeg_encoder_type: str,
    smiles_embed_dim: int,
    n_eeg_channels: int,
    device: torch.device,
    config: Dict = TRAIN_CONFIG,
    model_config: Dict = MODEL_CONFIG,
) -> Tuple[Dict[str, float], nn.Module]:
    """Train model for one fold.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        fusion_type: Type of fusion model.
        eeg_encoder_type: Type of EEG encoder.
        smiles_embed_dim: SMILES embedding dimension.
        n_eeg_channels: Number of EEG channels.
        device: Device to use.
        config: Training configuration.
        model_config: Model configuration.

    Returns:
        Tuple of (best metrics dict, trained model).
    """
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Create model
    is_moe = fusion_type == "fusemoe"
    window_chunk_size = CHUNK_SIZE_BY_ENCODER.get(eeg_encoder_type, 16)
    model = get_fusion_model(
        fusion_type=fusion_type,
        eeg_encoder_type=eeg_encoder_type,
        n_eeg_channels=n_eeg_channels,
        smiles_embed_dim=smiles_embed_dim,
        hidden_dim=model_config["hidden_dim"],
        num_classes=model_config["num_classes"],
        num_experts=model_config.get("num_experts", 4),
        top_k=model_config.get("top_k", 2),
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"],
        aux_loss_weight=model_config.get("aux_loss_weight", 0.1),
        window_chunk_size=window_chunk_size,
    )
    model = model.to(device)

    # Class weighting for imbalanced datasets
    train_labels = [train_dataset[i][3].item() for i in range(len(train_dataset))]
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    logger.debug(f"  Class weights: {class_weights.cpu().numpy()}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # Training loop
    best_auc = 0.0
    best_metrics = {}
    patience_counter = 0

    for epoch in range(config["epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, is_moe)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, is_moe)

        scheduler.step(val_metrics["auc"])

        # Log progress at intervals
        if epoch % 20 == 0 or epoch == config["epochs"] - 1:
            logger.debug(
                f"Epoch {epoch+1}/{config['epochs']}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"val_auc={val_metrics['auc']:.3f}"
            )

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_metrics = val_metrics.copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config["patience"]:
            logger.debug(f"Early stopping at epoch {epoch+1} (patience={config['patience']})")
            break

    return best_metrics, model


def run_cross_validation(
    eeg_data: Dict,
    smiles_embeddings: np.ndarray,
    smiles_indices: Dict,
    df,
    fusion_type: str,
    eeg_encoder_type: str,
    smiles_model: str,
    device: torch.device,
    config: Dict = TRAIN_CONFIG,
    verbose: bool = True,
) -> Dict:
    """Run k-fold cross validation.

    Args:
        eeg_data: Dict mapping patient ID to (windows, padding_mask).
        smiles_embeddings: SMILES embeddings array.
        smiles_indices: Dict mapping ASM name to embedding index.
        df: DataFrame with patient info.
        fusion_type: Type of fusion model.
        eeg_encoder_type: Type of EEG encoder.
        smiles_model: Name of SMILES model.
        device: Device to use.
        config: Training configuration.
        verbose: Whether to print progress.

    Returns:
        Results dictionary with metrics.
    """
    # Set seed
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    # Adjust batch size based on encoder type
    config = config.copy()
    config["batch_size"] = BATCH_SIZE_BY_ENCODER.get(eeg_encoder_type, config["batch_size"])

    # Get labels and determine dimensions
    labels = df["outcome"].values
    n_eeg_channels = get_max_channels(eeg_data)  # Use max channels across all data
    smiles_embed_dim = smiles_embeddings.shape[1]

    # K-fold cross validation
    kfold = StratifiedKFold(n_splits=config["n_folds"], shuffle=True, random_state=config["seed"])

    fold_metrics = {"accuracy": [], "auc": [], "f1": [], "f1_tuned": [], "balanced_acc_tuned": []}

    for fold, (train_idx, val_idx) in enumerate(kfold.split(df, labels)):
        if verbose:
            logger.info(f"  Fold {fold + 1}/{config['n_folds']} (train={len(train_idx)}, val={len(val_idx)})")

        # Create datasets (pass max_channels for consistent padding)
        train_ds, val_ds = create_datasets(
            eeg_data, smiles_embeddings, smiles_indices, df,
            train_idx, val_idx,
            max_channels=n_eeg_channels,
        )

        # Train
        try:
            metrics, _ = train_fold(
                train_ds, val_ds,
                fusion_type=fusion_type,
                eeg_encoder_type=eeg_encoder_type,
                smiles_embed_dim=smiles_embed_dim,
                n_eeg_channels=n_eeg_channels,
                device=device,
                config=config,
            )

            for key in fold_metrics:
                fold_metrics[key].append(metrics[key])

            if verbose:
                logger.info(
                    f"    Fold {fold + 1} results: AUC={metrics['auc']:.4f}, "
                    f"BalAcc_tuned={metrics['balanced_acc_tuned']:.4f}, "
                    f"F1_tuned={metrics['f1_tuned']:.4f} (thresh={metrics['optimal_threshold']:.3f})"
                )

        except Exception as e:
            logger.error(f"    Fold {fold + 1} failed: {type(e).__name__}: {e}")
            raise

    # Aggregate results
    results = {
        "experiment": f"exp2_{eeg_encoder_type}_{smiles_model}_{fusion_type}",
        "eeg_encoder": eeg_encoder_type,
        "smiles_model": smiles_model,
        "fusion_type": fusion_type,
        "n_samples": len(df),
        "n_folds": config["n_folds"],
        "config": {**config, **MODEL_CONFIG},
    }

    for metric in fold_metrics:
        values = np.array(fold_metrics[metric])
        results[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "per_fold": [float(v) for v in values],
        }

    if verbose:
        logger.info("Cross-validation complete:")
        logger.info(f"  AUC: {results['auc']['mean']:.4f} +/- {results['auc']['std']:.4f} (min={results['auc']['min']:.4f}, max={results['auc']['max']:.4f})")
        logger.info(f"  Balanced Acc: {results['balanced_acc_tuned']['mean']:.4f} +/- {results['balanced_acc_tuned']['std']:.4f} (min={results['balanced_acc_tuned']['min']:.4f}, max={results['balanced_acc_tuned']['max']:.4f})")
        logger.info(f"  F1 Tuned: {results['f1_tuned']['mean']:.4f} +/- {results['f1_tuned']['std']:.4f} (min={results['f1_tuned']['min']:.4f}, max={results['f1_tuned']['max']:.4f})")

    return results


if __name__ == "__main__":
    # Quick test with logging
    from .utils.logging_utils import setup_logging, log_environment_info

    logger = setup_logging("exp2_training_test", log_dir="logs")
    log_environment_info(logger)

    logger.info("Testing training pipeline...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Prepare data
    eeg_data, smiles_embeddings, smiles_indices, df = prepare_data(
        smiles_model="chemberta",
        cache_eeg=True,
    )

    # Run single experiment with SimpleCNN (faster for testing)
    results = run_cross_validation(
        eeg_data, smiles_embeddings, smiles_indices, df,
        fusion_type="mlp",
        eeg_encoder_type="simplecnn",
        smiles_model="chemberta",
        device=device,
        verbose=True,
    )

    logger.info(f"Results:")
    logger.info(f"  AUC: {results['auc']['mean']:.4f} +/- {results['auc']['std']:.4f}")
    logger.info(f"  Balanced Acc: {results['balanced_acc_tuned']['mean']:.4f} +/- {results['balanced_acc_tuned']['std']:.4f}")
    logger.info(f"  F1 Tuned: {results['f1_tuned']['mean']:.4f} +/- {results['f1_tuned']['std']:.4f}")
