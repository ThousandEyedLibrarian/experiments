"""Training utilities for Experiment 4: Clinical features baseline."""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from .config import CONFIG_4A, CONFIG_4B, CV_CONFIG
from .data_pipeline import ClinicalDataset, create_datasets, load_clinical_data
from .models import get_model

logger = logging.getLogger("exp4")


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch.

    Args:
        model: The model to train.
        dataloader: Training data loader.
        optimizer: Optimiser.
        criterion: Loss function.
        device: Device to use.

    Returns:
        Average training loss.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        features, labels = batch
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(features)
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
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model.

    Args:
        model: The model to evaluate.
        dataloader: Validation data loader.
        criterion: Loss function.
        device: Device to use.

    Returns:
        Tuple of (average loss, metrics dictionary).
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total_loss += loss.item()
            n_batches += 1

    # Compute metrics
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }

    # AUC requires both classes present
    if len(np.unique(all_labels)) > 1:
        metrics["auc"] = roc_auc_score(all_labels, all_probs)

        # Threshold tuning: find optimal threshold for balanced accuracy (Youden's J)
        fpr, tpr, thresholds_roc = roc_curve(all_labels, all_probs)
        youden_j = tpr - fpr  # Maximising J = maximising balanced accuracy
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
    train_dataset: ClinicalDataset,
    val_dataset: ClinicalDataset,
    model_type: str,
    device: torch.device,
    fold: int,
) -> Dict[str, float]:
    """Train and evaluate a single fold.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        model_type: Either 'mlp' or 'attention'.
        device: Device to use.
        fold: Fold number (for logging).

    Returns:
        Dictionary of best validation metrics.
    """
    config = CONFIG_4A if model_type == "mlp" else CONFIG_4B

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    # Create model
    model = get_model(model_type, device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model parameters: {n_params:,}")

    # Calculate class weights from training data (inverse frequency)
    train_labels = [train_dataset[i][1].item() for i in range(len(train_dataset))]
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    logger.info(f"  Class weights: {class_weights.cpu().numpy()}")

    # Optimiser and criterion
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Training loop with early stopping
    best_val_auc = 0.0
    best_metrics = {}
    patience_counter = 0

    for epoch in range(config["epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_metrics = val_metrics.copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"    Epoch {epoch + 1}: train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, val_auc={val_metrics['auc']:.4f}"
            )

        if patience_counter >= config["patience"]:
            logger.info(f"    Early stopping at epoch {epoch + 1}")
            break

    return best_metrics


def run_cross_validation(
    model_type: str = "mlp",
    device: torch.device = None,
) -> Dict[str, List[float]]:
    """Run 5-fold stratified cross-validation.

    Args:
        model_type: Either 'mlp' or 'attention'.
        device: Device to use.

    Returns:
        Dictionary mapping metric names to lists of per-fold values.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Running CV: model={model_type}")

    # Load data
    df = load_clinical_data()

    # Get outcomes for stratified split
    outcomes = df["outcome"].values

    # Cross-validation
    kfold = StratifiedKFold(
        n_splits=CV_CONFIG["n_splits"],
        shuffle=CV_CONFIG["shuffle"],
        random_state=CV_CONFIG["random_state"],
    )

    fold_metrics = {
        "auc": [],
        "accuracy": [],
        "f1": [],
        "f1_tuned": [],
        "balanced_acc_tuned": [],
    }

    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(outcomes)), outcomes)):
        logger.info(f"Fold {fold + 1}/{CV_CONFIG['n_splits']}")

        # Create datasets (preprocessor fit on training fold only)
        train_ds, val_ds, _ = create_datasets(df, train_idx, val_idx)
        logger.info(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

        # Train fold
        metrics = train_fold(train_ds, val_ds, model_type, device, fold)

        for key in fold_metrics:
            fold_metrics[key].append(metrics[key])

        logger.info(
            f"  Fold {fold + 1} results: AUC={metrics['auc']:.4f}, "
            f"Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, "
            f"BalAcc_tuned={metrics['balanced_acc_tuned']:.4f}, "
            f"F1_tuned={metrics['f1_tuned']:.4f} (thresh={metrics['optimal_threshold']:.3f})"
        )

    # Compute summary statistics
    logger.info("Cross-validation complete:")
    for key in fold_metrics:
        values = fold_metrics[key]
        mean, std = np.mean(values), np.std(values)
        min_val, max_val = np.min(values), np.max(values)
        logger.info(f"  {key}: {mean:.4f} +/- {std:.4f} (min={min_val:.4f}, max={max_val:.4f})")

    return fold_metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test training on MLP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = run_cross_validation(
        model_type="mlp",
        device=device,
    )
