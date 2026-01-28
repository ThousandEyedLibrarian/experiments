"""Training utilities for Experiment 5: Clinical + Single Modality Fusion."""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from .config import CV_CONFIG, TRAINING_CONFIG
from .data_pipeline import (
    ClinicalEEGDataset,
    ClinicalSMILESDataset,
    ClinicalTextDataset,
    create_clinical_eeg_datasets,
    create_clinical_smiles_datasets,
    create_clinical_text_datasets,
    prepare_clinical_eeg_data,
    prepare_clinical_smiles_data,
    prepare_clinical_text_data,
)
from .models import get_model

logger = logging.getLogger("exp5")


def train_epoch_smiles(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch (Clinical + SMILES)."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        clinical, smiles, labels = batch
        clinical = clinical.to(device)
        smiles = smiles.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(clinical, smiles)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def train_epoch_text(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch (Clinical + Text)."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        clinical, text, labels = batch
        clinical = clinical.to(device)
        text = text.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(clinical, text)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def train_epoch_eeg(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch (Clinical + EEG)."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        clinical, eeg_windows, padding_mask, labels = batch
        clinical = clinical.to(device)
        eeg_windows = eeg_windows.to(device)
        padding_mask = padding_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(clinical, eeg_windows, padding_mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate_smiles(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model (Clinical + SMILES)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            clinical, smiles, labels = batch
            clinical = clinical.to(device)
            smiles = smiles.to(device)
            labels = labels.to(device)

            logits = model(clinical, smiles)
            loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total_loss += loss.item()
            n_batches += 1

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    return total_loss / n_batches, metrics


def evaluate_text(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model (Clinical + Text)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            clinical, text, labels = batch
            clinical = clinical.to(device)
            text = text.to(device)
            labels = labels.to(device)

            logits = model(clinical, text)
            loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total_loss += loss.item()
            n_batches += 1

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    return total_loss / n_batches, metrics


def evaluate_eeg(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model (Clinical + EEG)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            clinical, eeg_windows, padding_mask, labels = batch
            clinical = clinical.to(device)
            eeg_windows = eeg_windows.to(device)
            padding_mask = padding_mask.to(device)
            labels = labels.to(device)

            logits = model(clinical, eeg_windows, padding_mask)
            loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total_loss += loss.item()
            n_batches += 1

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    return total_loss / n_batches, metrics


def compute_metrics(
    labels: List,
    preds: List,
    probs: List,
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    labels = np.array(labels)
    preds = np.array(preds)
    probs = np.array(probs)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
    }

    # AUC requires both classes present
    if len(np.unique(labels)) > 1:
        metrics["auc"] = roc_auc_score(labels, probs)

        # Threshold tuning: find optimal threshold for balanced accuracy (Youden's J)
        fpr, tpr, thresholds_roc = roc_curve(labels, probs)
        youden_j = tpr - fpr
        best_idx = np.argmax(youden_j)
        optimal_threshold = thresholds_roc[best_idx]
        tuned_preds = (probs >= optimal_threshold).astype(int)
        metrics["balanced_acc_tuned"] = balanced_accuracy_score(labels, tuned_preds)
        metrics["f1_tuned"] = f1_score(labels, tuned_preds, zero_division=0)
        metrics["optimal_threshold"] = optimal_threshold
    else:
        metrics["auc"] = 0.5
        metrics["balanced_acc_tuned"] = 0.5
        metrics["f1_tuned"] = 0.0
        metrics["optimal_threshold"] = 0.5

    return metrics


def train_fold(
    train_dataset,
    val_dataset,
    modality: str,
    smiles_model: str = None,
    text_model: str = None,
    eeg_model: str = None,
    device: torch.device = None,
    fold: int = 0,
) -> Dict[str, float]:
    """Train and evaluate a single fold."""
    config = TRAINING_CONFIG

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
    model = get_model(
        modality=modality,
        smiles_model=smiles_model,
        text_model=text_model,
        eeg_model=eeg_model,
        device=device,
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model parameters: {n_params:,}")

    # Calculate class weights from training data
    if modality == "smiles":
        train_labels = [train_dataset[i][2].item() for i in range(len(train_dataset))]
    elif modality == "text":
        train_labels = [train_dataset[i][2].item() for i in range(len(train_dataset))]
    else:  # eeg
        train_labels = [train_dataset[i][3].item() for i in range(len(train_dataset))]

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

    # Select train/eval functions based on modality
    if modality == "smiles":
        train_fn = train_epoch_smiles
        eval_fn = evaluate_smiles
    elif modality == "text":
        train_fn = train_epoch_text
        eval_fn = evaluate_text
    else:  # eeg
        train_fn = train_epoch_eeg
        eval_fn = evaluate_eeg

    # Training loop with early stopping
    best_val_auc = 0.0
    best_metrics = {}
    patience_counter = 0

    for epoch in range(config["epochs"]):
        train_loss = train_fn(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = eval_fn(model, val_loader, criterion, device)

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


def run_cross_validation_smiles(
    smiles_model: str = "chemberta",
    device: torch.device = None,
) -> Dict[str, List[float]]:
    """Run 5-fold CV for Clinical + SMILES."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Running CV: Clinical + SMILES ({smiles_model})")

    # Prepare data
    df, smiles_embeddings, smiles_indices = prepare_clinical_smiles_data(smiles_model)
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

        train_ds, val_ds, _ = create_clinical_smiles_datasets(
            df, smiles_embeddings, smiles_indices, train_idx, val_idx
        )
        logger.info(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

        metrics = train_fold(
            train_ds, val_ds,
            modality="smiles",
            smiles_model=smiles_model,
            device=device,
            fold=fold,
        )

        for key in fold_metrics:
            fold_metrics[key].append(metrics[key])

        logger.info(
            f"  Fold {fold + 1} results: AUC={metrics['auc']:.4f}, "
            f"BalAcc_tuned={metrics['balanced_acc_tuned']:.4f}"
        )

    log_cv_summary(fold_metrics)
    return fold_metrics


def run_cross_validation_text(
    text_model: str = "clinicalbert",
    device: torch.device = None,
) -> Dict[str, List[float]]:
    """Run 5-fold CV for Clinical + Text."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Running CV: Clinical + Text ({text_model})")

    # Prepare data
    df, text_embeddings = prepare_clinical_text_data(text_model)
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

        train_ds, val_ds, _ = create_clinical_text_datasets(
            df, text_embeddings, train_idx, val_idx
        )
        logger.info(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

        metrics = train_fold(
            train_ds, val_ds,
            modality="text",
            text_model=text_model,
            device=device,
            fold=fold,
        )

        for key in fold_metrics:
            fold_metrics[key].append(metrics[key])

        logger.info(
            f"  Fold {fold + 1} results: AUC={metrics['auc']:.4f}, "
            f"BalAcc_tuned={metrics['balanced_acc_tuned']:.4f}"
        )

    log_cv_summary(fold_metrics)
    return fold_metrics


def run_cross_validation_eeg(
    eeg_model: str = "simplecnn",
    device: torch.device = None,
) -> Dict[str, List[float]]:
    """Run 5-fold CV for Clinical + EEG."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Running CV: Clinical + EEG ({eeg_model})")

    # Prepare data
    df, eeg_data = prepare_clinical_eeg_data()
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

        train_ds, val_ds, _ = create_clinical_eeg_datasets(
            df, eeg_data, train_idx, val_idx
        )
        logger.info(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

        metrics = train_fold(
            train_ds, val_ds,
            modality="eeg",
            eeg_model=eeg_model,
            device=device,
            fold=fold,
        )

        for key in fold_metrics:
            fold_metrics[key].append(metrics[key])

        logger.info(
            f"  Fold {fold + 1} results: AUC={metrics['auc']:.4f}, "
            f"BalAcc_tuned={metrics['balanced_acc_tuned']:.4f}"
        )

    log_cv_summary(fold_metrics)
    return fold_metrics


def log_cv_summary(fold_metrics: Dict[str, List[float]]):
    """Log cross-validation summary."""
    logger.info("Cross-validation complete:")
    for key in fold_metrics:
        values = fold_metrics[key]
        mean, std = np.mean(values), np.std(values)
        min_val, max_val = np.min(values), np.max(values)
        logger.info(f"  {key}: {mean:.4f} +/- {std:.4f} (min={min_val:.4f}, max={max_val:.4f})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Test with SMILES
    results = run_cross_validation_smiles(smiles_model="chemberta", device=device)
