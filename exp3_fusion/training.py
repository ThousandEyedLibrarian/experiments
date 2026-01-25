"""Training utilities for Experiment 3: LLM + EEG + SMILES triple fusion."""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from .config import CONFIG_3A, CONFIG_3B, CV_CONFIG, EEG_ENCODER_CONFIG, SMILES_DIMS
from .data_pipeline import (
    TripleModalityDataset,
    create_datasets,
    get_max_channels,
    prepare_data,
)
from .models import TripleModalityMLP, TripleModalityFuseMoE

logger = logging.getLogger("exp3")


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    is_moe: bool = False,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        text_emb, eeg_windows, padding_mask, smiles_emb, labels = batch
        text_emb = text_emb.to(device)
        eeg_windows = eeg_windows.to(device)
        padding_mask = padding_mask.to(device)
        smiles_emb = smiles_emb.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if is_moe:
            logits, aux_loss = model(text_emb, eeg_windows, padding_mask, smiles_emb)
            loss = criterion(logits, labels) + aux_loss
        else:
            logits = model(text_emb, eeg_windows, padding_mask, smiles_emb)
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
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            text_emb, eeg_windows, padding_mask, smiles_emb, labels = batch
            text_emb = text_emb.to(device)
            eeg_windows = eeg_windows.to(device)
            padding_mask = padding_mask.to(device)
            smiles_emb = smiles_emb.to(device)
            labels = labels.to(device)

            if is_moe:
                logits, aux_loss = model(text_emb, eeg_windows, padding_mask, smiles_emb)
                loss = criterion(logits, labels) + aux_loss
            else:
                logits = model(text_emb, eeg_windows, padding_mask, smiles_emb)
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
    else:
        metrics["auc"] = 0.5

    return total_loss / n_batches, metrics


def get_model(
    fusion_type: str,
    text_dim: int,
    smiles_dim: int,
    device: torch.device,
) -> nn.Module:
    """Create fusion model based on type."""
    if fusion_type == "mlp":
        config = CONFIG_3A
        model = TripleModalityMLP(
            text_dim=text_dim,
            smiles_dim=smiles_dim,
            hidden_dim=config["hidden_dim"],
            num_classes=config["num_classes"],
            dropout=config["dropout"],
            eeg_encoder_type=EEG_ENCODER_CONFIG["encoder_type"],
            n_eeg_channels=EEG_ENCODER_CONFIG["n_channels"],
            n_eeg_times=EEG_ENCODER_CONFIG["n_times"],
            eeg_embed_dim=EEG_ENCODER_CONFIG["embed_dim"],
            num_heads=EEG_ENCODER_CONFIG["num_heads"],
            num_layers=EEG_ENCODER_CONFIG["num_layers"],
            max_windows=EEG_ENCODER_CONFIG["max_windows"],
            window_chunk_size=EEG_ENCODER_CONFIG["window_chunk_size"],
        )
    elif fusion_type == "fusemoe":
        config = CONFIG_3B
        model = TripleModalityFuseMoE(
            text_dim=text_dim,
            smiles_dim=smiles_dim,
            hidden_dim=config["hidden_dim"],
            num_classes=config["num_classes"],
            num_experts=config["num_experts"],
            top_k=config["top_k"],
            num_heads=config["num_heads"],
            num_moe_layers=config["num_moe_layers"],
            dropout=config["dropout"],
            aux_loss_weight=config["aux_loss_weight"],
            eeg_encoder_type=EEG_ENCODER_CONFIG["encoder_type"],
            n_eeg_channels=EEG_ENCODER_CONFIG["n_channels"],
            n_eeg_times=EEG_ENCODER_CONFIG["n_times"],
            eeg_embed_dim=EEG_ENCODER_CONFIG["embed_dim"],
            num_eeg_layers=EEG_ENCODER_CONFIG["num_layers"],
            max_windows=EEG_ENCODER_CONFIG["max_windows"],
            window_chunk_size=EEG_ENCODER_CONFIG["window_chunk_size"],
        )
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")

    return model.to(device)


def train_fold(
    train_dataset: TripleModalityDataset,
    val_dataset: TripleModalityDataset,
    fusion_type: str,
    text_dim: int,
    smiles_dim: int,
    device: torch.device,
    fold: int,
) -> Dict[str, float]:
    """Train and evaluate a single fold."""
    config = CONFIG_3A if fusion_type == "mlp" else CONFIG_3B
    is_moe = fusion_type == "fusemoe"

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
    model = get_model(fusion_type, text_dim, smiles_dim, device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model parameters: {n_params:,}")

    # Optimizer and criterion
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_auc = 0.0
    best_metrics = {}
    patience_counter = 0

    for epoch in range(config["epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, is_moe)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, is_moe)

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
    text_model: str,
    smiles_model: str,
    fusion_type: str,
    device: torch.device = None,
) -> Dict[str, List[float]]:
    """Run 5-fold cross-validation for a specific configuration."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Running CV: text={text_model}, smiles={smiles_model}, fusion={fusion_type}")

    # Prepare data
    text_emb, eeg_data, smiles_emb, smiles_idx, df = prepare_data(
        text_model=text_model,
        smiles_model=smiles_model,
        cache_eeg=True,
    )

    text_dim = 768  # All text models use 768
    smiles_dim = SMILES_DIMS[smiles_model]
    max_channels = get_max_channels(eeg_data)

    # Get outcomes for stratified split
    outcomes = df["outcome"].values

    # Cross-validation
    kfold = StratifiedKFold(
        n_splits=CV_CONFIG["n_splits"],
        shuffle=CV_CONFIG["shuffle"],
        random_state=CV_CONFIG["random_state"],
    )

    fold_metrics = {"auc": [], "accuracy": [], "f1": []}

    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(outcomes)), outcomes)):
        logger.info(f"Fold {fold + 1}/{CV_CONFIG['n_splits']}")

        # Create datasets
        train_ds, val_ds = create_datasets(
            text_emb, eeg_data, smiles_emb, smiles_idx, df,
            train_idx, val_idx, max_channels,
        )
        logger.info(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

        # Train fold
        metrics = train_fold(train_ds, val_ds, fusion_type, text_dim, smiles_dim, device, fold)

        for key in fold_metrics:
            fold_metrics[key].append(metrics[key])

        logger.info(
            f"  Fold {fold + 1} results: AUC={metrics['auc']:.4f}, "
            f"Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}"
        )

    # Compute summary statistics
    logger.info("Cross-validation complete:")
    for key in fold_metrics:
        values = fold_metrics[key]
        mean, std = np.mean(values), np.std(values)
        logger.info(f"  {key}: {mean:.4f} +/- {std:.4f}")

    return fold_metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test training on a single config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = run_cross_validation(
        text_model="clinicalbert",
        smiles_model="chemberta",
        fusion_type="mlp",
        device=device,
    )
