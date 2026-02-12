"""Training utilities for Experiment 10: Direct LLM Text Modality.

Handles frozen and fine-tuned LLM encoder training with
differential learning rates for the fine-tuning case.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

# Add parent directory for stratification imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from exp8_stratification.stratified_cv import get_multilabel_splits, get_outcome_only_splits

from .config import (
    CV_CONFIG,
    FINETUNE_CONFIG,
    TRAINING_CONFIG,
)
from .data_pipeline import create_datasets, prepare_data
from .models import ClinicalLLMFusion, get_model

logger = logging.getLogger("exp10")


# ============================================================================
# Training and Evaluation
# ============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        clinical, input_ids, attention_mask, labels = batch
        clinical = clinical.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(clinical, input_ids, attention_mask)
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
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            clinical, input_ids, attention_mask, labels = batch
            clinical = clinical.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(clinical, input_ids, attention_mask)
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
    """Compute evaluation metrics with threshold tuning."""
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

        # Threshold tuning: find optimal threshold via Youden's J
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


# ============================================================================
# Fold Training
# ============================================================================

def _build_optimiser(
    model: ClinicalLLMFusion,
    freeze: bool,
    config: Dict,
) -> torch.optim.Optimizer:
    """Build optimiser with differential LR for fine-tuning.

    When fine-tuning, the LLM encoder uses a lower learning rate
    than the classification head.
    """
    if freeze:
        # All trainable params use the same LR
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )

    # Fine-tuning: differential learning rates
    llm_params = list(model.llm_encoder.parameters())
    llm_param_ids = {id(p) for p in llm_params}
    head_params = [p for p in model.parameters() if id(p) not in llm_param_ids]

    param_groups = [
        {"params": [p for p in llm_params if p.requires_grad],
         "lr": config.get("encoder_lr", 2e-5)},
        {"params": head_params,
         "lr": config.get("head_lr", 1e-3)},
    ]

    return torch.optim.AdamW(
        param_groups,
        weight_decay=config["weight_decay"],
    )


def train_fold(
    train_dataset,
    val_dataset,
    llm_model: str,
    freeze: bool,
    unfreeze_layers: int = 0,
    device: torch.device = None,
    fold: int = 0,
) -> Dict[str, float]:
    """Train and evaluate a single fold.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        llm_model: LLM model key.
        freeze: Whether LLM encoder is frozen.
        unfreeze_layers: Number of layers to unfreeze.
        device: Device to use.
        fold: Fold index for logging.

    Returns:
        Best validation metrics for this fold.
    """
    config = FINETUNE_CONFIG if not freeze else TRAINING_CONFIG

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
        llm_model=llm_model,
        freeze=freeze,
        unfreeze_layers=unfreeze_layers,
        device=device,
    )

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {n_trainable:,} trainable / {n_total:,} total")

    # Calculate class weights from training data
    train_labels = train_dataset.labels.numpy()
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    logger.info(f"  Class weights: {class_weights.cpu().numpy()}")

    # Optimiser and criterion
    optimizer = _build_optimiser(model, freeze, config)
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


# ============================================================================
# Cross-Validation
# ============================================================================

def run_cross_validation(
    llm_model: str = "pubmedbert",
    freeze: bool = True,
    unfreeze_layers: int = 0,
    device: torch.device = None,
    use_multilabel_stratification: bool = True,
) -> Dict[str, List[float]]:
    """Run 5-fold CV for Clinical + Direct LLM.

    Args:
        llm_model: LLM model key from config.
        freeze: Whether to freeze the LLM encoder.
        unfreeze_layers: Number of layers to unfreeze if not frozen.
        device: Device to use.
        use_multilabel_stratification: Whether to use multilabel stratification.

    Returns:
        Dictionary of metric name -> list of per-fold values.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mode = "frozen" if freeze else f"finetune (last {unfreeze_layers} layers)"
    strat_type = "multilabel" if use_multilabel_stratification else "outcome-only"
    logger.info(
        f"Running CV: Clinical + {llm_model} ({mode}) "
        f"with {strat_type} stratification"
    )

    # Build a temporary LLM encoder to get the tokeniser
    from .models.llm_encoder import get_llm_encoder
    temp_encoder = get_llm_encoder(llm_model, freeze=True)
    tokeniser = temp_encoder.tokeniser

    # Prepare data (pre-tokenise all reports)
    df, input_ids, attention_masks = prepare_data(
        llm_model_name=llm_model,
        tokeniser=tokeniser,
        max_length=temp_encoder.max_length,
    )

    # Free the temporary encoder
    del temp_encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Cross-validation splits
    if use_multilabel_stratification:
        try:
            splits = list(get_multilabel_splits(
                df,
                stratify_cols=["outcome", "focal", "sex", "age_group"],
                n_splits=CV_CONFIG["n_splits"],
                shuffle=CV_CONFIG["shuffle"],
                random_state=CV_CONFIG["random_state"],
            ))
        except ImportError:
            logger.warning(
                "iterative-stratification not installed, "
                "falling back to outcome-only stratification"
            )
            splits = list(get_outcome_only_splits(
                df,
                n_splits=CV_CONFIG["n_splits"],
                shuffle=CV_CONFIG["shuffle"],
                random_state=CV_CONFIG["random_state"],
            ))
    else:
        splits = list(get_outcome_only_splits(
            df,
            n_splits=CV_CONFIG["n_splits"],
            shuffle=CV_CONFIG["shuffle"],
            random_state=CV_CONFIG["random_state"],
        ))

    fold_metrics = {
        "auc": [],
        "accuracy": [],
        "f1": [],
        "f1_tuned": [],
        "balanced_acc_tuned": [],
    }

    for fold, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"Fold {fold + 1}/{CV_CONFIG['n_splits']}")

        train_ds, val_ds, _ = create_datasets(
            df, input_ids, attention_masks, train_idx, val_idx
        )
        logger.info(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

        metrics = train_fold(
            train_ds,
            val_ds,
            llm_model=llm_model,
            freeze=freeze,
            unfreeze_layers=unfreeze_layers,
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
        logger.info(
            f"  {key}: {mean:.4f} +/- {std:.4f} "
            f"(min={min_val:.4f}, max={max_val:.4f})"
        )
