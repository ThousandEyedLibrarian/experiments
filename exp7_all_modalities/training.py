"""Training utilities for Experiment 7: All Four Modalities Fusion."""

import copy
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from .config import CV_CONFIG, MLP_CONFIG, MOE_CONFIG
from .data_pipeline import (
    create_quad_modality_datasets,
    prepare_quad_modality_data,
)
from .models import get_model

logger = logging.getLogger("exp7")


def train_epoch_mlp(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    asm_weighted: bool = False,
    class_weights: torch.Tensor = None,
) -> float:
    """Train for one epoch (MLP model).

    When ``asm_weighted`` is True the dataloader yields an extra
    per-sample weight tensor as the last element of each batch; the
    loss is computed via per-sample weighted cross-entropy
    (``shared.asm_balancing.weighted_cross_entropy``) using
    ``class_weights`` to preserve outcome-class balancing.
    """
    from shared.asm_balancing import weighted_cross_entropy
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        if asm_weighted:
            clinical, text, eeg, mask, smiles, labels, sample_weights = batch
            sample_weights = sample_weights.to(device)
        else:
            clinical, text, eeg, mask, smiles, labels = batch
            sample_weights = None
        clinical = clinical.to(device)
        text = text.to(device)
        eeg = eeg.to(device)
        mask = mask.to(device)
        smiles = smiles.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(clinical, text, eeg, mask, smiles)
        if asm_weighted:
            loss = weighted_cross_entropy(logits, labels, sample_weights, class_weight=class_weights)
        else:
            loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def train_epoch_moe(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    global_step: int = 0,
    asm_weighted: bool = False,
    class_weights: torch.Tensor = None,
) -> Tuple[float, int]:
    """Train for one epoch (MoE model with aux loss).

    Returns:
        Tuple of (avg_loss, updated_global_step).
    """
    from shared.asm_balancing import weighted_cross_entropy
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        if asm_weighted:
            clinical, text, eeg, mask, smiles, labels, sample_weights = batch
            sample_weights = sample_weights.to(device)
        else:
            clinical, text, eeg, mask, smiles, labels = batch
            sample_weights = None
        clinical = clinical.to(device)
        text = text.to(device)
        eeg = eeg.to(device)
        mask = mask.to(device)
        smiles = smiles.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, aux_loss = model(clinical, text, eeg, mask, smiles)
        if asm_weighted:
            loss = weighted_cross_entropy(logits, labels, sample_weights, class_weight=class_weights) + aux_loss
        else:
            loss = criterion(logits, labels) + aux_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Temperature annealing
        if hasattr(model, 'update_temperature'):
            model.update_temperature(global_step)
        global_step += 1

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches, global_step


def evaluate_mlp(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate MLP model."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            clinical, text, eeg, mask, smiles, labels = batch
            clinical = clinical.to(device)
            text = text.to(device)
            eeg = eeg.to(device)
            mask = mask.to(device)
            smiles = smiles.to(device)
            labels = labels.to(device)

            logits = model(clinical, text, eeg, mask, smiles)
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


def evaluate_moe(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate MoE model."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            clinical, text, eeg, mask, smiles, labels = batch
            clinical = clinical.to(device)
            text = text.to(device)
            eeg = eeg.to(device)
            mask = mask.to(device)
            smiles = smiles.to(device)
            labels = labels.to(device)

            logits, aux_loss = model(clinical, text, eeg, mask, smiles)
            loss = criterion(logits, labels) + aux_loss

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
        "y_prob": probs.tolist(),
        "y_true": labels.tolist(),
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
    fusion: str,
    text_model: str,
    smiles_model: str,
    device: torch.device,
    fold: int = 0,
    asm_balance_mode: str = "none",
) -> Dict[str, float]:
    """Train and evaluate a single fold.

    ``asm_balance_mode``:
        - "none" (default): standard training.
        - "weighted": inverse-sqrt sample weights via WeightedASMDataset.
        - "stratified_batch": StratifiedASMBatchSampler ensuring every
           mini-batch contains all ASMs.
    """
    from shared.asm_balancing import (
        WeightedASMDataset,
        StratifiedASMBatchSampler,
        compute_asm_sample_weights,
    )
    # Get config based on fusion type
    config = MLP_CONFIG if fusion == "mlp" else MOE_CONFIG

    # Optional ASM-balancing surgery on the training loader.
    asm_weighted = (asm_balance_mode == "weighted")
    asm_stratified = (asm_balance_mode == "stratified_batch")
    train_asm_labels = list(train_dataset.asm_drugs)

    if asm_weighted:
        weights = compute_asm_sample_weights(train_asm_labels)
        logger.info(f"  ASM-weighted training: mean={weights.mean():.3f}, min={weights.min():.3f}, max={weights.max():.3f}")
        train_dataset = WeightedASMDataset(train_dataset, weights)

    # Create dataloaders
    if asm_stratified:
        batch_sampler = StratifiedASMBatchSampler(
            train_asm_labels,
            batch_size=config["batch_size"],
            seed=fold,
        )
        logger.info(f"  Stratified ASM batch sampler: {len(batch_sampler)} batches, batch_size={config['batch_size']}, ASMs={len(batch_sampler.unique_asms)}")
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=0,
        )
    else:
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
        fusion=fusion,
        text_model=text_model,
        smiles_model=smiles_model,
        device=device,
    )

    # Disable temperature annealing for MoE (Exp12 finding)
    if fusion == "moe" and hasattr(model, 'fuse_moe'):
        model.fuse_moe.temperature_decay = 1.0

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model parameters: {n_params:,}")

    # Calculate class weights from training data
    train_labels = [train_dataset[i][5].item() for i in range(len(train_dataset))]
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

    # Select train/eval functions based on fusion type
    if fusion == "mlp":
        train_fn = train_epoch_mlp
        eval_fn = evaluate_mlp
    else:  # moe
        train_fn = train_epoch_moe
        eval_fn = evaluate_moe

    # Training loop with early stopping
    best_val_auc = 0.0
    best_metrics = {}
    patience_counter = 0
    global_step = 0

    for epoch in range(config["epochs"]):
        if fusion == "moe":
            train_loss, global_step = train_fn(
                model, train_loader, optimizer, criterion, device, global_step,
                asm_weighted=asm_weighted, class_weights=class_weights,
            )
        else:
            train_loss = train_fn(
                model, train_loader, optimizer, criterion, device,
                asm_weighted=asm_weighted, class_weights=class_weights,
            )
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


def run_cross_validation(
    fusion: str = "mlp",
    text_model: str = "clinicalbert",
    smiles_model: str = "chemberta",
    device: torch.device = None,
    asm_balance_mode: str = "none",
) -> Dict[str, List[float]]:
    """Run 5-fold CV for quad modality fusion."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Running CV: Quad Modality ({fusion}, {text_model}, {smiles_model})")

    # Prepare data
    df, smiles_embeddings, smiles_indices, text_embeddings, eeg_data = prepare_quad_modality_data(
        text_model, smiles_model
    )
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

        train_ds, val_ds, _ = create_quad_modality_datasets(
            df, smiles_embeddings, smiles_indices, text_embeddings, eeg_data, train_idx, val_idx
        )
        logger.info(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

        metrics = train_fold(
            train_ds, val_ds,
            fusion=fusion,
            text_model=text_model,
            smiles_model=smiles_model,
            device=device,
            fold=fold,
            asm_balance_mode=asm_balance_mode,
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


# ============================================================================
# Prediction-logging variant for downstream bootstrap CIs / clinical utility
# ============================================================================


def _predict_with_smiles_override(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    fusion: str,
    smiles_override: torch.Tensor = None,
) -> Tuple[List[str], List[int], List[float]]:
    """Run inference over ``val_loader`` and return pids, labels, probs.

    If ``smiles_override`` is provided (a 1D tensor of shape (smiles_dim,)),
    it is broadcast across the batch and substituted for the dataset's SMILES
    tensor. This is the counterfactual ASM-swap path. The val dataset must
    have been built with ``return_pid=True``.
    """
    model.eval()
    pids_out: List[str] = []
    y_true: List[int] = []
    y_prob: List[float] = []

    with torch.no_grad():
        for batch in val_loader:
            # Dataset built with return_pid=True yields a 7-tuple where the
            # final element is a tuple of pid strings produced by the default
            # collate.
            clinical, text, eeg, mask, smiles, labels, pids = batch
            clinical = clinical.to(device)
            text = text.to(device)
            eeg = eeg.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            if smiles_override is not None:
                batch_size = clinical.shape[0]
                smiles_in = smiles_override.to(device).unsqueeze(0).expand(batch_size, -1).contiguous()
            else:
                smiles_in = smiles.to(device)

            if fusion == "moe":
                logits, _aux = model(clinical, text, eeg, mask, smiles_in)
            else:
                logits = model(clinical, text, eeg, mask, smiles_in)

            probs = torch.softmax(logits, dim=1)[:, 1]
            y_prob.extend(probs.cpu().numpy().tolist())
            y_true.extend(labels.cpu().numpy().tolist())
            # The default collate turns a list of strings into a tuple/list.
            pids_out.extend([str(p) for p in pids])

    return pids_out, y_true, y_prob


def train_fold_with_predictions(
    train_dataset,
    val_dataset,
    fusion: str,
    text_model: str,
    smiles_model: str,
    device: torch.device,
    fold: int = 0,
    candidate_smiles: Dict[str, np.ndarray] = None,
    asm_balance_mode: str = "none",
) -> Dict[str, Any]:
    """Train one fold and return per-patient predictions plus ASM-swap predictions.

    Mirrors :func:`train_fold` but additionally:
      - Tracks the best ``model.state_dict()`` (not just metrics) by val AUC.
      - After training, restores the best weights and predicts on the val
        loader once for the prescribed ASM and once per candidate ASM with
        the SMILES tensor swapped at inference time.
      - Returns metrics, val pids, true labels, predicted probabilities under
        the prescribed ASM, and a dict mapping each candidate ASM to its
        per-patient predicted probabilities.

    The ``val_dataset`` must have been constructed with ``return_pid=True``.
    """
    if not getattr(val_dataset, "return_pid", False):
        raise ValueError("val_dataset must be built with return_pid=True for prediction logging.")

    config = MLP_CONFIG if fusion == "mlp" else MOE_CONFIG

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

    model = get_model(
        fusion=fusion,
        text_model=text_model,
        smiles_model=smiles_model,
        device=device,
    )

    if fusion == "moe" and hasattr(model, "fuse_moe"):
        model.fuse_moe.temperature_decay = 1.0

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model parameters: {n_params:,}")

    # Class weights from training labels. With return_pid, train_dataset
    # may also yield pids; the label is at index 5 either way.
    train_labels_list = [train_dataset[i][5].item() for i in range(len(train_dataset))]
    class_counts = np.bincount(train_labels_list)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    if fusion == "mlp":
        train_fn = train_epoch_mlp
        eval_fn = evaluate_mlp
    else:
        train_fn = train_epoch_moe
        eval_fn = evaluate_moe

    best_val_auc = -1.0
    best_metrics: Dict[str, float] = {}
    best_state_dict = None
    patience_counter = 0
    global_step = 0

    # During training we use the regular val_loader-style evaluation. The
    # train_dataset and val_dataset both yield a pid as a trailing element
    # when return_pid=True; train/eval loops here unpack only the first six,
    # so we wrap the loader with a custom collate that drops the pid for
    # the non-prediction passes.
    from shared.asm_balancing import (
        WeightedASMDataset,
        StratifiedASMBatchSampler,
        compute_asm_sample_weights,
    )
    asm_weighted = (asm_balance_mode == "weighted")
    asm_stratified = (asm_balance_mode == "stratified_batch")
    train_asm_labels = list(train_dataset.asm_drugs)

    train_no_pid = _DropPidWrapper(train_dataset)
    if asm_weighted:
        weights = compute_asm_sample_weights(train_asm_labels)
        logger.info(f"  ASM-weighted training: mean={weights.mean():.3f}, min={weights.min():.3f}, max={weights.max():.3f}")
        train_no_pid = WeightedASMDataset(train_no_pid, weights)

    if asm_stratified:
        batch_sampler = StratifiedASMBatchSampler(
            train_asm_labels,
            batch_size=config["batch_size"],
            seed=fold,
        )
        logger.info(f"  Stratified ASM batch sampler: {len(batch_sampler)} batches, ASMs={len(batch_sampler.unique_asms)}")
        train_loader_for_loss = DataLoader(
            train_no_pid,
            batch_sampler=batch_sampler,
            num_workers=0,
        )
    else:
        train_loader_for_loss = DataLoader(
            train_no_pid,
            batch_size=config["batch_size"],
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
    val_loader_for_loss = DataLoader(
        _DropPidWrapper(val_dataset),
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    for epoch in range(config["epochs"]):
        if fusion == "moe":
            train_loss, global_step = train_fn(
                model, train_loader_for_loss, optimizer, criterion, device, global_step,
                asm_weighted=asm_weighted, class_weights=class_weights,
            )
        else:
            train_loss = train_fn(
                model, train_loader_for_loss, optimizer, criterion, device,
                asm_weighted=asm_weighted, class_weights=class_weights,
            )
        val_loss, val_metrics = eval_fn(model, val_loader_for_loss, criterion, device)

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_metrics = val_metrics.copy()
            best_state_dict = copy.deepcopy(model.state_dict())
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

    # Restore best weights for inference.
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Per-patient predictions under the prescribed ASM.
    val_pids, val_y_true, val_y_prob = _predict_with_smiles_override(
        model, val_loader, device, fusion, smiles_override=None
    )

    # Counterfactual predictions under each candidate ASM.
    val_y_prob_per_asm: Dict[str, List[float]] = {}
    if candidate_smiles is not None:
        for asm_name, smiles_vec in candidate_smiles.items():
            override = torch.from_numpy(np.asarray(smiles_vec, dtype=np.float32))
            _, _, probs = _predict_with_smiles_override(
                model, val_loader, device, fusion, smiles_override=override
            )
            val_y_prob_per_asm[asm_name] = probs

    return {
        "metrics": best_metrics,
        "val_pids": val_pids,
        "val_y_true": val_y_true,
        "val_y_prob": val_y_prob,
        "val_y_prob_per_asm": val_y_prob_per_asm,
        "model_state_dict": best_state_dict,
    }


class _DropPidWrapper(torch.utils.data.Dataset):
    """Wrap a return_pid dataset so it yields the legacy 6-tuple.

    Used internally so the existing train/eval loops (which expect 6-tuples)
    can be reused without modification when the underlying dataset has been
    built with ``return_pid=True``.
    """

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        # If the base dataset returns pid, drop it; else pass through.
        if len(item) == 7:
            return item[:6]
        return item


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Test with MLP
    results = run_cross_validation(
        fusion="mlp",
        text_model="clinicalbert",
        smiles_model="chemberta",
        device=device,
    )
