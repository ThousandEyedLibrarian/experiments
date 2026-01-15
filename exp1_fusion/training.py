"""Training loop with 5-fold stratified cross-validation."""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

from .config import (
    CONFIG_1A, CONFIG_1B, CV_CONFIG, RESULTS_DIR,
    TEXT_DIM, SMILES_DIMS,
)
from .data_pipeline import get_full_dataset, load_csv_data
from .models import ConcatMLPClassifier, SimplifiedFuseMoE


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def create_model(
    fusion_type: str,
    text_dim: int,
    smiles_dim: int,
    config: Dict,
) -> nn.Module:
    """Create a model based on fusion type."""
    if fusion_type == 'mlp':
        return ConcatMLPClassifier(
            text_dim=text_dim,
            smiles_dim=smiles_dim,
            hidden_dims=config['hidden_dims'],
            dropout_rates=config['dropout_rates'],
            num_classes=config['num_classes'],
        )
    elif fusion_type == 'fusemoe':
        return SimplifiedFuseMoE(
            text_dim=text_dim,
            smiles_dim=smiles_dim,
            hidden_dim=config['hidden_dim'],
            num_experts=config['num_experts'],
            top_k=config['top_k'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            num_classes=config['num_classes'],
        )
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_aux_loss: bool = False,
    aux_loss_weight: float = 0.1,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        optimizer.zero_grad()

        text_emb = batch['text_emb'].to(device)
        smiles_emb = batch['smiles_emb'].to(device)
        labels = batch['label'].to(device)

        if use_aux_loss:
            logits, aux_loss = model(text_emb, smiles_emb)
            loss = criterion(logits, labels) + aux_loss_weight * aux_loss
        else:
            logits = model(text_emb, smiles_emb)
            loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    use_aux_loss: bool = False,
) -> Tuple[float, float, float, List[int], List[float], List[int]]:
    """
    Evaluate the model.

    Returns:
        accuracy, auc, f1, predictions, probabilities, labels
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            text_emb = batch['text_emb'].to(device)
            smiles_emb = batch['smiles_emb'].to(device)
            labels = batch['label']

            if use_aux_loss:
                logits, _ = model(text_emb, smiles_emb)
            else:
                logits = model(text_emb, smiles_emb)

            probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    # Handle edge case where all predictions are same class
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5  # Random baseline

    return accuracy, auc, f1, all_preds, all_probs, all_labels


def run_experiment(
    experiment_name: str,
    text_model: str,
    smiles_model: str,
    fusion_type: str,
    verbose: bool = True,
) -> Dict:
    """
    Run a single experiment with 5-fold cross-validation.

    Args:
        experiment_name: Name for the experiment
        text_model: 'clinicalbert' or 'pubmedbert'
        smiles_model: 'chemberta' or 'smilestrf'
        fusion_type: 'mlp' or 'fusemoe'
        verbose: Whether to print progress

    Returns:
        Dictionary with results
    """
    device = get_device()
    if verbose:
        print(f"\n{'='*60}")
        print(f"Experiment: {experiment_name}")
        print(f"Text: {text_model}, SMILES: {smiles_model}, Fusion: {fusion_type}")
        print(f"Device: {device}")
        print(f"{'='*60}")

    # Get config
    config = CONFIG_1A if fusion_type == 'mlp' else CONFIG_1B
    use_aux_loss = fusion_type == 'fusemoe'

    # Load data
    dataset, outcomes = get_full_dataset(text_model, smiles_model)
    smiles_dim = SMILES_DIMS[smiles_model]

    if verbose:
        print(f"Dataset size: {len(dataset)}")
        print(f"Class distribution: {np.bincount(outcomes)}")

    # Set up cross-validation
    skf = StratifiedKFold(
        n_splits=CV_CONFIG['n_splits'],
        shuffle=CV_CONFIG['shuffle'],
        random_state=CV_CONFIG['random_state'],
    )

    # Results storage
    results = {
        'experiment': experiment_name,
        'text_model': text_model,
        'smiles_model': smiles_model,
        'fusion_type': fusion_type,
        'n_samples': len(dataset),
        'n_folds': CV_CONFIG['n_splits'],
        'fold_accuracy': [],
        'fold_auc': [],
        'fold_f1': [],
        'config': config,
    }

    # Run cross-validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), outcomes)):
        if verbose:
            print(f"\n--- Fold {fold + 1}/{CV_CONFIG['n_splits']} ---")

        # Create dataloaders
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(
            train_subset,
            batch_size=config['batch_size'],
            shuffle=True,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=config['batch_size'],
            shuffle=False,
        )

        # Create model
        model = create_model(fusion_type, TEXT_DIM, smiles_dim, config)
        model.to(device)

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5,
        )

        # Loss function with class weights
        class_counts = np.bincount(outcomes[train_idx])
        class_weights = torch.FloatTensor([1.0 / c for c in class_counts])
        class_weights = class_weights / class_weights.sum()
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

        # Training loop with early stopping
        best_val_auc = 0.0
        patience_counter = 0
        best_model_state = None

        for epoch in range(config['epochs']):
            # Train
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device,
                use_aux_loss=use_aux_loss,
                aux_loss_weight=config.get('aux_loss_weight', 0.1),
            )

            # Evaluate
            val_acc, val_auc, val_f1, _, _, _ = evaluate(
                model, val_loader, device, use_aux_loss=use_aux_loss
            )

            scheduler.step(val_auc)

            # Early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= config['patience']:
                if verbose:
                    print(f"  Early stopping at epoch {epoch + 1}")
                break

            if verbose and (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch + 1}: Loss={train_loss:.4f}, "
                      f"Val AUC={val_auc:.4f}, Val F1={val_f1:.4f}")

        # Load best model and get final metrics
        model.load_state_dict(best_model_state)
        model.to(device)

        final_acc, final_auc, final_f1, _, _, _ = evaluate(
            model, val_loader, device, use_aux_loss=use_aux_loss
        )

        results['fold_accuracy'].append(final_acc)
        results['fold_auc'].append(final_auc)
        results['fold_f1'].append(final_f1)

        if verbose:
            print(f"  Fold {fold + 1} Results: Acc={final_acc:.4f}, "
                  f"AUC={final_auc:.4f}, F1={final_f1:.4f}")

    # Aggregate results
    results['accuracy'] = {
        'mean': float(np.mean(results['fold_accuracy'])),
        'std': float(np.std(results['fold_accuracy'])),
        'per_fold': results['fold_accuracy'],
    }
    results['auc'] = {
        'mean': float(np.mean(results['fold_auc'])),
        'std': float(np.std(results['fold_auc'])),
        'per_fold': results['fold_auc'],
    }
    results['f1'] = {
        'mean': float(np.mean(results['fold_f1'])),
        'std': float(np.std(results['fold_f1'])),
        'per_fold': results['fold_f1'],
    }

    # Clean up fold lists (now in nested dict)
    del results['fold_accuracy']
    del results['fold_auc']
    del results['fold_f1']

    if verbose:
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS: {experiment_name}")
        print(f"{'='*60}")
        print(f"Accuracy: {results['accuracy']['mean']:.4f} +/- {results['accuracy']['std']:.4f}")
        print(f"AUC:      {results['auc']['mean']:.4f} +/- {results['auc']['std']:.4f}")
        print(f"F1:       {results['f1']['mean']:.4f} +/- {results['f1']['std']:.4f}")

    # Add timestamp
    results['timestamp'] = datetime.now().isoformat()

    return results


def save_results(results: Dict, filename: Optional[str] = None) -> str:
    """Save results to JSON file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if filename is None:
        filename = f"{results['experiment']}.json"

    filepath = os.path.join(RESULTS_DIR, filename)

    # Convert config to serializable format
    results_copy = results.copy()
    results_copy['config'] = {k: v for k, v in results['config'].items()}

    with open(filepath, 'w') as f:
        json.dump(results_copy, f, indent=2)

    print(f"Results saved to: {filepath}")
    return filepath


if __name__ == '__main__':
    # Test training with a single experiment
    print("Testing training pipeline...")

    results = run_experiment(
        experiment_name='test_exp1a_pubmedbert_chemberta',
        text_model='pubmedbert',
        smiles_model='chemberta',
        fusion_type='mlp',
        verbose=True,
    )

    save_results(results, 'test_results.json')
    print("\nTraining test complete!")
