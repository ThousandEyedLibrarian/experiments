"""Run Experiment 11: EEG2Vec 128D re-runs with aggregator variants.

Re-runs exp3a (triple MLP), exp6b (clinical+SMILES+EEG), and exp7a (quad MLP)
with EEG2Vec encoder at 128D embedding dimension (based on exp9 findings).
Tests both Transformer and MeanMax aggregators.

Usage:
    python -m exp11_eeg_upgrade.run_experiments                    # All 16 configs
    python -m exp11_eeg_upgrade.run_experiments --base exp3a       # Only exp3a variants
    python -m exp11_eeg_upgrade.run_experiments --base exp6b       # Only exp6b variants
    python -m exp11_eeg_upgrade.run_experiments --base exp7a       # Only exp7a variants
    python -m exp11_eeg_upgrade.run_experiments --aggregator meanmax  # Only MeanMax
    python -m exp11_eeg_upgrade.run_experiments --dry-run           # Test data loading
"""

import argparse
import json
import logging
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

# Local config and models
from .config import (
    CONFIG_EXP3A,
    CONFIG_EXP6B,
    CONFIG_EXP7A,
    CV_CONFIG,
    EEG_CONFIG,
    EXPERIMENTS,
    RESULTS_DIR,
    SMILES_DIMS,
)
from .models import ClinicalEEGFusionv2, QuadMLPv2, TripleMLPv2

# Import data pipelines from parent experiments
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from exp3_fusion.data_pipeline import prepare_data as prepare_exp3_data, create_datasets as create_exp3_datasets, get_max_channels
from exp3_fusion.training import train_epoch as train_epoch_exp3, evaluate as evaluate_exp3
from exp6_clinical_triple.data_pipeline import prepare_clinical_smiles_eeg_data, create_clinical_smiles_eeg_datasets
from exp6_clinical_triple.training import train_epoch_eeg, evaluate_eeg
from exp7_all_modalities.data_pipeline import prepare_quad_modality_data, create_quad_modality_datasets
from exp7_all_modalities.training import train_epoch_mlp as train_epoch_exp7, evaluate_mlp as evaluate_exp7

logger = logging.getLogger("exp11")


def _train_fold_generic(model, train_loader, val_loader, config, device, train_fn, eval_fn):
    """Generic training fold with early stopping."""
    # Class weights from training data
    train_labels = []
    for batch in train_loader:
        train_labels.extend(batch[-1].numpy())

    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

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


def run_cv_exp3a(exp_config, device):
    """Run CV for exp3a-type experiment (Triple MLP: Text + EEG + SMILES)."""
    text_model = exp_config["text"]
    smiles_model = exp_config["smiles"]
    aggregator = exp_config["aggregator"]
    config = CONFIG_EXP3A

    logger.info(f"  Data: text={text_model}, smiles={smiles_model}")
    text_emb, eeg_data, smiles_emb, smiles_idx, df = prepare_exp3_data(
        text_model=text_model, smiles_model=smiles_model, cache_eeg=True,
    )
    smiles_dim = SMILES_DIMS[smiles_model]
    max_channels = get_max_channels(eeg_data)
    outcomes = df["outcome"].values

    kfold = StratifiedKFold(**CV_CONFIG)
    fold_metrics = {"auc": [], "accuracy": [], "f1": [], "f1_tuned": [], "balanced_acc_tuned": []}

    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(outcomes)), outcomes)):
        logger.info(f"  Fold {fold + 1}/{CV_CONFIG['n_splits']}")
        train_ds, val_ds = create_exp3_datasets(
            text_emb, eeg_data, smiles_emb, smiles_idx, df, train_idx, val_idx, max_channels,
        )

        model = TripleMLPv2(
            text_dim=768,
            smiles_dim=smiles_dim,
            hidden_dim=config["hidden_dim"],
            num_classes=config["num_classes"],
            dropout=config["dropout"],
            eeg_encoder_type=EEG_CONFIG["encoder_type"],
            eeg_embed_dim=EEG_CONFIG["embed_dim"],
            aggregator_type=aggregator,
            n_eeg_channels=EEG_CONFIG["n_channels"],
            n_eeg_times=EEG_CONFIG["n_times"],
            max_windows=EEG_CONFIG["max_windows"],
            window_chunk_size=EEG_CONFIG["window_chunk_size"],
        ).to(device)

        train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)

        metrics = _train_fold_generic(
            model, train_loader, val_loader, config, device,
            lambda m, dl, o, c, d: train_epoch_exp3(m, dl, o, c, d, is_moe=False, global_step=0)[0],
            lambda m, dl, c, d: evaluate_exp3(m, dl, c, d, is_moe=False),
        )

        for key in fold_metrics:
            fold_metrics[key].append(metrics[key])
        logger.info(f"    AUC={metrics['auc']:.4f}, BalAcc={metrics['balanced_acc_tuned']:.4f}")

    return fold_metrics


def run_cv_exp6b(exp_config, device):
    """Run CV for exp6b-type experiment (Clinical + SMILES + EEG)."""
    smiles_model = exp_config["smiles"]
    aggregator = exp_config["aggregator"]
    config = CONFIG_EXP6B

    logger.info(f"  Data: smiles={smiles_model}")
    df, smiles_embeddings, smiles_indices, eeg_data = prepare_clinical_smiles_eeg_data(smiles_model)
    smiles_dim = SMILES_DIMS[smiles_model]
    outcomes = df["outcome"].values

    kfold = StratifiedKFold(**CV_CONFIG)
    fold_metrics = {"auc": [], "accuracy": [], "f1": [], "f1_tuned": [], "balanced_acc_tuned": []}

    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(outcomes)), outcomes)):
        logger.info(f"  Fold {fold + 1}/{CV_CONFIG['n_splits']}")
        train_ds, val_ds, _ = create_clinical_smiles_eeg_datasets(
            df, smiles_embeddings, smiles_indices, eeg_data, train_idx, val_idx,
        )

        model = ClinicalEEGFusionv2(
            smiles_dim=smiles_dim,
            hidden_dim=config["hidden_dim"],
            num_classes=config["num_classes"],
            dropout=config["dropout"],
            eeg_encoder_type=EEG_CONFIG["encoder_type"],
            eeg_embed_dim=EEG_CONFIG["embed_dim"],
            aggregator_type=aggregator,
            n_channels=EEG_CONFIG["n_channels"],
            n_times=EEG_CONFIG["n_times"],
            max_windows=EEG_CONFIG["max_windows"],
            window_chunk_size=EEG_CONFIG["window_chunk_size"],
        ).to(device)

        batch_size = config["batch_size_eeg"]
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        metrics = _train_fold_generic(
            model, train_loader, val_loader, config, device,
            train_epoch_eeg, evaluate_eeg,
        )

        for key in fold_metrics:
            fold_metrics[key].append(metrics[key])
        logger.info(f"    AUC={metrics['auc']:.4f}, BalAcc={metrics['balanced_acc_tuned']:.4f}")

    return fold_metrics


def run_cv_exp7a(exp_config, device):
    """Run CV for exp7a-type experiment (Quad MLP: Clinical + Text + EEG + SMILES)."""
    text_model = exp_config["text"]
    smiles_model = exp_config["smiles"]
    aggregator = exp_config["aggregator"]
    config = CONFIG_EXP7A

    logger.info(f"  Data: text={text_model}, smiles={smiles_model}")
    df, smiles_emb, smiles_idx, text_emb, eeg_data = prepare_quad_modality_data(
        text_model, smiles_model,
    )
    smiles_dim = SMILES_DIMS[smiles_model]
    outcomes = df["outcome"].values

    kfold = StratifiedKFold(**CV_CONFIG)
    fold_metrics = {"auc": [], "accuracy": [], "f1": [], "f1_tuned": [], "balanced_acc_tuned": []}

    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(outcomes)), outcomes)):
        logger.info(f"  Fold {fold + 1}/{CV_CONFIG['n_splits']}")
        train_ds, val_ds, _ = create_quad_modality_datasets(
            df, smiles_emb, smiles_idx, text_emb, eeg_data, train_idx, val_idx,
        )

        model = QuadMLPv2(
            smiles_dim=smiles_dim,
            hidden_dim=config["hidden_dim"],
            num_classes=config["num_classes"],
            dropout=config["dropout"],
            eeg_encoder_type=EEG_CONFIG["encoder_type"],
            eeg_embed_dim=EEG_CONFIG["embed_dim"],
            aggregator_type=aggregator,
            n_channels=EEG_CONFIG["n_channels"],
            n_times=EEG_CONFIG["n_times"],
            max_windows=EEG_CONFIG["max_windows"],
            window_chunk_size=EEG_CONFIG["window_chunk_size"],
        ).to(device)

        train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)

        metrics = _train_fold_generic(
            model, train_loader, val_loader, config, device,
            train_epoch_exp7, evaluate_exp7,
        )

        for key in fold_metrics:
            fold_metrics[key].append(metrics[key])
        logger.info(f"    AUC={metrics['auc']:.4f}, BalAcc={metrics['balanced_acc_tuned']:.4f}")

    return fold_metrics


CV_RUNNERS = {
    "exp3a": run_cv_exp3a,
    "exp6b": run_cv_exp6b,
    "exp7a": run_cv_exp7a,
}


def main():
    parser = argparse.ArgumentParser(description="Experiment 11: EEG2Vec 128D Re-runs")
    parser.add_argument("--base", choices=["exp3a", "exp6b", "exp7a"], help="Only run this base experiment type")
    parser.add_argument("--aggregator", choices=["transformer", "meanmax"], help="Only run this aggregator")
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--output", type=str, help="Override output path")
    parser.add_argument("--dry-run", action="store_true", help="Test configuration only")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Filter experiments
    experiments = EXPERIMENTS
    if args.base:
        experiments = [e for e in experiments if e["base"] == args.base]
    if args.aggregator:
        experiments = [e for e in experiments if e["aggregator"] == args.aggregator]

    logger.info(f"Running {len(experiments)} experiments")

    if args.dry_run:
        for exp in experiments:
            logger.info(f"  {exp['name']}: base={exp['base']}, aggregator={exp['aggregator']}")
        logger.info("Dry run complete.")
        return

    # Run experiments
    all_results = {}
    for exp in experiments:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {exp['name']}")
        logger.info(f"{'='*60}")

        runner = CV_RUNNERS[exp["base"]]
        fold_metrics = runner(exp, device)

        # Compute summary
        summary = {}
        for key, values in fold_metrics.items():
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

        all_results[exp["name"]] = {
            "config": exp,
            "eeg_config": EEG_CONFIG,
            "fold_metrics": {k: [float(v) for v in vals] for k, vals in fold_metrics.items()},
            "summary": summary,
        }

        logger.info(
            f"  Result: AUC={summary['auc']['mean']:.4f} +/- {summary['auc']['std']:.4f}, "
            f"BalAcc={summary['balanced_acc_tuned']['mean']:.4f} +/- {summary['balanced_acc_tuned']['std']:.4f}"
        )

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or str(RESULTS_DIR / f"results_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    # Print summary table
    print(f"\n{'Experiment':<45} {'AUC':>12} {'Bal Acc':>12} {'F1 Tuned':>12}")
    print("-" * 85)
    for name, result in sorted(all_results.items(), key=lambda x: x[1]["summary"]["auc"]["mean"], reverse=True):
        s = result["summary"]
        print(
            f"{name:<45} "
            f"{s['auc']['mean']:.3f}+/-{s['auc']['std']:.3f} "
            f"{s['balanced_acc_tuned']['mean']:.3f}+/-{s['balanced_acc_tuned']['std']:.3f} "
            f"{s['f1_tuned']['mean']:.3f}+/-{s['f1_tuned']['std']:.3f}"
        )


if __name__ == "__main__":
    main()
