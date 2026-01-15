#!/usr/bin/env python
"""
Run all Experiment 1 fusion experiments.

Usage:
    python -m exp1_fusion.run_experiments           # Run all experiments
    python -m exp1_fusion.run_experiments --dry-run # Test data loading only
    python -m exp1_fusion.run_experiments --exp1a   # Run only Exp 1a (MLP)
    python -m exp1_fusion.run_experiments --exp1b   # Run only Exp 1b (FuseMoE)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import List, Dict

import numpy as np

from .config import EXPERIMENTS, RESULTS_DIR
from .data_pipeline import get_full_dataset, load_csv_data
from .training import run_experiment, save_results


def dry_run():
    """Test data loading and model creation without training."""
    print("="*60)
    print("DRY RUN: Testing data loading and model creation")
    print("="*60)

    # Test CSV loading
    print("\n[1] Loading CSV data...")
    df = load_csv_data()
    print(f"  Loaded {len(df)} samples")
    print(f"  Outcome distribution: {df['outcome'].value_counts().to_dict()}")
    print(f"  ASM distribution: {df['ASM'].value_counts().to_dict()}")

    # Test dataset creation for each combination
    print("\n[2] Testing dataset creation...")
    for text_model in ['clinicalbert', 'pubmedbert']:
        for smiles_model in ['chemberta', 'smilestrf']:
            print(f"\n  Testing {text_model} + {smiles_model}...")
            try:
                dataset, outcomes = get_full_dataset(text_model, smiles_model)
                sample = dataset[0]
                print(f"    Dataset size: {len(dataset)}")
                print(f"    Text shape: {sample['text_emb'].shape}")
                print(f"    SMILES shape: {sample['smiles_emb'].shape}")
                print(f"    Outcome distribution: {np.bincount(outcomes)}")
            except Exception as e:
                print(f"    ERROR: {e}")
                return False

    # Test model creation
    print("\n[3] Testing model creation...")
    import torch
    from .models import ConcatMLPClassifier, SimplifiedFuseMoE
    from .config import CONFIG_1A, CONFIG_1B, TEXT_DIM, SMILES_DIMS

    for smiles_model, smiles_dim in SMILES_DIMS.items():
        print(f"\n  Testing with smiles_dim={smiles_dim} ({smiles_model})...")

        # Test MLP
        mlp = ConcatMLPClassifier(
            text_dim=TEXT_DIM,
            smiles_dim=smiles_dim,
            hidden_dims=CONFIG_1A['hidden_dims'],
            dropout_rates=CONFIG_1A['dropout_rates'],
        )
        n_params_mlp = sum(p.numel() for p in mlp.parameters())
        print(f"    MLP parameters: {n_params_mlp:,}")

        # Test FuseMoE
        fusemoe = SimplifiedFuseMoE(
            text_dim=TEXT_DIM,
            smiles_dim=smiles_dim,
            hidden_dim=CONFIG_1B['hidden_dim'],
            num_experts=CONFIG_1B['num_experts'],
            top_k=CONFIG_1B['top_k'],
        )
        n_params_moe = sum(p.numel() for p in fusemoe.parameters())
        print(f"    FuseMoE parameters: {n_params_moe:,}")

        # Test forward pass
        text_emb = torch.randn(4, TEXT_DIM)
        smiles_emb = torch.randn(4, smiles_dim)

        mlp.eval()
        fusemoe.eval()
        with torch.no_grad():
            mlp_out = mlp(text_emb, smiles_emb)
            moe_out, aux = fusemoe(text_emb, smiles_emb)
            print(f"    MLP output shape: {mlp_out.shape}")
            print(f"    FuseMoE output shape: {moe_out.shape}")

    print("\n" + "="*60)
    print("DRY RUN COMPLETE: All checks passed!")
    print("="*60)
    return True


def run_all_experiments(
    experiments: List[Dict],
    verbose: bool = True,
) -> List[Dict]:
    """Run all specified experiments."""
    all_results = []

    print("="*60)
    print(f"Running {len(experiments)} experiments")
    print("="*60)

    for i, exp in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] {exp['name']}")

        results = run_experiment(
            experiment_name=exp['name'],
            text_model=exp['text'],
            smiles_model=exp['smiles'],
            fusion_type=exp['fusion'],
            verbose=verbose,
        )

        save_results(results)
        all_results.append(results)

    return all_results


def print_summary(results: List[Dict]):
    """Print a summary table of all results."""
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"{'Experiment':<40} {'Accuracy':<15} {'AUC':<15} {'F1':<15}")
    print("-"*80)

    for r in results:
        name = r['experiment']
        acc = f"{r['accuracy']['mean']:.4f}±{r['accuracy']['std']:.4f}"
        auc = f"{r['auc']['mean']:.4f}±{r['auc']['std']:.4f}"
        f1 = f"{r['f1']['mean']:.4f}±{r['f1']['std']:.4f}"
        print(f"{name:<40} {acc:<15} {auc:<15} {f1:<15}")

    print("="*80)

    # Find best model
    best_auc_idx = np.argmax([r['auc']['mean'] for r in results])
    best = results[best_auc_idx]
    print(f"\nBest model (by AUC): {best['experiment']}")
    print(f"  AUC: {best['auc']['mean']:.4f} ± {best['auc']['std']:.4f}")


def save_summary(results: List[Dict], filename: str = 'summary.json'):
    """Save summary of all results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_experiments': len(results),
        'experiments': [
            {
                'name': r['experiment'],
                'accuracy': r['accuracy'],
                'auc': r['auc'],
                'f1': r['f1'],
            }
            for r in results
        ],
    }

    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Run Experiment 1 fusion experiments')
    parser.add_argument('--dry-run', action='store_true',
                        help='Test data loading without training')
    parser.add_argument('--exp1a', action='store_true',
                        help='Run only Experiment 1a (MLP)')
    parser.add_argument('--exp1b', action='store_true',
                        help='Run only Experiment 1b (FuseMoE)')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')

    args = parser.parse_args()

    if args.dry_run:
        success = dry_run()
        sys.exit(0 if success else 1)

    # Filter experiments
    experiments = EXPERIMENTS
    if args.exp1a and not args.exp1b:
        experiments = [e for e in EXPERIMENTS if e['fusion'] == 'mlp']
    elif args.exp1b and not args.exp1a:
        experiments = [e for e in EXPERIMENTS if e['fusion'] == 'fusemoe']

    # Run experiments
    results = run_all_experiments(experiments, verbose=not args.quiet)

    # Print and save summary
    print_summary(results)
    save_summary(results)


if __name__ == '__main__':
    main()
