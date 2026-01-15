"""Run Experiment 2: EEG + SMILES fusion experiments."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from .config import EXPERIMENTS, OUTPUTS_DIR, SMILES_EMBED_DIMS
from .data_pipeline import prepare_data
from .training import run_cross_validation


def run_all_experiments(
    eeg_encoder: str = "labram",
    smiles_model: str = None,
    fusion_type: str = None,
    dry_run: bool = False,
    output_dir: Path = OUTPUTS_DIR / "exp2_results",
):
    """Run all configured experiments.

    Args:
        eeg_encoder: EEG encoder type to use ('labram', 'simplecnn').
        smiles_model: SMILES model filter (None = all).
        fusion_type: Fusion type filter (None = all).
        dry_run: If True, just print what would run.
        output_dir: Directory for results.
    """
    # Filter experiments
    experiments = []
    for exp in EXPERIMENTS:
        if smiles_model and exp["smiles_model"] != smiles_model:
            continue
        if fusion_type and exp["fusion"] != fusion_type:
            continue
        experiments.append({**exp, "eeg_model": eeg_encoder})

    if dry_run:
        print("Would run the following experiments:")
        for exp in experiments:
            print(f"  - {exp['eeg_model']} + {exp['smiles_model']} ({exp['fusion']})")
        return

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Running {len(experiments)} experiments\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    # Run experiments
    for i, exp in enumerate(experiments):
        print(f"[{i+1}/{len(experiments)}] {exp['eeg_model']} + {exp['smiles_model']} ({exp['fusion']})")

        # Load data for this SMILES model
        smiles_model_name = exp["smiles_model"]
        eeg_data, smiles_embeddings, smiles_indices, df = prepare_data(
            smiles_model=smiles_model_name,
            cache_eeg=True,
        )

        # Run cross-validation
        results = run_cross_validation(
            eeg_data=eeg_data,
            smiles_embeddings=smiles_embeddings,
            smiles_indices=smiles_indices,
            df=df,
            fusion_type=exp["fusion"],
            eeg_encoder_type=exp["eeg_model"],
            smiles_model=smiles_model_name,
            device=device,
            verbose=True,
        )

        results["timestamp"] = datetime.now().isoformat()
        all_results.append(results)

        # Save individual result
        result_file = output_dir / f"{results['experiment']}.json"
        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"  -> Acc: {results['accuracy']['mean']:.3f}, AUC: {results['auc']['mean']:.3f}, F1: {results['f1']['mean']:.3f}\n")

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_experiments": len(all_results),
        "experiments": all_results,
    }
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to {output_dir}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Experiment':<40} {'Acc':<10} {'AUC':<10} {'F1':<10}")
    print("-" * 70)
    for r in sorted(all_results, key=lambda x: -x["auc"]["mean"]):
        name = r["experiment"]
        acc = f"{r['accuracy']['mean']:.3f}"
        auc = f"{r['auc']['mean']:.3f}"
        f1 = f"{r['f1']['mean']:.3f}"
        print(f"{name:<40} {acc:<10} {auc:<10} {f1:<10}")


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 2: EEG + SMILES fusion")
    parser.add_argument("--eeg-encoder", type=str, default="simplecnn",
                        choices=["labram", "simplecnn"],
                        help="EEG encoder type (default: simplecnn)")
    parser.add_argument("--smiles-model", type=str, default=None,
                        choices=["chemberta", "smilestrf"],
                        help="SMILES model to use (default: all)")
    parser.add_argument("--fusion", type=str, default=None,
                        choices=["mlp", "fusemoe"],
                        help="Fusion type (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print experiments without running")

    args = parser.parse_args()

    run_all_experiments(
        eeg_encoder=args.eeg_encoder,
        smiles_model=args.smiles_model,
        fusion_type=args.fusion,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
