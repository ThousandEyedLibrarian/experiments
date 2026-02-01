#!/usr/bin/env python3
"""Run Experiment 8: Stratification Analysis.

Compares outcome-only stratification (baseline) with multi-label
stratification to assess impact on cross-validation stability.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .config import (
    CV_CONFIG,
    EXPERIMENTS,
    RESULTS_DIR,
    TRAINING_CONFIG,
)
from .data_pipeline import prepare_quad_modality_data_with_df
from .feature_analysis import (
    analyse_all_features,
    generate_analysis_report,
    recommend_stratification_strategy,
)
from .stratified_cv import (
    analyse_fold_balance,
    get_multilabel_splits,
    get_outcome_only_splits,
)
from .training import compute_summary, run_cv_experiment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("exp8")


def run_experiments(
    text_model: str = "clinicalbert",
    smiles_model: str = "chemberta",
):
    """Run stratification comparison experiments.

    Args:
        text_model: Text embedding model to use.
        smiles_model: SMILES embedding model to use.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading data...")
    df, smiles_emb, smiles_idx, text_emb, eeg_data = prepare_quad_modality_data_with_df(
        text_model=text_model,
        smiles_model=smiles_model,
    )

    # Feature analysis
    logger.info("\n" + "=" * 60)
    logger.info("FEATURE ANALYSIS")
    logger.info("=" * 60)
    feature_analysis = analyse_all_features(df)
    print(generate_analysis_report(feature_analysis))

    recommendations = recommend_stratification_strategy(feature_analysis)
    logger.info(f"\nRecommended stratification: {recommendations['recommended_features']}")
    logger.info(f"Severely imbalanced features: {recommendations['severely_imbalanced']}")

    # Results storage
    all_results = {}

    # Experiment 1: Baseline (outcome-only stratification)
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT: Baseline (outcome-only stratification)")
    logger.info("=" * 60)

    baseline_splits = list(get_outcome_only_splits(df))

    # Analyse baseline fold balance
    baseline_balance = analyse_fold_balance(df, baseline_splits)
    logger.info("\nBaseline fold balance:")
    print(baseline_balance.to_string(index=False))

    # Run baseline experiment
    baseline_metrics = run_cv_experiment(
        df=df,
        splits=baseline_splits,
        smiles_embeddings=smiles_emb,
        smiles_indices=smiles_idx,
        text_embeddings=text_emb,
        eeg_data=eeg_data,
        config=TRAINING_CONFIG,
        device=device,
    )
    baseline_summary = compute_summary(baseline_metrics)

    all_results["baseline_outcome_only"] = {
        "config": {
            "name": "baseline_outcome_only",
            "stratification": "outcome_only",
            "text_model": text_model,
            "smiles_model": smiles_model,
        },
        "fold_metrics": baseline_metrics,
        "summary": baseline_summary,
        "fold_balance": baseline_balance.to_dict(orient="records"),
    }

    logger.info("\nBaseline Results:")
    for metric, stats in baseline_summary.items():
        logger.info(f"  {metric}: {stats['mean']:.4f} +/- {stats['std']:.4f}")

    # Experiment 2: Multi-label stratification
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT: Multi-label stratification (outcome + focal + sex)")
    logger.info("=" * 60)

    try:
        multilabel_splits = list(get_multilabel_splits(df))

        # Analyse multi-label fold balance
        multilabel_balance = analyse_fold_balance(df, multilabel_splits)
        logger.info("\nMulti-label fold balance:")
        print(multilabel_balance.to_string(index=False))

        # Run multi-label experiment
        multilabel_metrics = run_cv_experiment(
            df=df,
            splits=multilabel_splits,
            smiles_embeddings=smiles_emb,
            smiles_indices=smiles_idx,
            text_embeddings=text_emb,
            eeg_data=eeg_data,
            config=TRAINING_CONFIG,
            device=device,
        )
        multilabel_summary = compute_summary(multilabel_metrics)

        all_results["multilabel_stratification"] = {
            "config": {
                "name": "multilabel_stratification",
                "stratification": "multilabel",
                "stratify_cols": ["outcome", "focal", "sex"],
                "text_model": text_model,
                "smiles_model": smiles_model,
            },
            "fold_metrics": multilabel_metrics,
            "summary": multilabel_summary,
            "fold_balance": multilabel_balance.to_dict(orient="records"),
        }

        logger.info("\nMulti-label Results:")
        for metric, stats in multilabel_summary.items():
            logger.info(f"  {metric}: {stats['mean']:.4f} +/- {stats['std']:.4f}")

    except ImportError as e:
        logger.error(f"Multi-label stratification failed: {e}")
        all_results["multilabel_stratification"] = {"error": str(e)}

    # Comparison
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON: Baseline vs Multi-label")
    logger.info("=" * 60)

    if "error" not in all_results.get("multilabel_stratification", {}):
        for metric in ["auc", "balanced_acc_tuned"]:
            base_mean = baseline_summary[metric]["mean"]
            base_std = baseline_summary[metric]["std"]
            multi_mean = multilabel_summary[metric]["mean"]
            multi_std = multilabel_summary[metric]["std"]

            diff = multi_mean - base_mean
            std_change = multi_std - base_std

            logger.info(f"\n{metric}:")
            logger.info(f"  Baseline:    {base_mean:.4f} +/- {base_std:.4f}")
            logger.info(f"  Multi-label: {multi_mean:.4f} +/- {multi_std:.4f}")
            logger.info(f"  Difference:  {diff:+.4f} (std change: {std_change:+.4f})")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"results_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nResults saved to: {results_file}")

    return all_results


def main():
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Exp8 stratification analysis")
    parser.add_argument(
        "--text-model",
        default="clinicalbert",
        choices=["clinicalbert", "pubmedbert"],
        help="Text embedding model",
    )
    parser.add_argument(
        "--smiles-model",
        default="chemberta",
        choices=["chemberta"],
        help="SMILES embedding model",
    )

    args = parser.parse_args()

    run_experiments(
        text_model=args.text_model,
        smiles_model=args.smiles_model,
    )


if __name__ == "__main__":
    main()
