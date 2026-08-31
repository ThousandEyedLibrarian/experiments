"""Run Experiment 16 (reduced-capacity quad-modal fusion) with per-patient
prediction logging, one OOF file per size variant.

Usage:
    python -m exp16_reduced_capacity.run_experiments \\
        --seed 42 --asm-balance none --mode predictions --deterministic \\
        --output-dir outputs/exp16_predictions
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from .config import ASM_NAME_MAPPING, CV_CONFIG, VARIANTS
from .data_pipeline import create_quad_modality_datasets, prepare_quad_modality_data
from .training import train_fold_with_predictions

logger = logging.getLogger("exp16")


def _save_predictions_json(payload: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    with open(path, "w") as f:
        json.dump(_convert(payload), f, indent=2)
    logger.info(f"Predictions saved to {path}")


def _top_n_asms_for_cohort(df, n: int = 5) -> List[str]:
    counts: Dict[str, int] = {}
    short_for_canon: Dict[str, str] = {}
    for raw in df["ASM"].astype(str):
        short = raw.strip()
        canon = ASM_NAME_MAPPING.get(short, short)
        preferred = short.upper() if short.upper() in ASM_NAME_MAPPING else short
        counts[canon] = counts.get(canon, 0) + 1
        short_for_canon.setdefault(canon, preferred)
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return [short_for_canon[canon] for canon, _ in ordered[:n]]


def _build_candidate_smiles(top_asms, smiles_embeddings, smiles_indices) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for short in top_asms:
        canon = ASM_NAME_MAPPING.get(short, short)
        if canon not in smiles_indices:
            logger.warning(f"  Skipping ASM '{short}' (canon '{canon}') - not in smiles_indices.")
            continue
        out[short] = np.asarray(smiles_embeddings[smiles_indices[canon]], dtype=np.float32)
    return out


def run_exp16_with_predictions(output_dir: Path, top_n_asms: int, device, asm_balance_mode: str,
                               seed: int, output_suffix: str = "") -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cohort + modalities are shared across variants -> load once.
    df, smiles_embeddings, smiles_indices, text_embeddings, eeg_data = prepare_quad_modality_data(
        text_model="clinicalbert", smiles_model="chemberta",
    )
    outcomes = df["outcome"].values
    logger.info(f"  Cohort size: {len(df)} patients")

    top_asms = _top_n_asms_for_cohort(df, n=top_n_asms)
    candidate_smiles = _build_candidate_smiles(top_asms, smiles_embeddings, smiles_indices)
    asms_used = list(candidate_smiles.keys())

    kfold = StratifiedKFold(
        n_splits=CV_CONFIG["n_splits"], shuffle=CV_CONFIG["shuffle"],
        random_state=CV_CONFIG["random_state"],
    )
    splits = list(kfold.split(np.zeros(len(outcomes)), outcomes))

    for variant in VARIANTS:
        logger.info(f"=== Variant {variant['name']} (hidden_dim={variant['hidden_dim']}, "
                    f"eeg_embed_dim={variant['eeg_embed_dim']}, agg={variant['aggregator_type']}) ===")
        folds_payload: List[Dict] = []
        for fold, (train_idx, val_idx) in enumerate(splits):
            from shared.determinism import enable_determinism
            enable_determinism(seed + fold)

            train_ds, val_ds, _ = create_quad_modality_datasets(
                df, smiles_embeddings, smiles_indices, text_embeddings, eeg_data,
                train_idx, val_idx, return_pid=True,
            )
            result = train_fold_with_predictions(
                train_ds, val_ds, variant=variant, device=device, fold=fold,
                candidate_smiles=candidate_smiles, asm_balance_mode=asm_balance_mode,
            )
            scalar_metrics = {
                k: float(v) for k, v in result["metrics"].items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            folds_payload.append({
                "fold": fold,
                "metrics": scalar_metrics,
                "pids": result["val_pids"],
                "y_true": result["val_y_true"],
                "y_prob": result["val_y_prob"],
                "y_prob_per_asm": {a: result["val_y_prob_per_asm"].get(a, []) for a in asms_used},
            })
            logger.info(f"  Fold {fold + 1}: AUC={result['metrics'].get('auc', float('nan')):.4f}")

        oof_payload = {
            "experiment": variant["name"],
            "asm_balance_mode": asm_balance_mode,
            "seed": seed,
            "hidden_dim": variant["hidden_dim"],
            "eeg_embed_dim": variant["eeg_embed_dim"],
            "aggregator_type": variant["aggregator_type"],
            "text_model": variant["text_model"],
            "smiles_model": variant["smiles_model"],
            "asms": asms_used,
            "cv_random_state": CV_CONFIG["random_state"],
            "n_splits": CV_CONFIG["n_splits"],
            "folds": folds_payload,
        }
        oof_path = output_dir / f"predictions_oof_{variant['name']}{output_suffix}.json"
        _save_predictions_json(oof_payload, oof_path)


def main():
    parser = argparse.ArgumentParser(description="Exp16: reduced-capacity quad-modal fusion")
    parser.add_argument("--mode", choices=["predictions"], default="predictions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", "--output_dir", type=str, default=None)
    parser.add_argument("--asm-balance", choices=["none", "weighted", "stratified_batch"], default="none")
    parser.add_argument("--top-n-asms", type=int, default=5)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if args.deterministic:
        from shared.determinism import enable_determinism
        enable_determinism(args.seed)

    device = torch.device(args.device) if args.device else (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Using device: {device}")

    out_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).parent.parent / "outputs" / "exp16_predictions"
    )
    suffix = {"weighted": "_asmweighted", "stratified_batch": "_asmstratbatch"}.get(args.asm_balance, "")
    run_exp16_with_predictions(
        output_dir=out_dir, top_n_asms=args.top_n_asms, device=device,
        asm_balance_mode=args.asm_balance, seed=args.seed, output_suffix=suffix,
    )


if __name__ == "__main__":
    main()
