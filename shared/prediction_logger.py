"""Per-fold prediction logging for experiments that need patient-level OOF outputs.

Use this helper to dump a ``predictions_oof.json`` file with the same schema
as ``outputs/exp7_predictions/predictions_oof.json`` (which is consumed by
``thesisStandalone/analysis/compute_bootstrap_cis.py`` and the planned
all-pairs DeLong statistical-comparison script).

Usage pattern inside an experiment's training script:

    from shared.prediction_logger import PredictionLogger

    logger = PredictionLogger(exp_id="exp4a_mlp", output_dir=OUTPUT_DIR)
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(...)):
        # ... train ...
        val_y_prob = model.predict_proba(...)[:, 1]
        val_pids = pid_array[val_idx]
        logger.log_fold(
            fold=fold_idx,
            pids=val_pids,
            y_true=val_y_true,
            y_prob=val_y_prob,
            threshold=fold_threshold,
        )
    logger.save()

The resulting JSON has the schema:

    {
        "exp_id": "exp4a_mlp",
        "n_folds": 5,
        "folds": [
            {"fold": 0, "pids": [...], "y_true": [...], "y_prob": [...], "threshold": 0.42},
            ...
        ],
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


class PredictionLogger:
    """Accumulate per-fold OOF predictions and dump as JSON.

    The logger is intentionally minimal: it stores native-Python lists
    (not numpy arrays) so the JSON dump is portable across environments.
    """

    def __init__(self, exp_id: str, output_dir: str | Path, filename: str = "predictions_oof.json"):
        self.exp_id = exp_id
        self.output_path = Path(output_dir) / filename
        self.folds: list[dict] = []

    def log_fold(
        self,
        fold: int,
        pids: Iterable,
        y_true: Iterable,
        y_prob: Iterable,
        threshold: float | None = None,
    ) -> None:
        """Append one fold's held-out predictions to the accumulator."""
        pids_list = [str(p) for p in pids]
        y_true_list = [int(v) for v in y_true]
        y_prob_list = [float(v) for v in y_prob]
        n = len(y_prob_list)
        if not (len(pids_list) == len(y_true_list) == n):
            raise ValueError(
                f"PredictionLogger.log_fold: length mismatch for fold {fold}: "
                f"pids={len(pids_list)}, y_true={len(y_true_list)}, y_prob={n}"
            )
        entry = {
            "fold": int(fold),
            "n": n,
            "pids": pids_list,
            "y_true": y_true_list,
            "y_prob": y_prob_list,
        }
        if threshold is not None:
            entry["threshold"] = float(threshold)
        self.folds.append(entry)

    def save(self) -> Path:
        """Write the accumulated payload to ``predictions_oof.json``."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "exp_id": self.exp_id,
            "n_folds": len(self.folds),
            "folds": self.folds,
        }
        with self.output_path.open("w") as handle:
            json.dump(payload, handle, indent=2)
        return self.output_path
