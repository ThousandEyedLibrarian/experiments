"""ASM-class-balancing utilities for Stage B.

Addresses the per-Duong-email follow-up to the #26 best-ASM
counterfactual finding (Phase 3b): the LEV-recommendation skew is
attributable to the LEV-dominated training distribution.

Two strategies are exposed, both directly per Duong's spec:

1. ``compute_asm_sample_weights`` - inverse-square-root sample weighting.
   Each sample s gets w_s = 1 / sqrt(n_s), where n_s is the number of
   training samples sharing s's ASM. Weights are then normalised so the
   training-set mean is 1. Apply in the loss via
   ``loss_per_sample = CrossEntropyLoss(reduction='none')(...)``
   and ``loss = (loss_per_sample * weights).mean()``.

2. ``StratifiedASMBatchSampler`` - PyTorch BatchSampler that constructs
   every mini-batch to contain at least one sample from every ASM
   present in the training set. Rare ASMs (e.g., TPM with n=1) are
   resampled with replacement when they would otherwise be exhausted
   within an epoch.

Per the planning decision, we keep PTN (n=2) and TPM (n=1) in training
and accept the resulting heavy upweighting for those samples. Set
``min_count_floor`` in ``compute_asm_sample_weights`` to a positive
integer to cap the upweighting if instability appears in practice.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler


def normalise_asm(value: object) -> str:
    """Collapse case typos (e.g. 'cBZ' -> 'CBZ') and trim whitespace."""
    if value is None:
        return ""
    return str(value).strip().upper()


def compute_asm_sample_weights(
    asm_labels: Sequence[object],
    min_count_floor: int = 0,
) -> np.ndarray:
    """Inverse-square-root sample weights per Duong's spec.

    Args:
        asm_labels: per-sample ASM identifier (any hashable, normalised
            internally via :func:`normalise_asm`).
        min_count_floor: if > 0, treat any per-ASM count below this as
            the floor when computing ``1/sqrt(n)``. Caps single-sample
            upweighting (TPM at n=1). Default 0 keeps Duong's literal
            spec; the planning answer chose this default.

    Returns:
        ``np.ndarray`` of float weights, one per input sample, with
        mean equal to 1.0 (within rounding).
    """
    normalised = [normalise_asm(a) for a in asm_labels]
    counts = Counter(normalised)
    effective_counts = {a: max(c, min_count_floor) for a, c in counts.items()}
    raw = np.asarray(
        [1.0 / math.sqrt(effective_counts[a]) for a in normalised],
        dtype=np.float64,
    )
    mean = float(raw.mean()) if raw.size else 1.0
    if mean == 0:
        return np.ones_like(raw)
    return (raw / mean).astype(np.float32)


class StratifiedASMBatchSampler(Sampler[list[int]]):
    """BatchSampler ensuring each mini-batch contains every ASM.

    Each batch is constructed by:
      1. Drawing one sample per ASM from the training set (random within
         each ASM, with replacement when an ASM has fewer remaining
         indices than required to complete the epoch).
      2. Filling the remainder of ``batch_size`` with random samples
         from the global training pool (no stratification, with
         replacement only at the per-ASM level above).

    Epoch length is set so each training sample is expected to be drawn
    approximately once (``ceil(n_samples / batch_size)`` batches). Rare
    ASMs are oversampled relative to their natural frequency by design.

    Args:
        asm_labels: per-sample ASM identifier aligned with the dataset
            indices passed via ``__iter__``.
        batch_size: mini-batch size. Must be >= number of distinct ASMs
            in ``asm_labels``.
        n_batches_per_epoch: explicit override; default is
            ``ceil(len(asm_labels) / batch_size)``.
        seed: random seed for reproducible shuffling.
        drop_last: ignored; kept for interface symmetry with the
            built-in BatchSampler.
    """

    def __init__(
        self,
        asm_labels: Sequence[object],
        batch_size: int,
        n_batches_per_epoch: int | None = None,
        seed: int = 0,
        drop_last: bool = False,
    ):
        del drop_last  # API symmetry only.
        self.asm_labels = [normalise_asm(a) for a in asm_labels]
        self.batch_size = int(batch_size)
        # Coerce to non-negative; negative seeds (e.g. fold=-1 for the
        # refit pass in exp7) crash numpy.random.default_rng.
        self.seed = abs(int(seed))
        # Index pool per ASM.
        self.indices_by_asm: dict[str, list[int]] = defaultdict(list)
        for idx, asm in enumerate(self.asm_labels):
            self.indices_by_asm[asm].append(idx)
        self.unique_asms = sorted(self.indices_by_asm.keys())
        if self.batch_size < len(self.unique_asms):
            raise ValueError(
                f"batch_size={self.batch_size} is smaller than the number "
                f"of distinct ASMs ({len(self.unique_asms)}); cannot satisfy "
                "the at-least-one-per-ASM constraint."
            )
        n_total = len(self.asm_labels)
        if n_batches_per_epoch is None:
            n_batches_per_epoch = max(1, math.ceil(n_total / self.batch_size))
        self.n_batches = int(n_batches_per_epoch)
        self._epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        # Maintain a shuffled queue per ASM that we refill on exhaustion.
        queues = {
            asm: list(rng.permutation(self.indices_by_asm[asm]))
            for asm in self.unique_asms
        }
        all_indices = np.arange(len(self.asm_labels))
        for _ in range(self.n_batches):
            batch: list[int] = []
            # Slot 1..k: one per ASM.
            for asm in self.unique_asms:
                if not queues[asm]:
                    queues[asm] = list(rng.permutation(self.indices_by_asm[asm]))
                batch.append(queues[asm].pop())
            # Fill remaining slots with uniform-random draws from the
            # global pool. Without replacement within the same batch to
            # avoid trivial duplicates.
            fill = self.batch_size - len(batch)
            if fill > 0:
                candidates = rng.permutation(all_indices)
                k = 0
                while fill > 0 and k < len(candidates):
                    cand = int(candidates[k])
                    k += 1
                    if cand in batch:
                        continue
                    batch.append(cand)
                    fill -= 1
            yield batch

    def __len__(self) -> int:
        return self.n_batches


class WeightedASMDataset(torch.utils.data.Dataset):
    """Wrapper that appends a per-sample weight tensor to each item.

    The base dataset's ``__getitem__`` is unchanged structurally; this
    wrapper simply concatenates a 0-dim tensor weight as the final
    element of the returned tuple. The default PyTorch ``collate_fn``
    stacks the weights into a ``(batch,)`` tensor automatically.

    In the training loop, unpack as::

        *features, labels, asm_weights = batch
    """

    def __init__(self, base_dataset: torch.utils.data.Dataset, weights: Sequence[float]):
        if len(weights) != len(base_dataset):
            raise ValueError(
                f"weights length {len(weights)} != dataset length {len(base_dataset)}"
            )
        self.base = base_dataset
        self.weights = torch.tensor(list(weights), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple:
        item = self.base[idx]
        if not isinstance(item, tuple):
            item = (item,)
        return (*item, self.weights[idx])


def weighted_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_weights: torch.Tensor,
    class_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-sample weighted cross-entropy loss.

    Args:
        logits: ``(batch, n_classes)`` model outputs.
        targets: ``(batch,)`` integer class labels.
        sample_weights: ``(batch,)`` per-sample weights (e.g. from
            :func:`compute_asm_sample_weights` indexed at the batch's
            global indices).
        class_weight: optional ``(n_classes,)`` class-imbalance weight
            passed through to ``F.cross_entropy``; preserves the
            existing outcome-class balancing used by every experiment.

    Returns:
        Scalar loss, equal to the weighted mean of per-sample CE.
    """
    per_sample = torch.nn.functional.cross_entropy(
        logits,
        targets,
        weight=class_weight,
        reduction="none",
    )
    weighted = per_sample * sample_weights
    return weighted.mean()
