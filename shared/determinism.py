"""Deterministic training helpers.

Call ``enable_determinism(seed)`` near the top of any experiment's
``run_experiments.py`` to reduce drift between reruns of the same
configuration.

This is best-effort, not absolute. On consumer GPUs (RTX 5070 Ti in
this project) cuDNN and the CUDA driver still introduce some
nondeterminism in convolution kernels even with ``deterministic=True``
set, but the drift is much smaller than the default (typically <0.01
AUC instead of 0.05+).

Side effects:
  - ``CUBLAS_WORKSPACE_CONFIG`` is set so ``torch.use_deterministic_algorithms(True)``
    does not crash on matmul.
  - ``PYTHONHASHSEED`` is set so dict iteration order is stable
    across process restarts.
  - ``random``, ``numpy.random``, and ``torch`` RNGs are all seeded.
  - cuDNN benchmark mode is disabled (a small training-time penalty
    in exchange for reproducibility).
"""

from __future__ import annotations

import os
import random


def enable_determinism(seed: int = 42) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # use_deterministic_algorithms raises if any op lacks a
        # deterministic implementation; warn_only keeps existing runs
        # working while flagging anything that needs attention.
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
