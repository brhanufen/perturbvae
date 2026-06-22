"""Non-learned baselines for perturbation-response prediction.

These are the sanity checks the literature shows are surprisingly hard to beat.
Any learned model must clear at least ``mean_of_training`` to be worth anything.
All baselines predict a per-gene delta (expression change vs control).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def control_mean_delta(n_genes: int) -> np.ndarray:
    """Predict no change at all (the perturbation does nothing)."""
    return np.zeros(int(n_genes), dtype=np.float64)


def mean_of_training_delta(train_deltas: Sequence[np.ndarray]) -> np.ndarray:
    """Predict the average perturbation delta observed in training. A single
    fixed vector applied to every test perturbation; strong because most
    perturbations share a common stress-response component."""
    if len(train_deltas) == 0:
        raise ValueError("train_deltas is empty")
    return np.mean(np.stack([np.asarray(d, dtype=np.float64).ravel() for d in train_deltas], 0), axis=0)
