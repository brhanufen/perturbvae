"""Evaluation metrics for perturbation-response prediction.

Predictions and ground truth are per-gene expression-change vectors (the
"delta": mean expression of a perturbation minus the mean of non-targeting
controls). All metrics operate on these delta vectors, which is the standard
pseudobulk evaluation in the perturbation-prediction literature.
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np


def delta_pearson(pred_delta: np.ndarray, true_delta: np.ndarray) -> float:
    """Pearson correlation between predicted and observed expression change."""
    pred_delta = np.asarray(pred_delta, dtype=np.float64).ravel()
    true_delta = np.asarray(true_delta, dtype=np.float64).ravel()
    if pred_delta.std() == 0 or true_delta.std() == 0:
        return float("nan")
    return float(np.corrcoef(pred_delta, true_delta)[0, 1])


def delta_mse(pred_delta: np.ndarray, true_delta: np.ndarray) -> float:
    pred_delta = np.asarray(pred_delta, dtype=np.float64).ravel()
    true_delta = np.asarray(true_delta, dtype=np.float64).ravel()
    return float(np.mean((pred_delta - true_delta) ** 2))


def _top_deg(delta: np.ndarray, k: int) -> set:
    """Indices of the top-k differentially expressed genes (by |delta|)."""
    delta = np.abs(np.asarray(delta, dtype=np.float64).ravel())
    k = min(k, delta.size)
    return set(np.argsort(delta)[-k:].tolist())


def deg_jaccard(pred_delta: np.ndarray, true_delta: np.ndarray, k: int = 20) -> float:
    """Overlap (Jaccard) of the top-k predicted vs observed DE genes."""
    p, t = _top_deg(pred_delta, k), _top_deg(true_delta, k)
    union = p | t
    return float(len(p & t) / len(union)) if union else float("nan")


def score_condition(pred_delta: np.ndarray, true_delta: np.ndarray, k: int = 20) -> Dict[str, float]:
    return {
        "pearson": delta_pearson(pred_delta, true_delta),
        "mse": delta_mse(pred_delta, true_delta),
        "jaccard": deg_jaccard(pred_delta, true_delta, k=k),
    }


def summarize(per_condition: Mapping[str, Mapping[str, float]]) -> Dict[str, float]:
    """Mean of each metric across conditions (nan-safe)."""
    keys = ("pearson", "mse", "jaccard")
    out: Dict[str, float] = {"n_conditions": float(len(per_condition))}
    for k in keys:
        vals = [v[k] for v in per_condition.values() if k in v]
        out[f"mean_{k}"] = float(np.nanmean(vals)) if vals else float("nan")
    return out


def leakage_gap(summary_random: Mapping[str, float], summary_function: Mapping[str, float]) -> Dict[str, float]:
    """The headline number: how much performance drops from the random split (A)
    to the leakage-free function-grouped split (B)."""
    return {
        "pearson_random": summary_random.get("mean_pearson", float("nan")),
        "pearson_function": summary_function.get("mean_pearson", float("nan")),
        "pearson_gap": summary_random.get("mean_pearson", float("nan"))
        - summary_function.get("mean_pearson", float("nan")),
    }
