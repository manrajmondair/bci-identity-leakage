"""Bootstrap confidence intervals.

Cheap percentile bootstrap on a per-window (or per-trial) outcome array.
Used everywhere we report a metric — task accuracy, attack top-1, EER,
membership-inference advantage, etc.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapCI:
    point: float
    low: float
    high: float
    alpha: float
    n_resamples: int

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.low:.3f}, {self.high:.3f}]"


def bootstrap_ci(
    values: np.ndarray,
    *,
    statistic=np.mean,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapCI:
    """Percentile bootstrap CI of a statistic over per-sample outcomes.

    For top-1 accuracy, pass `values = (preds == truth).astype(float)`.
    For EER, pass per-pair scores+labels and use a custom statistic.
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    n = len(values)
    samples = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        samples[i] = float(statistic(values[idx]))
    point = float(statistic(values))
    low, high = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapCI(point=point, low=float(low), high=float(high),
                       alpha=alpha, n_resamples=n_resamples)


def grouped_bootstrap_ci(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    statistic=np.mean,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapCI:
    """Bootstrap that resamples whole *groups* (e.g., trials) instead of
    individual samples. Required when within-group correlations would
    otherwise produce artificially tight CIs — e.g., 3 windows per trial
    are not independent observations of trial-level identity.
    """
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    samples = np.empty(n_resamples, dtype=np.float64)
    # Pre-bucket per group for fast resampling
    buckets = {g: np.where(groups == g)[0] for g in unique_groups}
    for i in range(n_resamples):
        chosen_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        idx = np.concatenate([buckets[g] for g in chosen_groups])
        samples[i] = float(statistic(values[idx]))
    point = float(statistic(values))
    low, high = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapCI(point=point, low=float(low), high=float(high),
                       alpha=alpha, n_resamples=n_resamples)
