"""Helpers for cross-dataset experiments — channel subsetting + resampling.

The cross-dataset A4 experiment trains a contrastive embedding on PhysioNet
imagery and tests verification on BCI IV-2a session 1. PhysioNet has 64
channels at 160 Hz; IV-2a has 22 channels at 250 Hz. Empirically the IV-2a
22-channel set is a strict subset of PhysioNet's 64 (verified at runtime).
We resample IV-2a from 250 Hz to 160 Hz so the network input shape matches
between train and test.

Functions here are pure NumPy where possible so callers can pass in
arbitrary (n_windows, n_channels, n_times) arrays from either loader.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import scipy.signal

from preprocess.windows import WindowedDataset


def normalize_ch_names(names: Iterable[str]) -> list[str]:
    """Uppercase + strip trailing dots (PhysioNet uses 'Fcz.' for some channels)."""
    return [c.upper().rstrip(".") for c in names]


def channel_intersection(a: Iterable[str], b: Iterable[str]) -> list[str]:
    """Return channels present in both, in the order they appear in `a`."""
    a_norm = normalize_ch_names(a)
    b_norm = set(normalize_ch_names(b))
    return [ch for ch in a_norm if ch in b_norm]


def subset_channels(ds: WindowedDataset, target_channels: list[str]) -> WindowedDataset:
    """Return a WindowedDataset restricted to `target_channels` (preserving
    target_channels order). target_channels are matched case-insensitively
    after stripping trailing dots."""
    src_norm = normalize_ch_names(ds.channel_names)
    tgt_norm = normalize_ch_names(target_channels)
    src_idx_by_name = {ch: i for i, ch in enumerate(src_norm)}
    pick = [src_idx_by_name[ch] for ch in tgt_norm if ch in src_idx_by_name]
    if len(pick) != len(tgt_norm):
        missing = [ch for ch in tgt_norm if ch not in src_idx_by_name]
        raise KeyError(f"missing channels in source: {missing}")
    X_sub = ds.X[:, pick, :]
    new_names = tuple(target_channels)
    return WindowedDataset(
        X=X_sub.astype(np.float32, copy=False),
        y=ds.y, subject_ids=ds.subject_ids,
        trial_ids=ds.trial_ids, run_ids=ds.run_ids,
        sfreq=ds.sfreq, channel_names=new_names,
    )


def resample_windows(ds: WindowedDataset, target_sfreq: float) -> WindowedDataset:
    """Resample windows along the time axis from ds.sfreq to target_sfreq."""
    if abs(ds.sfreq - target_sfreq) < 1e-3:
        return ds
    n, c, t = ds.X.shape
    new_t = int(round(t * target_sfreq / ds.sfreq))
    # scipy resample uses Fourier method; OK for our short windows
    X_new = scipy.signal.resample(ds.X.astype(np.float64), num=new_t, axis=2)
    return WindowedDataset(
        X=X_new.astype(np.float32, copy=False),
        y=ds.y, subject_ids=ds.subject_ids,
        trial_ids=ds.trial_ids, run_ids=ds.run_ids,
        sfreq=float(target_sfreq), channel_names=ds.channel_names,
    )

