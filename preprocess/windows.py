"""Epoching and sliding-window extraction.

PhysioNet motor-imagery runs annotate three event codes:
  T0 = rest (1 s)
  T1 = first task class for the run
  T2 = second task class for the run

Each task trial is 4 s long. Standard motor-imagery preprocessing:
  1. Find T1/T2 events across all task runs.
  2. Epoch [0, 4 s] after event onset.
  3. (Optional) cut each 4-s epoch into 2-s sliding windows w/ 1-s stride
     so EEGNet sees more examples per trial.

This module returns numpy arrays (X, y, subject_ids, trial_ids) instead of
mne.Epochs so the rest of the pipeline can be framework-agnostic.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np

from config import (
    BANDPASS_HIGH,
    BANDPASS_LOW,
    CACHE_DIR,
    EPOCH_TMAX,
    EPOCH_TMIN,
    SAMPLING_RATE,
    WINDOW_SECONDS,
    WINDOW_STRIDE_SECONDS,
)
from data.physionet_loader import SubjectRecording, load_subject, run_label_pair
from preprocess.filtering import bandpass


@dataclass(frozen=True)
class WindowedDataset:
    """A flat collection of fixed-length EEG windows.

    Attributes
    ----------
    X : np.ndarray, shape (n_windows, n_channels, n_times)
        Float32. n_times = WINDOW_SECONDS * SAMPLING_RATE.
    y : np.ndarray, shape (n_windows,)
        Int64 class labels in {0, 1, 2, 3} for motor imagery.
    subject_ids : np.ndarray, shape (n_windows,)
        Int subject identifier (PhysioNet 1-based).
    trial_ids : np.ndarray, shape (n_windows,)
        Int unique trial identifier within (subject, dataset). Used for
        leave-one-trial-out splits and for grouping windows so we never
        leak windows from the same trial across train/test.
    run_ids : np.ndarray, shape (n_windows,)
        Int run identifier (PhysioNet 1-based). Used for run-level splits
        and for the cross-task (A2) attack: train on execution runs,
        test on imagery runs.
    sfreq : float
        Sampling rate (Hz). Same for every window.
    channel_names : tuple[str, ...]
        Channel labels in the order matching X's channel axis.
    """
    X: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray
    trial_ids: np.ndarray
    run_ids: np.ndarray
    sfreq: float
    channel_names: tuple[str, ...]

    @property
    def n_windows(self) -> int:
        return len(self.X)

    @property
    def n_channels(self) -> int:
        return self.X.shape[1]

    @property
    def n_times(self) -> int:
        return self.X.shape[2]

    def filter_subjects(self, subject_ids: list[int]) -> WindowedDataset:
        mask = np.isin(self.subject_ids, np.asarray(subject_ids))
        return self._mask(mask)

    def filter_runs(self, run_ids: list[int]) -> WindowedDataset:
        mask = np.isin(self.run_ids, np.asarray(run_ids))
        return self._mask(mask)

    def filter_classes(self, classes: list[int]) -> WindowedDataset:
        mask = np.isin(self.y, np.asarray(classes))
        return self._mask(mask)

    def _mask(self, mask: np.ndarray) -> WindowedDataset:
        return WindowedDataset(
            X=self.X[mask],
            y=self.y[mask],
            subject_ids=self.subject_ids[mask],
            trial_ids=self.trial_ids[mask],
            run_ids=self.run_ids[mask],
            sfreq=self.sfreq,
            channel_names=self.channel_names,
        )


def _events_with_run_id(raw: mne.io.BaseRaw) -> tuple[np.ndarray, dict]:
    """mne.events_from_annotations on a concatenated recording uses string
    codes 'T0','T1','T2'. Return events plus the dict mapping."""
    return mne.events_from_annotations(raw, event_id=dict(T0=0, T1=1, T2=2),
                                       verbose=False)


def epoch_subject(
    rec: SubjectRecording,
    *,
    bandpass_low: float | None = 4.0,
    bandpass_high: float | None = 40.0,
    drop_rest: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Epoch one subject's recording into per-trial 4-s segments.

    Returns
    -------
    X_trials : (n_trials, n_channels, n_times_per_trial) float32
    y_trials : (n_trials,) int64 class labels in {0, 1, 2, 3}
    trial_ids : (n_trials,) int64 unique within subject
    run_ids : (n_trials,) int (which PhysioNet run produced the trial)
    channel_names : channel order matching X
    """
    raw = rec.raw
    if bandpass_low is not None and bandpass_high is not None:
        raw = bandpass(raw, bandpass_low, bandpass_high, copy=True)

    # PhysioNet annotations use 'T0','T1','T2' on the *concatenated* raw, so we
    # only know which RUN each event came from by checking the sample index
    # against rec.run_boundaries.
    events, _ = _events_with_run_id(raw)
    if drop_rest:
        events = events[events[:, 2] != 0]  # drop T0 (rest)

    sfreq = raw.info["sfreq"]
    n_samples_epoch = int(round((EPOCH_TMAX - EPOCH_TMIN) * sfreq))

    X_list, y_list, trial_list, run_list = [], [], [], []
    data = raw.get_data()  # (n_channels, n_samples)
    next_trial_id = 0

    for ev_sample, _, ev_code in events:
        # Find which run this event came from
        run_idx_in_rec = None
        for i, (lo, hi) in enumerate(rec.run_boundaries):
            if lo <= ev_sample < hi:
                run_idx_in_rec = i
                break
        if run_idx_in_rec is None:
            continue
        run_number = rec.runs[run_idx_in_rec]

        # Map T1/T2 → 4-class label using per-run convention
        try:
            t1_class, t2_class = run_label_pair(run_number)
        except KeyError:
            continue  # baseline run with no task labels

        if ev_code == 1:
            y = t1_class
        elif ev_code == 2:
            y = t2_class
        else:
            continue

        # Slice the trial; skip if it would run past the recording
        start = ev_sample + int(round(EPOCH_TMIN * sfreq))
        stop = start + n_samples_epoch
        if stop > data.shape[1]:
            continue

        X_list.append(data[:, start:stop].astype(np.float32, copy=False))
        y_list.append(y)
        trial_list.append(next_trial_id)
        run_list.append(run_number)
        next_trial_id += 1

    if not X_list:
        raise RuntimeError(f"No trials extracted for subject {rec.subject_id}")

    X = np.stack(X_list, axis=0)  # (n_trials, n_channels, n_times)
    y = np.asarray(y_list, dtype=np.int64)
    trial_ids = np.asarray(trial_list, dtype=np.int64)
    run_ids = np.asarray(run_list, dtype=np.int64)
    channel_names = tuple(raw.ch_names)
    return X, y, trial_ids, run_ids, channel_names


def slide_windows(
    X_trials: np.ndarray,
    y_trials: np.ndarray,
    trial_ids: np.ndarray,
    run_ids: np.ndarray,
    *,
    window_seconds: float = WINDOW_SECONDS,
    stride_seconds: float = WINDOW_STRIDE_SECONDS,
    sfreq: float = SAMPLING_RATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cut each per-trial epoch into overlapping fixed-length windows."""
    n_trials, n_channels, n_times = X_trials.shape
    win = int(round(window_seconds * sfreq))
    stride = int(round(stride_seconds * sfreq))
    if win > n_times:
        raise ValueError(f"Window ({win}) longer than trial ({n_times} samples)")

    starts = np.arange(0, n_times - win + 1, stride)
    n_per_trial = len(starts)

    # Vectorized slice: result shape (n_trials, n_per_trial, n_channels, win)
    windows = np.stack([X_trials[:, :, s:s + win] for s in starts], axis=1)
    X = windows.reshape(n_trials * n_per_trial, n_channels, win).astype(np.float32, copy=False)

    y = np.repeat(y_trials, n_per_trial)
    t = np.repeat(trial_ids, n_per_trial)
    r = np.repeat(run_ids, n_per_trial)
    return X, y, t, r


def windowed_subject(
    subject_id: int,
    *,
    runs: str | tuple[int, ...] = "imagery",
    bandpass_low: float | None = 4.0,
    bandpass_high: float | None = 40.0,
    window_seconds: float = WINDOW_SECONDS,
    stride_seconds: float = WINDOW_STRIDE_SECONDS,
) -> WindowedDataset:
    """Convenience: load → bandpass → epoch → slide windows for one subject."""
    rec = load_subject(subject_id, runs=runs)
    X_trials, y_trials, trial_ids, run_ids, ch_names = epoch_subject(
        rec, bandpass_low=bandpass_low, bandpass_high=bandpass_high,
    )
    X, y, t, r = slide_windows(
        X_trials, y_trials, trial_ids, run_ids,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        sfreq=rec.raw.info["sfreq"],
    )
    s = np.full_like(t, subject_id)
    return WindowedDataset(
        X=X, y=y, subject_ids=s, trial_ids=t, run_ids=r,
        sfreq=rec.raw.info["sfreq"], channel_names=ch_names,
    )


def _cache_key(subject_ids: list[int], runs: str | tuple[int, ...],
               bandpass_low: float | None, bandpass_high: float | None,
               window_seconds: float, stride_seconds: float) -> str:
    """Stable hash of the configuration that produced a windowed dataset.

    Lets us reuse cached arrays across attacks as long as preprocessing is
    identical, and forces a recompute when any preprocessing knob changes.
    """
    h = hashlib.sha256()
    h.update(repr(sorted(subject_ids)).encode())
    h.update(repr(runs if isinstance(runs, str) else tuple(sorted(runs))).encode())
    h.update(repr((bandpass_low, bandpass_high, window_seconds, stride_seconds)).encode())
    return h.hexdigest()[:16]


def windowed_subjects(
    subject_ids: list[int],
    *,
    runs: str | tuple[int, ...] = "imagery",
    bandpass_low: float | None = BANDPASS_LOW,
    bandpass_high: float | None = BANDPASS_HIGH,
    window_seconds: float = WINDOW_SECONDS,
    stride_seconds: float = WINDOW_STRIDE_SECONDS,
    cache: bool = True,
) -> WindowedDataset:
    """Same as windowed_subject but pooled across many subjects.

    Trial IDs are made unique across subjects by offsetting per-subject IDs
    into disjoint integer ranges (subject_id * 100_000 + trial_id_within_subj).

    If `cache=True` (default) the result is persisted under
    cache/windows/<hash>.npz and reused on subsequent calls with the same
    config. The cache is a few hundred MB for the full 104-subject set.
    """
    if cache:
        key = _cache_key(subject_ids, runs, bandpass_low, bandpass_high,
                         window_seconds, stride_seconds)
        cache_path = Path(CACHE_DIR) / "windows" / f"{key}.npz"
        if cache_path.exists():
            data = np.load(cache_path, allow_pickle=False)
            return WindowedDataset(
                X=data["X"], y=data["y"],
                subject_ids=data["subject_ids"], trial_ids=data["trial_ids"],
                run_ids=data["run_ids"],
                sfreq=float(data["sfreq"].item()),
                channel_names=tuple(data["channel_names"].tolist()),
            )

    parts = []
    for s in subject_ids:
        ds = windowed_subject(
            s, runs=runs,
            bandpass_low=bandpass_low, bandpass_high=bandpass_high,
            window_seconds=window_seconds, stride_seconds=stride_seconds,
        )
        offset = s * 100_000
        parts.append((ds.X, ds.y, ds.subject_ids, ds.trial_ids + offset, ds.run_ids,
                      ds.sfreq, ds.channel_names))

    X = np.concatenate([p[0] for p in parts], axis=0)
    y = np.concatenate([p[1] for p in parts], axis=0)
    s = np.concatenate([p[2] for p in parts], axis=0)
    t = np.concatenate([p[3] for p in parts], axis=0)
    r = np.concatenate([p[4] for p in parts], axis=0)
    sfreq = parts[0][5]
    ch_names = parts[0][6]
    for p in parts[1:]:
        assert p[6] == ch_names, "Channel mismatch between subjects"

    result = WindowedDataset(
        X=X, y=y, subject_ids=s, trial_ids=t, run_ids=r,
        sfreq=sfreq, channel_names=ch_names,
    )

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path,
                 X=X, y=y, subject_ids=s, trial_ids=t, run_ids=r,
                 sfreq=np.asarray(sfreq),
                 channel_names=np.asarray(ch_names))
    return result

