"""BCI Competition IV dataset 2a loader.

This dataset has 9 subjects each recorded across **two sessions on different
days** (training session 'T', evaluation session 'E'), 22 EEG channels at
250 Hz. It is the only commonly-used motor-imagery dataset that supplies
a true cross-session split — required for the realistic
biometric-linkage attack (A3).

We expose it through moabb.datasets.BNCI2014_001, which wraps the official
BNCI Horizon 2020 distribution. moabb caches the raw files in
~/mne_data/MNE-bnci-data/ on first use.

Class mapping (matches the BCI competition's official labels):
    0 = left hand
    1 = right hand
    2 = both feet
    3 = tongue

These are different motor-imagery classes than PhysioNet's 4-class scheme,
so we never mix the two datasets in a single classifier — IV-2a is used
exclusively for cross-session re-ID.
"""
from __future__ import annotations

import time

import mne
import numpy as np

from preprocess.filtering import bandpass
from preprocess.windows import WindowedDataset, slide_windows

_BCI2014_001_CLASSES = {
    "left_hand": 0,
    "right_hand": 1,
    "feet": 2,
    "tongue": 3,
}


def _load_dataset():
    """Lazy import — moabb pulls a lot of code, so don't import at module top."""
    from moabb.datasets import BNCI2014_001
    return BNCI2014_001()


def _get_subject_data_with_retry(
    subject_id: int,
    *,
    max_attempts: int = 4,
    backoff_seconds: float = 4.0,
) -> dict:
    """Wrap moabb.get_data with retries.

    moabb's pooch backend uses requests.iter_content without auto-retry, so a
    mid-stream connection drop (`IncompleteRead`) on the IV-2a host kills
    the whole experiment. We've seen this happen on Colab. Retrying after a
    short backoff resumes from where pooch left off (the partial file is
    discarded; the next attempt re-downloads the failing .mat from scratch).
    """
    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            ds = _load_dataset()
            return ds.get_data(subjects=[subject_id])
        except Exception as exc:
            last_err = exc
            if attempt < max_attempts:
                wait = backoff_seconds * (2 ** (attempt - 1))
                print(f"    IV-2a get_data for subject {subject_id} "
                      f"failed (attempt {attempt}/{max_attempts}): "
                      f"{type(exc).__name__}: {exc}. "
                      f"Retrying in {wait:.0f}s ...", flush=True)
                time.sleep(wait)
    assert last_err is not None
    raise last_err


def load_subject_session(
    subject_id: int,
    session: str = "0train",
    *,
    bandpass_low: float | None = 4.0,
    bandpass_high: float | None = 40.0,
    window_seconds: float = 2.0,
    stride_seconds: float = 1.0,
) -> WindowedDataset:
    """Load one (subject, session) pair as windowed data.

    moabb's BNCI2014_001 returns a nested dict:
        data[subject_id][session_name][run_name] = mne.Raw
    where session_name is one of {'0train', '1test'} (training vs. evaluation
    session, recorded on different days) and run_name is run-by-run within
    that session. We concatenate all runs in the requested session.
    """
    if session not in ("0train", "1test"):
        raise ValueError(f"session must be '0train' or '1test', got {session!r}")

    data = _get_subject_data_with_retry(subject_id)  # downloads on first use
    session_runs = data[subject_id][session]  # dict[run_name -> Raw]
    raws = list(session_runs.values())
    raw = mne.concatenate_raws(raws, verbose=False)

    if bandpass_low is not None and bandpass_high is not None:
        raw = bandpass(raw, bandpass_low, bandpass_high, copy=True)

    # moabb annotations come pre-coded with class labels as strings.
    # We map them to our 0–3 integer scheme.
    ann_to_class = {
        "left_hand": 0, "right_hand": 1, "feet": 2, "tongue": 3,
        # Some moabb versions use these aliases:
        "769": 0, "770": 1, "771": 2, "772": 3,
    }
    event_id = {k: v for k, v in ann_to_class.items()
                if k in {a["description"] for a in raw.annotations}}
    if not event_id:
        # Fallback: try moabb's default mapping
        event_id = {k: v for k, v in ann_to_class.items() if k in ann_to_class}

    events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)
    if len(events) == 0:
        raise RuntimeError(
            f"No labeled trials found for IV-2a subject {subject_id} session {session}. "
            f"Available annotations: {set(a['description'] for a in raw.annotations)}"
        )

    sfreq = raw.info["sfreq"]
    # IV-2a trial structure: cue at t=0, motor imagery 0–4 s after cue.
    # Use the same 4-s post-cue epoch as PhysioNet so the EEG-window length
    # is comparable.
    EPOCH_TMIN, EPOCH_TMAX = 0.0, 4.0
    n_samples = int(round((EPOCH_TMAX - EPOCH_TMIN) * sfreq))

    eeg_data = raw.get_data(picks="eeg")
    X_trials, y_trials, trial_ids, run_ids = [], [], [], []
    for i, (sample, _, code) in enumerate(events):
        start = sample + int(round(EPOCH_TMIN * sfreq))
        stop = start + n_samples
        if stop > eeg_data.shape[1]:
            continue
        X_trials.append(eeg_data[:, start:stop].astype(np.float32, copy=False))
        y_trials.append(int(code))
        trial_ids.append(i)
        # Encode session as a "run" id so downstream filter_runs() works:
        #   session_T -> run 1, session_E -> run 2.
        run_ids.append(1 if session == "0train" else 2)

    X_trials = np.stack(X_trials, axis=0)
    y_trials = np.asarray(y_trials, dtype=np.int64)
    trial_ids = np.asarray(trial_ids, dtype=np.int64)
    run_ids = np.asarray(run_ids, dtype=np.int64)

    X, y, t, r = slide_windows(
        X_trials, y_trials, trial_ids, run_ids,
        window_seconds=window_seconds, stride_seconds=stride_seconds, sfreq=sfreq,
    )
    s = np.full_like(t, subject_id)
    eeg_ch_names = tuple(ch for ch, ty in zip(raw.ch_names, raw.get_channel_types())
                         if ty == "eeg")
    return WindowedDataset(
        X=X, y=y, subject_ids=s, trial_ids=t, run_ids=r,
        sfreq=sfreq, channel_names=eeg_ch_names,
    )


def valid_subjects() -> list[int]:
    """All 9 IV-2a subjects."""
    return list(range(1, 10))

