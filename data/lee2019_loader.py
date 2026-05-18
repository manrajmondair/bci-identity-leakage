"""Lee 2019 OpenBMI motor-imagery loader (54 subjects × 2 sessions).

Lee et al. 2019 (DOI 10.1093/gigascience/giz002) — "EEG dataset and OpenBMI
toolbox for three BCI paradigms." We use only the motor-imagery paradigm:
binary left/right hand, 62 channels @ 1000 Hz (downsampled by moabb to
the dataset's published rate), two sessions on different days per
subject. This makes it the second public motor-imagery dataset in the
project that allows true cross-session re-identification (the first is
BCI Competition IV-2a, n=9).

We expose it through moabb.datasets.Lee2019_MI, which wraps the official
OpenBMI distribution. moabb caches the raw files under
~/mne_data/MNE-lee2019-data/ on first use.

Class mapping:
    0 = left_hand
    1 = right_hand
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import mne
import numpy as np

from preprocess.filtering import bandpass
from preprocess.windows import WindowedDataset, slide_windows


_LEE2019_CLASSES = {"left_hand": 0, "right_hand": 1}


@dataclass(frozen=True)
class Lee2019Recording:
    subject_id: int
    session: str        # 'session_1' (day 1) or 'session_2' (day 2)
    raw: mne.io.BaseRaw # concatenated across runs within session


def _load_dataset():
    """Lazy import — moabb pulls a lot of code, so don't import at module top."""
    from moabb.datasets import Lee2019_MI
    return Lee2019_MI()


def _get_subject_data_with_retry(
    subject_id: int,
    *,
    max_attempts: int = 4,
    backoff_seconds: float = 6.0,
) -> dict:
    """moabb's pooch backend can drop mid-stream on the OpenBMI host. Retry
    after a short backoff (mirror of the IV-2a loader's strategy)."""
    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            ds = _load_dataset()
            return ds.get_data(subjects=[subject_id])
        except Exception as exc:
            last_err = exc
            if attempt < max_attempts:
                wait = backoff_seconds * (2 ** (attempt - 1))
                print(f"    Lee2019 get_data for subject {subject_id} "
                      f"failed (attempt {attempt}/{max_attempts}): "
                      f"{type(exc).__name__}: {exc}. "
                      f"Retrying in {wait:.0f}s ...", flush=True)
                time.sleep(wait)
    assert last_err is not None
    raise last_err


def load_subject_session(
    subject_id: int,
    session: str = "session_1",
    *,
    bandpass_low: float | None = 4.0,
    bandpass_high: float | None = 40.0,
    window_seconds: float = 2.0,
    stride_seconds: float = 1.0,
    target_sfreq: float | None = 250.0,
) -> WindowedDataset:
    """Load one (subject, session) pair as windowed data.

    moabb's Lee2019_MI returns:
        data[subject_id][session_name][run_name] = mne.Raw
    where session_name is one of {'0', '1'} (or 'session_1'/'session_2'
    depending on moabb version) and run_name is per-run within that
    session. We concatenate all runs in the requested session.

    Lee2019 native sampling rate is 1000 Hz, which gives 2000-sample
    windows — far more than EEGNet was trained for. We downsample to
    250 Hz to match BCI IV-2a's sampling rate (and then apply the same
    2-s window with 1-s stride, giving 500 samples per window).
    """
    data = _get_subject_data_with_retry(subject_id)
    # moabb naming has shifted across versions; tolerate both.
    sessions = data[subject_id]
    if session in sessions:
        runs = sessions[session]
    else:
        # Fallback: pick by ordering (sessions are returned ordered)
        ordered = list(sessions.keys())
        if session in ("session_1", "0", "0train", "train"):
            runs = sessions[ordered[0]]
        elif session in ("session_2", "1", "1test", "test"):
            runs = sessions[ordered[1]]
        else:
            raise ValueError(
                f"Lee2019: unknown session {session!r}. "
                f"Available: {list(sessions.keys())}"
            )

    raws = list(runs.values())
    raw = mne.concatenate_raws(raws, verbose=False)

    if target_sfreq is not None and abs(raw.info["sfreq"] - target_sfreq) > 0.5:
        raw = raw.copy().resample(target_sfreq, verbose=False)

    if bandpass_low is not None and bandpass_high is not None:
        raw = bandpass(raw, bandpass_low, bandpass_high, copy=True)

    # Lee2019 annotations: left_hand / right_hand
    ann_to_class = dict(_LEE2019_CLASSES)
    available_ann = {a["description"] for a in raw.annotations}
    event_id = {k: v for k, v in ann_to_class.items() if k in available_ann}
    if not event_id:
        # Last-resort: try numeric codes
        event_id = {"1": 0, "2": 1}
        event_id = {k: v for k, v in event_id.items() if k in available_ann}
    if not event_id:
        raise RuntimeError(
            f"Lee2019: no recognised motor-imagery annotations for "
            f"subject {subject_id} session {session}. "
            f"Available: {available_ann}"
        )

    events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)
    if len(events) == 0:
        raise RuntimeError(
            f"Lee2019: no labeled trials extracted for subject {subject_id} "
            f"session {session}. Annotations were {available_ann}."
        )

    sfreq = raw.info["sfreq"]
    EPOCH_TMIN, EPOCH_TMAX = 0.0, 4.0
    n_samples = int(round((EPOCH_TMAX - EPOCH_TMIN) * sfreq))

    eeg_data = raw.get_data(picks="eeg")
    X_trials, y_trials, trial_ids, run_ids = [], [], [], []
    session_run_id = 1 if session in ("session_1", "0", "0train", "train") else 2
    for i, (sample, _, code) in enumerate(events):
        start = sample + int(round(EPOCH_TMIN * sfreq))
        stop = start + n_samples
        if stop > eeg_data.shape[1]:
            continue
        X_trials.append(eeg_data[:, start:stop].astype(np.float32, copy=False))
        y_trials.append(int(code))
        trial_ids.append(i)
        run_ids.append(session_run_id)

    if not X_trials:
        raise RuntimeError(
            f"Lee2019: no trials survived epoching for subject {subject_id} "
            f"session {session}."
        )

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
    """All 54 Lee2019 motor-imagery subjects."""
    return list(range(1, 55))


# ---------- compact-cache loader ----------
#
# When data/lee2019_prefetch.py has been run, each (subject, session) pair
# lives as a small float16 .npz on disk. This is dramatically faster (and
# disk-cheaper) than re-downloading + decoding the original ~600 MB .mat
# through moabb on every Colab session.

import os as _os  # noqa: E402

from preprocess.windows import WindowedDataset as _WD  # noqa: E402


def _compact_cache_dir() -> "os.PathLike | None":
    """Return the Drive cache root if set via env, else None.

    The env var matches what `data.lee2019_prefetch` writes to.
    """
    cd = _os.environ.get("BCI_LEE2019_CACHE")
    if cd:
        return _os.path.expanduser(cd)
    return None


def _compact_path(subject_id: int, session: str) -> "os.PathLike | None":
    cd = _compact_cache_dir()
    if cd is None:
        return None
    s_num = 1 if session in ("session_1", "0", "0train", "train") else 2
    path = _os.path.join(cd, "lee2019_mi", "windowed",
                         f"subj{subject_id:02d}_sess{s_num}.npz")
    return path if _os.path.isfile(path) else None


def load_subject_session_compact(subject_id: int, session: str = "session_1") -> _WD:
    """Load one (subject, session) pair from the compact .npz cache.

    Falls back to the moabb path (`load_subject_session`) if the compact
    cache is unset or missing.
    """
    path = _compact_path(subject_id, session)
    if path is None:
        return load_subject_session(subject_id, session=session)

    with np.load(path, allow_pickle=False) as z:
        X = z["X"].astype(np.float32)
        y = z["y"].astype(np.int64)
        trial_ids = z["trial_ids"].astype(np.int64)
        run_ids = z["run_ids"].astype(np.int64)
        chans = tuple(name.decode("utf-8").strip() for name in z["channel_names"])
        sfreq = float(z["sfreq"][0])

    s = np.full_like(trial_ids, subject_id, dtype=np.int64)
    return _WD(
        X=X, y=y, subject_ids=s, trial_ids=trial_ids, run_ids=run_ids,
        sfreq=sfreq, channel_names=chans,
    )
