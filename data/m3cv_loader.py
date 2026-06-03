"""M3CV (Wang et al. 2022) optional-corpus loader.

The M3CV biometrics challenge dataset (NeuroImage 264:119666) is a
multi-subject, multi-session, multi-task EEG biometric corpus designed
specifically for cross-task biometric evaluation. Cohort: 95 subjects,
several recording sessions per subject, six distinct paradigms.

The primary distribution is on Science Data Bank
(https://www.scidb.cn/en/detail?dataSetId=778773169668231168) which
hosts the original .npz / .mat archive. This module exposes a thin
loader so that, when the corpus is mirrored into the cache path under
`BCI_M3CV_CACHE`, the rest of the codebase can consume it as a
second-corpus alongside Lee 2019 and IV-2a.

This module is CACHE-ONLY: it contains no download or .mat/.npz decoding
code. It only reads a pre-built compact cache that an external mirroring
step must produce under `BCI_M3CV_CACHE`; until that cache exists,
`is_available()` returns False and the corpus is simply skipped. M3CV is an
optional/aspirational second corpus, not wired into any committed result.

The expected on-disk layout is documented below; `load_subject_session_compact`
is the entry point used by experiments when the cache is present.

Expected on-disk layout (after mirroring):

    <BCI_M3CV_CACHE>/m3cv/windowed/subj{ID:03d}_sess{S}_task{TASK}.npz
        keys: X (n_windows, 64, n_times) float16
              y (n_windows,)               int8         task class
              trial_ids (n_windows,)       int32
              run_ids (n_windows,)         int8
              channel_names (64,)          S8

We deliberately keep the on-disk schema identical to Lee 2019's compact
cache so the same WindowedDataset machinery in `preprocess.windows` /
the experiment scripts can consume both without branching.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from preprocess.windows import WindowedDataset

SUPPORTED_PARADIGMS = (
    "resting",         # eyes-open / eyes-closed
    "imagery",         # motor imagery (left/right hand, feet)
    "ssvep",           # steady-state visual evoked
    "p300",            # auditory / visual oddball
    "selfregulation",  # self-regulation BCI
    "go_nogo",         # go / no-go
)


def _cache_root() -> Path | None:
    cd = os.environ.get("BCI_M3CV_CACHE")
    return Path(os.path.expanduser(cd)) if cd else None


def is_available() -> bool:
    """Check whether the M3CV compact cache has at least one subject."""
    root = _cache_root()
    if root is None:
        return False
    win_dir = root / "m3cv" / "windowed"
    if not win_dir.is_dir():
        return False
    return any(win_dir.glob("subj*_sess*.npz"))


def valid_subjects() -> list[int]:
    """Subjects discovered in the local M3CV compact cache.

    Returns an empty list if the cache is missing -- callers can use
    `is_available()` to decide whether to proceed.
    """
    root = _cache_root()
    if root is None:
        return []
    win_dir = root / "m3cv" / "windowed"
    if not win_dir.is_dir():
        return []
    seen: set[int] = set()
    for f in win_dir.glob("subj*_sess*.npz"):
        # Filename is subj{ID}_sess{S}_task{TASK}.npz
        sid = f.stem.split("_")[0][4:]
        try:
            seen.add(int(sid))
        except ValueError:
            continue
    return sorted(seen)


def load_subject_session_compact(
    subject_id: int,
    session: int = 1,
    paradigm: str = "imagery",
) -> WindowedDataset:
    """Load one (subject, session, paradigm) window pack from the compact cache."""
    root = _cache_root()
    if root is None:
        raise RuntimeError(
            "BCI_M3CV_CACHE env var is not set. Set it to the Drive path "
            "where the M3CV mirror lives (e.g. "
            "/content/drive/MyDrive/bci_cache)."
        )
    path = (root / "m3cv" / "windowed"
                 / f"subj{subject_id:03d}_sess{session}_task{paradigm}.npz")
    if not path.is_file():
        raise FileNotFoundError(
            f"M3CV compact cache miss: {path}. Either the subject does "
            f"not exist in the mirror, or the paradigm name does not match "
            f"the on-disk convention. Try is_available() / valid_subjects() "
            f"to discover what is present."
        )
    with np.load(path, allow_pickle=False) as z:
        X = z["X"].astype(np.float32)
        y = z["y"].astype(np.int64)
        trial_ids = z["trial_ids"].astype(np.int64)
        run_ids = z["run_ids"].astype(np.int64)
        chans = tuple(name.decode("utf-8").strip() for name in z["channel_names"])
        sfreq = float(z["sfreq"][0]) if "sfreq" in z.files else 1000.0
    s = np.full_like(trial_ids, subject_id, dtype=np.int64)
    return WindowedDataset(
        X=X, y=y, subject_ids=s, trial_ids=trial_ids, run_ids=run_ids,
        sfreq=sfreq, channel_names=chans,
    )

