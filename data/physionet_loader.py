"""PhysioNet EEG Motor Movement/Imagery Dataset loader.

Wraps mne.datasets.eegbci.load_data with caching, drop-list handling,
and consistent run/task indexing. We intentionally do NOT do any
preprocessing here — that lives in preprocess/. This module's job is
just "give me Raw objects (or epoched arrays) for subject X, runs Y."

Reference: Schalk et al., "BCI2000", IEEE TBME 2004.
"""
from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import mne
import numpy as np
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

from config import (
    PHYSIONET_DROP_SUBJECTS,
    PHYSIONET_N_SUBJECTS,
    PHYSIONET_RUNS_BASELINE,
    PHYSIONET_RUNS_EXECUTION,
    PHYSIONET_RUNS_IMAGERY,
    SAMPLING_RATE,
)

# Suppress mne's RuntimeWarning about unsigned events — known and harmless.
warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")


# ---- Run-level event labels ---------------------------------------------------
# Each task block has 2 trial classes (in addition to "rest"=T0).
# We map to a unified 4-class label space for motor imagery decoding:
#   0: left fist (imagery)
#   1: right fist (imagery)
#   2: both fists (imagery)
#   3: both feet (imagery)
# For execution runs we use the same label space (real movement instead of imagery).
#
# Mapping per run, T1 / T2 → class index:
#   Runs 3,4,7,8,11,12 (one fist):    T1 = left fist (0), T2 = right fist (1)
#   Runs 5,6,9,10,13,14 (fists/feet): T1 = both fists (2), T2 = both feet (3)
_RUN_T1_T2 = {
    3: (0, 1), 4: (0, 1), 7: (0, 1), 8: (0, 1), 11: (0, 1), 12: (0, 1),
    5: (2, 3), 6: (2, 3), 9: (2, 3), 13: (2, 3), 10: (2, 3), 14: (2, 3),
}


@dataclass(frozen=True)
class SubjectRecording:
    """Concatenated raw recording for one subject across requested runs."""
    subject_id: int
    runs: tuple[int, ...]
    raw: mne.io.BaseRaw
    # Per-run boundaries (sample indices) — needed if we later want to split
    # train/test by run rather than by epoch.
    run_boundaries: tuple[tuple[int, int], ...]


def valid_subjects(n: int = PHYSIONET_N_SUBJECTS) -> list[int]:
    """Return the canonical 104-subject list (drops known-bad subjects)."""
    return [s for s in range(1, n + 1) if s not in PHYSIONET_DROP_SUBJECTS]


def _normalize_montage(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """PhysioNet EDF channel names have trailing dots and use BCI2000's
    truncated 10-10 names ('Fcz.', 'Af3.', etc). Strip dots, lowercase,
    then assign the standard 10-05 montage."""
    eegbci.standardize(raw)  # in-place rename and montage attachment
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing="ignore")
    return raw


def load_subject(
    subject_id: int,
    runs: Iterable[int] | str = "imagery",
    *,
    cache_dir: Path | None = None,
    preload: bool = True,
    verbose: bool = False,
) -> SubjectRecording:
    """Load one subject's raw recording for the given runs.

    runs: an iterable of 1-based run indices, or one of:
          "imagery"   → all 6 motor-imagery runs
          "execution" → all 6 motor-execution runs
          "baseline"  → 2 baseline runs (eyes open / eyes closed)
          "all"       → all 14 runs
    """
    if subject_id in PHYSIONET_DROP_SUBJECTS:
        raise ValueError(f"Subject {subject_id} is in the project drop-list.")

    if isinstance(runs, str):
        runs = {
            "imagery": PHYSIONET_RUNS_IMAGERY,
            "execution": PHYSIONET_RUNS_EXECUTION,
            "baseline": PHYSIONET_RUNS_BASELINE,
            "all": tuple(range(1, 15)),
        }[runs]
    runs = tuple(int(r) for r in runs)

    paths = eegbci.load_data(subject_id, list(runs), path=str(cache_dir) if cache_dir else None,
                             update_path=False, verbose=verbose)
    raws = [read_raw_edf(p, preload=preload, verbose=verbose) for p in paths]

    # Track per-run sample boundaries before concatenation
    boundaries: list[tuple[int, int]] = []
    cursor = 0
    for r in raws:
        n = r.n_times
        boundaries.append((cursor, cursor + n))
        cursor += n

    raw = concatenate_raws(raws)
    raw = _normalize_montage(raw)

    # PhysioNet is 160 Hz; assert in case mne ever changes defaults.
    assert int(round(raw.info["sfreq"])) == SAMPLING_RATE, (
        f"Unexpected sample rate {raw.info['sfreq']}Hz for subject {subject_id}"
    )

    return SubjectRecording(
        subject_id=subject_id,
        runs=runs,
        raw=raw,
        run_boundaries=tuple(boundaries),
    )


def run_label_pair(run: int) -> tuple[int, int]:
    """Class indices that T1 and T2 map to for a given run number."""
    return _RUN_T1_T2[run]


# Demographic metadata. PhysioNet stores age/sex in each subject's EDF header
# under "subject_info" → dict with keys 'his_id', 'sex', 'age', 'birthday'.
# `sex`: 1 = male, 2 = female, 0 = unknown (mne convention).
@lru_cache(maxsize=None)
def subject_metadata(subject_id: int, cache_dir: Path | None = None) -> dict:
    """Return {'subject_id', 'age_years', 'sex', 'sex_code'} for a subject.

    Subject metadata lives in every EDF's header; reads whichever run is
    already cached so we don't trigger ~100 MB of baseline-run downloads
    for subjects whose imagery runs are already on disk. Falls back to
    R01 if nothing else is cached.
    """
    # Prefer an already-cached run. Imagery runs are typically present
    # because A1 caches them; baseline (1, 2) tend not to be.
    candidate_runs = list(PHYSIONET_RUNS_IMAGERY) + [1, 2]
    base = (Path(cache_dir) if cache_dir
            else Path.home() / "mne_data" / "MNE-eegbci-data"
            / "files" / "eegmmidb" / "1.0.0")
    chosen_run = None
    for r in candidate_runs:
        candidate = base / f"S{subject_id:03d}" / f"S{subject_id:03d}R{r:02d}.edf"
        if candidate.exists():
            chosen_run = r
            break
    if chosen_run is None:
        chosen_run = 1  # trigger download as fallback

    paths = eegbci.load_data(subject_id, [chosen_run],
                             path=str(cache_dir) if cache_dir else None,
                             update_path=False, verbose=False)
    raw = read_raw_edf(paths[0], preload=False, verbose=False)
    info = raw.info["subject_info"] or {}
    sex_code = int(info.get("sex", 0))
    sex = {0: "unknown", 1: "male", 2: "female"}[sex_code]
    # PhysioNet ages are integers stored as 'age' in years.
    age = info.get("age")
    return {
        "subject_id": subject_id,
        "age_years": int(age) if age is not None else None,
        "sex": sex,
        "sex_code": sex_code,
    }


def all_subject_metadata(cache_dir: Path | None = None) -> list[dict]:
    """Bulk-load metadata for every valid subject. Used by subgroup analysis."""
    return [subject_metadata(s, cache_dir=cache_dir) for s in valid_subjects()]


def smoke_test() -> None:
    """Sanity check: load subject 1 imagery runs and print shape."""
    rec = load_subject(1, "imagery")
    print(f"Subject {rec.subject_id} | runs {rec.runs}")
    print(f"Raw: {rec.raw}")
    print(f"Channels: {len(rec.raw.ch_names)} | sfreq: {rec.raw.info['sfreq']} Hz")
    print(f"Duration: {rec.raw.times[-1]:.1f} s across {len(rec.runs)} runs")
    meta = subject_metadata(1)
    print(f"Metadata: {meta}")


if __name__ == "__main__":
    smoke_test()
