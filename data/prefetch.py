"""Parallel prefetcher for PhysioNet EEG-MMIDB.

`mne.datasets.eegbci.load_data` downloads serially, one EDF at a time.
For 104 subjects × 6 imagery runs = 624 files that's > 1 hour over a
typical link. We thread the per-subject calls (subjects are independent;
pooch handles caching safely) so end-to-end download is bounded by the
slowest few subjects rather than their sum.

Usage
-----
    python -m data.prefetch --runs imagery
    python -m data.prefetch --runs imagery execution baseline
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from mne.datasets import eegbci

from config import (
    PHYSIONET_RUNS_BASELINE,
    PHYSIONET_RUNS_EXECUTION,
    PHYSIONET_RUNS_IMAGERY,
)
from data.physionet_loader import valid_subjects


_RUN_GROUPS = {
    "imagery": PHYSIONET_RUNS_IMAGERY,
    "execution": PHYSIONET_RUNS_EXECUTION,
    "baseline": PHYSIONET_RUNS_BASELINE,
}


def _fetch_one(subject_id: int, runs: tuple[int, ...]) -> tuple[int, float, str | None]:
    t0 = time.time()
    try:
        eegbci.load_data(subject_id, list(runs), update_path=False, verbose=False)
        return subject_id, time.time() - t0, None
    except Exception as exc:  # pragma: no cover — network errors are the point of catching here
        return subject_id, time.time() - t0, str(exc)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["imagery"],
                   choices=list(_RUN_GROUPS.keys()))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--subjects", type=int, nargs="*", default=None,
                   help="Restrict prefetch to a subject list (default: all 104 valid).")
    args = p.parse_args()

    runs: tuple[int, ...] = tuple(sorted({r for g in args.runs for r in _RUN_GROUPS[g]}))
    subjects = args.subjects or valid_subjects()

    print(f"Prefetching {len(subjects)} subjects × {len(runs)} runs each "
          f"({len(subjects) * len(runs)} files) with {args.workers} workers")
    print(f"Runs: {runs}")

    t0 = time.time()
    failures: list[tuple[int, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_fetch_one, s, runs) for s in subjects]
        for i, fut in enumerate(as_completed(futures), 1):
            subject_id, dt, err = fut.result()
            if err:
                failures.append((subject_id, err))
                print(f"  [{i:3d}/{len(subjects)}]  S{subject_id:03d}  FAILED  ({dt:.1f}s): {err}")
            else:
                print(f"  [{i:3d}/{len(subjects)}]  S{subject_id:03d}  ok  ({dt:.1f}s)")

    print(f"\nDone in {time.time() - t0:.1f}s. {len(failures)} failures.")
    if failures:
        for s, e in failures:
            print(f"  S{s:03d}: {e}")


if __name__ == "__main__":
    main()
