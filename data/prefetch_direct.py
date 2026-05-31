"""Direct parallel HTTPS downloader for PhysioNet EEG-MMIDB.

The thread-pool prefetcher backed by mne.datasets.eegbci
serializes badly because pooch holds a per-dataset lock when checking
its registry. We saw ~1 subject/min on 8 workers, which would take 100+
minutes for the full corpus. This direct version downloads each EDF
straight from physionet.org via httpx into mne's cache directory layout,
where mne will subsequently discover them on read.

Usage
-----
    python -m data.prefetch_direct --runs imagery --workers 32
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from config import (
    PHYSIONET_RUNS_BASELINE,
    PHYSIONET_RUNS_EXECUTION,
    PHYSIONET_RUNS_IMAGERY,
)
from data.physionet_loader import valid_subjects

# archive.physionet.org serves the same EDF bytes ~10x faster per connection
# than the versioned URL on the main physionet.org host (measured ~2 MB/s vs
# ~200 KB/s on a single curl from a US-residential connection in May 2026).
# The trailing layout matches: S{NNN}/S{NNN}R{NN}.edf identical filenames.
BASE_URL = "https://archive.physionet.org/pn4/eegmmidb"
CACHE_ROOT = Path(os.path.expanduser("~/mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0"))

_RUN_GROUPS = {
    "imagery": PHYSIONET_RUNS_IMAGERY,
    "execution": PHYSIONET_RUNS_EXECUTION,
    "baseline": PHYSIONET_RUNS_BASELINE,
}


def _file_urls(subject_id: int, runs: tuple[int, ...]) -> list[tuple[str, Path]]:
    """For one subject, return the (URL, dest) pairs for each requested run."""
    out = []
    subj_dir = CACHE_ROOT / f"S{subject_id:03d}"
    for r in runs:
        fname = f"S{subject_id:03d}R{r:02d}.edf"
        url = f"{BASE_URL}/S{subject_id:03d}/{fname}"
        dest = subj_dir / fname
        out.append((url, dest))
    return out


def _fetch_one(
    client: httpx.Client,
    url: str,
    dest: Path,
    *,
    max_attempts: int = 5,
    backoff_base: float = 2.0,
) -> tuple[Path, int, str | None]:
    if dest.exists() and dest.stat().st_size > 0:
        return dest, dest.stat().st_size, None
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: str | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            with client.stream("GET", url, timeout=httpx.Timeout(connect=30.0,
                                                                  read=300.0,
                                                                  write=30.0,
                                                                  pool=60.0)) as r:
                r.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in r.iter_bytes(chunk_size=1 << 16):
                        f.write(chunk)
            tmp.rename(dest)
            return dest, dest.stat().st_size, None
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            if attempt < max_attempts:
                time.sleep(backoff_base ** (attempt - 1))
    return dest, 0, last_err


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["imagery"],
                   choices=list(_RUN_GROUPS.keys()))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--subjects", type=int, nargs="*", default=None)
    args = p.parse_args()

    runs: tuple[int, ...] = tuple(sorted({r for g in args.runs for r in _RUN_GROUPS[g]}))
    subjects = args.subjects or valid_subjects()

    pairs: list[tuple[str, Path]] = []
    for s in subjects:
        pairs.extend(_file_urls(s, runs))

    # Filter out files that already exist
    todo = [(u, d) for (u, d) in pairs if not (d.exists() and d.stat().st_size > 0)]
    print(f"{len(pairs) - len(todo)} of {len(pairs)} EDFs already cached.", flush=True)
    print(f"Downloading {len(todo)} EDFs with {args.workers} workers ...", flush=True)
    if not todo:
        return

    t0 = time.time()
    bytes_dl = 0
    failures: list[tuple[str, str]] = []
    limits = httpx.Limits(max_connections=args.workers, max_keepalive_connections=args.workers)
    with httpx.Client(http2=False, limits=limits, follow_redirects=True) as client:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_fetch_one, client, u, d): (u, d) for (u, d) in todo}
            done = 0
            for fut in as_completed(futures):
                u, d = futures[fut]
                _, size, err = fut.result()
                done += 1
                if err:
                    failures.append((u, err))
                    print(f"  [{done:4d}/{len(todo)}]  FAIL  {d.name}: {err}", flush=True)
                else:
                    bytes_dl += size
                    if done % 25 == 0 or done == len(todo):
                        elapsed = time.time() - t0
                        mbps = (bytes_dl / 1e6) / max(elapsed, 1e-9)
                        print(f"  [{done:4d}/{len(todo)}]  ok  "
                              f"{bytes_dl / 1e6:.0f} MB | {elapsed:.0f}s | "
                              f"{mbps:.1f} MB/s", flush=True)

    print(f"\nDone in {time.time() - t0:.1f}s. {len(failures)} failures.", flush=True)
    for u, e in failures[:10]:
        print(f"  {u}: {e}", flush=True)


if __name__ == "__main__":
    main()
