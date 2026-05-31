"""Parallel range-request downloader + on-the-fly preprocessor for Lee 2019.

The OpenBMI Lee2019 corpus lives on a Wasabi S3 bucket in ap-northeast-1
(Tokyo). Single-connection download throughput from a US-residential or
Colab US-host averages ~3 MB/s, which makes a serial moabb fetch of the
~65 GB raw corpus impractical within a single Colab wall-clock budget.

This script does two things that together make Lee2019 tractable on free-
tier Colab + free-tier Drive:

  1.  Multi-range parallel download per file (8-16 concurrent Range
      requests against the same Wasabi object). The same file streams
      ~5-8x faster end-to-end versus a single connection.

  2.  Stream-and-compact per subject. Each (subject, session) .mat is
      decoded, bandpassed, windowed, and written as a compact float16 .npz
      to the cache directory, then the raw .mat is deleted. Total cache
      footprint after the prefetch run is ~3-5 GB rather than ~65 GB.

Cache layout (under --cache-dir):

    <cache>/lee2019_mi/raw/sess{1,2}/sess{NN}_subj{NN}_EEG_MI.mat
        ephemeral; deleted after the matching compact npz lands.

    <cache>/lee2019_mi/windowed/subj{NN}_sess{1,2}.npz
        keys: X (n_windows, 62, n_times) float16
              y (n_windows,)             int8       0=left, 1=right
              trial_ids (n_windows,)     int32
              run_ids (n_windows,)       int8       1 or 2
              channel_names (62,)        S8

Usage
-----
    python -m data.lee2019_prefetch --subjects 1-54 --cache-dir /content/drive/MyDrive/bci_cache --workers 8
    python -m data.lee2019_prefetch --resume --cache-dir /content/drive/MyDrive/bci_cache --workers 8
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

LEE2019_URL = (
    "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/"
    "10.5524/100001_101000/100542/"
)
EXPECTED_SUBJECTS = list(range(1, 55))
SESSIONS = (1, 2)

# Preprocessing constants (mirror the IV-2a path so cross-dataset
# comparisons are apples-to-apples).
TARGET_SFREQ = 250.0
BANDPASS_LOW = 4.0
BANDPASS_HIGH = 40.0
EPOCH_TMIN = 0.0
EPOCH_TMAX = 4.0
WINDOW_SECONDS = 2.0
WINDOW_STRIDE_SECONDS = 1.0


def _file_url(subject: int, session: int) -> str:
    return (
        f"{LEE2019_URL}session{session}/s{subject}/"
        f"sess{session:02d}_subj{subject:02d}_EEG_MI.mat"
    )


def _raw_dest(cache_dir: Path, subject: int, session: int) -> Path:
    return (
        cache_dir
        / "lee2019_mi"
        / "raw"
        / f"sess{session}"
        / f"sess{session:02d}_subj{subject:02d}_EEG_MI.mat"
    )


def _windowed_dest(cache_dir: Path, subject: int, session: int) -> Path:
    return cache_dir / "lee2019_mi" / "windowed" / f"subj{subject:02d}_sess{session}.npz"


def _content_length(client, url: str, *, attempts: int = 5) -> int:
    import httpx
    for k in range(attempts):
        try:
            r = client.head(url, timeout=httpx.Timeout(30.0))
            r.raise_for_status()
            return int(r.headers["Content-Length"])
        except Exception:
            time.sleep(2 ** k)
    raise RuntimeError(f"HEAD failed for {url}")


def _download_one_range(client, url: str, lo: int, hi: int, *,
                        attempts: int = 5) -> bytes:
    import httpx
    last_err: Exception | None = None
    for k in range(attempts):
        try:
            r = client.get(
                url,
                headers={"Range": f"bytes={lo}-{hi}"},
                timeout=httpx.Timeout(connect=30.0, read=600.0,
                                      write=30.0, pool=60.0),
            )
            r.raise_for_status()
            if r.status_code not in (200, 206):
                raise RuntimeError(f"HTTP {r.status_code}")
            return r.content
        except Exception as exc:
            last_err = exc
            time.sleep(2 ** k)
    raise RuntimeError(f"Range {lo}-{hi} failed for {url}: {last_err}")


def download_parallel(url: str, dest: Path, *,
                      chunks: int = 8, attempts: int = 5) -> None:
    """Download `url` to `dest` using `chunks` parallel Range requests."""
    import httpx
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    limits = httpx.Limits(max_connections=chunks + 2,
                          max_keepalive_connections=chunks + 2)
    with httpx.Client(http2=False, limits=limits, follow_redirects=True) as client:
        total = _content_length(client, url, attempts=attempts)
        # Build chunk boundaries.
        edges = np.linspace(0, total, chunks + 1, dtype=np.int64).tolist()
        ranges = [(edges[i], edges[i + 1] - 1) for i in range(chunks)]
        buffers: list[bytes | None] = [None] * chunks

        def _do(i: int) -> tuple[int, bytes]:
            lo, hi = ranges[i]
            return i, _download_one_range(client, url, lo, hi, attempts=attempts)

        with ThreadPoolExecutor(max_workers=chunks) as ex:
            for fut in as_completed([ex.submit(_do, i) for i in range(chunks)]):
                i, data = fut.result()
                buffers[i] = data

    with tmp.open("wb") as f:
        for buf in buffers:
            assert buf is not None
            f.write(buf)
    if tmp.stat().st_size != total:
        tmp.unlink()
        raise RuntimeError(
            f"Size mismatch after download: "
            f"got {tmp.stat().st_size} expected {total}"
        )
    tmp.replace(dest)


def _load_mat_session(mat_path: Path) -> tuple[np.ndarray, np.ndarray,
                                                np.ndarray, list[str], float]:
    """Decode a Lee2019 MI .mat into (X, y, trial_ids, channel_names, sfreq).

    Lee2019 nests:
        mat['EEG_MI_train'][0,0] / mat['EEG_MI_test'][0,0]
            .x           (n_samples, 62)  float64
            .chan        (62,)            S
            .fs          scalar           int
            .t           (n_trials,)      int   event onset samples
            .y_dec       (n_trials,)      int   class id (1=right, 2=left in their numbering)
            .class       (n_classes,)     pairs of (class name, dec id)

    We concatenate train and test halves end-to-end so the full session
    contains all trials (the test half is unlabeled for the online-BCI
    competition format but for MI the labels are intact via y_dec).
    """
    from scipy.io import loadmat

    mat = loadmat(str(mat_path), squeeze_me=False)
    halves = []
    for key in ("EEG_MI_train", "EEG_MI_test"):
        if key in mat:
            halves.append(mat[key][0, 0])
    if not halves:
        raise KeyError(f"No EEG_MI_train/test in {mat_path}")

    sfreq = float(halves[0]["fs"].squeeze())
    chan_field = halves[0]["chan"]
    chans = [str(c[0]).strip() for c in chan_field.squeeze()]

    # Build the y_dec → {0, 1} remap from the class table so we are robust to
    # Lee 2019's native ordering (1=right, 2=left in their .mat) and match the
    # left=0 / right=1 convention used everywhere else in the project.
    name_to_id: dict[str, int] = {}
    try:
        # moabb pattern: `for v, c in data['class']` where v is dec_id, c is name.
        for v, c in halves[0]["class"]:
            dec_id = int(np.asarray(v).item())
            name = str(np.asarray(c).item()).strip().lower()
            name_to_id[name] = dec_id
    except Exception:
        name_to_id = {}

    if any("left" in k for k in name_to_id) and any("right" in k for k in name_to_id):
        left_id = next(v for k, v in name_to_id.items() if "left" in k)
        right_id = next(v for k, v in name_to_id.items() if "right" in k)
    else:
        # Fallback: Lee 2019 native convention (right=1, left=2).
        right_id, left_id = 1, 2
    remap = {left_id: 0, right_id: 1}

    Xs, Ys, Ts = [], [], []
    offset = 0
    for half in halves:
        x = np.asarray(half["x"], dtype=np.float32)   # (n_samples, 62)
        t = np.asarray(half["t"], dtype=np.int64).reshape(-1)   # onset samples
        y_raw = np.asarray(half["y_dec"], dtype=np.int64).reshape(-1)
        y = np.array([remap.get(int(v), -1) for v in y_raw], dtype=np.int64)
        keep = y >= 0
        Xs.append(x)
        Ys.append(y[keep])
        Ts.append(t[keep] + offset)
        offset += x.shape[0]
    X_concat = np.concatenate(Xs, axis=0)    # (n_samples_total, 62)
    y_concat = np.concatenate(Ys)
    t_concat = np.concatenate(Ts)

    return X_concat.T, y_concat, t_concat, chans, sfreq


def _bandpass(X: np.ndarray, sfreq: float,
              low: float, high: float) -> np.ndarray:
    """4-40 Hz zero-phase Butterworth bandpass, channel-wise."""
    from scipy.signal import butter, sosfiltfilt
    nyq = sfreq / 2.0
    sos = butter(4, [low / nyq, high / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, X, axis=-1).astype(np.float32, copy=False)


def _resample_axis(X: np.ndarray, src_sfreq: float, tgt_sfreq: float) -> np.ndarray:
    """Polyphase resampling along the last axis."""
    if abs(src_sfreq - tgt_sfreq) < 0.5:
        return X
    from math import gcd

    from scipy.signal import resample_poly
    g = gcd(int(round(src_sfreq)), int(round(tgt_sfreq)))
    up = int(round(tgt_sfreq)) // g
    down = int(round(src_sfreq)) // g
    return resample_poly(X, up, down, axis=-1).astype(np.float32, copy=False)


def _window_trials(X: np.ndarray, t_events: np.ndarray, y_events: np.ndarray,
                   sfreq: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Epoch each event into [tmin, tmax] then 2-s sliding windows w/ 1-s stride."""
    epoch_n = int(round((EPOCH_TMAX - EPOCH_TMIN) * sfreq))
    win_n = int(round(WINDOW_SECONDS * sfreq))
    stride_n = int(round(WINDOW_STRIDE_SECONDS * sfreq))

    win_X, win_y, win_t = [], [], []
    for trial_idx, (sample, code) in enumerate(zip(t_events, y_events)):
        start = sample + int(round(EPOCH_TMIN * sfreq))
        stop = start + epoch_n
        if stop > X.shape[1] or start < 0:
            continue
        epoch = X[:, start:stop]                     # (62, epoch_n)
        # slide
        for w_start in range(0, epoch_n - win_n + 1, stride_n):
            win_X.append(epoch[:, w_start:w_start + win_n])
            win_y.append(int(code))
            win_t.append(trial_idx)
    if not win_X:
        raise RuntimeError("No windows produced — corrupt session?")
    return (
        np.stack(win_X, axis=0).astype(np.float32, copy=False),
        np.asarray(win_y, dtype=np.int64),
        np.asarray(win_t, dtype=np.int64),
    )


def _save_compact_npz(dest: Path,
                      X: np.ndarray, y: np.ndarray, trial_ids: np.ndarray,
                      session: int, channel_names: list[str], sfreq: float) -> None:
    # Quantize to float16 — EEG is wide-dynamic-range microvolt-scale so float16
    # preserves ~3 sig-figs which is fine for downstream filtering/embedding.
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        dest,
        X=X.astype(np.float16),
        y=y.astype(np.int8),
        trial_ids=trial_ids.astype(np.int32),
        run_ids=np.full_like(trial_ids, fill_value=session, dtype=np.int8),
        channel_names=np.asarray(channel_names, dtype="S8"),
        sfreq=np.asarray([sfreq], dtype=np.float32),
    )


def process_subject_session(cache_dir: Path, subject: int, session: int,
                            *, workers: int, keep_raw: bool = False) -> dict:
    """Download, decode, preprocess, save compact. Idempotent."""
    out_npz = _windowed_dest(cache_dir, subject, session)
    if out_npz.exists():
        return {"subject": subject, "session": session, "status": "cached"}

    raw_path = _raw_dest(cache_dir, subject, session)
    url = _file_url(subject, session)

    t0 = time.time()
    if not (raw_path.exists() and raw_path.stat().st_size > 0):
        download_parallel(url, raw_path, chunks=workers)
    t_dl = time.time() - t0
    raw_mb = raw_path.stat().st_size / 1e6 if raw_path.exists() else 0.0

    t = time.time()
    X, y, t_events, chans, sfreq = _load_mat_session(raw_path)
    X = _bandpass(X, sfreq, BANDPASS_LOW, BANDPASS_HIGH)
    X = _resample_axis(X, sfreq, TARGET_SFREQ)
    # Resample the event sample-indices to the new rate too.
    t_events_resampled = np.round(t_events * (TARGET_SFREQ / sfreq)).astype(np.int64)
    Xw, yw, trial_ids = _window_trials(X, t_events_resampled, y, TARGET_SFREQ)
    _save_compact_npz(out_npz, Xw, yw, trial_ids, session, chans, TARGET_SFREQ)
    t_proc = time.time() - t

    if not keep_raw and raw_path.exists():
        raw_path.unlink()

    return {
        "subject": subject, "session": session, "status": "ok",
        "n_windows": int(Xw.shape[0]),
        "download_seconds": round(t_dl, 1),
        "preprocess_seconds": round(t_proc, 1),
        "mb_downloaded": round(raw_mb, 1),
    }


def _parse_subjects(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted({s for s in out if s in EXPECTED_SUBJECTS})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", required=True, type=Path,
                   help="Where to write the windowed/.npz cache. Should be on Drive.")
    p.add_argument("--subjects", default="1-54",
                   help='Subject range, e.g. "1-54" or "1,5,10-20"')
    p.add_argument("--sessions", default="1,2",
                   help='Comma-separated session list, default "1,2"')
    p.add_argument("--workers", type=int, default=8,
                   help="Range-chunks per file download")
    p.add_argument("--keep-raw", action="store_true",
                   help="Do not delete the raw .mat after compaction")
    p.add_argument("--resume", action="store_true",
                   help="Skip subjects whose compact npz already exists (default behavior)")
    args = p.parse_args()

    subjects = _parse_subjects(args.subjects)
    sessions = [int(s) for s in args.sessions.split(",")]

    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cache root:   {cache_dir}")
    print(f"Subjects:     {len(subjects)} ({subjects[0]}..{subjects[-1]})")
    print(f"Sessions:     {sessions}")
    print(f"Workers/file: {args.workers}")
    print()

    todo: list[tuple[int, int]] = []
    for s in subjects:
        for sess in sessions:
            if not _windowed_dest(cache_dir, s, sess).exists():
                todo.append((s, sess))
    print(f"Already cached: {len(subjects) * len(sessions) - len(todo)} "
          f"({100 - 100 * len(todo) / (len(subjects) * len(sessions)):.0f}%)")
    print(f"To download:    {len(todo)}\n", flush=True)
    if not todo:
        print("Nothing to do — cache complete.")
        return

    t0 = time.time()
    total_mb = 0.0
    for idx, (subject, session) in enumerate(todo, start=1):
        try:
            info = process_subject_session(
                cache_dir, subject, session,
                workers=args.workers, keep_raw=args.keep_raw,
            )
        except Exception as exc:
            print(f"  [{idx:3d}/{len(todo)}]  FAIL subj{subject:02d} sess{session}: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            continue
        total_mb += info.get("mb_downloaded", 0.0)
        elapsed = time.time() - t0
        rate = total_mb / max(elapsed, 1e-9)
        eta_min = (len(todo) - idx) * (elapsed / idx) / 60.0
        print(
            f"  [{idx:3d}/{len(todo)}]  {info['status']:6s}  "
            f"subj{subject:02d} sess{session}  "
            f"win={info.get('n_windows', '-'):>4}  "
            f"dl={info.get('download_seconds', 0):>4}s  "
            f"pp={info.get('preprocess_seconds', 0):>4}s  "
            f"avg={rate:.2f} MB/s  ETA={eta_min:.0f} min",
            flush=True,
        )

    print(f"\nDone in {(time.time()-t0)/60:.1f} min. "
          f"Total downloaded: {total_mb:.0f} MB.")


if __name__ == "__main__":
    sys.exit(main())
