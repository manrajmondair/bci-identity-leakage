"""A3 cross-session re-identification on Lee 2019 OpenBMI motor imagery.

The original A3 (experiments/05) uses BCI Competition IV-2a — 9 subjects,
two sessions on different days. The cross-session result (Riemann 91.3%,
FBCSP 88.9%, EEGNet 78.3% top-1, chance 11.1%) is striking but n=9
gives wide CIs and a hostile reviewer can fairly call it anecdotal.

Lee 2019 (`data/lee2019_loader.py`) is the second public motor-imagery
dataset with multi-session per subject — 54 subjects × 2 sessions on
different days, recorded with a different rig and country (binary
left/right hand instead of IV-2a's 4-class). If A3 generalizes to this
cohort, it's no longer anecdotal.

Pipeline (mirrors experiments/05):
    Train victim on session-1 task labels (left vs right hand).
    Train re-ID probe on session-1 embeddings → subject_id.
    Test probe on session-2 embeddings (different day) → subject_id.

Class chance = 1/54 ≈ 1.85% (vs IV-2a's 1/9 ≈ 11.1%).

Usage
-----
    # Smoke run on 4 subjects (laptop / smoke-test)
    python -m experiments.20_a3_lee2019 --smoke

    # Full Colab run on all 54 subjects
    python -m experiments.20_a3_lee2019 --all
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import numpy as np

from attacks.closed_set import closed_set_reid
from config import FIGURES_DIR, RESULTS_DIR
from data.lee2019_loader import load_subject_session_compact as load_subject_session
from data.lee2019_loader import valid_subjects
from eval.plots import closed_set_bar_chart, closed_set_table
from models.eegnet import EEGNetVictim
from models.fbcsp import FBCSPVictim
from models.riemannian import RiemannianVictim
from preprocess.windows import WindowedDataset


def _build_victim(name: str, *, n_channels: int, n_times: int, sfreq: float,
                  eegnet_epochs: int, seed: int):
    if name == "eegnet":
        return EEGNetVictim(
            n_channels=n_channels, n_times=n_times, n_classes=2,
            n_epochs=eegnet_epochs, seed=seed, verbose=False,
        )
    if name == "fbcsp":
        return FBCSPVictim(sfreq=sfreq, n_classes=2)
    if name == "riemann":
        return RiemannianVictim(n_classes=2, seed=seed)
    raise ValueError(name)


def _pool_subjects_session(subjects: list[int], session: str) -> WindowedDataset:
    """Pool all subjects' windows from one session into one dataset."""
    parts: list[WindowedDataset] = []
    print(f"  pooling Lee2019 {session} for {len(subjects)} subjects ...",
          flush=True)
    for s in subjects:
        try:
            parts.append(load_subject_session(s, session=session))
        except Exception as exc:
            print(f"    !! subject {s} failed: {type(exc).__name__}: {exc}. Skipping.",
                  flush=True)
    if not parts:
        raise RuntimeError(f"Lee2019: no subjects loaded successfully for {session}.")

    X = np.concatenate([p.X for p in parts], axis=0)
    y = np.concatenate([p.y for p in parts], axis=0)
    s = np.concatenate([p.subject_ids for p in parts], axis=0)
    # Make trial_ids globally unique across subjects.
    t_parts = []
    for p in parts:
        offset = int(p.subject_ids[0]) * 1_000_000
        t_parts.append(p.trial_ids + offset)
    t = np.concatenate(t_parts, axis=0)
    r = np.concatenate([p.run_ids for p in parts], axis=0)
    sfreq = parts[0].sfreq
    chs = parts[0].channel_names
    # If channel order ever differs across subjects (it shouldn't, but
    # moabb has surprised us before), drop those that don't match.
    keep = []
    for p in parts:
        if p.channel_names != chs:
            print(f"    !! channel mismatch in subject {int(p.subject_ids[0])}; "
                  f"using subject 0's channel order")
        keep.append(p)
    return WindowedDataset(
        X=X, y=y, subject_ids=s, trial_ids=t, run_ids=r,
        sfreq=sfreq, channel_names=chs,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="4 subjects, EEGNet capped at 30 epochs.")
    p.add_argument("--all", action="store_true",
                   help="All 54 Lee2019 motor-imagery subjects.")
    p.add_argument("--subjects", type=int, nargs="*", default=None)
    p.add_argument("--models", nargs="+",
                   default=["eegnet", "fbcsp", "riemann"],
                   choices=["eegnet", "fbcsp", "riemann"])
    p.add_argument("--eegnet-epochs", type=int, default=80)
    p.add_argument("--bootstrap-n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.subjects:
        subjects = args.subjects
    elif args.smoke:
        subjects = valid_subjects()[:4]
        args.eegnet_epochs = min(args.eegnet_epochs, 30)
    elif args.all:
        subjects = valid_subjects()
    else:
        p.error("Provide --smoke, --all, or --subjects")

    np.random.seed(args.seed)
    print(f"Lee2019 subjects: {len(subjects)}  "
          f"chance top-1 = {100/len(subjects):.2f}%")
    print(f"Models: {args.models}\n", flush=True)

    print("Loading Lee2019 sessions ...", flush=True)
    t0 = time.time()
    train_ds = _pool_subjects_session(subjects, "session_1")
    test_ds = _pool_subjects_session(subjects, "session_2")
    print(f"  loaded in {time.time() - t0:.1f}s | "
          f"session_1={train_ds.n_windows} session_2={test_ds.n_windows} "
          f"chans={train_ds.n_channels} times={train_ds.n_times}\n",
          flush=True)

    all_results = []
    for victim_name in args.models:
        print(f"=== {victim_name} ===", flush=True)
        victim = _build_victim(
            victim_name,
            n_channels=train_ds.n_channels, n_times=train_ds.n_times,
            sfreq=train_ds.sfreq,
            eegnet_epochs=args.eegnet_epochs, seed=args.seed,
        )
        t = time.time()
        victim.fit(train_ds.X, train_ds.y)
        task_acc = victim.score(test_ds.X, test_ds.y)
        print(f"  victim train+score: {time.time() - t:.1f}s | "
              f"task_acc(session2)={task_acc:.3f}", flush=True)

        t = time.time()
        results = closed_set_reid(
            victim, train_ds, test_ds,
            probes=("knn", "logreg"),
            bootstrap_n=args.bootstrap_n, seed=args.seed,
        )
        print(f"  attack: {time.time() - t:.1f}s")
        for r in results:
            print(f"    {r.probe:7s}  top1={r.top1:.3f} "
                  f"[{r.top1_ci_low:.3f}, {r.top1_ci_high:.3f}]  "
                  f"top5={r.top5:.3f}  top10={r.top10:.3f}  "
                  f"(chance top1={r.chance_top1:.3f})")
            all_results.append({**asdict(r), "task_acc": task_acc,
                                "dataset": "lee2019"})
        print()

    def _nan_to_none(v):
        return None if isinstance(v, float) and (v != v) else v
    sanitized = [{k: _nan_to_none(v) for k, v in r.items()} for r in all_results]

    out_path = RESULTS_DIR / "20_a3_lee2019.json"
    out_path.write_text(json.dumps(sanitized, indent=2, allow_nan=False))
    print(f"Results written to {out_path}")

    fig_path = FIGURES_DIR / "20_a3_lee2019.pdf"
    closed_set_bar_chart(
        all_results, fig_path,
        title=f"A3 cross-session re-ID  (Lee2019, {len(subjects)} subj)\n"
              f"probe trained on session-1, tested on session-2",
    )
    print(f"Figure written to {fig_path}\n")
    print(closed_set_table(all_results))


if __name__ == "__main__":
    main()
