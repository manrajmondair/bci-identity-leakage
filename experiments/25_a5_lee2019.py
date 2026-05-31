"""A5 — membership inference on Lee 2019 OpenBMI motor imagery.

Second-corpus replication of the headline A5 result. PhysioNet's
experiment 08 reports MI AUC = 0.845 on EEGNet shadow models trained on
104 subjects. This script runs the same Shokri-style shadow methodology
on Lee 2019 (54 subjects, binary left/right hand, ~600 windows/subject).

Each shadow trains on a random 50% subject split (27 of 54). Attack
features are (mean per-window cross-entropy loss, mean max-softmax)
computed on every subject's session_1 windows. Logistic-regression
attack classifier on the shadow rows; evaluation on a held-out target
EEGNet trained on its own random 50% split.

Reads from the compact cache produced by `data.lee2019_prefetch`. Set
BCI_LEE2019_CACHE to the cache root.

Usage
-----
    python -m experiments.25_a5_lee2019 --smoke
    python -m experiments.25_a5_lee2019 --all
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import numpy as np
import torch

from attacks.membership_inference import membership_inference
from config import RESULTS_DIR
from data.lee2019_loader import (
    load_subject_session_compact,
    valid_subjects,
)
from preprocess.windows import WindowedDataset


def _pool(subjects: list[int], session: str = "session_1") -> WindowedDataset:
    parts: list[WindowedDataset] = []
    for s in subjects:
        try:
            parts.append(load_subject_session_compact(s, session=session))
        except Exception as exc:
            print(f"    !! subj{s} {session} failed: {type(exc).__name__}: {exc}",
                  flush=True)
    if not parts:
        raise RuntimeError("no Lee 2019 subjects loaded")
    X = np.concatenate([p.X for p in parts], axis=0)
    y = np.concatenate([p.y for p in parts], axis=0)
    s = np.concatenate([p.subject_ids for p in parts], axis=0)
    t_parts = []
    for p in parts:
        offset = int(p.subject_ids[0]) * 1_000_000
        t_parts.append(p.trial_ids + offset)
    t = np.concatenate(t_parts, axis=0)
    r = np.concatenate([p.run_ids for p in parts], axis=0)
    return WindowedDataset(
        X=X, y=y, subject_ids=s, trial_ids=t, run_ids=r,
        sfreq=parts[0].sfreq, channel_names=parts[0].channel_names,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="20 subjects, 6 shadows, 15 epochs each")
    p.add_argument("--all", action="store_true",
                   help="All 54 Lee 2019 subjects")
    p.add_argument("--n-shadows", type=int, default=12)
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.smoke:
        subjects = valid_subjects()[:20]
        args.n_shadows = 6
        args.n_epochs = 15
    elif args.all:
        subjects = valid_subjects()
    else:
        p.error("Provide --smoke or --all")

    print(f"Subjects: {len(subjects)}")
    print(f"Shadows: {args.n_shadows}  Epochs/shadow: {args.n_epochs}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n", flush=True)

    print("Loading Lee 2019 session_1 windows ...", flush=True)
    t0 = time.time()
    full = _pool(subjects, "session_1")
    print(f"  loaded in {time.time() - t0:.1f}s | windows={full.n_windows} "
          f"chans={full.n_channels} times={full.n_times}\n", flush=True)

    t0 = time.time()
    result = membership_inference(
        X_train=full.X, y_train=full.y, subject_ids_train=full.subject_ids,
        X_eval=full.X, y_eval=full.y, subject_ids_eval=full.subject_ids,
        all_subjects=np.asarray(subjects),
        n_chans=full.n_channels, n_times=full.n_times, n_classes=2,
        n_shadows=args.n_shadows, n_epochs=args.n_epochs,
        member_frac=0.5, seed=args.seed, verbose=True,
    )
    dt = time.time() - t0

    print(f"\n  AUC = {result.auc:.4f} "
          f"[{result.auc_ci_low:.4f}, {result.auc_ci_high:.4f}]")
    print(f"  Advantage (TPR-FPR) = {result.advantage:.4f} "
          f"@ threshold {result.advantage_threshold:.3f}")
    print(f"  Members={result.n_target_members}  "
          f"Non-members={result.n_target_nonmembers}")
    print(f"  Total seconds: {dt:.1f}")

    out_path = RESULTS_DIR / "25_a5_lee2019.json"
    out = {**asdict(result), "dataset": "lee2019", "seed": args.seed}
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
