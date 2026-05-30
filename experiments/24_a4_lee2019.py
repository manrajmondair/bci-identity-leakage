"""A4 — open-set verification on Lee 2019 OpenBMI (held-out unseen subjects).

Second-corpus replication of the headline A4 result. PhysioNet's
experiment 06 trains a contrastive EEGNet on 80 of 104 subjects and
verifies identity on the 24 unseen subjects at AUC = 0.925. This script
re-runs the same protocol on Lee 2019 motor imagery (54 subjects total)
with a 40 train / 14 held-out split — if AUC on the unseen Lee 2019
subjects is in the same range, the "EEG functions as a biometric
template for arbitrary individuals" claim is no longer
PhysioNet-specific.

Within-session evaluation by default (probe and target both come from
session_1). The `--cross-session` flag tests the harder case: train on
session_1 of the training subjects, verify on session_2 of the
held-out subjects (different recording day).

Reads from the compact cache produced by `data.lee2019_prefetch`. Set
the BCI_LEE2019_CACHE env var to the cache root.

Usage
-----
    python -m experiments.24_a4_lee2019 --smoke
    python -m experiments.24_a4_lee2019 --all
    python -m experiments.24_a4_lee2019 --all --cross-session
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import numpy as np
import torch

from attacks.verification import open_set_verification
from config import FIGURES_DIR, RESULTS_DIR
from data.lee2019_loader import (
    load_subject_session_compact,
    valid_subjects,
)
from eval.plots import verification_panel
from preprocess.windows import WindowedDataset


def _pool(subjects: list[int], session: str) -> WindowedDataset:
    parts: list[WindowedDataset] = []
    for s in subjects:
        try:
            parts.append(load_subject_session_compact(s, session=session))
        except Exception as exc:
            print(f"    !! subj{s} {session} failed: {type(exc).__name__}: {exc}",
                  flush=True)
    if not parts:
        raise RuntimeError(f"no Lee 2019 subjects loaded for {session}")
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
                   help="20 subjects (16 train / 4 held-out), 10 epochs")
    p.add_argument("--all", action="store_true",
                   help="All 54 Lee 2019 subjects (40 train / 14 held-out)")
    p.add_argument("--n-train-subjects", type=int, default=40)
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--n-pairs", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cross-session", action="store_true",
                   help="Train on session_1 of training subjects; "
                        "verify on session_2 of held-out subjects")
    args = p.parse_args()

    if args.smoke:
        subjects = valid_subjects()[:20]
        args.n_train_subjects = 16
        args.n_epochs = 10
    elif args.all:
        subjects = valid_subjects()
    else:
        p.error("Provide --smoke or --all")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(np.asarray(subjects))
    train_subjects = sorted(int(s) for s in perm[: args.n_train_subjects])
    test_subjects = sorted(int(s) for s in perm[args.n_train_subjects:])

    print(f"Total Lee 2019 subjects: {len(subjects)}")
    print(f"Train: {len(train_subjects)}  Held-out (unseen): {len(test_subjects)}")
    print(f"Epochs: {args.n_epochs}  Pairs sampled: {args.n_pairs}")
    print(f"Cross-session verification: {args.cross_session}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n", flush=True)

    print("Loading Lee 2019 windows ...", flush=True)
    t0 = time.time()
    train_ds = _pool(train_subjects, "session_1")
    eval_session = "session_2" if args.cross_session else "session_1"
    test_ds = _pool(test_subjects, eval_session)
    print(f"  loaded in {time.time() - t0:.1f}s | "
          f"train_windows={train_ds.n_windows} test_windows={test_ds.n_windows} "
          f"chans={train_ds.n_channels} times={train_ds.n_times}\n", flush=True)

    t0 = time.time()
    result, scores, labels = open_set_verification(
        train_ds.X, train_ds.subject_ids,
        test_ds.X, test_ds.subject_ids,
        trial_test=test_ds.trial_ids,
        n_chans=train_ds.n_channels, n_times=train_ds.n_times,
        n_epochs=args.n_epochs, n_pairs=args.n_pairs,
        seed=args.seed, device=device, verbose=True,
    )
    dt = time.time() - t0
    print(f"\n  AUC = {result.auc:.4f} "
          f"[{result.auc_ci_low:.4f}, {result.auc_ci_high:.4f}]")
    print(f"  EER = {result.eer:.4f}  (threshold = {result.eer_threshold:.4f})")
    print(f"  total seconds: {dt:.1f}")

    tag = "cross_session" if args.cross_session else "within_session"
    out_path = RESULTS_DIR / f"24_a4_lee2019_{tag}.json"
    out = {**asdict(result),
           "dataset": "lee2019",
           "eval_session": eval_session,
           "cross_session": args.cross_session,
           "train_subjects": train_subjects,
           "test_subjects": test_subjects,
           "seed": args.seed}
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults written to {out_path}")

    scores_path = RESULTS_DIR / f"24_a4_lee2019_{tag}_scores.npz"
    np.savez(scores_path, scores=scores, labels=labels)
    print(f"Per-pair scores written to {scores_path}")

    fig_path = FIGURES_DIR / f"24_a4_lee2019_{tag}.pdf"
    verification_panel(
        scores=scores, labels=labels,
        auc=result.auc, eer=result.eer, out_path=fig_path,
        title=(f"A4 open-set verification on Lee 2019 "
               f"({result.n_test_subjects} unseen subjects, "
               f"{result.n_pairs:,} pairs, {tag.replace('_', '-')})"),
    )
    print(f"Figure written to {fig_path}")


if __name__ == "__main__":
    main()
