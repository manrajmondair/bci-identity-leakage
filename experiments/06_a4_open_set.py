"""A4 — open-set verification on UNSEEN subjects (the headline novel attack).

Splits PhysioNet's 104 valid subjects into ~80 training / ~24 held-out.
Trains a contrastive EEG embedding (EEGNet backbone + projection head,
batch-hard triplet loss, subject ids as supervision) on the training
subjects' imagery windows. Then on the held-out subjects — whom the
network has NEVER seen during training — it evaluates whether two EEG
windows can be linked as same-vs-different person.

This is the strongest evidence that EEG functions as a biometric template:
the embedding must generalize across people. AUC = 0.5 means no identity
signal; AUC -> 1.0 means perfect verification on novel users.

Usage
-----
    python -m experiments.06_a4_open_set --smoke   # 20 subj, 16 train / 4 test
    python -m experiments.06_a4_open_set --all     # 104 subj, 80 train / 24 test
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
from data.physionet_loader import valid_subjects
from eval.plots import verification_panel
from preprocess.windows import windowed_subjects


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="20 subjects (16 train / 4 held-out), 10 epochs.")
    p.add_argument("--all", action="store_true",
                   help="All 104 PhysioNet subjects (80 train / 24 held-out).")
    p.add_argument("--n-train-subjects", type=int, default=80)
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--n-pairs", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=0)
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

    print(f"Total subjects: {len(subjects)}")
    print(f"Train: {len(train_subjects)} | Held-out (unseen): {len(test_subjects)}")
    print(f"Epochs: {args.n_epochs} | Pairs sampled: {args.n_pairs}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n", flush=True)

    print("Loading imagery windows ...", flush=True)
    t0 = time.time()
    train_ds = windowed_subjects(train_subjects, runs="imagery")
    test_ds = windowed_subjects(test_subjects, runs="imagery")
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

    print(f"\n  AUC = {result.auc:.4f} [{result.auc_ci_low:.4f}, {result.auc_ci_high:.4f}]")
    print(f"  EER = {result.eer:.4f}  (threshold = {result.eer_threshold:.4f})")
    print(f"  total seconds: {dt:.1f}")

    out_path = RESULTS_DIR / "06_a4_open_set.json"
    out = {
        **asdict(result),
        "train_subjects": train_subjects,
        "test_subjects": test_subjects,
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults written to {out_path}")

    # Also dump per-pair scores so the rich verification figure can be
    # regenerated locally without re-running the experiment.
    scores_path = RESULTS_DIR / "06_a4_open_set_scores.npz"
    np.savez(scores_path, scores=scores, labels=labels)
    print(f"Per-pair scores written to {scores_path}")

    fig_path = FIGURES_DIR / "06_a4_open_set.pdf"
    verification_panel(
        scores=scores, labels=labels,
        auc=result.auc, eer=result.eer, out_path=fig_path,
        title=(f"A4 open-set verification on PhysioNet "
               f"({result.n_test_subjects} unseen subjects, "
               f"{result.n_pairs:,} pairs)"),
    )
    print(f"Figure written to {fig_path}")


if __name__ == "__main__":
    main()

