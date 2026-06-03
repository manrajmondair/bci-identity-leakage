"""A5 — per-subject membership inference on PhysioNet motor imagery.

Trains 20 shadow EEGNets on random 50% subject splits + a separate
target. Asks the trained attack model to predict, for each of the 104
subjects, whether their data was used to train the target.

Reports AUC and the (TPR - FPR) attack advantage. Attack works (i.e.,
membership leaks) iff AUC > 0.5 with non-trivial advantage.

Designed to fit the 1-hour budget on Colab L4: 21 EEGNet trainings ×
30 epochs each at ~1 minute apiece.
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
from data.physionet_loader import valid_subjects
from preprocess.windows import windowed_subjects

VICTIM_TRAIN_RUNS = (4, 6, 8, 10)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="20 subjects, 6 shadows, 15 epochs each.")
    p.add_argument("--all", action="store_true",
                   help="All 104 PhysioNet subjects.")
    p.add_argument("--n-shadows", type=int, default=20)
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
    print(f"Shadows: {args.n_shadows} | epochs/shadow: {args.n_epochs}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n", flush=True)

    print("Loading imagery windows ...", flush=True)
    t0 = time.time()
    full = windowed_subjects(subjects, runs="imagery")
    train = full.filter_runs(list(VICTIM_TRAIN_RUNS))
    print(f"  loaded in {time.time() - t0:.1f}s | "
          f"train_windows={train.n_windows} chans={train.n_channels}\n",
          flush=True)

    t0 = time.time()
    result = membership_inference(
        X_train=train.X, y_train=train.y, subject_ids_train=train.subject_ids,
        # Same windows for evaluation: the attacker has labeled training-set
        # windows for each candidate subject and asks "is this in the model's
        # training pool?" -- the canonical MIA threat model.
        X_eval=train.X, y_eval=train.y, subject_ids_eval=train.subject_ids,
        all_subjects=np.asarray(subjects),
        n_chans=train.n_channels, n_times=train.n_times, n_classes=4,
        n_shadows=args.n_shadows, n_epochs=args.n_epochs,
        member_frac=0.5, seed=args.seed, verbose=True,
    )
    dt = time.time() - t0

    print(f"\n  AUC = {result.auc:.4f} [{result.auc_ci_low:.4f}, {result.auc_ci_high:.4f}]")
    print(f"  Advantage (TPR - FPR) = {result.advantage:.4f} "
          f"@ threshold {result.advantage_threshold:.3f}")
    print(f"  Members={result.n_target_members}  "
          f"Non-members={result.n_target_nonmembers}")
    print(f"  Total seconds: {dt:.1f}")

    out_path = RESULTS_DIR / "08_a5_membership_inference.json"
    out_path.write_text(json.dumps(asdict(result), indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()

