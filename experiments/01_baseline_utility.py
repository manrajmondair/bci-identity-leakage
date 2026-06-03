"""Baseline BCI-decoder accuracy.

Trains each of the three victim models (EEGNet, FBCSP+LDA, Riemannian
tangent-space LR) on PhysioNet motor-imagery for one or many subjects,
using a within-subject train/test split that holds out the *last two
imagery runs* (runs 12 and 14) as test. This is the standard BCI eval
protocol — within-subject, run-level holdout — and it's the same split
the closed-set re-identification attack will later use, so the task and
attack scores are directly comparable.

Usage
-----
    # smoke test on subject 1 only — should finish in < 2 min
    python -m experiments.01_baseline_utility --smoke

    # all valid subjects, all three models
    python -m experiments.01_baseline_utility --all
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import numpy as np

from config import RESULTS_DIR
from data.physionet_loader import valid_subjects
from models.eegnet import EEGNetVictim
from models.fbcsp import FBCSPVictim
from models.riemannian import RiemannianVictim
from preprocess.windows import windowed_subject

# Train on runs {4, 6, 8, 10}, test on the held-out imagery runs {12, 14}.
# This is the standard within-subject / cross-run BCI evaluation.
TRAIN_RUNS = (4, 6, 8, 10)
TEST_RUNS = (12, 14)


@dataclass
class SubjectResult:
    subject_id: int
    model: str
    n_train: int
    n_test: int
    n_classes: int
    accuracy: float
    seconds: float


def _train_eval_one(
    subject_id: int,
    model_name: str,
    *,
    eegnet_epochs: int,
    seed: int,
    verbose: bool,
) -> SubjectResult:
    ds = windowed_subject(subject_id, runs="imagery")
    train = ds.filter_runs(list(TRAIN_RUNS))
    test = ds.filter_runs(list(TEST_RUNS))

    if model_name == "eegnet":
        model = EEGNetVictim(
            n_channels=ds.n_channels, n_times=ds.n_times,
            n_classes=4, n_epochs=eegnet_epochs, seed=seed, verbose=verbose,
        )
    elif model_name == "fbcsp":
        model = FBCSPVictim(sfreq=ds.sfreq, n_classes=4)
    elif model_name == "riemann":
        model = RiemannianVictim(n_classes=4, seed=seed)
    else:
        raise ValueError(model_name)

    t0 = time.time()
    model.fit(train.X, train.y)
    acc = model.score(test.X, test.y)
    dt = time.time() - t0

    return SubjectResult(
        subject_id=subject_id,
        model=model_name,
        n_train=int(train.n_windows),
        n_test=int(test.n_windows),
        n_classes=4,
        accuracy=float(acc),
        seconds=dt,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="Subject 1 only; minimal training; sanity-check the pipeline.")
    p.add_argument("--all", action="store_true",
                   help="All 104 valid PhysioNet subjects.")
    p.add_argument("--subjects", type=int, nargs="*", default=None,
                   help="Explicit subject list (overrides --smoke / --all).")
    p.add_argument("--models", nargs="+",
                   default=["eegnet", "fbcsp", "riemann"],
                   choices=["eegnet", "fbcsp", "riemann"])
    p.add_argument("--eegnet-epochs", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.subjects:
        subjects = args.subjects
    elif args.smoke:
        subjects = [1]
        args.eegnet_epochs = min(args.eegnet_epochs, 40)
    elif args.all:
        subjects = valid_subjects()
    else:
        p.error("Provide --smoke, --all, or --subjects")

    np.random.seed(args.seed)

    print(f"Subjects: {len(subjects)} | Models: {args.models}")
    print(f"Train runs {TRAIN_RUNS} | Test runs {TEST_RUNS}\n")

    results: list[SubjectResult] = []
    for s in subjects:
        for m in args.models:
            r = _train_eval_one(
                s, m,
                eegnet_epochs=args.eegnet_epochs,
                seed=args.seed,
                verbose=args.verbose,
            )
            results.append(r)
            print(f"  S{s:03d} | {m:8s} | acc={r.accuracy:.3f} "
                  f"| n_train={r.n_train} n_test={r.n_test} | {r.seconds:.1f}s")

    # Per-model summary
    print("\nPer-model mean accuracy:")
    for m in args.models:
        accs = [r.accuracy for r in results if r.model == m]
        print(f"  {m:8s}  mean={np.mean(accs):.3f}  std={np.std(accs):.3f}  "
              f"n={len(accs)}")

    # Persist for later analyses
    out_path = RESULTS_DIR / "01_baseline_utility.json"
    out_path.write_text(json.dumps([asdict(r) for r in results], indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()

