"""A1 — closed-set subject re-identification across all 3 victim models.

Trains one cross-subject task decoder per victim family on PhysioNet
motor-imagery train_runs, then probes the resulting embeddings for
subject identity. Reports top-1 / top-5 / top-10 attack accuracy with
bootstrap CIs.

Usage
-----
    python -m experiments.02_closed_set_reid --smoke   # 10 subjects, fast
    python -m experiments.02_closed_set_reid --all     # 104 subjects
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import numpy as np

from attacks.closed_set import closed_set_reid
from config import RESULTS_DIR
from data.physionet_loader import valid_subjects
from models.eegnet import EEGNetVictim
from models.fbcsp import FBCSPVictim
from models.riemannian import RiemannianVictim
from preprocess.windows import windowed_subjects

TRAIN_RUNS = (4, 6, 8, 10)
TEST_RUNS = (12, 14)


def _build_victim(name: str, *, n_channels: int, n_times: int, sfreq: float,
                  eegnet_epochs: int, seed: int):
    if name == "eegnet":
        return EEGNetVictim(
            n_channels=n_channels, n_times=n_times, n_classes=4,
            n_epochs=eegnet_epochs, seed=seed, verbose=False,
        )
    if name == "fbcsp":
        return FBCSPVictim(sfreq=sfreq, n_classes=4)
    if name == "riemann":
        return RiemannianVictim(n_classes=4, seed=seed)
    raise ValueError(name)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="10 subjects, faster training. Pipeline-validation run.")
    p.add_argument("--all", action="store_true",
                   help="All 104 valid PhysioNet subjects.")
    p.add_argument("--subjects", type=int, nargs="*", default=None)
    p.add_argument("--models", nargs="+",
                   default=["eegnet", "fbcsp", "riemann"],
                   choices=["eegnet", "fbcsp", "riemann"])
    p.add_argument("--eegnet-epochs", type=int, default=60)
    p.add_argument("--bootstrap-n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.subjects:
        subjects = args.subjects
    elif args.smoke:
        subjects = valid_subjects()[:10]
        args.eegnet_epochs = min(args.eegnet_epochs, 30)
    elif args.all:
        subjects = valid_subjects()
    else:
        p.error("Provide --smoke, --all, or --subjects")

    np.random.seed(args.seed)
    print(f"Subjects: {len(subjects)} (chance top-1 = {100/len(subjects):.2f}%)")
    print(f"Models: {args.models}")
    print(f"Train runs {TRAIN_RUNS}  Test runs {TEST_RUNS}\n")

    print("Loading windowed data ...")
    t0 = time.time()
    full = windowed_subjects(subjects, runs="imagery")
    train_set = full.filter_runs(list(TRAIN_RUNS))
    test_set = full.filter_runs(list(TEST_RUNS))
    print(f"  loaded in {time.time() - t0:.1f}s | "
          f"train={train_set.n_windows} test={test_set.n_windows} "
          f"channels={train_set.n_channels} times={train_set.n_times}\n")

    all_results = []
    for victim_name in args.models:
        print(f"=== {victim_name} ===")
        victim = _build_victim(
            victim_name,
            n_channels=train_set.n_channels,
            n_times=train_set.n_times,
            sfreq=train_set.sfreq,
            eegnet_epochs=args.eegnet_epochs,
            seed=args.seed,
        )
        t0 = time.time()
        # Train on motor-imagery TASK labels (not subject IDs)
        victim.fit(train_set.X, train_set.y)
        task_acc = victim.score(test_set.X, test_set.y)
        print(f"  victim train+score: {time.time() - t0:.1f}s | "
              f"task_acc(test)={task_acc:.3f}")

        t0 = time.time()
        results = closed_set_reid(
            victim, train_set, test_set,
            probes=("knn", "logreg"),
            bootstrap_n=args.bootstrap_n, seed=args.seed,
        )
        print(f"  attack: {time.time() - t0:.1f}s")
        for r in results:
            print(f"    {r.probe:7s}  top1={r.top1:.3f} "
                  f"[{r.top1_ci_low:.3f}, {r.top1_ci_high:.3f}]  "
                  f"top5={r.top5:.3f}  top10={r.top10:.3f}  "
                  f"(chance top1={r.chance_top1:.3f})")
            all_results.append({**asdict(r), "task_acc": task_acc})
        print()

    out_path = RESULTS_DIR / "02_closed_set_reid.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
