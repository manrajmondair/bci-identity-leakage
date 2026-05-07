"""D2 — DANN adversarial subject-invariant training.

Train EEGNet with two heads: task (motor imagery, 4-way) and subject
(104-way) connected to the encoder via a Gradient Reversal Layer.
λ controls how strongly the encoder is pressured to be subject-invariant.

Sweeps λ ∈ {0, 0.1, 0.5, 1.0} on PhysioNet motor imagery. λ=0 reduces to
vanilla EEGNet (= the A1 EEGNet baseline). For each λ, runs the same A1
closed-set re-ID attack on the trained encoder and reports task accuracy
+ re-ID top-1 with bootstrap CIs.

Designed to fit the 1-hour budget on Colab L4: 4 λ-values × ~10 min
training each (DANN's two heads add ~50% over plain EEGNet) + ~3 min
attack each = ~55 min total.
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
from defenses.dann import DANNVictim
from preprocess.windows import windowed_subjects

VICTIM_TRAIN_RUNS = (4, 6, 8, 10)
VICTIM_TEST_RUNS = (12, 14)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="10 subjects, 1 lambda (0.5), 15 epochs.")
    p.add_argument("--all", action="store_true",
                   help="All 104 PhysioNet subjects.")
    p.add_argument("--lambdas", type=float, nargs="+",
                   default=[0.0, 0.1, 0.5, 1.0])
    p.add_argument("--n-epochs", type=int, default=50,
                   help="Reduced from 80 to fit the 4-lambda sweep in 1 hour.")
    p.add_argument("--bootstrap-n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.smoke:
        subjects = valid_subjects()[:10]
        args.lambdas = [0.0, 0.5]
        args.n_epochs = 15
    elif args.all:
        subjects = valid_subjects()
    else:
        p.error("Provide --smoke or --all")

    np.random.seed(args.seed)
    print(f"Subjects: {len(subjects)} (chance top-1 = {100/len(subjects):.2f}%)")
    print(f"λ sweep: {args.lambdas} | epochs/condition: {args.n_epochs}\n",
          flush=True)

    print("Loading windowed data ...", flush=True)
    t0 = time.time()
    full = windowed_subjects(subjects, runs="imagery")
    train = full.filter_runs(list(VICTIM_TRAIN_RUNS))
    test = full.filter_runs(list(VICTIM_TEST_RUNS))
    print(f"  loaded in {time.time() - t0:.1f}s | "
          f"train={train.n_windows} test={test.n_windows} chans={train.n_channels}\n",
          flush=True)

    all_results = []
    for lam in args.lambdas:
        print(f"--- λ = {lam} ---", flush=True)
        victim = DANNVictim(
            n_channels=train.n_channels, n_times=train.n_times,
            n_classes_task=4, n_epochs=args.n_epochs,
            lambda_=float(lam), seed=args.seed, verbose=False,
        )
        t0 = time.time()
        victim.fit(train.X, train.y, subject_ids=train.subject_ids)
        task_acc = victim.score(test.X, test.y)
        print(f"  victim train+score: {time.time() - t0:.1f}s | "
              f"task_acc(test)={task_acc:.3f}", flush=True)

        t0 = time.time()
        results = closed_set_reid(
            victim, train, test,
            probes=("knn", "logreg"),
            bootstrap_n=args.bootstrap_n, seed=args.seed,
        )
        print(f"  attack: {time.time() - t0:.1f}s")
        for r in results:
            row = {**asdict(r), "lambda": float(lam),
                   "defense": f"dann_lam{lam}", "task_acc": float(task_acc)}
            all_results.append(row)
            if r.probe == "logreg":
                print(f"    logreg  top1={r.top1:.3f} "
                      f"[{r.top1_ci_low:.3f}, {r.top1_ci_high:.3f}]",
                      flush=True)
        print()

    out_path = RESULTS_DIR / "09_d2_dann.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"Results written to {out_path}\n")

    # Quick text summary
    print("| Victim | λ | Top-1 (logreg, 95% CI) | Task acc | Chance |")
    print("|---|---|---|---|---|")
    for r in sorted(all_results, key=lambda x: (x["probe"], x["lambda"])):
        if r["probe"] != "logreg":
            continue
        ci = f"{r['top1']:.3f} [{r['top1_ci_low']:.3f}, {r['top1_ci_high']:.3f}]"
        print(f"| {r['victim']} | {r['lambda']} | {ci} | "
              f"{r['task_acc']:.3f} | {r['chance_top1']:.3f} |")


if __name__ == "__main__":
    main()
