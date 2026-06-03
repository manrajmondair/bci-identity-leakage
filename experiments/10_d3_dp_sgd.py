"""D3 — DP-SGD via Opacus on EEGNet.

Trains EEGNet under (ε, δ)-differential privacy at three settings and
re-runs the A1 closed-set re-ID attack on each. Defender's claim: any
adversary observing only the trained-model API cannot determine any
single training sample's contribution with likelihood-ratio better
than e^ε.

Sweeps target_epsilon ∈ {None (no DP), 10.0, 3.0} at 40 epochs each.
ε=None recovers vanilla SGD-trained EEGNet (separate from A1's AdamW
trained one — different optimizer, so the baseline is internal).

Designed to fit the 1-hour budget on Colab L4: DP-SGD is ~3-5× slower
than vanilla SGD due to per-sample gradient computation, so 40 epochs
× 3 conditions ≈ 35-45 min training + 10-15 min attacks.
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
from defenses.dp_sgd import DPSGDVictim
from preprocess.windows import windowed_subjects

VICTIM_TRAIN_RUNS = (4, 6, 8, 10)
VICTIM_TEST_RUNS = (12, 14)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="10 subjects, only ε=10, 10 epochs.")
    p.add_argument("--all", action="store_true",
                   help="All 104 PhysioNet subjects.")
    p.add_argument("--epsilons", type=str, nargs="+",
                   default=["none", "10.0", "3.0"],
                   help="Target epsilons. 'none' means no DP.")
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--n-epochs", type=int, default=40)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1.0)
    p.add_argument("--bootstrap-n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.smoke:
        subjects = valid_subjects()[:10]
        args.epsilons = ["10.0"]
        args.n_epochs = 10
    elif args.all:
        subjects = valid_subjects()
    else:
        p.error("Provide --smoke or --all")

    # Parse epsilons
    epsilons: list[float | None] = []
    for s in args.epsilons:
        s = s.lower()
        epsilons.append(None if s in ("none", "inf", "no_dp") else float(s))

    np.random.seed(args.seed)
    print(f"Subjects: {len(subjects)} (chance top-1 = {100/len(subjects):.2f}%)")
    print(f"ε sweep: {epsilons}  δ={args.delta:.0e}  epochs={args.n_epochs}")
    print(f"max_grad_norm={args.max_grad_norm}  batch={args.batch_size}  lr={args.lr}",
          flush=True)

    print("\nLoading windowed data ...", flush=True)
    t0 = time.time()
    full = windowed_subjects(subjects, runs="imagery")
    train = full.filter_runs(list(VICTIM_TRAIN_RUNS))
    test = full.filter_runs(list(VICTIM_TEST_RUNS))
    print(f"  loaded in {time.time() - t0:.1f}s | "
          f"train={train.n_windows} test={test.n_windows} chans={train.n_channels}\n",
          flush=True)

    all_results = []
    for eps in epsilons:
        eps_label = "no_dp" if eps is None else f"eps_{eps}"
        print(f"--- ε = {eps if eps is not None else '∞ (no DP)'} ---", flush=True)
        victim = DPSGDVictim(
            n_channels=train.n_channels, n_times=train.n_times, n_classes=4,
            n_epochs=args.n_epochs, batch_size=args.batch_size, lr=args.lr,
            target_epsilon=eps, target_delta=args.delta,
            max_grad_norm=args.max_grad_norm,
            seed=args.seed, verbose=False,
        )
        t0 = time.time()
        victim.fit(train.X, train.y)
        task_acc = victim.score(test.X, test.y)
        eps_final = victim.final_epsilon_
        print(f"  victim train+score: {time.time() - t0:.1f}s | "
              f"task_acc(test)={task_acc:.3f} | "
              f"final_ε={eps_final:.2f}" if eps_final is not None
              else f"  victim train+score: {time.time() - t0:.1f}s | "
                   f"task_acc(test)={task_acc:.3f} | (no DP)", flush=True)

        t0 = time.time()
        results = closed_set_reid(
            victim, train, test, probes=("knn", "logreg"),
            bootstrap_n=args.bootstrap_n, seed=args.seed,
        )
        print(f"  attack: {time.time() - t0:.1f}s")
        for r in results:
            row = {
                **asdict(r),
                "target_epsilon": eps,
                "final_epsilon": eps_final,
                "delta": float(args.delta) if eps is not None else None,
                "defense": eps_label,
                "task_acc": float(task_acc),
            }
            all_results.append(row)
            if r.probe == "logreg":
                print(f"    logreg  top1={r.top1:.3f} "
                      f"[{r.top1_ci_low:.3f}, {r.top1_ci_high:.3f}]",
                      flush=True)
        print()

    out_path = RESULTS_DIR / "10_d3_dp_sgd.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"Results written to {out_path}\n")

    print("| Defense | ε (target) | ε (final) | Top-1 (logreg) | Task acc |")
    print("|---|---|---|---|---|")
    for r in sorted(all_results, key=lambda x: (x["probe"], x["target_epsilon"] if x["target_epsilon"] else 99999)):
        if r["probe"] != "logreg":
            continue
        eps_t = "—" if r["target_epsilon"] is None else f"{r['target_epsilon']:.1f}"
        eps_f = "—" if r["final_epsilon"] is None else f"{r['final_epsilon']:.2f}"
        ci = f"{r['top1']:.3f} [{r['top1_ci_low']:.3f}, {r['top1_ci_high']:.3f}]"
        print(f"| {r['defense']} | {eps_t} | {eps_f} | {ci} | {r['task_acc']:.3f} |")


if __name__ == "__main__":
    main()

