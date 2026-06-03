"""DP-SGD architecture ablation.

The D3 result reports re-ID top-1 ≈ 2% at ε=3, vs ≈41% for the no-defense
A1 baseline (AdamW EEGNet with BatchNorm). The naive interpretation is
"DP-SGD provides ~39 pp of privacy". But Opacus forces a BatchNorm →
GroupNorm replacement (BN's batch statistics are not per-sample
gradient compatible) and uses SGD instead of AdamW. We need to
disentangle two contributions:

  - architecture / optimizer:  GroupNorm + SGD (no DP)
  - noise mechanism:           per-sample gradient clipping + Gaussian
                               noise (the actual DP guarantee)

This experiment trains EEGNet with the same architectural surgery and
the same SGD optimizer that DP-SGD uses, but **without** the Opacus
PrivacyEngine — i.e., no per-sample gradient clipping, no Gaussian
noise, infinite ε. We then run A1 closed-set re-ID on its embeddings.

If A1 top-1 here is close to the DP-SGD ε=3 number (~2%), the privacy
came from the architecture; if it's close to the AdamW BatchNorm A1
baseline (~41%), the privacy came from the noise. The truth is almost
certainly between, and the breakdown determines the contribution of
the formal DP mechanism on top of the architectural change.

Outputs:
  results/19_dp_sgd_arch_ablation.json  — A1 top-1 / task acc for the
                                          GN+SGD-no-DP configuration

Pair this with results/10_d3_dp_sgd.json (DP at ε=3) and
results/02_closed_set_reid.json (AdamW + BN baseline) to read the
breakdown.
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
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--n-epochs", type=int, default=40,
                   help="Same as the DP-SGD sweep at ε=3 for fairness.")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1.0,
                   help="Same SGD LR as the DP-SGD sweep.")
    p.add_argument("--bootstrap-n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.smoke:
        subjects = valid_subjects()[:10]
        args.n_epochs = 10
    elif args.all:
        subjects = valid_subjects()
    else:
        p.error("Provide --smoke or --all")

    np.random.seed(args.seed)
    print(f"Subjects: {len(subjects)} (chance top-1 = {100/len(subjects):.2f}%)")
    print(f"Configuration: GroupNorm-EEGNet + SGD (no DP)  "
          f"epochs={args.n_epochs}  batch={args.batch_size}  lr={args.lr}",
          flush=True)

    print("\nLoading windowed data ...", flush=True)
    t0 = time.time()
    full = windowed_subjects(subjects, runs="imagery")
    train = full.filter_runs(list(VICTIM_TRAIN_RUNS))
    test = full.filter_runs(list(VICTIM_TEST_RUNS))
    print(f"  loaded in {time.time() - t0:.1f}s | "
          f"train={train.n_windows} test={test.n_windows} chans={train.n_channels}\n",
          flush=True)

    # The trick: DPSGDVictim with target_epsilon=None gives us the same
    # GroupNorm-EEGNet built via ModuleValidator.fix, but with the
    # PrivacyEngine bypassed. Same SGD optimizer, same hyperparameters
    # except no per-sample gradient clipping and no Gaussian noise.
    print("=== training GroupNorm-EEGNet (SGD, no DP) ===", flush=True)
    victim = DPSGDVictim(
        n_channels=train.n_channels, n_times=train.n_times, n_classes=4,
        n_epochs=args.n_epochs, batch_size=args.batch_size, lr=args.lr,
        target_epsilon=None, target_delta=1e-5,
        max_grad_norm=1.0,
        seed=args.seed, verbose=False,
    )
    t0 = time.time()
    victim.fit(train.X, train.y)
    task_acc = victim.score(test.X, test.y)
    print(f"  victim train+score: {time.time() - t0:.1f}s | "
          f"task_acc(test)={task_acc:.3f}", flush=True)

    t0 = time.time()
    results = closed_set_reid(
        victim, train, test, probes=("knn", "logreg"),
        bootstrap_n=args.bootstrap_n, seed=args.seed,
    )
    print(f"  attack: {time.time() - t0:.1f}s", flush=True)

    serialized = []
    for r in results:
        row = {
            **asdict(r),
            "configuration": "groupnorm_sgd_no_dp",
            "n_epochs": int(args.n_epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "task_acc": float(task_acc),
        }
        serialized.append(row)
        if r.probe == "logreg":
            print(f"    logreg  top1={r.top1:.3f} "
                  f"[{r.top1_ci_low:.3f}, {r.top1_ci_high:.3f}]  "
                  f"task={task_acc:.3f}", flush=True)

    out_path = RESULTS_DIR / "19_dp_sgd_arch_ablation.json"
    out_path.write_text(json.dumps(serialized, indent=2))
    print(f"\nResults written to {out_path}")
    print()
    print("Compare against:")
    print("  results/02_closed_set_reid.json  (AdamW + BatchNorm baseline; ~41%)")
    print("  results/10_d3_dp_sgd.json        (DP-SGD ε=3; ~2%)")
    print("Privacy attributable to architecture = (this top1) − (DP ε=3 top1)")
    print("Privacy attributable to noise        = (AdamW+BN top1) − (this top1)")


if __name__ == "__main__":
    main()

