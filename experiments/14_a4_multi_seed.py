"""A4 multi-seed — robustness CI on the AUC headline.

The original A4 reports AUC = 0.925 [0.923, 0.928] with bootstrap CI
over verification PAIRS. That CI tells us about pair-level sampling
noise, not about the variance across train/test subject splits. This
script reports both: 5 different random splits of the 104 subjects into
80 train / 24 held-out, retraining the contrastive embedding from
scratch for each. Across-seed mean ± std on the AUC and EER is the
robustness number.

Designed to fit the 1-hour Colab L4 budget: 5 seeds × 22 epochs each
~ 55 min. Reduced from the original 30 epochs of A4; the per-seed AUC
will be slightly lower than 0.925 but the cross-seed std is what
matters for the robustness claim.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import numpy as np
import torch

from attacks.verification import open_set_verification
from config import RESULTS_DIR
from data.physionet_loader import valid_subjects
from preprocess.windows import windowed_subjects


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="20 subjects (16 train / 4 unseen), 2 seeds, 8 epochs.")
    p.add_argument("--all", action="store_true",
                   help="Full multi-seed eval: 5 seeds × 80 train / 24 held-out.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n-train-subjects", type=int, default=80)
    p.add_argument("--n-epochs", type=int, default=22,
                   help="Reduced from 30 (A4 default) so the 5-seed sweep "
                        "fits ~60 min on L4.")
    p.add_argument("--n-pairs", type=int, default=50_000)
    args = p.parse_args()

    if args.smoke:
        subjects = valid_subjects()[:20]
        args.n_train_subjects = 16
        args.seeds = [0, 1]
        args.n_epochs = 8
    elif args.all:
        subjects = valid_subjects()
    else:
        p.error("Provide --smoke or --all")

    print(f"Total subjects: {len(subjects)}")
    print(f"Seeds: {args.seeds}  (n_train={args.n_train_subjects}, "
          f"n_epochs={args.n_epochs}, pairs={args.n_pairs})", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n", flush=True)

    print("Loading imagery windows once (subjects regrouped per seed) ...",
          flush=True)
    t0 = time.time()
    full = windowed_subjects(subjects, runs="imagery")
    print(f"  loaded in {time.time() - t0:.1f}s | "
          f"windows={full.n_windows}\n", flush=True)

    seed_results: list[dict] = []
    for seed in args.seeds:
        print(f"=== seed = {seed} ===", flush=True)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(np.asarray(subjects))
        train_subjects = sorted(int(s) for s in perm[: args.n_train_subjects])
        test_subjects = sorted(int(s) for s in perm[args.n_train_subjects:])

        train = full.filter_subjects(train_subjects)
        test = full.filter_subjects(test_subjects)

        t0 = time.time()
        result, _scores, _labels = open_set_verification(
            train.X, train.subject_ids,
            test.X, test.subject_ids,
            trial_test=test.trial_ids,
            n_chans=train.n_channels, n_times=train.n_times,
            n_epochs=args.n_epochs, n_pairs=args.n_pairs,
            seed=seed, device=device, verbose=False,
        )
        dt = time.time() - t0
        print(f"  AUC = {result.auc:.4f} "
              f"[{result.auc_ci_low:.4f}, {result.auc_ci_high:.4f}]  "
              f"EER = {result.eer:.4f}  ({dt:.0f}s)", flush=True)

        row = {
            **asdict(result),
            "seed": int(seed),
            "train_subjects_count": len(train_subjects),
            "test_subjects_count": len(test_subjects),
            "wall_seconds": float(dt),
        }
        seed_results.append(row)

    # Cross-seed aggregate
    aucs = np.array([r["auc"] for r in seed_results])
    eers = np.array([r["eer"] for r in seed_results])

    aggregate = {
        "n_seeds": len(seed_results),
        "n_epochs_per_seed": int(args.n_epochs),
        "auc_mean": float(aucs.mean()),
        "auc_std": float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.0,
        "auc_min": float(aucs.min()),
        "auc_max": float(aucs.max()),
        "eer_mean": float(eers.mean()),
        "eer_std": float(eers.std(ddof=1)) if len(eers) > 1 else 0.0,
    }
    print()
    print(f"Cross-seed: AUC = {aggregate['auc_mean']:.4f} ± "
          f"{aggregate['auc_std']:.4f}  "
          f"(min {aggregate['auc_min']:.4f}, max {aggregate['auc_max']:.4f})")
    print(f"Cross-seed: EER = {aggregate['eer_mean']:.4f} ± "
          f"{aggregate['eer_std']:.4f}", flush=True)

    out = {"per_seed": seed_results, "aggregate": aggregate}
    out_path = RESULTS_DIR / "14_a4_multi_seed.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()

