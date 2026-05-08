"""W5.1 EEGNet age-effect replication across seeds.

Experiment 17 reported a Mann-Whitney p=0.044 for "EEGNet attack accuracy
in the youngest age tertile vs the oldest age tertile" on a single
training seed (seed=0). p=0.044 is right at the α=0.05 boundary; with
multiple comparisons or a different random initialization, this could
easily flip. Honest reporting requires replication across seeds.

Pipeline (mirrors experiments/17_subgroup_fairness_eegnet but loops):
    For seed ∈ {0, 1, 2, 3, 4}:
        Train EEGNet on imagery train_runs (4, 6, 8, 10).
        Per-subject A1 attack accuracy on the test_runs (12, 14).
        Compute Mann-Whitney p for {age_low, sex_diff} vs {age_high, F}.
    Report per-seed p-values, mean ± std of the age and sex effect
    sizes, and Fisher's combined p across seeds.

Outputs: results/22_eegnet_age_seeds.json with per-seed details + the
across-seed aggregate. The interpretation is:
  - if all 5 seeds give p < 0.05 → robust effect
  - if median p < 0.05 but variance is wide → suggestive, hedge
  - if p flips above 0.05 in ≥ 1 seed → not robust at α=0.05; report
    as "borderline, not replicated"
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import scipy.stats
from sklearn.linear_model import LogisticRegression

from config import RESULTS_DIR, ROOT
from data.physionet_loader import valid_subjects
from models.eegnet import EEGNetVictim
from preprocess.windows import windowed_subjects

VICTIM_TRAIN_RUNS = (4, 6, 8, 10)
VICTIM_TEST_RUNS = (12, 14)
PARTICIPANTS_TSV = ROOT / "data" / "external" / "openneuro_ds004362_participants.tsv"


def load_demographics() -> dict[int, dict]:
    out: dict[int, dict] = {}
    with PARTICIPANTS_TSV.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sid_str = row["participant_id"]
            try:
                sid = int(sid_str.replace("sub-", ""))
            except ValueError:
                continue
            sex = row["Gender"] if row["Gender"] in ("M", "F") else None
            try:
                age = int(row["Age"])
            except (TypeError, ValueError):
                age = None
            out[sid] = {"sex": sex, "age": age}
    return out


def _per_seed_eegnet(train, test, *, seed: int, n_epochs: int):
    victim = EEGNetVictim(
        n_channels=train.n_channels, n_times=train.n_times, n_classes=4,
        n_epochs=n_epochs, seed=seed, verbose=False,
    )
    victim.fit(train.X, train.y)
    task_acc = float(victim.score(test.X, test.y))
    Z_train = victim.embed(train.X)
    Z_test = victim.embed(test.X)
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0).fit(
        Z_train, train.subject_ids,
    )
    preds = clf.predict(Z_test)
    return task_acc, preds


def _per_seed_subgroup(test, preds, demo, *, seed: int):
    correct = (preds == test.subject_ids).astype(np.float64)
    rows = []
    for s in np.unique(test.subject_ids):
        mask = test.subject_ids == s
        d = demo.get(int(s), {})
        rows.append({
            "subject_id": int(s),
            "attack_acc": float(correct[mask].mean()),
            "sex": d.get("sex"),
            "age": d.get("age"),
        })

    male = np.array([r["attack_acc"] for r in rows if r["sex"] == "M"])
    female = np.array([r["attack_acc"] for r in rows if r["sex"] == "F"])
    ages_known = sorted(r["age"] for r in rows if r["age"] is not None)
    low_cut = float(np.percentile(ages_known, 100 / 3)) if ages_known else 0.0
    high_cut = float(np.percentile(ages_known, 200 / 3)) if ages_known else 0.0
    age_low = np.array([r["attack_acc"] for r in rows
                        if r["age"] is not None and r["age"] <= low_cut])
    age_high = np.array([r["attack_acc"] for r in rows
                         if r["age"] is not None and r["age"] > high_cut])

    sex_p = (float(scipy.stats.mannwhitneyu(male, female,
                                            alternative="two-sided").pvalue)
             if len(male) and len(female) else None)
    age_p = (float(scipy.stats.mannwhitneyu(age_low, age_high,
                                            alternative="two-sided").pvalue)
             if len(age_low) and len(age_high) else None)

    attack = np.array([r["attack_acc"] for r in rows])
    sorted_rows = sorted(rows, key=lambda r: r["attack_acc"])
    n_decile = max(1, len(sorted_rows) // 10)
    most = np.array([r["attack_acc"] for r in sorted_rows[-n_decile:]])
    least = np.array([r["attack_acc"] for r in sorted_rows[:n_decile]])

    return {
        "seed": int(seed),
        "n_subjects": int(len(rows)),
        "attack_acc_mean": float(attack.mean()),
        "attack_acc_std": float(attack.std()),
        "decile_gap": float(most.mean() - least.mean()),
        "sex_M_mean": float(male.mean()) if len(male) else None,
        "sex_F_mean": float(female.mean()) if len(female) else None,
        "sex_diff_M_minus_F": (float(male.mean() - female.mean())
                                if len(male) and len(female) else None),
        "sex_p": sex_p,
        "age_low_mean": float(age_low.mean()) if len(age_low) else None,
        "age_high_mean": float(age_high.mean()) if len(age_high) else None,
        "age_diff_low_minus_high": (float(age_low.mean() - age_high.mean())
                                     if len(age_low) and len(age_high) else None),
        "age_p": age_p,
        "age_low_n": int(len(age_low)),
        "age_high_n": int(len(age_high)),
        "low_cut": low_cut,
        "high_cut": high_cut,
    }


def _fisher_combined(p_values: list[float]) -> float:
    """Fisher's method for combining independent p-values."""
    arr = np.array([p for p in p_values if p is not None and p > 0],
                   dtype=np.float64)
    if len(arr) == 0:
        return float("nan")
    chi2 = -2.0 * np.sum(np.log(arr))
    df = 2 * len(arr)
    return float(scipy.stats.chi2.sf(chi2, df))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="10 subjects, 2 seeds, 30 epochs.")
    p.add_argument("--all", action="store_true",
                   help="All 104 PhysioNet subjects.")
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[0, 1, 2, 3, 4])
    p.add_argument("--n-epochs", type=int, default=80)
    args = p.parse_args()

    if args.smoke:
        subjects = valid_subjects()[:10]
        args.seeds = [0, 1]
        args.n_epochs = 30
    elif args.all:
        subjects = valid_subjects()
    else:
        p.error("Provide --smoke or --all")

    print(f"Subjects: {len(subjects)}  seeds: {args.seeds}  "
          f"epochs: {args.n_epochs}\n", flush=True)
    demo = load_demographics()

    print("Loading imagery windows (cached) ...", flush=True)
    t0 = time.time()
    full = windowed_subjects(subjects, runs="imagery")
    train = full.filter_runs(list(VICTIM_TRAIN_RUNS))
    test = full.filter_runs(list(VICTIM_TEST_RUNS))
    print(f"  loaded in {time.time() - t0:.1f}s | train={train.n_windows} "
          f"test={test.n_windows}\n", flush=True)

    per_seed = []
    for seed in args.seeds:
        print(f"=== seed {seed} ===", flush=True)
        t0 = time.time()
        task_acc, preds = _per_seed_eegnet(train, test, seed=seed,
                                           n_epochs=args.n_epochs)
        row = _per_seed_subgroup(test, preds, demo, seed=seed)
        row["task_acc"] = task_acc
        row["wall_seconds"] = float(time.time() - t0)
        per_seed.append(row)
        ap = "n/a" if row["age_p"] is None else f"{row['age_p']:.3f}"
        sp = "n/a" if row["sex_p"] is None else f"{row['sex_p']:.3f}"
        print(f"  task={task_acc:.3f}  attack_mean={row['attack_acc_mean']:.3f}"
              f"  age_p={ap}  sex_p={sp}"
              f"  ({row['wall_seconds']:.0f}s)", flush=True)

    age_ps = [r["age_p"] for r in per_seed]
    sex_ps = [r["sex_p"] for r in per_seed]
    age_diffs = [r["age_diff_low_minus_high"] for r in per_seed]
    sex_diffs = [r["sex_diff_M_minus_F"] for r in per_seed]
    decile_gaps = [r["decile_gap"] for r in per_seed]

    aggregate = {
        "n_seeds": len(args.seeds),
        "age_p_per_seed": age_ps,
        "sex_p_per_seed": sex_ps,
        "age_p_median": float(np.nanmedian([p for p in age_ps if p is not None])),
        "age_p_max": float(np.nanmax([p for p in age_ps if p is not None])),
        "sex_p_median": float(np.nanmedian([p for p in sex_ps if p is not None])),
        "fisher_age_p": _fisher_combined(age_ps),
        "fisher_sex_p": _fisher_combined(sex_ps),
        "age_diff_mean": float(np.nanmean([d for d in age_diffs if d is not None])),
        "age_diff_std": float(np.nanstd([d for d in age_diffs if d is not None])),
        "sex_diff_mean": float(np.nanmean([d for d in sex_diffs if d is not None])),
        "sex_diff_std": float(np.nanstd([d for d in sex_diffs if d is not None])),
        "decile_gap_mean": float(np.nanmean(decile_gaps)),
        "decile_gap_std": float(np.nanstd(decile_gaps)),
        "n_seeds_age_p_below_05": int(sum(1 for p in age_ps
                                          if p is not None and p < 0.05)),
        "n_seeds_sex_p_below_05": int(sum(1 for p in sex_ps
                                          if p is not None and p < 0.05)),
    }

    out = {"per_seed": per_seed, "aggregate": aggregate}
    out_path = RESULTS_DIR / "22_eegnet_age_seeds.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nResults written to {out_path}")
    print()
    print(f"Across {aggregate['n_seeds']} seeds:")
    print(f"  age effect: Δ={aggregate['age_diff_mean']:+.3f} ± "
          f"{aggregate['age_diff_std']:.3f}, "
          f"Fisher p={aggregate['fisher_age_p']:.4f}, "
          f"{aggregate['n_seeds_age_p_below_05']}/{aggregate['n_seeds']} seeds p<0.05")
    print(f"  sex effect: Δ={aggregate['sex_diff_mean']:+.3f} ± "
          f"{aggregate['sex_diff_std']:.3f}, "
          f"Fisher p={aggregate['fisher_sex_p']:.4f}, "
          f"{aggregate['n_seeds_sex_p_below_05']}/{aggregate['n_seeds']} seeds p<0.05")


if __name__ == "__main__":
    main()
