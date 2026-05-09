"""W5.1 subgroup fairness on EEGNet — completes the demographic table.

The local-Mac fairness tool covers FBCSP+LDA and Riemann tangent-space.
EEGNet needs a GPU pass (80 epochs cross-subject training), so we ship
that case here as a Colab notebook backed by this script.

Same protocol as `tools/subgroup_fairness.py`:
  1. Train cross-subject EEGNet (A1 baseline, 80 epochs) with input_scale=1e6.
  2. Per-subject A1 attack accuracy via the embedding-+-logreg probe.
  3. Demographic stratification (M vs F, age tertile) using OpenNeuro
     ds004362 demographics on the 104-subject analysis cohort.
  4. Per-subject heterogeneity (decile gap, distribution stats).

Output: results/17_subgroup_fairness_eegnet.json — same schema as
results/12_subgroup_fairness.json so the report can be compared head-to-
head with the classical pipelines.
"""
from __future__ import annotations

import argparse
import csv
import json
import time

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


def _bootstrap_mean_ci(values, *, n_resamples=1000, seed=0):
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, len(values), size=len(values))
        samples[i] = values[idx].mean()
    return (float(values.mean()),
            float(np.percentile(samples, 2.5)),
            float(np.percentile(samples, 97.5)))


def _bootstrap_diff_ci(a, b, *, n_resamples=1000, seed=0):
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        ai = rng.integers(0, len(a), size=len(a))
        bi = rng.integers(0, len(b), size=len(b))
        samples[i] = a[ai].mean() - b[bi].mean()
    return (float(a.mean() - b.mean()),
            float(np.percentile(samples, 2.5)),
            float(np.percentile(samples, 97.5)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="10 subjects, 30 epochs.")
    p.add_argument("--all", action="store_true")
    p.add_argument("--n-epochs", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.smoke:
        subjects = valid_subjects()[:10]
        args.n_epochs = 30
    elif args.all:
        subjects = valid_subjects()
    else:
        p.error("Provide --smoke or --all")

    print(f"Subjects: {len(subjects)}", flush=True)
    demo = load_demographics()
    sex_counts = {"M": 0, "F": 0, "unknown": 0}
    for s in subjects:
        d = demo.get(s, {})
        sex_counts[d.get("sex") or "unknown"] += 1
    print(f"  sex distribution: {sex_counts}", flush=True)

    print("\nLoading imagery windows ...", flush=True)
    full = windowed_subjects(subjects, runs="imagery")
    train = full.filter_runs(list(VICTIM_TRAIN_RUNS))
    test = full.filter_runs(list(VICTIM_TEST_RUNS))
    print(f"  train={train.n_windows} test={test.n_windows}\n", flush=True)

    print(f"Training cross-subject EEGNet ({args.n_epochs} epochs) ...",
          flush=True)
    t0 = time.time()
    victim = EEGNetVictim(
        n_channels=train.n_channels, n_times=train.n_times, n_classes=4,
        n_epochs=args.n_epochs, seed=args.seed, verbose=False,
    )
    victim.fit(train.X, train.y)
    task_acc = victim.score(test.X, test.y)
    print(f"  trained in {time.time() - t0:.1f}s | task_acc={task_acc:.3f}",
          flush=True)

    print("\nPer-subject A1 attack accuracy ...", flush=True)
    Z_train = victim.embed(train.X)
    Z_test = victim.embed(test.X)
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0).fit(
        Z_train, train.subject_ids,
    )
    preds = clf.predict(Z_test)
    correct = (preds == test.subject_ids).astype(np.float64)

    rows = []
    for s in np.unique(test.subject_ids):
        mask = test.subject_ids == s
        subj_acc = float(correct[mask].mean())
        d = demo.get(int(s), {})
        rows.append({
            "subject_id": int(s),
            "attack_acc": subj_acc,
            "task_acc": float((victim.predict(test.X[mask]) == test.y[mask]).mean()),
            "sex": d.get("sex"),
            "age": d.get("age"),
        })

    # ---- Demographic stratification ----
    male = np.array([r["attack_acc"] for r in rows if r["sex"] == "M"])
    female = np.array([r["attack_acc"] for r in rows if r["sex"] == "F"])
    ages_known = sorted(r["age"] for r in rows if r["age"] is not None)
    low_cut = float(np.percentile(ages_known, 100 / 3)) if ages_known else 0.0
    high_cut = float(np.percentile(ages_known, 200 / 3)) if ages_known else 0.0

    def _bucket(age):
        if age is None: return "unknown"
        if age <= low_cut: return "low"
        if age <= high_cut: return "mid"
        return "high"
    for r in rows:
        r["age_bucket"] = _bucket(r["age"])
    age_low = np.array([r["attack_acc"] for r in rows if r["age_bucket"] == "low"])
    age_high = np.array([r["attack_acc"] for r in rows if r["age_bucket"] == "high"])

    # ---- Heterogeneity ----
    attack = np.array([r["attack_acc"] for r in rows])
    sorted_rows = sorted(rows, key=lambda r: r["attack_acc"])
    n_decile = max(1, len(sorted_rows) // 10)
    most = np.array([r["attack_acc"] for r in sorted_rows[-n_decile:]])
    least = np.array([r["attack_acc"] for r in sorted_rows[:n_decile]])

    sex_diff = _bootstrap_diff_ci(male, female, seed=args.seed) if (
        len(male) and len(female)
    ) else (None, None, None)
    age_diff = _bootstrap_diff_ci(age_low, age_high, seed=args.seed) if (
        len(age_low) and len(age_high)
    ) else (None, None, None)

    sex_p = float(scipy.stats.mannwhitneyu(male, female,
                                            alternative="two-sided").pvalue) \
        if len(male) and len(female) else None
    age_p = float(scipy.stats.mannwhitneyu(age_low, age_high,
                                            alternative="two-sided").pvalue) \
        if len(age_low) and len(age_high) else None

    out = {
        "victim": "eegnet",
        "task_acc": float(task_acc),
        "attack_acc_mean": float(attack.mean()),
        "attack_acc_std": float(attack.std()),
        "decile_gap": float(most.mean() - least.mean()),
        "sex": {
            "M": {"n": int(len(male)), "mean": float(male.mean()) if len(male) else None,
                  "ci": _bootstrap_mean_ci(male, seed=args.seed)[1:] if len(male) else None},
            "F": {"n": int(len(female)),
                  "mean": float(female.mean()) if len(female) else None,
                  "ci": _bootstrap_mean_ci(female, seed=args.seed)[1:] if len(female) else None},
            "diff_M_minus_F": {
                "point": sex_diff[0], "ci_low": sex_diff[1], "ci_high": sex_diff[2],
                "mannwhitneyu_p": sex_p,
            },
        },
        "age": {
            "low": {"n": int(len(age_low)),
                    "mean": float(age_low.mean()) if len(age_low) else None,
                    "cutoff": low_cut},
            "high": {"n": int(len(age_high)),
                     "mean": float(age_high.mean()) if len(age_high) else None,
                     "cutoff": high_cut},
            "diff_low_minus_high": {
                "point": age_diff[0], "ci_low": age_diff[1], "ci_high": age_diff[2],
                "mannwhitneyu_p": age_p,
            },
        },
        "per_subject": rows,
    }

    print()
    print(f"Mean attack acc: {attack.mean():.3f} ± {attack.std():.3f}  "
          f"decile gap: {(most.mean() - least.mean()):+.3f}")
    if len(male) and len(female):
        print(f"M ({len(male)})={male.mean():.3f}  F ({len(female)})="
              f"{female.mean():.3f}  diff={sex_diff[0]:+.3f}  p={sex_p:.3f}")
    if len(age_low) and len(age_high):
        print(f"age low ({len(age_low)})={age_low.mean():.3f}  high "
              f"({len(age_high)})={age_high.mean():.3f}  "
              f"diff={age_diff[0]:+.3f}  p={age_p:.3f}")

    out_path = RESULTS_DIR / "17_subgroup_fairness_eegnet.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
