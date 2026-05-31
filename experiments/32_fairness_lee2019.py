"""Fairness analysis on Lee 2019 OpenBMI.

Two analyses run unconditionally:

  (1) Per-subject A1 attack accuracy distribution across 54 Lee 2019
      subjects (Riemann + FBCSP + EEGNet victims). Reports decile gap
      (most-leaked minus least-leaked decile), inter-quartile range,
      and the Mann-Whitney U on the upper-half vs lower-half subject
      split. These metrics do NOT require demographic metadata and
      are the within-cohort heterogeneity counterpart of the
      PhysioNet subgroup fairness analysis.

  (2) Demographic stratification (sex, age, handedness) IF the user
      has populated `data/external/lee2019_demographics.tsv` from a
      per-subject release. The Lee 2019 paper only publishes
      cohort-level aggregates (25F/29M, ages 24-35), so for now this
      branch reports "demographic stratification skipped" cleanly.

Reads from the compact npz cache produced by `data.lee2019_prefetch`.
Set BCI_LEE2019_CACHE to the cache root.

Usage
-----
    python -m experiments.32_fairness_lee2019 --smoke
    python -m experiments.32_fairness_lee2019 --all
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression

from config import RESULTS_DIR, ROOT
from data.lee2019_loader import (
    load_subject_session_compact,
    valid_subjects,
)
from models.eegnet import EEGNetVictim
from models.fbcsp import FBCSPVictim
from models.riemannian import RiemannianVictim
from preprocess.windows import WindowedDataset

LEE2019_DEMOGRAPHICS_TSV = ROOT / "data" / "external" / "lee2019_demographics.tsv"


def _pool(subjects: list[int], session: str = "session_1") -> WindowedDataset:
    parts: list[WindowedDataset] = []
    for s in subjects:
        try:
            parts.append(load_subject_session_compact(s, session=session))
        except Exception as exc:
            print(f"    !! subj{s} failed: {exc}", flush=True)
    if not parts:
        raise RuntimeError("no Lee 2019 subjects loaded")
    X = np.concatenate([p.X for p in parts], axis=0)
    y = np.concatenate([p.y for p in parts], axis=0)
    s = np.concatenate([p.subject_ids for p in parts], axis=0)
    t_parts = []
    for p in parts:
        offset = int(p.subject_ids[0]) * 1_000_000
        t_parts.append(p.trial_ids + offset)
    t = np.concatenate(t_parts, axis=0)
    r = np.concatenate([p.run_ids for p in parts], axis=0)
    return WindowedDataset(
        X=X, y=y, subject_ids=s, trial_ids=t, run_ids=r,
        sfreq=parts[0].sfreq, channel_names=parts[0].channel_names,
    )


def _load_demographics() -> dict[int, dict] | None:
    """Return {subject_id: {sex,age,handedness}} or None if the TSV is empty."""
    if not LEE2019_DEMOGRAPHICS_TSV.exists():
        return None
    out: dict[int, dict] = {}
    with LEE2019_DEMOGRAPHICS_TSV.open() as f:
        reader = csv.reader(f, delimiter="\t")
        header_seen = False
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if not header_seen:
                header_seen = True
                continue
            if len(row) < 4:
                continue
            sid = int(row[0])
            sex = row[1] if row[1] not in ("n/a", "") else None
            try:
                age: int | None = int(row[2])
            except (ValueError, TypeError):
                age = None
            handedness = row[3] if row[3] not in ("n/a", "") else None
            out[sid] = {"sex": sex, "age": age, "handedness": handedness}
    # If every row is fully n/a, the TSV is unpopulated — treat as missing.
    has_any = any(v["sex"] is not None or v["age"] is not None
                  or v["handedness"] is not None for v in out.values())
    return out if has_any else None


def _per_subject_a1_accuracy(victim, train_ds: WindowedDataset,
                             test_ds: WindowedDataset) -> dict[int, float]:
    """A1 logreg probe + per-subject window-level top-1 accuracy."""
    Z_train = victim.embed(train_ds.X)
    Z_test = victim.embed(test_ds.X)
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0)
    clf.fit(Z_train, train_ds.subject_ids)
    preds = clf.predict(Z_test)
    correct = (preds == test_ds.subject_ids).astype(np.float64)
    out: dict[int, float] = {}
    for s in np.unique(test_ds.subject_ids):
        mask = test_ds.subject_ids == s
        out[int(s)] = float(correct[mask].mean()) if mask.any() else float("nan")
    return out


def _build_victim(name: str, *, n_channels: int, n_times: int,
                  sfreq: float, eegnet_epochs: int, seed: int):
    if name == "eegnet":
        return EEGNetVictim(
            n_channels=n_channels, n_times=n_times, n_classes=2,
            n_epochs=eegnet_epochs, seed=seed, verbose=False,
        )
    if name == "fbcsp":
        return FBCSPVictim(sfreq=sfreq, n_classes=2)
    if name == "riemann":
        return RiemannianVictim(n_classes=2, seed=seed)
    raise ValueError(name)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--models", nargs="+",
                   default=["fbcsp", "riemann", "eegnet"],
                   choices=["fbcsp", "riemann", "eegnet"])
    p.add_argument("--eegnet-epochs", type=int, default=80)
    p.add_argument("--bootstrap-n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.smoke:
        subjects = valid_subjects()[:10]
        args.eegnet_epochs = 20
    elif args.all:
        subjects = valid_subjects()
    else:
        p.error("Provide --smoke or --all")

    np.random.seed(args.seed)
    print(f"Subjects: {len(subjects)}")
    print(f"Models: {args.models}")

    demo = _load_demographics()
    if demo is None:
        print("No populated Lee 2019 demographics TSV — skipping stratification "
              "(per-subject heterogeneity still reported).\n", flush=True)
    else:
        sex_count = Counter(v["sex"] for v in demo.values())
        age_known = sum(1 for v in demo.values() if v["age"] is not None)
        print(f"Lee 2019 demographics: sex {dict(sex_count)} | "
              f"age-known={age_known}/{len(demo)}\n", flush=True)

    print("Loading Lee 2019 session_1 ...", flush=True)
    t0 = time.time()
    full = _pool(subjects, "session_1")
    print(f"  loaded in {time.time() - t0:.1f}s | windows={full.n_windows} "
          f"chans={full.n_channels}\n", flush=True)

    # Within-session train/test split: shuffle trials, take 2/3 for train.
    rng = np.random.default_rng(args.seed)
    unique_trials = sorted(set(int(t) for t in full.trial_ids))
    rng.shuffle(unique_trials)
    n_train = int(round(len(unique_trials) * 2 / 3))
    train_trials = set(unique_trials[:n_train])
    train_mask = np.array([int(t) in train_trials for t in full.trial_ids])
    train_ds = WindowedDataset(
        X=full.X[train_mask], y=full.y[train_mask],
        subject_ids=full.subject_ids[train_mask],
        trial_ids=full.trial_ids[train_mask],
        run_ids=full.run_ids[train_mask],
        sfreq=full.sfreq, channel_names=full.channel_names,
    )
    test_ds = WindowedDataset(
        X=full.X[~train_mask], y=full.y[~train_mask],
        subject_ids=full.subject_ids[~train_mask],
        trial_ids=full.trial_ids[~train_mask],
        run_ids=full.run_ids[~train_mask],
        sfreq=full.sfreq, channel_names=full.channel_names,
    )
    print(f"  train trials={n_train}  test trials={len(unique_trials) - n_train}",
          flush=True)
    print(f"  train_windows={train_ds.n_windows}  test_windows={test_ds.n_windows}\n",
          flush=True)

    out: dict = {"dataset": "lee2019", "subjects": sorted(int(s) for s in subjects),
                 "victim_results": {}}

    for victim_name in args.models:
        print(f"=== {victim_name} ===", flush=True)
        victim = _build_victim(
            victim_name,
            n_channels=train_ds.n_channels, n_times=train_ds.n_times,
            sfreq=train_ds.sfreq,
            eegnet_epochs=args.eegnet_epochs, seed=args.seed,
        )
        t = time.time()
        victim.fit(train_ds.X, train_ds.y)
        task_acc = victim.score(test_ds.X, test_ds.y)
        print(f"  victim train+score: {time.time() - t:.0f}s  "
              f"task_acc={task_acc:.3f}", flush=True)

        # Per-subject A1 attack accuracy
        t = time.time()
        per_subj = _per_subject_a1_accuracy(victim, train_ds, test_ds)
        accs = np.asarray([v for v in per_subj.values()])
        print(f"  per-subject A1: mean={accs.mean():.3f}  "
              f"std={accs.std():.3f}  "
              f"decile_gap={(np.percentile(accs, 90) - np.percentile(accs, 10)):.3f}  "
              f"({time.time() - t:.0f}s)", flush=True)

        # Demographic stratification (if available)
        strat = None
        if demo is not None:
            strat = {}
            # Sex
            sex_groups: dict[str, list[float]] = {"M": [], "F": []}
            for sid, acc in per_subj.items():
                sex = demo.get(int(sid), {}).get("sex")
                if sex in ("M", "F"):
                    sex_groups[sex].append(float(acc))
            if len(sex_groups["M"]) >= 5 and len(sex_groups["F"]) >= 5:
                stat, pv = mannwhitneyu(sex_groups["M"], sex_groups["F"],
                                         alternative="two-sided")
                strat["sex"] = {
                    "n_M": int(len(sex_groups["M"])),
                    "n_F": int(len(sex_groups["F"])),
                    "mean_M": float(np.mean(sex_groups["M"])),
                    "mean_F": float(np.mean(sex_groups["F"])),
                    "diff_M_minus_F": float(np.mean(sex_groups["M"]) - np.mean(sex_groups["F"])),
                    "mannwhitney_p": float(pv),
                }
            # Age tertiles
            ages = {int(sid): demo[int(sid)]["age"] for sid in per_subj
                    if demo.get(int(sid), {}).get("age") is not None}
            if len(ages) >= 12:
                age_vals = sorted(ages.values())
                lo_cut = age_vals[len(age_vals) // 3]
                hi_cut = age_vals[2 * len(age_vals) // 3]
                lo_acc = [float(per_subj[sid]) for sid, age in ages.items() if age <= lo_cut]
                hi_acc = [float(per_subj[sid]) for sid, age in ages.items() if age > hi_cut]
                if len(lo_acc) >= 5 and len(hi_acc) >= 5:
                    stat, pv = mannwhitneyu(lo_acc, hi_acc, alternative="two-sided")
                    strat["age"] = {
                        "lo_cut": int(lo_cut), "hi_cut": int(hi_cut),
                        "n_lo": int(len(lo_acc)), "n_hi": int(len(hi_acc)),
                        "mean_lo": float(np.mean(lo_acc)),
                        "mean_hi": float(np.mean(hi_acc)),
                        "diff_lo_minus_hi": float(np.mean(lo_acc) - np.mean(hi_acc)),
                        "mannwhitney_p": float(pv),
                    }

        out["victim_results"][victim_name] = {
            "task_acc": float(task_acc),
            "per_subject_accuracy": {int(k): float(v) for k, v in per_subj.items()},
            "heterogeneity": {
                "mean": float(accs.mean()),
                "std": float(accs.std()),
                "decile_gap": float(np.percentile(accs, 90) - np.percentile(accs, 10)),
                "iqr": float(np.percentile(accs, 75) - np.percentile(accs, 25)),
                "min": float(accs.min()), "max": float(accs.max()),
            },
            "demographic_stratification": strat,
        }
        print()

    out_path = RESULTS_DIR / "32_fairness_lee2019.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
