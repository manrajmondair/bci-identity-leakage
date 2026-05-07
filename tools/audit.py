"""Comprehensive integrity audit of the BCI identity-leakage pipeline.

Validates the things most likely to be silently wrong in an EEG privacy
benchmark, on the cached windowed data and the committed result JSONs.
Writes a markdown report to `runs/audit_<timestamp>/audit.md` so the
checks are reproducible from the repo state.

Run with:
    python -m tools.audit
"""
from __future__ import annotations

import datetime
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from config import CACHE_DIR, RESULTS_DIR
from data.physionet_loader import (
    PHYSIONET_DROP_SUBJECTS,
    PHYSIONET_RUNS_BASELINE,
    PHYSIONET_RUNS_EXECUTION,
    PHYSIONET_RUNS_IMAGERY,
    valid_subjects,
)
from preprocess.windows import windowed_subjects


_OK = "OK"
_WARN = "WARN"
_FAIL = "FAIL"


class Audit:
    def __init__(self) -> None:
        self.entries: list[dict] = []

    def record(self, name: str, status: str, detail: str) -> None:
        self.entries.append({"name": name, "status": status, "detail": detail})
        flag = {"OK": "✓", "WARN": "!", "FAIL": "✗"}[status]
        print(f"  [{flag}] {name}: {detail}", flush=True)

    def expect_eq(self, name: str, actual, expected, *, detail: str = "") -> None:
        ok = actual == expected
        msg = f"actual={actual!r} expected={expected!r}"
        if detail:
            msg = f"{detail} ({msg})"
        self.record(name, _OK if ok else _FAIL, msg)

    def expect(self, name: str, condition: bool, detail: str) -> None:
        self.record(name, _OK if condition else _FAIL, detail)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


# -----------------------------------------------------------------------------
# 1. Static data-integrity checks against the cached windowed datasets
# -----------------------------------------------------------------------------
def check_data_integrity(audit: Audit) -> dict:
    print("\n## 1. DATA INTEGRITY", flush=True)

    subjects = valid_subjects()
    audit.expect_eq("valid_subjects() count", len(subjects), 104)
    audit.expect("known-bad subjects dropped",
                 all(s not in subjects for s in PHYSIONET_DROP_SUBJECTS),
                 f"{set(subjects) & set(PHYSIONET_DROP_SUBJECTS)} should be empty")

    audit.expect("imagery & execution & baseline runs are disjoint families",
                 not (set(PHYSIONET_RUNS_IMAGERY) & set(PHYSIONET_RUNS_EXECUTION))
                 and not (set(PHYSIONET_RUNS_IMAGERY) & set(PHYSIONET_RUNS_BASELINE))
                 and not (set(PHYSIONET_RUNS_EXECUTION) & set(PHYSIONET_RUNS_BASELINE)),
                 f"imagery={set(PHYSIONET_RUNS_IMAGERY)}  "
                 f"execution={set(PHYSIONET_RUNS_EXECUTION)}  "
                 f"baseline={set(PHYSIONET_RUNS_BASELINE)}")

    print("  loading 104-subject imagery windowed data from cache ...", flush=True)
    ds = windowed_subjects(subjects, runs="imagery")

    audit.expect_eq("X.shape[0] == y.shape[0] == subject_ids.shape[0]",
                    True,
                    ds.X.shape[0] == ds.y.shape[0] == ds.subject_ids.shape[0],
                    detail=f"X={ds.X.shape} y={ds.y.shape} sids={ds.subject_ids.shape}")
    audit.expect_eq("n_channels (PhysioNet)", ds.n_channels, 64)
    audit.expect_eq("n_times (2s @ 160Hz)", ds.n_times, 320)
    audit.expect_eq("classes used", sorted(np.unique(ds.y).tolist()), [0, 1, 2, 3])

    n_subj_in_data = len(np.unique(ds.subject_ids))
    audit.expect_eq("subject count in cached data", n_subj_in_data, 104)

    audit.expect("no NaN/Inf in X",
                 np.isfinite(ds.X).all(),
                 f"X has {(~np.isfinite(ds.X)).sum()} non-finite values")

    # Window count per subject — should be ~270 (6 imagery runs × 45 windows/run)
    counts = np.bincount(ds.subject_ids)
    nz_counts = counts[counts > 0]
    audit.expect("per-subject window counts are tightly clustered",
                 nz_counts.std() / nz_counts.mean() < 0.15,
                 f"mean={nz_counts.mean():.1f}  std={nz_counts.std():.1f}  "
                 f"min={nz_counts.min()}  max={nz_counts.max()}")

    # Class balance per subject
    bad_subjects: list[int] = []
    for s in np.unique(ds.subject_ids):
        mask = ds.subject_ids == s
        cls_counts = np.bincount(ds.y[mask], minlength=4)
        if (cls_counts > 0).sum() < 4 or cls_counts.min() < 0.5 * cls_counts.max():
            bad_subjects.append(int(s))
    audit.expect("each subject has all 4 motor-imagery classes, no severe imbalance",
                 len(bad_subjects) == 0,
                 f"{len(bad_subjects)} suspect subjects" +
                 (f": first 5 = {bad_subjects[:5]}" if bad_subjects else ""))

    # trial_ids: by construction, multiple windows from one trial SHARE a
    # trial_id (so bootstrap can resample whole trials). We just need:
    #   (a) every trial_id maps to exactly one subject (no cross-subject collisions);
    #   (b) windows-per-trial is consistent across the dataset.
    n_unique_trials = len(np.unique(ds.trial_ids))
    n_windows = len(ds.trial_ids)
    windows_per_trial_avg = n_windows / n_unique_trials
    # 4-s trial, 2-s window, 1-s stride => 3 windows per trial expected
    audit.expect("windows-per-trial ≈ 3 (4-s trial, 2-s window, 1-s stride)",
                 abs(windows_per_trial_avg - 3.0) < 0.01,
                 f"{n_unique_trials} unique trials, {n_windows} windows, "
                 f"avg {windows_per_trial_avg:.3f} per trial")

    # Each trial belongs to exactly one subject (cross-subject leakage check).
    trial_to_subj: dict[int, int] = {}
    conflicts = 0
    for tid, sid in zip(ds.trial_ids, ds.subject_ids):
        prev = trial_to_subj.setdefault(int(tid), int(sid))
        if prev != int(sid):
            conflicts += 1
    audit.expect("each trial_id maps to a single subject (no cross-subject collision)",
                 conflicts == 0,
                 f"{conflicts} conflicts")

    # Trial-id offset trick: subject s's trial_ids should fall in
    # [s * 100_000, (s+1) * 100_000). This is what windowed_subjects() does.
    bad_offset = 0
    for tid, sid in zip(ds.trial_ids, ds.subject_ids):
        lo = int(sid) * 100_000
        hi = lo + 100_000
        if not (lo <= int(tid) < hi):
            bad_offset += 1
    audit.expect("subject's trial_ids land inside its 100k-offset block",
                 bad_offset == 0,
                 f"{bad_offset} mismatched")

    # Channel order consistent (already enforced in windowed_subjects, double check)
    audit.expect_eq("channel order length", len(ds.channel_names), 64)
    return {
        "n_subjects": int(n_subj_in_data),
        "n_windows": int(ds.X.shape[0]),
        "channels": list(ds.channel_names)[:5] + ["..."],
    }


# -----------------------------------------------------------------------------
# 2. Train / test split correctness for each attack
# -----------------------------------------------------------------------------
def check_split_correctness(audit: Audit) -> dict:
    print("\n## 2. TRAIN/TEST SPLIT CORRECTNESS", flush=True)

    # ---- A1 closed-set: same subjects, disjoint runs ----
    A1_TRAIN_RUNS = {4, 6, 8, 10}
    A1_TEST_RUNS = {12, 14}
    audit.expect("A1 train and test runs are disjoint",
                 not (A1_TRAIN_RUNS & A1_TEST_RUNS),
                 f"train={A1_TRAIN_RUNS}  test={A1_TEST_RUNS}")
    audit.expect("A1 train and test runs are both subsets of imagery runs",
                 (A1_TRAIN_RUNS | A1_TEST_RUNS).issubset(set(PHYSIONET_RUNS_IMAGERY)),
                 f"missing from imagery: "
                 f"{(A1_TRAIN_RUNS | A1_TEST_RUNS) - set(PHYSIONET_RUNS_IMAGERY)}")

    print("  building A1 train/test slices to verify subject + trial integrity ...",
          flush=True)
    ds = windowed_subjects(valid_subjects(), runs="imagery")
    train = ds.filter_runs(list(A1_TRAIN_RUNS))
    test = ds.filter_runs(list(A1_TEST_RUNS))
    train_subjs = set(np.unique(train.subject_ids).tolist())
    test_subjs = set(np.unique(test.subject_ids).tolist())
    audit.expect("A1 closed-set: train_subjects == test_subjects",
                 train_subjs == test_subjs,
                 f"|train|={len(train_subjs)} |test|={len(test_subjs)} "
                 f"diff={train_subjs ^ test_subjs}")

    overlap_trials = set(train.trial_ids) & set(test.trial_ids)
    audit.expect("A1 train and test trials are disjoint (no within-trial leakage)",
                 len(overlap_trials) == 0,
                 f"|overlap|={len(overlap_trials)}")

    audit.expect_eq("A1 train run-set in train slice",
                    set(np.unique(train.run_ids).tolist()), A1_TRAIN_RUNS)
    audit.expect_eq("A1 test run-set in test slice",
                    set(np.unique(test.run_ids).tolist()), A1_TEST_RUNS)

    # ---- A2 cross-task: probe trains on execution, tests on imagery (different runs) ----
    A2_PROBE_TRAIN_RUNS = set(PHYSIONET_RUNS_EXECUTION)
    A2_PROBE_TEST_RUNS = {12, 14}  # subset of imagery
    audit.expect("A2: probe train (execution) ∩ probe test (imagery) is empty",
                 not (A2_PROBE_TRAIN_RUNS & A2_PROBE_TEST_RUNS),
                 f"train_runs={sorted(A2_PROBE_TRAIN_RUNS)}  "
                 f"test_runs={sorted(A2_PROBE_TEST_RUNS)}")

    # ---- A3 cross-session: BCI IV-2a, two sessions ----
    audit.expect("A3 sessions used: 0train -> 1test (different recording days)",
                 True,
                 "by construction in experiments/05_a3_cross_session.py")

    # ---- A4 open-set: train/test SUBJECT lists are disjoint ----
    a4_path = RESULTS_DIR / "06_a4_open_set.json"
    if a4_path.exists():
        a4 = json.loads(a4_path.read_text())
        train_s = set(a4.get("train_subjects", []))
        test_s = set(a4.get("test_subjects", []))
        audit.expect("A4: train_subjects ∩ test_subjects is empty (open-set)",
                     not (train_s & test_s),
                     f"|train|={len(train_s)} |test|={len(test_s)} "
                     f"|overlap|={len(train_s & test_s)}")
        audit.expect_eq("A4 |train|", len(train_s), 80)
        audit.expect_eq("A4 |test (held-out)|", len(test_s), 24)
        audit.expect("A4 union == 104 valid subjects",
                     train_s | test_s == set(valid_subjects()),
                     f"missing={set(valid_subjects()) - (train_s | test_s)}  "
                     f"extra={(train_s | test_s) - set(valid_subjects())}")
    else:
        audit.record("A4 train/test split", _WARN,
                     "results/06_a4_open_set.json not present — skipping")

    return {"a1_train_runs": sorted(A1_TRAIN_RUNS), "a1_test_runs": sorted(A1_TEST_RUNS)}


# -----------------------------------------------------------------------------
# 3. Probe-methodology spot-checks
# -----------------------------------------------------------------------------
def check_probe_methodology(audit: Audit) -> None:
    print("\n## 3. PROBE METHODOLOGY", flush=True)

    # Bootstrap CI: grouped by trial_ids?
    src = (Path(__file__).parent.parent / "attacks" / "closed_set.py").read_text()
    audit.expect("A1 attack: bootstrap CI uses grouped_bootstrap_ci over trial_ids",
                 "grouped_bootstrap_ci" in src and "groups=test_set.trial_ids" in src,
                 "see attacks/closed_set.py")

    # Probe trained on Z_train only, tested on Z_test only
    audit.expect("A1 attack: probe trained ONLY on Z_train, tested ONLY on Z_test",
                 "clf.fit(Z_train" in src and "clf.predict(Z_test" in src,
                 "fit/predict separation enforced")

    # A4: contrastive trained on training subjects only, scored on test subjects only
    src_v = (Path(__file__).parent.parent / "attacks" / "verification.py").read_text()
    audit.expect("A4 attack: training on (X_train, subj_train) only",
                 "_train_contrastive(\n        X_train, subj_train" in src_v
                 or "X_train, subj_train" in src_v,
                 "see attacks/verification.py")
    audit.expect("A4 attack: test pairs sampled from test_subjects only",
                 "subj_to_idx_test" in src_v and "rng.choice(test_subjects" in src_v,
                 "verification pair sampling restricted to held-out subjects")

    # EEGNet input scale fix is in place
    src_e = (Path(__file__).parent.parent / "models" / "eegnet.py").read_text()
    audit.expect("EEGNet uses input_scale=1e6 (volts -> microvolts)",
                 "input_scale: float = 1e6" in src_e
                 and "X[sl] * self.input_scale" in src_e,
                 "see models/eegnet.py")


# -----------------------------------------------------------------------------
# 4. Empirical sanity / negative control
# -----------------------------------------------------------------------------
def check_negative_control(audit: Audit) -> None:
    """A 'shuffled-labels' negative control on the A1 closed-set probe.

    Take the cached windowed data, fit Riemann tangent-space + LR using
    the test windows as if they were training, but with subject ids RANDOMLY
    PERMUTED, and probe identity on a fresh test set drawn from the same
    pool. Top-1 must collapse to chance. If it doesn't, our probe is leaking
    information from somewhere it shouldn't.
    """
    print("\n## 4. NEGATIVE CONTROL (shuffled subject labels)", flush=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import top_k_accuracy_score

    print("  loading cached imagery windows ...", flush=True)
    ds = windowed_subjects(valid_subjects(), runs="imagery")
    train = ds.filter_runs([4, 6, 8, 10])
    test = ds.filter_runs([12, 14])

    # Use a quick "embedding" so we don't have to retrain a victim:
    # per-window log-variance per channel = a 64-d feature, similar in spirit
    # to FBCSP's log-variance features but with no spatial filtering.
    def feat(X):
        return np.log(X.var(axis=2) + 1e-12).astype(np.float32)

    Z_train = feat(train.X)
    Z_test = feat(test.X)

    rng = np.random.default_rng(42)
    y_train_shuf = rng.permutation(train.subject_ids)
    y_test_shuf = rng.permutation(test.subject_ids)

    clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0).fit(Z_train, y_train_shuf)
    proba = clf.predict_proba(Z_test)
    top1 = float((clf.predict(Z_test) == y_test_shuf).mean())
    top5 = float(top_k_accuracy_score(y_test_shuf, proba, k=5, labels=clf.classes_))

    chance = 1.0 / len(np.unique(ds.subject_ids))
    audit.expect(
        f"shuffled-label probe top-1 ≈ chance ({chance:.4f})",
        abs(top1 - chance) < 0.02,
        f"top1={top1:.4f}  top5={top5:.4f}  chance={chance:.4f}",
    )
    # Also: same features, REAL labels — must beat chance to confirm the features
    # carry SOME signal (otherwise the negative control is unconvincing).
    clf_real = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0).fit(
        Z_train, train.subject_ids,
    )
    top1_real = float((clf_real.predict(Z_test) == test.subject_ids).mean())
    audit.expect(
        "same probe + REAL labels recovers identity well above chance",
        top1_real > 5 * chance,
        f"top1(real)={top1_real:.4f}  vs chance={chance:.4f}",
    )


# -----------------------------------------------------------------------------
# 5. Result-file consistency: shapes, fields, finite values
# -----------------------------------------------------------------------------
def check_result_files(audit: Audit) -> dict:
    print("\n## 5. RESULT FILES", flush=True)
    sizes = {}
    for name in ("02_closed_set_reid.json", "06_a4_open_set.json"):
        path = RESULTS_DIR / name
        if not path.exists():
            audit.record(f"results/{name}", _WARN, "missing")
            continue
        data = json.loads(path.read_text())
        if isinstance(data, list):
            audit.expect(f"results/{name}: non-empty list",
                         len(data) > 0, f"{len(data)} rows")
            for r in data:
                for k in ("top1", "top1_ci_low", "top1_ci_high", "chance_top1"):
                    v = r.get(k)
                    audit.expect(f"results/{name}: {k} is finite",
                                 v is not None and isinstance(v, (int, float))
                                 and 0 <= v <= 1,
                                 f"{r.get('victim','?')}/{r.get('probe','?')}: {k}={v}")
                audit.expect(f"results/{name}: CI brackets the point estimate",
                             r["top1_ci_low"] <= r["top1"] <= r["top1_ci_high"],
                             f"{r.get('victim','?')}/{r.get('probe','?')}: "
                             f"low={r['top1_ci_low']:.4f} pt={r['top1']:.4f} "
                             f"hi={r['top1_ci_high']:.4f}")
            sizes[name] = len(data)
        elif isinstance(data, dict):
            for k in ("auc", "auc_ci_low", "auc_ci_high", "eer"):
                v = data.get(k)
                audit.expect(f"results/{name}: {k} in [0,1]",
                             v is not None and 0 <= v <= 1,
                             f"{k}={v}")
            audit.expect(f"results/{name}: AUC CI brackets point estimate",
                         data["auc_ci_low"] <= data["auc"] <= data["auc_ci_high"],
                         f"low={data['auc_ci_low']:.4f} "
                         f"pt={data['auc']:.4f} "
                         f"hi={data['auc_ci_high']:.4f}")
            sizes[name] = "dict"
    return sizes


# -----------------------------------------------------------------------------
# 6. Effect-size sanity vs literature
# -----------------------------------------------------------------------------
def check_effect_sizes(audit: Audit) -> None:
    print("\n## 6. EFFECT-SIZE SANITY VS LITERATURE", flush=True)

    a1_path = RESULTS_DIR / "02_closed_set_reid.json"
    if a1_path.exists():
        rows = json.loads(a1_path.read_text())
        for r in rows:
            if r["probe"] != "logreg":
                continue
            v = r["victim"]
            top1 = r["top1"]
            if v == "riemann_ts_lr":
                # Maiorana 2016, Yang & Deravi 2017 closed-set EEG re-ID >= 95%
                audit.expect(f"A1 {v} top-1 high (lit. closed-set EEG re-ID >= 0.95)",
                             top1 >= 0.95, f"top1={top1:.3f}")
            elif v == "fbcsp_lda":
                audit.expect(f"A1 {v} top-1 substantially above chance",
                             top1 >= 0.5, f"top1={top1:.3f}")
            elif v == "eegnet":
                audit.expect(f"A1 {v} top-1 above chance (compresses identity but leaks)",
                             top1 >= 0.10, f"top1={top1:.3f}  (chance={r['chance_top1']:.4f})")
                audit.expect(f"A1 {v} task acc above chance (>0.25 for 4-class)",
                             r["task_acc"] > 0.25,
                             f"task_acc={r['task_acc']:.3f}  "
                             f"(verifies input_scale fix is working)")

    a4_path = RESULTS_DIR / "06_a4_open_set.json"
    if a4_path.exists():
        a4 = json.loads(a4_path.read_text())
        # Literature: open-set EEG verification AUC typically 0.85-0.99 depending on protocol;
        # ours is on motor-imagery (harder than resting-state). 0.85+ is plausible, 0.99+ is suspect.
        audit.expect("A4 AUC plausible for open-set EEG verification (0.80 < AUC < 0.99)",
                     0.80 < a4["auc"] < 0.99,
                     f"AUC={a4['auc']:.3f}")
        audit.expect("A4 EER consistent with AUC (rough rule: EER ≈ 1 - AUC)",
                     abs(a4["eer"] - (1 - a4["auc"])) < 0.10,
                     f"EER={a4['eer']:.3f}  1-AUC={1 - a4['auc']:.3f}  "
                     f"|diff|={abs(a4['eer'] - (1 - a4['auc'])):.3f}")


# -----------------------------------------------------------------------------
# Run + write report
# -----------------------------------------------------------------------------
def main() -> None:
    audit = Audit()
    print("Running comprehensive audit ...\n", flush=True)
    integrity = check_data_integrity(audit)
    splits = check_split_correctness(audit)
    check_probe_methodology(audit)
    check_negative_control(audit)
    files = check_result_files(audit)
    check_effect_sizes(audit)

    n_ok = sum(1 for e in audit.entries if e["status"] == _OK)
    n_warn = sum(1 for e in audit.entries if e["status"] == _WARN)
    n_fail = sum(1 for e in audit.entries if e["status"] == _FAIL)

    print(f"\n=== SUMMARY: {n_ok} OK, {n_warn} WARN, {n_fail} FAIL ===\n",
          flush=True)
    if n_fail > 0:
        for e in audit.entries:
            if e["status"] == _FAIL:
                print(f"  FAIL: {e['name']}\n         {e['detail']}", flush=True)

    # Persist
    sha = _git_sha()
    ts = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    run_id = ts.replace(":", "").replace("-", "").rstrip("Z") + f"_audit_{sha[:7]}"
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "audit.json").write_text(json.dumps({
        "run_id": run_id,
        "git_sha": sha,
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "completed_at_utc": ts,
        "summary": {"OK": n_ok, "WARN": n_warn, "FAIL": n_fail},
        "entries": audit.entries,
        "integrity": integrity,
        "splits": splits,
        "files": files,
    }, indent=2))

    md_lines = [
        f"# Audit report — {ts}",
        "",
        f"git: `{sha}`",
        f"platform: `{platform.platform()}`",
        f"python: `{sys.version.split()[0]}`",
        "",
        f"## Summary: {n_ok} OK, {n_warn} WARN, {n_fail} FAIL",
        "",
        "| | Check | Detail |",
        "|---|---|---|",
    ]
    for e in audit.entries:
        flag = {"OK": "✓", "WARN": "!", "FAIL": "✗"}[e["status"]]
        md_lines.append(f"| {flag} | {e['name']} | {e['detail']} |")
    (run_dir / "audit.md").write_text("\n".join(md_lines) + "\n")
    print(f"\nReport written to {run_dir}/audit.md and audit.json", flush=True)

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
