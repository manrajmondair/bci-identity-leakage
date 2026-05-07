"""W5.1 — subgroup fairness analysis (demographic + within-cohort heterogeneity).

PhysioNet's EDFs are scrubbed (`his_id: X, sex: 0` everywhere). The OpenNeuro
BIDS conversion of the same source data (ds004362) re-publishes a
participants.tsv with Gender, Age, and Handedness for 95 of the 109 subjects.
Both releases use the same BCI2000 source files and the same subject indexing
(sub-001 ↔ S001), so we can map demographics back onto the PhysioNet release
we run our experiments on.

We do TWO complementary analyses:

  1. Demographic stratification (the original W5.1 plan).
     A1 attack accuracy by sex (M vs F) and by age tertile (low / mid / high
     cutoffs computed within our cohort). Bootstrap CIs on group means,
     bootstrap CIs on group differences, and Mann-Whitney U on the per-
     subject distributions.

  2. Per-subject heterogeneity (within-cohort fairness).
     Distribution of A1 attack accuracy across subjects: mean, std, deciles,
     most- vs least-leaked decile. Tests whether the privacy threat is
     uniformly distributed regardless of demographic axis.

Run locally on Mac (FBCSP + Riemann are CPU-fast):
    python -m tools.subgroup_fairness --models fbcsp riemann

EEGNet subgroup numbers come from a separate Colab notebook in the next
batch (it needs a GPU pass to train one A1 EEGNet at the standard hparams).

Sources
-------
- OpenNeuro ds004362 participants.tsv (BIDS conversion of PhysioNet EEG-MMIDB)
- This is committed to the repo at data/external/openneuro_ds004362_participants.tsv
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression

from config import FIGURES_DIR, RESULTS_DIR, ROOT
from data.physionet_loader import valid_subjects
from eval.plots import _setup_axes
from models.fbcsp import FBCSPVictim
from models.riemannian import RiemannianVictim
from preprocess.windows import windowed_subjects

VICTIM_TRAIN_RUNS = (4, 6, 8, 10)
VICTIM_TEST_RUNS = (12, 14)
PARTICIPANTS_TSV = ROOT / "data" / "external" / "openneuro_ds004362_participants.tsv"


# ---------------------------------------------------------------------------
# Demographics (from OpenNeuro ds004362 participants.tsv)
# ---------------------------------------------------------------------------
def load_demographics() -> dict[int, dict]:
    """{subject_id: {'sex': 'M'|'F'|None, 'age': int|None, 'handedness': 'L'|'R'|None}}.

    OpenNeuro stores 'n/a' for missing values; we normalize to None.
    """
    out: dict[int, dict] = {}
    if not PARTICIPANTS_TSV.exists():
        raise FileNotFoundError(
            f"{PARTICIPANTS_TSV} missing — fetch with:\n"
            f"  curl -s https://s3.amazonaws.com/openneuro.org/ds004362/participants.tsv "
            f"-o {PARTICIPANTS_TSV}"
        )

    def _maybe_str(s, valid: tuple) -> str | None:
        return s if s in valid else None

    def _maybe_int(s) -> int | None:
        try:
            return int(s)
        except (TypeError, ValueError):
            return None

    with PARTICIPANTS_TSV.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sid_str = row["participant_id"]            # e.g. "sub-001"
            try:
                sid = int(sid_str.replace("sub-", ""))
            except ValueError:
                continue
            # Defensive parsing: the OpenNeuro TSV has at least one row
            # (sub-044) with a transcription error (Age='M', Gender='n/a').
            # Coerce anything not in the expected value space to None.
            out[sid] = {
                "sex": _maybe_str(row.get("Gender"), ("M", "F")),
                "age": _maybe_int(row.get("Age")),
                "handedness": _maybe_str(row.get("Handedness"), ("L", "R", "A")),
            }
    return out


# ---------------------------------------------------------------------------
# Attack accuracy per subject
# ---------------------------------------------------------------------------
def per_subject_attack_acc(victim, train_set, test_set) -> dict[int, float]:
    Z_train = victim.embed(train_set.X)
    Z_test = victim.embed(test_set.X)
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0).fit(
        Z_train, train_set.subject_ids,
    )
    preds = clf.predict(Z_test)
    correct = (preds == test_set.subject_ids).astype(np.float64)
    out: dict[int, float] = {}
    for s in np.unique(test_set.subject_ids):
        mask = test_set.subject_ids == s
        out[int(s)] = float(correct[mask].mean())
    return out


def per_subject_task_acc(victim, test_set) -> dict[int, float]:
    preds = victim.predict(test_set.X)
    correct = (preds == test_set.y).astype(np.float64)
    out: dict[int, float] = {}
    for s in np.unique(test_set.subject_ids):
        mask = test_set.subject_ids == s
        out[int(s)] = float(correct[mask].mean())
    return out


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------
def _bootstrap_mean_ci(values: np.ndarray, *, n_resamples: int = 1000,
                       seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    samples = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        samples[i] = values[idx].mean()
    return (float(values.mean()),
            float(np.percentile(samples, 2.5)),
            float(np.percentile(samples, 97.5)))


def _bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, *,
                       n_resamples: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        ai = rng.integers(0, len(a), size=len(a))
        bi = rng.integers(0, len(b), size=len(b))
        samples[i] = a[ai].mean() - b[bi].mean()
    return (float(a.mean() - b.mean()),
            float(np.percentile(samples, 2.5)),
            float(np.percentile(samples, 97.5)))


def _distribution_summary(values: np.ndarray) -> dict:
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
    }


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------
def stratify(rows: list[dict], *, seed: int) -> dict:
    """Group attack_acc by sex / age tertile; compute group means + tests."""
    male = np.array([r["attack_acc"] for r in rows if r["sex"] == "M"])
    female = np.array([r["attack_acc"] for r in rows if r["sex"] == "F"])

    ages_known = sorted(r["age"] for r in rows if r["age"] is not None)
    if len(ages_known) >= 6:
        low_cut = float(np.percentile(ages_known, 100 / 3))
        high_cut = float(np.percentile(ages_known, 200 / 3))
    else:
        low_cut, high_cut = 0.0, 0.0

    def _bucket(age):
        if age is None:
            return "unknown"
        if age <= low_cut:
            return "low"
        if age <= high_cut:
            return "mid"
        return "high"

    for r in rows:
        r["age_bucket"] = _bucket(r["age"])

    age_low = np.array([r["attack_acc"] for r in rows if r["age_bucket"] == "low"])
    age_mid = np.array([r["attack_acc"] for r in rows if r["age_bucket"] == "mid"])
    age_high = np.array([r["attack_acc"] for r in rows if r["age_bucket"] == "high"])

    def _grp(values: np.ndarray, label: str) -> dict:
        if len(values) == 0:
            return {"label": label, "n": 0}
        m, lo, hi = _bootstrap_mean_ci(values, seed=seed)
        return {"label": label, "n": int(len(values)),
                "mean": m, "ci_low": lo, "ci_high": hi}

    sex_diff_pt, sex_diff_lo, sex_diff_hi = _bootstrap_diff_ci(
        male, female, seed=seed,
    ) if len(male) and len(female) else (None, None, None)
    age_diff_pt, age_diff_lo, age_diff_hi = _bootstrap_diff_ci(
        age_low, age_high, seed=seed,
    ) if len(age_low) and len(age_high) else (None, None, None)

    return {
        "age_tertile_cutoffs": {"low_le": low_cut, "high_ge": high_cut},
        "sex": {
            "M": _grp(male, "M"),
            "F": _grp(female, "F"),
            "diff_M_minus_F": {
                "point": sex_diff_pt, "ci_low": sex_diff_lo, "ci_high": sex_diff_hi,
                "mannwhitneyu_p": (
                    float(mannwhitneyu(male, female, alternative="two-sided").pvalue)
                    if len(male) and len(female) else None
                ),
            },
        },
        "age": {
            "low": _grp(age_low, "low"),
            "mid": _grp(age_mid, "mid"),
            "high": _grp(age_high, "high"),
            "diff_low_minus_high": {
                "point": age_diff_pt, "ci_low": age_diff_lo, "ci_high": age_diff_hi,
                "mannwhitneyu_p": (
                    float(mannwhitneyu(age_low, age_high, alternative="two-sided").pvalue)
                    if len(age_low) and len(age_high) else None
                ),
            },
        },
    }


# ---------------------------------------------------------------------------
# Per-subject heterogeneity (decile gap)
# ---------------------------------------------------------------------------
def heterogeneity(rows: list[dict]) -> dict:
    attack = np.array([r["attack_acc"] for r in rows])
    task = np.array([r["task_acc"] for r in rows])
    sorted_rows = sorted(rows, key=lambda r: r["attack_acc"])
    n_decile = max(1, len(sorted_rows) // 10)
    most = sorted_rows[-n_decile:]
    least = sorted_rows[:n_decile]
    most_attack = np.array([r["attack_acc"] for r in most])
    least_attack = np.array([r["attack_acc"] for r in least])
    most_task = np.array([r["task_acc"] for r in most])
    least_task = np.array([r["task_acc"] for r in least])
    return {
        "attack_acc_distribution": _distribution_summary(attack),
        "task_acc_distribution": _distribution_summary(task),
        "most_leaked_decile": {
            "n": int(n_decile),
            "mean_attack": float(most_attack.mean()),
            "mean_task": float(most_task.mean()),
            "subject_ids": [r["subject_id"] for r in most],
        },
        "least_leaked_decile": {
            "n": int(n_decile),
            "mean_attack": float(least_attack.mean()),
            "mean_task": float(least_task.mean()),
            "subject_ids": [r["subject_id"] for r in least],
        },
        "decile_gap_attack": float(most_attack.mean() - least_attack.mean()),
        "pearson_attack_vs_task": float(np.corrcoef(attack, task)[0, 1]),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_subgroup(per_subject_by_victim: dict[str, list[dict]], out_path) -> None:
    plt.rcParams.update(_setup_axes())
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.6))
    colors_v = {"fbcsp_lda": "#7f8c8d", "riemann_ts_lr": "#2c3e50"}

    # (0,0) histogram of per-subject attack accuracy by victim
    ax = axes[0, 0]
    bins = np.linspace(0, 1.001, 25)
    for v_name, rows in per_subject_by_victim.items():
        accs = np.array([r["attack_acc"] for r in rows])
        ax.hist(accs, bins=bins, alpha=0.6, color=colors_v.get(v_name, "#000"),
                edgecolor="white", linewidth=0.5, label=v_name)
    ax.set_xlabel("Per-subject A1 attack accuracy")
    ax.set_ylabel("# subjects")
    ax.set_title("Per-subject heterogeneity (attack acc)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(linestyle=":", linewidth=0.4, alpha=0.5)

    # (0,1) per-subject task vs attack scatter
    ax = axes[0, 1]
    for v_name, rows in per_subject_by_victim.items():
        xs = [r["task_acc"] for r in rows]
        ys = [r["attack_acc"] for r in rows]
        ax.scatter(xs, ys, s=18, alpha=0.5, color=colors_v.get(v_name, "#000"),
                   edgecolor="white", linewidth=0.4, label=v_name)
    ax.set_xlabel("Per-subject task accuracy")
    ax.set_ylabel("Per-subject A1 attack accuracy")
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
    ax.set_title("Task vs attack accuracy (per subject)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(linestyle=":", linewidth=0.4, alpha=0.5)

    # (1,0) by sex
    ax = axes[1, 0]
    for v_name, rows in per_subject_by_victim.items():
        m_accs = [r["attack_acc"] for r in rows if r["sex"] == "M"]
        f_accs = [r["attack_acc"] for r in rows if r["sex"] == "F"]
        positions = [0 if v_name == "fbcsp_lda" else 1,
                     2.5 if v_name == "fbcsp_lda" else 3.5]
        bp = ax.boxplot([m_accs, f_accs], positions=positions, widths=0.7,
                        patch_artist=True, showfliers=True,
                        flierprops={"marker": ".", "markersize": 3, "alpha": 0.5})
        for patch in bp["boxes"]:
            patch.set_facecolor(colors_v.get(v_name, "#000"))
            patch.set_alpha(0.6)
    ax.set_xticks([0.5, 3.0])
    ax.set_xticklabels(["Male", "Female"])
    ax.set_ylabel("A1 attack accuracy")
    ax.set_title("By sex (n = M, F per victim)")
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)

    # (1,1) by age tertile
    ax = axes[1, 1]
    for v_name, rows in per_subject_by_victim.items():
        for ti, tlabel in enumerate(["low", "mid", "high"]):
            accs = [r["attack_acc"] for r in rows if r.get("age_bucket") == tlabel]
            if not accs:
                continue
            offset = -0.2 if v_name == "fbcsp_lda" else 0.2
            ax.boxplot([accs], positions=[ti + offset], widths=0.35,
                       patch_artist=True, showfliers=True,
                       flierprops={"marker": ".", "markersize": 3, "alpha": 0.5},
                       boxprops={"facecolor": colors_v.get(v_name, "#000"),
                                 "alpha": 0.6})
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Low", "Mid", "High"])
    ax.set_ylabel("A1 attack accuracy")
    ax.set_xlabel("Age tertile")
    ax.set_title("By age tertile")
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)

    fig.suptitle("Subgroup fairness — A1 attack accuracy across PhysioNet 104-subject cohort",
                 y=1.005, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["fbcsp", "riemann"],
                   choices=["fbcsp", "riemann"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    subjects = valid_subjects()
    print(f"Subjects in analysis cohort: {len(subjects)}", flush=True)

    print("Loading demographics from OpenNeuro ds004362 participants.tsv ...",
          flush=True)
    demo = load_demographics()
    in_cohort_demo = {s: demo[s] for s in subjects if s in demo}
    sex_counts = {"M": 0, "F": 0, "unknown": 0}
    age_known = []
    for d in in_cohort_demo.values():
        sex_counts[d["sex"] if d["sex"] in ("M", "F") else "unknown"] += 1
        if d["age"] is not None:
            age_known.append(d["age"])
    print(f"  in-cohort sex distribution: {sex_counts}")
    print(f"  in-cohort age coverage: {len(age_known)} subjects, "
          f"range [{min(age_known)}, {max(age_known)}], "
          f"median {int(np.median(age_known))}\n", flush=True)

    print("Loading 104-subject windowed data from cache ...", flush=True)
    t0 = time.time()
    full = windowed_subjects(subjects, runs="imagery")
    train_set = full.filter_runs(list(VICTIM_TRAIN_RUNS))
    test_set = full.filter_runs(list(VICTIM_TEST_RUNS))
    print(f"  loaded in {time.time() - t0:.1f}s\n", flush=True)

    out: dict[str, dict] = {"summary": {}, "per_subject_by_victim": {},
                            "demographics_source":
                                "OpenNeuro ds004362 participants.tsv"}
    for victim_name in args.models:
        print(f"=== {victim_name} ===", flush=True)
        if victim_name == "fbcsp":
            victim = FBCSPVictim(sfreq=train_set.sfreq, n_classes=4)
        else:
            victim = RiemannianVictim(n_classes=4, seed=args.seed)
        t0 = time.time()
        victim.fit(train_set.X, train_set.y)
        print(f"  victim fit: {time.time() - t0:.0f}s", flush=True)

        t0 = time.time()
        attack = per_subject_attack_acc(victim, train_set, test_set)
        task = per_subject_task_acc(victim, test_set)
        print(f"  scoring: {time.time() - t0:.0f}s", flush=True)

        rows = []
        for sid in sorted(attack):
            d = in_cohort_demo.get(sid, {"sex": None, "age": None, "handedness": None})
            rows.append({
                "subject_id": int(sid),
                "attack_acc": float(attack[sid]),
                "task_acc": float(task[sid]),
                "sex": d["sex"],
                "age": d["age"],
                "handedness": d["handedness"],
            })

        het = heterogeneity(rows)
        sub = stratify(rows, seed=args.seed)
        out[victim_name] = {
            "demographic_stratification": sub,
            "heterogeneity": het,
        }
        out["per_subject_by_victim"][victim_name] = rows
        out["summary"][victim_name] = {
            "mean_attack_acc": het["attack_acc_distribution"]["mean"],
            "std_attack_acc": het["attack_acc_distribution"]["std"],
            "decile_gap_attack": het["decile_gap_attack"],
            "pearson_attack_vs_task": het["pearson_attack_vs_task"],
            "sex_M_mean": sub["sex"]["M"].get("mean"),
            "sex_F_mean": sub["sex"]["F"].get("mean"),
            "sex_diff_M_minus_F_pt": sub["sex"]["diff_M_minus_F"]["point"],
            "sex_diff_M_minus_F_ci": (sub["sex"]["diff_M_minus_F"]["ci_low"],
                                     sub["sex"]["diff_M_minus_F"]["ci_high"]),
            "sex_diff_p": sub["sex"]["diff_M_minus_F"]["mannwhitneyu_p"],
            "age_low_mean": sub["age"]["low"].get("mean"),
            "age_high_mean": sub["age"]["high"].get("mean"),
            "age_low_minus_high_pt": sub["age"]["diff_low_minus_high"]["point"],
            "age_diff_p": sub["age"]["diff_low_minus_high"]["mannwhitneyu_p"],
        }
        # Quick log
        s = out["summary"][victim_name]
        print(f"  attack mean ± std: {s['mean_attack_acc']:.3f} ± "
              f"{s['std_attack_acc']:.3f}   "
              f"decile gap: {s['decile_gap_attack']:+.3f}", flush=True)
        m_n = sub["sex"]["M"]["n"]
        f_n = sub["sex"]["F"]["n"]
        print(f"  by sex: M(n={m_n})={s['sex_M_mean']:.3f}  "
              f"F(n={f_n})={s['sex_F_mean']:.3f}  "
              f"diff={s['sex_diff_M_minus_F_pt']:+.3f}  p={s['sex_diff_p']:.3f}",
              flush=True)
        print(f"  by age tertile: low(n={sub['age']['low']['n']})="
              f"{s['age_low_mean']:.3f}  high(n={sub['age']['high']['n']})="
              f"{s['age_high_mean']:.3f}  "
              f"diff(low-high)={s['age_low_minus_high_pt']:+.3f}  "
              f"p={s['age_diff_p']:.3f}\n", flush=True)

    out_path = RESULTS_DIR / "12_subgroup_fairness.json"
    Path(out_path).write_text(json.dumps(out, indent=2, default=float))
    print(f"Results written to {out_path}")

    fig_path = FIGURES_DIR / "12_subgroup_fairness.pdf"
    plot_subgroup(out["per_subject_by_victim"], fig_path)
    print(f"Figure written to {fig_path}")


if __name__ == "__main__":
    main()
