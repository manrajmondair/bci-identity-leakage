"""A1b — within-subject task + closed-set re-identification.

Trains one personal motor-imagery decoder per subject (104 EEGNets,
104 FBCSPs, 104 Riemannian classifiers), measures per-subject task
accuracy, then runs the all-models re-ID attack: feed each test window
through every personal decoder and attribute by argmax-confidence.

This complements 02_closed_set_reid.py (cross-subject pooled victim) by
covering the within-subject deployment that BCI services actually ship.
It is also the only setup where vanilla EEGNet learns motor imagery on
PhysioNet at above-chance accuracy, so it is the only setup where we
can fairly measure EEGNet privacy leakage.

Usage
-----
    python -m experiments.03_within_subject_reid --smoke   # 10 subjects
    python -m experiments.03_within_subject_reid --all     # 104 subjects
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import numpy as np

from attacks.per_subject import per_subject_closed_set_reid
from config import FIGURES_DIR, RESULTS_DIR
from data.physionet_loader import valid_subjects
from eval.plots import _setup_axes
from models.base import VictimModel
from models.eegnet import EEGNetVictim
from models.fbcsp import FBCSPVictim
from models.riemannian import RiemannianVictim
from preprocess.windows import windowed_subjects

TRAIN_RUNS = (4, 6, 8, 10)
TEST_RUNS = (12, 14)


@dataclass
class PerSubjectTaskRow:
    subject_id: int
    victim_family: str
    task_acc: float
    n_train: int
    n_test: int
    fit_seconds: float


def _build_victim(name: str, *, n_channels: int, n_times: int, sfreq: float,
                  eegnet_epochs: int, seed: int) -> VictimModel:
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


def _train_one_subject(
    subject_data: dict, victim_name: str,
    *, eegnet_epochs: int, seed: int,
) -> tuple[VictimModel, PerSubjectTaskRow]:
    """Train one personal decoder for one subject and score on its own held-out runs."""
    sid = subject_data["subject_id"]
    Xtr, ytr = subject_data["X_train"], subject_data["y_train"]
    Xte, yte = subject_data["X_test"], subject_data["y_test"]

    victim = _build_victim(
        victim_name,
        n_channels=Xtr.shape[1], n_times=Xtr.shape[2],
        sfreq=subject_data["sfreq"], eegnet_epochs=eegnet_epochs, seed=seed,
    )
    t0 = time.time()
    victim.fit(Xtr, ytr)
    acc = victim.score(Xte, yte)
    dt = time.time() - t0
    return victim, PerSubjectTaskRow(
        subject_id=sid, victim_family=victim_name,
        task_acc=float(acc), n_train=int(len(Xtr)), n_test=int(len(Xte)),
        fit_seconds=dt,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="10 subjects, EEGNet capped at 30 epochs.")
    p.add_argument("--all", action="store_true",
                   help="All 104 valid PhysioNet subjects.")
    p.add_argument("--subjects", type=int, nargs="*", default=None)
    p.add_argument("--models", nargs="+",
                   default=["eegnet", "fbcsp", "riemann"],
                   choices=["eegnet", "fbcsp", "riemann"])
    p.add_argument("--eegnet-epochs", type=int, default=80)
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
    print(f"Subjects: {len(subjects)}  Models: {args.models}")
    print(f"Train runs {TRAIN_RUNS}  Test runs {TEST_RUNS}\n", flush=True)

    # ---- Load all subjects' windowed data once (reuse the cached .npz) ----
    print("Loading windowed data ...", flush=True)
    t0 = time.time()
    full = windowed_subjects(subjects, runs="imagery")
    print(f"  loaded in {time.time() - t0:.1f}s | "
          f"chans={full.n_channels} times={full.n_times}\n", flush=True)

    # Bucket by subject so we don't keep slicing the global array repeatedly
    per_subject: dict[int, dict] = {}
    for s in subjects:
        ds = full.filter_subjects([s])
        train = ds.filter_runs(list(TRAIN_RUNS))
        test = ds.filter_runs(list(TEST_RUNS))
        per_subject[s] = dict(
            subject_id=s,
            X_train=train.X, y_train=train.y,
            X_test=test.X, y_test=test.y,
            test_trial_ids=test.trial_ids,
            sfreq=full.sfreq,
        )

    # ---- For each victim family: train per-subject + run attacks ----
    all_task_rows: list[PerSubjectTaskRow] = []
    all_attack_rows: list[dict] = []
    for victim_name in args.models:
        print(f"=== {victim_name} (per-subject training) ===", flush=True)
        victims_by_sid: dict[int, VictimModel] = {}
        train_X_by_sid: dict[int, np.ndarray] = {}
        test_X_by_sid: dict[int, np.ndarray] = {}
        test_trials_by_sid: dict[int, np.ndarray] = {}
        t_train_start = time.time()
        for s in subjects:
            data = per_subject[s]
            v, row = _train_one_subject(
                data, victim_name,
                eegnet_epochs=args.eegnet_epochs, seed=args.seed,
            )
            victims_by_sid[s] = v
            train_X_by_sid[s] = data["X_train"]
            test_X_by_sid[s] = data["X_test"]
            test_trials_by_sid[s] = data["test_trial_ids"]
            all_task_rows.append(row)
            if (s in (subjects[0], subjects[-1])) or (s % 10 == 0):
                print(f"  S{s:03d}  task_acc={row.task_acc:.3f}  "
                      f"({row.fit_seconds:.1f}s, {len(victims_by_sid)}/{len(subjects)})",
                      flush=True)
        train_time = time.time() - t_train_start

        accs = [r.task_acc for r in all_task_rows if r.victim_family == victim_name]
        mean_task = float(np.mean(accs))
        std_task = float(np.std(accs))
        print(f"  trained {len(subjects)} subjects in {train_time:.0f}s | "
              f"task_acc mean={mean_task:.3f} std={std_task:.3f}", flush=True)

        # ---- Run all-models re-ID attacks (argmax + softmax probe) ----
        t_attack = time.time()
        results = per_subject_closed_set_reid(
            victims_by_sid, train_X_by_sid, test_X_by_sid, test_trials_by_sid,
            bootstrap_n=args.bootstrap_n, seed=args.seed,
        )
        attack_dt = time.time() - t_attack
        for r in results:
            d = asdict(r)
            d["mean_within_subject_task_acc"] = mean_task
            d["std_within_subject_task_acc"] = std_task
            all_attack_rows.append(d)
            print(f"  {r.attack:14s}  top1={r.top1:.3f} "
                  f"[{r.top1_ci_low:.3f}, {r.top1_ci_high:.3f}]  "
                  f"top5={r.top5:.3f}  top10={r.top10:.3f}  "
                  f"(chance={r.chance_top1:.3f})", flush=True)
        print(f"  attack runtime: {attack_dt:.1f}s\n", flush=True)

    # ---- Persist + plot ----
    out = {
        "subjects": subjects,
        "task_rows": [asdict(r) for r in all_task_rows],
        "attack_rows": all_attack_rows,
    }
    out_path = RESULTS_DIR / "03_within_subject_reid.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Results written to {out_path}", flush=True)

    _plot_within_subject_summary(all_task_rows, all_attack_rows,
                                 FIGURES_DIR / "03_within_subject_reid.pdf",
                                 n_subjects=len(subjects))
    print(f"Figure written to {FIGURES_DIR / '03_within_subject_reid.pdf'}", flush=True)

    print("\n--- Within-subject summary ---")
    print("| Victim | Attack | Mean ± SD task | Re-ID top-1 (95% CI) | Top-5 | Top-10 | Chance |")
    print("|---|---|---|---|---|---|---|")
    for r in all_attack_rows:
        print(f"| {r['victim_family']} | {r['attack']} | "
              f"{r['mean_within_subject_task_acc']:.3f} ± {r['std_within_subject_task_acc']:.3f} | "
              f"{r['top1']:.3f} [{r['top1_ci_low']:.3f}, {r['top1_ci_high']:.3f}] | "
              f"{r['top5']:.3f} | {r['top10']:.3f} | {r['chance_top1']:.3f} |")


def _plot_within_subject_summary(task_rows, attack_rows, out_path, *, n_subjects: int) -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update(_setup_axes())
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(7.5, 3.6))

    families = sorted({r.victim_family for r in task_rows})

    # Left: violin / scatter of per-subject task accuracy
    x_pos = np.arange(len(families))
    for i, fam in enumerate(families):
        accs = [r.task_acc for r in task_rows if r.victim_family == fam]
        ax_l.scatter([i + 0.05 * (np.random.rand(len(accs)) - 0.5)],
                     accs, s=8, alpha=0.5, color="#7f8c8d")
        m = np.mean(accs)
        ax_l.hlines(m, i - 0.2, i + 0.2, colors="#2c3e50", linewidth=1.2)
    ax_l.set_xticks(x_pos)
    ax_l.set_xticklabels(families)
    ax_l.axhline(0.25, color="#c0392b", linestyle="--", linewidth=0.8,
                 label="chance = 0.25")
    ax_l.set_ylabel("Per-subject task accuracy")
    ax_l.set_ylim(0, 1.05)
    ax_l.set_title(f"Within-subject motor-imagery (n={n_subjects})")
    ax_l.legend(frameon=False, loc="upper left", fontsize=8)
    ax_l.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)

    # Right: per-family re-ID top-1, two bars per family (one per attack variant)
    families = sorted({r["victim_family"] for r in attack_rows})
    attacks = ["argmax_conf", "softmax_probe"]
    n_a = len(attacks)
    bar_w = 0.8 / n_a
    fills = ["#7f8c8d", "#2c3e50"]
    bx = np.arange(len(families))
    for k, atk in enumerate(attacks):
        ys = []
        ylo = []
        yhi = []
        for fam in families:
            row = next((r for r in attack_rows
                       if r["victim_family"] == fam and r["attack"] == atk), None)
            ys.append(row["top1"] if row else np.nan)
            ylo.append(row["top1"] - row["top1_ci_low"] if row else 0)
            yhi.append(row["top1_ci_high"] - row["top1"] if row else 0)
        offsets = (k - (n_a - 1) / 2) * bar_w
        ax_r.bar(bx + offsets, ys, bar_w, yerr=[ylo, yhi], color=fills[k],
                 edgecolor="white", linewidth=0.8, label=atk,
                 error_kw=dict(elinewidth=0.6, capsize=2, capthick=0.6))
    chance = attack_rows[0]["chance_top1"]
    ax_r.axhline(chance, color="#c0392b", linestyle="--", linewidth=0.8,
                 label=f"chance = {chance:.3f}")
    ax_r.set_xticks(bx)
    ax_r.set_xticklabels(families)
    ax_r.set_ylim(0, 1.05)
    ax_r.set_ylabel("Re-ID top-1 (all-models attack)")
    ax_r.set_title("Within-subject re-identification")
    ax_r.legend(frameon=False, loc="upper left", fontsize=8)
    ax_r.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
