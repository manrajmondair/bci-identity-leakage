"""A2 — cross-task subject re-identification.

Threat model: an attacker has access to a deployed motor-imagery decoder
plus labeled EEG from real-movement (motor-execution) sessions of the
enrolled users — and wants to attribute new EEG windows of imagined
movement back to those users.

Pipeline:
    Train victim on imagery train_runs           (same as A1)
    Train re-ID probe on EXECUTION-runs embeddings  →  subject_id
    Test probe on IMAGERY test_runs embeddings      →  subject_id

The victim's task-trained features are extracted from EEG of one
cognitive task (real movement), the probe is forced to attribute EEG
from a different cognitive task (imagined movement) of the same
subjects. Tests whether identity rides on cognitive-task-orthogonal
components — the realistic threat where a BCI service trained for one
task is deployed against EEG from another.

Numbers are directly comparable to A1 (same chance top-1, same victim,
same imagery test_runs) — the only thing different is what the probe
was trained on.

Usage
-----
    python -m experiments.04_a2_cross_task --smoke   # 10 subjects
    python -m experiments.04_a2_cross_task --all     # 104 subjects
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import numpy as np

from attacks.closed_set import closed_set_reid
from config import FIGURES_DIR, RESULTS_DIR
from data.physionet_loader import valid_subjects
from eval.plots import closed_set_bar_chart, closed_set_table
from models.eegnet import EEGNetVictim
from models.fbcsp import FBCSPVictim
from models.riemannian import RiemannianVictim
from preprocess.windows import windowed_subjects

VICTIM_TRAIN_RUNS = (4, 6, 8, 10)              # imagery train (same as A1)
PROBE_TRAIN_RUNS_EXEC = (3, 5, 7, 9, 11, 13)   # all execution runs
PROBE_TEST_RUNS_IMG = (12, 14)                 # imagery held-out (same as A1)


def _build_victim(name: str, *, n_channels: int, n_times: int, sfreq: float,
                  eegnet_epochs: int, seed: int):
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
    print(f"Subjects: {len(subjects)} (chance top-1 = {100/len(subjects):.2f}%)")
    print(f"Models: {args.models}", flush=True)
    print(f"Victim train runs (imagery)  {VICTIM_TRAIN_RUNS}")
    print(f"Probe train runs (execution) {PROBE_TRAIN_RUNS_EXEC}")
    print(f"Probe test runs  (imagery)   {PROBE_TEST_RUNS_IMG}\n", flush=True)

    print("Loading imagery + execution windows ...", flush=True)
    t0 = time.time()
    imagery = windowed_subjects(subjects, runs="imagery")
    execution = windowed_subjects(subjects, runs="execution")
    victim_train = imagery.filter_runs(list(VICTIM_TRAIN_RUNS))
    probe_train = execution.filter_runs(list(PROBE_TRAIN_RUNS_EXEC))
    probe_test = imagery.filter_runs(list(PROBE_TEST_RUNS_IMG))
    print(f"  loaded in {time.time() - t0:.1f}s | "
          f"victim_train={victim_train.n_windows} "
          f"probe_train(exec)={probe_train.n_windows} "
          f"probe_test(img)={probe_test.n_windows} "
          f"chans={imagery.n_channels} times={imagery.n_times}\n", flush=True)

    all_results = []
    for victim_name in args.models:
        print(f"=== {victim_name} ===", flush=True)
        victim = _build_victim(
            victim_name,
            n_channels=victim_train.n_channels, n_times=victim_train.n_times,
            sfreq=victim_train.sfreq,
            eegnet_epochs=args.eegnet_epochs, seed=args.seed,
        )
        t0 = time.time()
        victim.fit(victim_train.X, victim_train.y)
        task_acc = victim.score(probe_test.X, probe_test.y)
        print(f"  victim train+score: {time.time() - t0:.1f}s | "
              f"task_acc(imagery_test)={task_acc:.3f}", flush=True)

        t0 = time.time()
        results = closed_set_reid(
            victim, probe_train, probe_test,
            probes=("knn", "logreg"),
            bootstrap_n=args.bootstrap_n, seed=args.seed,
        )
        print(f"  attack: {time.time() - t0:.1f}s")
        for r in results:
            print(f"    {r.probe:7s}  top1={r.top1:.3f} "
                  f"[{r.top1_ci_low:.3f}, {r.top1_ci_high:.3f}]  "
                  f"top5={r.top5:.3f}  top10={r.top10:.3f}  "
                  f"(chance top1={r.chance_top1:.3f})")
            all_results.append({**asdict(r), "task_acc": task_acc})
        print()

    out_path = RESULTS_DIR / "04_a2_cross_task.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"Results written to {out_path}")

    fig_path = FIGURES_DIR / "04_a2_cross_task.pdf"
    closed_set_bar_chart(
        all_results, fig_path,
        title=f"A2 cross-task re-ID  ({len(subjects)} subj)\n"
              f"probe trained on execution-runs, tested on imagery-runs",
    )
    print(f"Figure written to {fig_path}\n")
    print(closed_set_table(all_results))


if __name__ == "__main__":
    main()

