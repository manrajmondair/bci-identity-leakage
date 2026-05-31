"""A2 vs resting-state — clean cross-task re-identification.

The original A2 (experiments/04) trains the re-ID probe on motor-EXECUTION
embeddings and tests on motor-IMAGERY embeddings. This contrasts two
related motor tasks; both share substantial premotor / SMA neural
activation, so the "task-orthogonal identity signal" claim is softer
than the headline framing suggests.

This experiment uses a much cleaner contrast: probe trained on
RESTING-STATE EEG (PhysioNet runs 1 = eyes open, 2 = eyes closed) and
tested on motor-imagery EEG. There's no shared task structure between
"sit still and stare" and "imagine moving your hand"; if the probe
still recovers identity at high accuracy, the identity signal really
is task-independent rather than just motor-shared.

Pipeline:
    Train victim on imagery train_runs (4, 6, 8, 10) — same as A1.
    Slide windows across the resting-state runs (1, 2) — these have no
        T1/T2 events; we just chop the continuous signal.
    Train re-ID probe on resting-state embeddings → subject_id.
    Test probe on imagery test_runs (12, 14) → subject_id.
    Numbers directly comparable to A1 / A2 (same chance top-1, same
    test-set windows).

Usage
-----
    python -m experiments.21_a2_vs_rest --smoke
    python -m experiments.21_a2_vs_rest --all
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import numpy as np

from attacks.closed_set import closed_set_reid
from config import (
    BANDPASS_HIGH,
    BANDPASS_LOW,
    FIGURES_DIR,
    PHYSIONET_RUNS_BASELINE,
    RESULTS_DIR,
    WINDOW_SECONDS,
    WINDOW_STRIDE_SECONDS,
)
from data.physionet_loader import load_subject, valid_subjects
from eval.plots import closed_set_bar_chart, closed_set_table
from models.eegnet import EEGNetVictim
from models.fbcsp import FBCSPVictim
from models.riemannian import RiemannianVictim
from preprocess.filtering import bandpass
from preprocess.windows import WindowedDataset, windowed_subjects

VICTIM_TRAIN_RUNS = (4, 6, 8, 10)
PROBE_TEST_RUNS_IMG = (12, 14)


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


def _slide_resting_subject(subject_id: int) -> WindowedDataset:
    """Slide fixed-length windows across the subject's resting-state runs.

    PhysioNet baseline runs (1 = eyes open, 2 = eyes closed) have no
    T1/T2 task annotations, so we just bandpass and chop the continuous
    signal into 2-s windows with 1-s stride. Per-window class label is
    set to the run index encoded as a placeholder (0 for run 1, 1 for
    run 2) — the probe ignores this and uses subject_id only.
    """
    rec = load_subject(subject_id, runs="baseline")
    raw = bandpass(rec.raw, BANDPASS_LOW, BANDPASS_HIGH, copy=True)
    sfreq = raw.info["sfreq"]
    win = int(round(WINDOW_SECONDS * sfreq))
    stride = int(round(WINDOW_STRIDE_SECONDS * sfreq))
    data = raw.get_data()  # (n_channels, n_total_samples)

    X_list, y_list, t_list, r_list = [], [], [], []
    next_trial = 0
    for run_idx_in_rec, (lo, hi) in enumerate(rec.run_boundaries):
        run_no = rec.runs[run_idx_in_rec]
        if run_no not in PHYSIONET_RUNS_BASELINE:
            continue
        # Slide windows across this run only
        starts = np.arange(lo, hi - win + 1, stride)
        for st in starts:
            X_list.append(data[:, st:st + win].astype(np.float32, copy=False))
            # Pseudo-label: 0 for eyes-open, 1 for eyes-closed.
            y_list.append(0 if run_no == 1 else 1)
            t_list.append(next_trial)
            r_list.append(run_no)
            next_trial += 1

    if not X_list:
        raise RuntimeError(f"Subject {subject_id}: no resting-state windows.")

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.int64)
    t = np.asarray(t_list, dtype=np.int64)
    r = np.asarray(r_list, dtype=np.int64)
    s = np.full_like(t, subject_id)
    chs = tuple(raw.ch_names)
    return WindowedDataset(
        X=X, y=y, subject_ids=s, trial_ids=t, run_ids=r,
        sfreq=sfreq, channel_names=chs,
    )


def _resting_state_pool(subjects: list[int]) -> WindowedDataset:
    """Pool every subject's resting-state windows into a single dataset.

    Globally-unique trial_ids (subject*100k + within-subject trial) so
    bootstrap-by-trial groups don't collide across subjects.
    """
    parts = []
    for s in subjects:
        try:
            parts.append(_slide_resting_subject(s))
        except Exception as exc:
            print(f"    !! subject {s}: {type(exc).__name__}: {exc}", flush=True)
    if not parts:
        raise RuntimeError("No resting-state subjects loaded.")
    X = np.concatenate([p.X for p in parts], axis=0)
    y = np.concatenate([p.y for p in parts], axis=0)
    s = np.concatenate([p.subject_ids for p in parts], axis=0)
    t = np.concatenate([p.trial_ids + int(p.subject_ids[0]) * 100_000
                        for p in parts], axis=0)
    r = np.concatenate([p.run_ids for p in parts], axis=0)
    sfreq = parts[0].sfreq
    chs = parts[0].channel_names
    return WindowedDataset(
        X=X, y=y, subject_ids=s, trial_ids=t, run_ids=r,
        sfreq=sfreq, channel_names=chs,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="10 subjects, EEGNet capped at 30 epochs.")
    p.add_argument("--all", action="store_true")
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
    print(f"Victim train runs (imagery)        {VICTIM_TRAIN_RUNS}")
    print(f"Probe train runs (resting-state)   {PHYSIONET_RUNS_BASELINE}")
    print(f"Probe test runs (imagery)          {PROBE_TEST_RUNS_IMG}\n", flush=True)

    print("Loading imagery + resting-state windows ...", flush=True)
    t0 = time.time()
    imagery = windowed_subjects(subjects, runs="imagery")
    rest = _resting_state_pool(subjects)
    victim_train = imagery.filter_runs(list(VICTIM_TRAIN_RUNS))
    probe_test = imagery.filter_runs(list(PROBE_TEST_RUNS_IMG))
    print(f"  loaded in {time.time() - t0:.1f}s | "
          f"victim_train={victim_train.n_windows} "
          f"probe_train(rest)={rest.n_windows} "
          f"probe_test(img)={probe_test.n_windows} "
          f"chans={imagery.n_channels}\n", flush=True)

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
            victim, rest, probe_test,
            probes=("knn", "logreg"),
            bootstrap_n=args.bootstrap_n, seed=args.seed,
        )
        print(f"  attack: {time.time() - t0:.1f}s")
        for r in results:
            print(f"    {r.probe:7s}  top1={r.top1:.3f} "
                  f"[{r.top1_ci_low:.3f}, {r.top1_ci_high:.3f}]  "
                  f"top5={r.top5:.3f}  top10={r.top10:.3f}  "
                  f"(chance top1={r.chance_top1:.3f})")
            all_results.append({**asdict(r), "task_acc": task_acc,
                                "probe_train_runs": list(PHYSIONET_RUNS_BASELINE)})
        print()

    out_path = RESULTS_DIR / "21_a2_vs_rest.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"Results written to {out_path}")

    fig_path = FIGURES_DIR / "21_a2_vs_rest.pdf"
    closed_set_bar_chart(
        all_results, fig_path,
        title=f"A2 cross-task re-ID  ({len(subjects)} subj)\n"
              f"probe trained on RESTING-STATE, tested on motor-imagery",
    )
    print(f"Figure written to {fig_path}\n")
    print(closed_set_table(all_results))


if __name__ == "__main__":
    main()
