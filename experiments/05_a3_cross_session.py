"""A3 — cross-session subject re-identification on BCI Competition IV-2a.

Threat model: an attacker has access to a deployed motor-imagery decoder
trained on session-1 EEG of N enrolled users plus their labeled session-1
windows. They are now given an unlabeled EEG window from a session
recorded a different day, and want to attribute it to one of the
enrolled users.

This is the realistic biometric-linkage scenario. PhysioNet has a single
session per subject; BCI IV-2a is the only widely-used motor-imagery
dataset with two sessions on different days, so it is the only public
testbed where we can answer this question directly.

Pipeline:
    Train victim on all 9 subjects' session-1 (training session)
    Train re-ID probe on session-1 embeddings  →  subject_id
    Test probe on session-2 embeddings (different day)  →  subject_id
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import numpy as np

from attacks.closed_set import closed_set_reid
from config import FIGURES_DIR, RESULTS_DIR
from data.bciiv2a_loader import load_subject_session, valid_subjects
from eval.plots import closed_set_bar_chart, closed_set_table
from models.eegnet import EEGNetVictim
from models.fbcsp import FBCSPVictim
from models.riemannian import RiemannianVictim
from preprocess.windows import WindowedDataset


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


def _pool_subjects_session(subjects: list[int], session: str) -> WindowedDataset:
    """Pool all the subjects' windows for one session into a single dataset."""
    parts = [load_subject_session(s, session=session) for s in subjects]
    X = np.concatenate([p.X for p in parts], axis=0)
    y = np.concatenate([p.y for p in parts], axis=0)
    s = np.concatenate([p.subject_ids for p in parts], axis=0)
    # Make trial_ids globally unique by offsetting per subject
    t_parts = []
    for p in parts:
        offset = int(p.subject_ids[0]) * 100_000
        t_parts.append(p.trial_ids + offset)
    t = np.concatenate(t_parts, axis=0)
    r = np.concatenate([p.run_ids for p in parts], axis=0)
    sfreq = parts[0].sfreq
    chs = parts[0].channel_names
    for p in parts[1:]:
        assert p.channel_names == chs, "Channel order mismatch across subjects"
    return WindowedDataset(
        X=X, y=y, subject_ids=s, trial_ids=t, run_ids=r,
        sfreq=sfreq, channel_names=chs,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="3 subjects, EEGNet capped at 30 epochs.")
    p.add_argument("--all", action="store_true",
                   help="All 9 IV-2a subjects.")
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
        subjects = valid_subjects()[:3]
        args.eegnet_epochs = min(args.eegnet_epochs, 30)
    elif args.all:
        subjects = valid_subjects()
    else:
        p.error("Provide --smoke, --all, or --subjects")

    np.random.seed(args.seed)
    print(f"BCI IV-2a subjects: {len(subjects)} "
          f"(chance top-1 = {100/len(subjects):.2f}%)")
    print(f"Models: {args.models}\n", flush=True)

    # The loader downloads on first use via moabb. Both sessions of all
    # requested subjects pulled here; cached under ~/mne_data/MNE-bnci-data/.
    print("Loading BCI IV-2a sessions ...", flush=True)
    t0 = time.time()
    train_ds = _pool_subjects_session(subjects, "0train")
    test_ds = _pool_subjects_session(subjects, "1test")
    print(f"  loaded in {time.time() - t0:.1f}s | "
          f"session_T={train_ds.n_windows} session_E={test_ds.n_windows} "
          f"chans={train_ds.n_channels} times={train_ds.n_times}\n",
          flush=True)

    all_results = []
    for victim_name in args.models:
        print(f"=== {victim_name} ===", flush=True)
        victim = _build_victim(
            victim_name,
            n_channels=train_ds.n_channels, n_times=train_ds.n_times,
            sfreq=train_ds.sfreq,
            eegnet_epochs=args.eegnet_epochs, seed=args.seed,
        )
        t = time.time()
        # Victim trained on session-1 task labels (motor-imagery classes)
        victim.fit(train_ds.X, train_ds.y)
        task_acc = victim.score(test_ds.X, test_ds.y)
        print(f"  victim train+score: {time.time() - t:.1f}s | "
              f"task_acc(session2)={task_acc:.3f}", flush=True)

        t = time.time()
        # Probe trained on session-1 embeddings, tested on session-2
        results = closed_set_reid(
            victim, train_ds, test_ds,
            probes=("knn", "logreg"),
            bootstrap_n=args.bootstrap_n, seed=args.seed,
        )
        print(f"  attack: {time.time() - t:.1f}s")
        for r in results:
            print(f"    {r.probe:7s}  top1={r.top1:.3f} "
                  f"[{r.top1_ci_low:.3f}, {r.top1_ci_high:.3f}]  "
                  f"top5={r.top5:.3f}  top10={r.top10:.3f}  "
                  f"(chance top1={r.chance_top1:.3f})")
            all_results.append({**asdict(r), "task_acc": task_acc})
        print()

    out_path = RESULTS_DIR / "05_a3_cross_session.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"Results written to {out_path}")

    fig_path = FIGURES_DIR / "05_a3_cross_session.pdf"
    closed_set_bar_chart(
        all_results, fig_path,
        title=f"A3 cross-session re-ID  (BCI IV-2a, {len(subjects)} subj)\n"
              f"probe trained on session-1, tested on session-2",
    )
    print(f"Figure written to {fig_path}\n")
    print(closed_set_table(all_results))


if __name__ == "__main__":
    main()
