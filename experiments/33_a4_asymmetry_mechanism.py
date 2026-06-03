"""Lee 2019 -> PhysioNet asymmetry mechanism test.

Experiment 26 reported that Lee 2019 -> PhysioNet A4 verification
collapses to AUC = 0.499 (chance), while the reverse direction holds
at AUC = 0.814 and IV-2a -> PhysioNet succeeds at 0.838 with only 9
training subjects. The working hypothesis is that the Lee 2019
training task (binary left/right hand) is too easy to induce
subject-discriminative features that transfer across recording
protocols. Richer-task corpora (PhysioNet 4-class, IV-2a 4-class)
appear to drive the embedder toward features that generalise.

This experiment falsifies-or-confirms that hypothesis with one
controlled change. We re-train the Lee 2019 contrastive embedder
exactly as in experiment 26, but with a synthetic 4-class label
constructed from each trial's (hand, half) pair:

    class 0 = left-hand,  first-half-of-trial
    class 1 = left-hand,  second-half-of-trial
    class 2 = right-hand, first-half-of-trial
    class 3 = right-hand, second-half-of-trial

The "half" axis is purely temporal (windows from t in [0, 2s) vs
t in [2s, 4s)) and carries no additional task semantics. It does,
however, force the contrastive's batch-hard triplet objective to
separate four distinct task contexts per subject instead of two,
which is the same structural pressure PhysioNet's 4-class label
applies during contrastive training.

If AUC on Lee 2019 -> PhysioNet rises materially above the original
0.499 with no other change, the task-complexity hypothesis is
supported. If it stays near chance, the asymmetry has a different
cause (recording-rig spectral content, channel set, sampling-rate
artefact post-resample).

Reads from the Lee 2019 compact cache and PhysioNet on-disk EDFs.
Set BCI_LEE2019_CACHE.

Usage
-----
    python -m experiments.33_a4_asymmetry_mechanism --smoke
    python -m experiments.33_a4_asymmetry_mechanism --all
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
from data.channel_subset import (
    channel_intersection,
    resample_windows,
    subset_channels,
)
from data.lee2019_loader import (
    load_subject_session_compact,
)
from data.lee2019_loader import (
    valid_subjects as lee_subjects,
)
from data.physionet_loader import valid_subjects as physionet_subjects
from preprocess.windows import WindowedDataset, windowed_subjects


def _pool_lee(subjects: list[int]) -> WindowedDataset:
    parts: list[WindowedDataset] = []
    for s in subjects:
        try:
            parts.append(load_subject_session_compact(s, session="session_1"))
        except Exception as exc:
            print(f"    !! lee subj{s} failed: {exc}", flush=True)
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


def _synthetic_4class(ds: WindowedDataset) -> np.ndarray:
    """Build (hand x first/second half) label per window.

    Each trial contributes three windows under the 2-s window / 1-s
    stride scheme. We split into early (first window) and late (last
    window); the middle window is dropped from the synthetic-label
    pool so the half assignment is unambiguous. Returns a per-window
    integer label in {0, 1, 2, 3} or -1 for windows to drop.
    """
    out = np.full(len(ds.y), fill_value=-1, dtype=np.int64)
    unique_trials = np.unique(ds.trial_ids)
    for t in unique_trials:
        mask = ds.trial_ids == t
        idxs = np.where(mask)[0]
        if len(idxs) < 2:
            continue
        first = idxs.min()
        last = idxs.max()
        hand = int(ds.y[first])      # 0 = left, 1 = right (per lee2019_prefetch convention)
        out[first] = 2 * hand + 0    # early
        out[last] = 2 * hand + 1     # late
    return out


def _offset_subject_ids(ds: WindowedDataset, offset: int) -> WindowedDataset:
    return WindowedDataset(
        X=ds.X, y=ds.y, subject_ids=ds.subject_ids + offset,
        trial_ids=ds.trial_ids, run_ids=ds.run_ids,
        sfreq=ds.sfreq, channel_names=ds.channel_names,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--n-train-subjects-lee", type=int, default=40)
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--n-pairs", type=int, default=50_000)
    p.add_argument("--target-sfreq", type=float, default=160.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.smoke:
        n_train_lee = 8
        phys = physionet_subjects()[:20]
        args.n_epochs = 10
    elif args.all:
        n_train_lee = args.n_train_subjects_lee
        phys = physionet_subjects()
    else:
        p.error("Provide --smoke or --all")

    rng = np.random.default_rng(args.seed)
    lee_all = lee_subjects()
    lee_train = sorted(int(s) for s in rng.choice(lee_all, size=n_train_lee, replace=False))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Asymmetry mechanism: Lee 2019 -> PhysioNet with synthetic 4-class label")
    print(f"Lee 2019 train subjects: {len(lee_train)}  |  PhysioNet test subjects: {len(phys)}")
    print(f"Target sfreq: {args.target_sfreq} Hz  Epochs: {args.n_epochs}  Pairs: {args.n_pairs}")
    print(f"Device: {device}\n", flush=True)

    print("Loading Lee 2019 session_1 windows ...", flush=True)
    t0 = time.time()
    train_lee = _pool_lee(lee_train)
    print(f"  loaded in {time.time() - t0:.1f}s  windows={train_lee.n_windows}",
          flush=True)

    print("Building synthetic 4-class label (hand x first/second-half) ...",
          flush=True)
    syn_y = _synthetic_4class(train_lee)
    keep = syn_y >= 0
    train_lee = WindowedDataset(
        X=train_lee.X[keep], y=syn_y[keep],
        subject_ids=train_lee.subject_ids[keep],
        trial_ids=train_lee.trial_ids[keep],
        run_ids=train_lee.run_ids[keep],
        sfreq=train_lee.sfreq, channel_names=train_lee.channel_names,
    )
    # Histogram so we can confirm the 4-class labels balance.
    counts = {int(c): int((train_lee.y == c).sum()) for c in sorted(np.unique(train_lee.y))}
    print(f"  synthetic label histogram: {counts}\n", flush=True)

    print("Loading PhysioNet imagery for the eval pool ...", flush=True)
    t0 = time.time()
    test_phys = windowed_subjects(phys, runs="imagery")
    test_phys = _offset_subject_ids(test_phys, 10_000)
    print(f"  loaded in {time.time() - t0:.1f}s  windows={test_phys.n_windows}\n",
          flush=True)

    # Align channel set + sample rate (mirror experiment 26).
    common = channel_intersection(train_lee.channel_names, test_phys.channel_names)
    train_a = subset_channels(train_lee, common)
    test_a = subset_channels(test_phys, common)
    train_a = resample_windows(train_a, args.target_sfreq)
    test_a = resample_windows(test_a, args.target_sfreq)
    print(f"Aligned to {len(common)} common channels at {args.target_sfreq} Hz")
    print(f"  train: {train_a.X.shape}  test: {test_a.X.shape}\n", flush=True)

    t0 = time.time()
    result, scores, labels = open_set_verification(
        train_a.X, train_a.subject_ids,
        test_a.X, test_a.subject_ids,
        trial_test=test_a.trial_ids,
        n_chans=train_a.n_channels, n_times=train_a.n_times,
        n_epochs=args.n_epochs, n_pairs=args.n_pairs,
        seed=args.seed, device=device, verbose=True,
    )
    dt = time.time() - t0
    print(f"\n  AUC = {result.auc:.4f}  "
          f"[{result.auc_ci_low:.4f}, {result.auc_ci_high:.4f}]")
    print(f"  EER = {result.eer:.4f}  total seconds: {dt:.1f}")

    # Comparison vs experiment 26's binary-task baseline
    # (Lee 2019 -> PhysioNet, results/26_a4_xds_lee2019_to_physionet.json, seed 0)
    BINARY_BASELINE_AUC = 0.4986738224
    auc_lift = result.auc - BINARY_BASELINE_AUC

    out = {
        **asdict(result),
        "direction": "lee2019_to_physionet_synthetic4class",
        "train_dataset": "Lee 2019 OpenBMI motor imagery (session 1), synthetic 4-class label",
        "test_dataset": "PhysioNet EEG-MMIDB (motor imagery)",
        "synthetic_label_histogram": counts,
        "common_channel_count": len(common),
        "common_channels": common,
        "target_sfreq_hz": float(args.target_sfreq),
        "binary_baseline_auc": BINARY_BASELINE_AUC,
        "auc_lift_over_binary_baseline": float(auc_lift),
        "hypothesis_supported": bool(auc_lift >= 0.10),
        "seed": args.seed,
    }
    out_path = RESULTS_DIR / "33_a4_asymmetry_mechanism.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults written to {out_path}")
    print(f"\nBinary baseline (exp 26):  AUC = {BINARY_BASELINE_AUC:.4f}")
    print(f"Synthetic 4-class:         AUC = {result.auc:.4f}")
    print(f"Lift:                       {auc_lift:+.4f} "
          f"({'HYPOTHESIS SUPPORTED' if auc_lift >= 0.10 else 'HYPOTHESIS NOT SUPPORTED'})")


if __name__ == "__main__":
    main()

