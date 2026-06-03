"""A4 cross-dataset, symmetric — extends experiment 13 to three directions.

Experiment 13 covered PhysioNet -> IV-2a (AUC 0.726). A reviewer can fairly
ask: is that direction-specific? Does the transfer claim survive when we
swap train and test, or when we substitute a third dataset?

This script supports four named directions, each a separate run:

    iv2a_to_physionet      IV-2a 9 subjects (train) -> PhysioNet 104 unseen (eval)
    physionet_to_lee2019   PhysioNet 80 subj (train) -> Lee 2019 54 unseen (eval)
    lee2019_to_physionet   Lee 2019 40 subj (train) -> PhysioNet 104 unseen (eval)
    iv2a_to_lee2019        IV-2a 9 subj (train)    -> Lee 2019 54 unseen (eval)

For each direction we compute the channel intersection between the two
datasets, resample to a shared sampling rate, train the contrastive
EEGNet on the source side, then verify identity on the held-out subjects
of the destination side. Reports ROC-AUC and EER, same plotting / scores
file as experiment 13.

Reads Lee 2019 from the compact npz cache (set BCI_LEE2019_CACHE).

Usage
-----
    python -m experiments.26_a4_cross_dataset_symmetric --direction iv2a_to_physionet --all
    python -m experiments.26_a4_cross_dataset_symmetric --direction physionet_to_lee2019 --all
    python -m experiments.26_a4_cross_dataset_symmetric --direction lee2019_to_physionet --all
    python -m experiments.26_a4_cross_dataset_symmetric --direction iv2a_to_lee2019 --all
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
from data.bciiv2a_loader import load_subject_session as iv2a_load
from data.bciiv2a_loader import valid_subjects as iv2a_subjects
from data.channel_subset import (
    channel_intersection,
    resample_windows,
    subset_channels,
)
from data.lee2019_loader import load_subject_session_compact as lee_load
from data.lee2019_loader import valid_subjects as lee_subjects
from data.physionet_loader import valid_subjects as physionet_subjects
from preprocess.windows import WindowedDataset, windowed_subjects


def _pool_iv2a(subjects: list[int]) -> WindowedDataset:
    parts: list[WindowedDataset] = []
    for s in subjects:
        for sess in ("0train", "1test"):
            try:
                parts.append(iv2a_load(s, session=sess))
            except Exception as exc:
                print(f"    !! iv2a subj{s} {sess} failed: {exc}", flush=True)
    if not parts:
        raise RuntimeError("no IV-2a windows loaded")
    return _stack_with_unique_trial_ids(parts)


def _pool_lee(subjects: list[int], session: str = "session_1") -> WindowedDataset:
    parts: list[WindowedDataset] = []
    for s in subjects:
        try:
            parts.append(lee_load(s, session=session))
        except Exception as exc:
            print(f"    !! lee subj{s} {session} failed: {exc}", flush=True)
    if not parts:
        raise RuntimeError("no Lee 2019 windows loaded")
    return _stack_with_unique_trial_ids(parts)


def _stack_with_unique_trial_ids(parts: list[WindowedDataset]) -> WindowedDataset:
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


def _offset_subject_ids(ds: WindowedDataset, offset: int) -> WindowedDataset:
    return WindowedDataset(
        X=ds.X, y=ds.y, subject_ids=ds.subject_ids + offset,
        trial_ids=ds.trial_ids, run_ids=ds.run_ids,
        sfreq=ds.sfreq, channel_names=ds.channel_names,
    )


def _align(train: WindowedDataset, test: WindowedDataset,
           target_sfreq: float) -> tuple[WindowedDataset, WindowedDataset, list[str]]:
    """Compute channel intersection, subset both sides, resample both sides."""
    common = channel_intersection(train.channel_names, test.channel_names)
    if not common:
        raise RuntimeError("no common channels between datasets")
    train_a = subset_channels(train, common)
    test_a = subset_channels(test, common)
    train_a = resample_windows(train_a, target_sfreq)
    test_a = resample_windows(test_a, target_sfreq)
    return train_a, test_a, common


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--direction", required=True,
                   choices=["iv2a_to_physionet",
                            "physionet_to_lee2019",
                            "lee2019_to_physionet",
                            "iv2a_to_lee2019"])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--n-train-subjects-physionet", type=int, default=80)
    p.add_argument("--n-train-subjects-lee", type=int, default=40)
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--n-pairs", type=int, default=50_000)
    p.add_argument("--target-sfreq", type=float, default=160.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not (args.smoke or args.all):
        p.error("Provide --smoke or --all")

    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Direction: {args.direction}")
    print(f"Device: {device}")
    print(f"Target sfreq: {args.target_sfreq} Hz")
    print(f"Epochs: {args.n_epochs}  Pairs: {args.n_pairs}\n", flush=True)

    t0 = time.time()

    if args.direction == "iv2a_to_physionet":
        iv = iv2a_subjects() if args.all else iv2a_subjects()[:3]
        phys = physionet_subjects() if args.all else physionet_subjects()[:20]
        train_ds = _pool_iv2a(iv)
        test_ds_full = windowed_subjects(phys, runs="imagery")
        # Offset PhysioNet subject ids so they can't collide with IV-2a's
        # 1..9 range during pair sampling logic.
        test_ds_full = _offset_subject_ids(test_ds_full, 10_000)
        train_dataset_label = "BCI Competition IV-2a (both sessions, pooled)"
        test_dataset_label = "PhysioNet EEG-MMIDB (motor imagery)"

    elif args.direction == "iv2a_to_lee2019":
        iv = iv2a_subjects() if args.all else iv2a_subjects()[:3]
        lee_subj_pool = lee_subjects() if args.all else lee_subjects()[:6]
        train_ds = _pool_iv2a(iv)
        test_ds_full = _pool_lee(lee_subj_pool, "session_1")
        test_ds_full = _offset_subject_ids(test_ds_full, 20_000)
        train_dataset_label = "BCI Competition IV-2a (both sessions, pooled)"
        test_dataset_label = "Lee 2019 OpenBMI motor imagery (session 1)"

    elif args.direction == "physionet_to_lee2019":
        phys_all = physionet_subjects()
        n_pt = args.n_train_subjects_physionet if args.all else 16
        phys_train = sorted(int(s) for s in rng.choice(phys_all, size=n_pt, replace=False))
        lee_subj_pool = lee_subjects() if args.all else lee_subjects()[:6]
        train_ds = windowed_subjects(phys_train, runs="imagery")
        test_ds_full = _pool_lee(lee_subj_pool, "session_1")
        test_ds_full = _offset_subject_ids(test_ds_full, 20_000)
        train_dataset_label = (f"PhysioNet EEG-MMIDB imagery "
                               f"({n_pt}-subject random subset)")
        test_dataset_label = "Lee 2019 OpenBMI motor imagery (session 1)"

    elif args.direction == "lee2019_to_physionet":
        lee_all = lee_subjects()
        n_lt = args.n_train_subjects_lee if args.all else 8
        lee_train = sorted(int(s) for s in rng.choice(lee_all, size=n_lt, replace=False))
        phys = physionet_subjects() if args.all else physionet_subjects()[:20]
        train_ds = _pool_lee(lee_train, "session_1")
        test_ds_full = windowed_subjects(phys, runs="imagery")
        test_ds_full = _offset_subject_ids(test_ds_full, 10_000)
        train_dataset_label = (f"Lee 2019 OpenBMI motor imagery (session 1) "
                               f"({n_lt}-subject random subset)")
        test_dataset_label = "PhysioNet EEG-MMIDB (motor imagery)"

    print(f"Loaded train+test in {time.time() - t0:.1f}s\n"
          f"  train: {train_ds.X.shape} @ {train_ds.sfreq:.0f} Hz "
          f"({len(np.unique(train_ds.subject_ids))} subj, "
          f"{len(train_ds.channel_names)} chan)\n"
          f"  test : {test_ds_full.X.shape} @ {test_ds_full.sfreq:.0f} Hz "
          f"({len(np.unique(test_ds_full.subject_ids))} subj, "
          f"{len(test_ds_full.channel_names)} chan)",
          flush=True)

    # Align: channel intersection + resample to target_sfreq
    train_a, test_a, common = _align(train_ds, test_ds_full,
                                     target_sfreq=args.target_sfreq)
    print(f"\nAligned to common 22+ channel set: {len(common)} channels "
          f"@ {args.target_sfreq} Hz\n"
          f"  train aligned: {train_a.X.shape}\n"
          f"  test  aligned: {test_a.X.shape}\n", flush=True)

    t0 = time.time()
    result, scores, labels = open_set_verification(
        train_a.X, train_a.subject_ids,
        test_a.X, test_a.subject_ids,
        trial_test=test_a.trial_ids,
        n_chans=train_a.n_channels, n_times=train_a.n_times,
        n_epochs=args.n_epochs, n_pairs=args.n_pairs,
        seed=args.seed, device=device, verbose=True,
    )
    print(f"\n  AUC = {result.auc:.4f} "
          f"[{result.auc_ci_low:.4f}, {result.auc_ci_high:.4f}]")
    print(f"  EER = {result.eer:.4f}  total seconds: {time.time() - t0:.1f}")

    out = {
        **asdict(result),
        "direction": args.direction,
        "train_dataset": train_dataset_label,
        "test_dataset": test_dataset_label,
        "common_channel_count": len(common),
        "common_channels": common,
        "target_sfreq_hz": float(args.target_sfreq),
        "seed": args.seed,
    }
    out_path = RESULTS_DIR / f"26_a4_xds_{args.direction}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults written to {out_path}")

    scores_path = RESULTS_DIR / f"26_a4_xds_{args.direction}_scores.npz"
    np.savez(scores_path, scores=scores, labels=labels)
    print(f"Per-pair scores written to {scores_path}")


if __name__ == "__main__":
    main()

