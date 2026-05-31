"""A4 cross-dataset — does the EEG biometric template transfer across
datasets, devices, and cognitive tasks?

Train a contrastive EEGNet on PhysioNet imagery (80 subjects, 22 channels
matching the BCI IV-2a montage, resampled to 160 Hz). Evaluate same/
different verification on BCI IV-2a session-1 (9 subjects, 22 channels,
resampled to 160 Hz, completely different subjects, completely different
recording rig + country + cognitive-task class set).

If AUC stays substantially above 0.5, the biometric claim transcends the
specific dataset: it is a property of EEG itself, not a PhysioNet
artifact. This is the strongest single test of the central claim.

Reports ROC-AUC, EER on (X_test, subj_test) pairs sampled balanced
same vs different.
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
from data.bciiv2a_loader import load_subject_session
from data.bciiv2a_loader import valid_subjects as iv2a_subjects
from data.channel_subset import (
    channel_intersection,
    resample_windows,
    subset_channels,
)
from data.physionet_loader import valid_subjects as physionet_subjects
from preprocess.windows import windowed_subjects


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="20 PhysioNet train + 3 IV-2a test, 10 epochs.")
    p.add_argument("--all", action="store_true",
                   help="Full cross-dataset eval.")
    p.add_argument("--n-train-subjects", type=int, default=80,
                   help="PhysioNet subjects for contrastive training.")
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--n-pairs", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.smoke:
        physionet_train = physionet_subjects()[:20]
        iv2a_test = iv2a_subjects()[:3]
        args.n_epochs = 10
    elif args.all:
        physionet_train = physionet_subjects()[:args.n_train_subjects]
        iv2a_test = iv2a_subjects()
    else:
        p.error("Provide --smoke or --all")

    print(f"Train (PhysioNet imagery): {len(physionet_train)} subjects")
    print(f"Test  (BCI IV-2a session 1): {len(iv2a_test)} subjects")
    print(f"Epochs: {args.n_epochs}  pairs: {args.n_pairs}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n", flush=True)

    # ---- Load IV-2a session 1 once to discover the channel set / sfreq ----
    print("Loading IV-2a session 1 ...", flush=True)
    iv2a_per_subj = []
    for s in iv2a_test:
        iv2a_per_subj.append(load_subject_session(s, session="0train"))
    iv2a_chs = iv2a_per_subj[0].channel_names
    iv2a_sfreq = iv2a_per_subj[0].sfreq
    print(f"  IV-2a sfreq: {iv2a_sfreq} Hz   channels: {len(iv2a_chs)}", flush=True)

    # ---- Load PhysioNet imagery for the train subjects ----
    print("Loading PhysioNet imagery ...", flush=True)
    t0 = time.time()
    phys = windowed_subjects(physionet_train, runs="imagery")
    print(f"  PhysioNet sfreq: {phys.sfreq} Hz   channels: {phys.n_channels}   "
          f"windows: {phys.n_windows}", flush=True)

    # ---- Identify common channels (verified empirically: IV-2a ⊂ PhysioNet) ----
    common = channel_intersection(iv2a_chs, phys.channel_names)
    if len(common) != len(iv2a_chs):
        raise RuntimeError(
            f"Expected IV-2a channels to be a subset of PhysioNet's; got "
            f"intersection of {len(common)} but IV-2a has {len(iv2a_chs)}"
        )
    print(f"  Common channel set: {len(common)} channels (used as cross-"
          f"dataset shared montage)", flush=True)

    # ---- Subset PhysioNet to those 22 channels ----
    phys_22 = subset_channels(phys, common)
    print(f"  PhysioNet (subset to common): shape {phys_22.X.shape} "
          f"@ {phys_22.sfreq} Hz", flush=True)

    # ---- Resample IV-2a to PhysioNet's 160 Hz, then concatenate across subjects ----
    print("Resampling IV-2a to 160 Hz and pooling across subjects ...",
          flush=True)
    target_sfreq = phys_22.sfreq
    iv2a_resampled = []
    for ds in iv2a_per_subj:
        # IV-2a's loader already returns the 22 EEG channels in standard
        # order; subset_channels here is a guard against ordering drift.
        ds_sub = subset_channels(ds, common)
        ds_resampled = resample_windows(ds_sub, target_sfreq)
        iv2a_resampled.append(ds_resampled)

    # Pool IV-2a windows across subjects
    X_iv = np.concatenate([d.X for d in iv2a_resampled], axis=0)
    subj_iv = np.concatenate([d.subject_ids for d in iv2a_resampled], axis=0)
    # Per-window trial ids so same-subject verification pairs come from
    # different trials (within-subject uniqueness is all the sampler needs).
    trial_iv = np.concatenate([d.trial_ids for d in iv2a_resampled], axis=0)
    # Differentiate IV-2a subject ids from PhysioNet's by adding 10000 offset
    # so the embedder doesn't accidentally see them as PhysioNet labels.
    subj_iv_offset = subj_iv + 10000
    print(f"  IV-2a (resampled): shape {X_iv.shape} @ {target_sfreq} Hz   "
          f"subjects: {sorted(set(int(s) for s in subj_iv))}\n", flush=True)

    # The two arrays now have the SAME shape along axes 1 and 2:
    assert X_iv.shape[1] == phys_22.X.shape[1] == len(common)
    assert X_iv.shape[2] == phys_22.X.shape[2]
    print(f"Architecture-matched: train {phys_22.X.shape}, "
          f"test {X_iv.shape}\n", flush=True)
    print(f"Total load + prep time: {time.time() - t0:.1f}s\n", flush=True)

    # ---- Train contrastive on PhysioNet, eval on IV-2a ----
    t0 = time.time()
    result, scores, labels = open_set_verification(
        phys_22.X, phys_22.subject_ids,
        X_iv, subj_iv_offset,
        trial_test=trial_iv,
        n_chans=phys_22.n_channels, n_times=phys_22.n_times,
        n_epochs=args.n_epochs, n_pairs=args.n_pairs,
        seed=args.seed, device=device, verbose=True,
    )
    dt = time.time() - t0
    print(f"\n  AUC = {result.auc:.4f} [{result.auc_ci_low:.4f}, "
          f"{result.auc_ci_high:.4f}]")
    print(f"  EER = {result.eer:.4f}")
    print(f"  Total seconds: {dt:.1f}")

    out = {
        **asdict(result),
        "train_dataset": "PhysioNet EEG-MMIDB imagery",
        "test_dataset": "BCI Competition IV-2a session 1",
        "train_subjects": [int(s) for s in physionet_train],
        "test_subjects": [int(s) for s in iv2a_test],
        "common_channel_count": len(common),
        "common_channels": common,
        "shared_sfreq_hz": float(target_sfreq),
        "iv2a_native_sfreq_hz": float(iv2a_sfreq),
    }
    out_path = RESULTS_DIR / "13_a4_cross_dataset.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults written to {out_path}")

    scores_path = RESULTS_DIR / "13_a4_cross_dataset_scores.npz"
    np.savez(scores_path, scores=scores, labels=labels)
    print(f"Per-pair scores written to {scores_path}")


if __name__ == "__main__":
    main()
