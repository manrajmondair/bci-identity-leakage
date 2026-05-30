"""A4 — open-set subject verification.

Trains a contrastive EEG embedding on a subset of subjects, then asks
whether two windows from a *held-out* set of subjects (whom the embedding
network has never seen) can be linked as same-vs-different person. This
is the strongest test of "EEG functions as a biometric template": the
embedding must generalize across people, not just across windows.

Reports ROC-AUC and Equal Error Rate (EER). AUC = 0.5 means the embedding
provides no identity signal on unseen subjects; AUC -> 1.0 means perfect
verification.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from models.contrastive import ContrastiveEEGNet, batch_hard_triplet_loss


@dataclass
class VerificationResult:
    n_train_subjects: int
    n_test_subjects: int
    n_train_windows: int
    n_test_windows: int
    n_pairs: int
    auc: float
    auc_ci_low: float
    auc_ci_high: float
    eer: float
    eer_threshold: float


def _train_contrastive(
    X: np.ndarray,
    subj: np.ndarray,
    *,
    n_chans: int,
    n_times: int,
    embed_dim: int = 64,
    n_epochs: int = 30,
    batch_size_subjects: int = 8,
    samples_per_subject: int = 4,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    margin: float = 0.2,
    device: str = "cuda",
    seed: int = 0,
    verbose: bool = True,
) -> ContrastiveEEGNet:
    """Train a contrastive embedding via batch-hard triplet loss."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = ContrastiveEEGNet(n_chans=n_chans, n_times=n_times,
                              embed_dim=embed_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    unique_subjects = np.unique(subj)
    subj_to_idx = {int(s): np.where(subj == s)[0] for s in unique_subjects}

    # An "epoch" here just means roughly enough batches to touch every window once.
    n_batches = max(1, len(X) // (batch_size_subjects * samples_per_subject))

    model.train()
    for epoch in range(n_epochs):
        running = 0.0
        for _ in range(n_batches):
            chosen_subjects = rng.choice(unique_subjects, size=batch_size_subjects,
                                         replace=False)
            batch_idx: list[int] = []
            batch_labels: list[int] = []
            for s in chosen_subjects:
                pool = subj_to_idx[int(s)]
                if len(pool) < samples_per_subject:
                    pick = rng.choice(pool, size=samples_per_subject, replace=True)
                else:
                    pick = rng.choice(pool, size=samples_per_subject, replace=False)
                batch_idx.extend(pick.tolist())
                batch_labels.extend([int(s)] * samples_per_subject)
            xb = torch.from_numpy(X[batch_idx].astype(np.float32, copy=False)).to(device)
            yb = torch.tensor(batch_labels, device=device)

            opt.zero_grad()
            emb = model(xb)
            loss = batch_hard_triplet_loss(emb, yb, margin=margin)
            loss.backward()
            opt.step()
            running += loss.item()
        if verbose and (epoch % 5 == 0 or epoch == n_epochs - 1):
            print(f"  epoch {epoch:3d}  loss={running / n_batches:.4f}", flush=True)

    model.eval()
    return model


@torch.no_grad()
def _embed_all(model: ContrastiveEEGNet, X: np.ndarray, *,
               device: str = "cuda", batch_size: int = 256) -> np.ndarray:
    parts = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size].astype(np.float32, copy=False)).to(device)
        parts.append(model(xb).cpu().numpy())
    return np.concatenate(parts, axis=0)


def _bootstrap_auc(
    scores: np.ndarray, labels: np.ndarray,
    *, n_resamples: int = 1000, seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(labels)
    samples = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        try:
            samples[i] = roc_auc_score(labels[idx], scores[idx])
        except ValueError:
            samples[i] = float("nan")
    return float(np.nanpercentile(samples, 2.5)), float(np.nanpercentile(samples, 97.5))


def open_set_verification(
    X_train: np.ndarray, subj_train: np.ndarray,
    X_test: np.ndarray, subj_test: np.ndarray,
    *,
    n_chans: int, n_times: int,
    trial_test: np.ndarray | None = None,
    embed_dim: int = 64,
    n_epochs: int = 30,
    n_pairs: int = 50_000,
    seed: int = 0,
    device: str = "cuda",
    verbose: bool = True,
) -> tuple[VerificationResult, np.ndarray, np.ndarray]:
    """Train a contrastive embedding on `*_train`, evaluate verification on `*_test`.

    Returns the result struct plus the (scores, labels) arrays for the
    sampled pairs so the caller can plot ROC / similarity histograms.

    Pass `trial_test` (the per-window trial id of every test window) so that
    same-subject verification pairs are drawn from two DIFFERENT trials. With
    2-s windows at 1-s stride there are ~3 heavily-overlapping windows per 4-s
    trial; a "same" pair drawn at the window level can be two near-duplicate
    windows of one trial, which is trivially similar and inflates AUC/EER. When
    `trial_test` is None the sampler falls back to window-level pairs but still
    forbids window-vs-itself pairs.
    """
    if verbose:
        print(f"Training contrastive embedding on "
              f"{len(np.unique(subj_train))} subjects "
              f"({len(X_train)} windows) ...", flush=True)
    model = _train_contrastive(
        X_train, subj_train,
        n_chans=n_chans, n_times=n_times, embed_dim=embed_dim,
        n_epochs=n_epochs, device=device, seed=seed, verbose=verbose,
    )
    if verbose:
        print(f"Embedding {len(X_test)} test windows from "
              f"{len(np.unique(subj_test))} unseen subjects ...", flush=True)
    Z_test = _embed_all(model, X_test, device=device)

    rng = np.random.default_rng(seed)
    test_subjects = np.unique(subj_test)
    subj_to_idx_test = {int(s): np.where(subj_test == s)[0] for s in test_subjects}
    half = n_pairs // 2

    # --- Same-subject pairs -------------------------------------------------
    # A "same" pair is two DISTINCT windows of one subject and, when trial ids
    # are supplied, from two DIFFERENT trials (so overlapping within-trial
    # windows can't pad the positive distribution). Pairs are collected into
    # lists and only valid ones are kept, so a subject with too few windows is
    # skipped rather than silently emitting a window-vs-itself pair at index 0.
    if trial_test is not None:
        trial_test = np.asarray(trial_test)
        subj_trial_idx: dict[int, dict[int, np.ndarray]] = {}
        for s in test_subjects:
            s = int(s)
            si = subj_to_idx_test[s]
            tids = trial_test[si]
            subj_trial_idx[s] = {int(t): si[tids == t] for t in np.unique(tids)}
        eligible_same = [s for s in subj_trial_idx if len(subj_trial_idx[s]) >= 2]
    else:
        eligible_same = [int(s) for s in test_subjects
                         if len(subj_to_idx_test[int(s)]) >= 2]

    if not eligible_same:
        raise ValueError(
            "open_set_verification: no test subject has two distinct "
            + ("trials" if trial_test is not None else "windows")
            + " to form a same-subject pair."
        )

    same_a_list: list[int] = []
    same_b_list: list[int] = []
    max_attempts = half * 50 + 1000
    attempts = 0
    while len(same_a_list) < half and attempts < max_attempts:
        attempts += 1
        s = int(rng.choice(eligible_same))
        if trial_test is not None:
            trials = list(subj_trial_idx[s].keys())
            ta, tb = rng.choice(len(trials), size=2, replace=False)
            a = int(rng.choice(subj_trial_idx[s][trials[ta]]))
            b = int(rng.choice(subj_trial_idx[s][trials[tb]]))
        else:
            a, b = (int(v) for v in rng.choice(subj_to_idx_test[s],
                                               size=2, replace=False))
        same_a_list.append(a)
        same_b_list.append(b)

    n_same = len(same_a_list)  # < half only on a pathologically small cohort

    # --- Different-subject pairs (matched count, for a balanced ROC) --------
    diff_a_list: list[int] = []
    diff_b_list: list[int] = []
    for _ in range(n_same):
        s_a, s_b = rng.choice(test_subjects, size=2, replace=False)
        diff_a_list.append(int(rng.choice(subj_to_idx_test[int(s_a)])))
        diff_b_list.append(int(rng.choice(subj_to_idx_test[int(s_b)])))

    same_a = np.asarray(same_a_list, dtype=np.int64)
    same_b = np.asarray(same_b_list, dtype=np.int64)
    diff_a = np.asarray(diff_a_list, dtype=np.int64)
    diff_b = np.asarray(diff_b_list, dtype=np.int64)

    # cosine similarity (Z is L2-normalized, so dot product = cosine)
    same_sim = (Z_test[same_a] * Z_test[same_b]).sum(axis=1)
    diff_sim = (Z_test[diff_a] * Z_test[diff_b]).sum(axis=1)
    scores = np.concatenate([same_sim, diff_sim])
    labels = np.concatenate([np.ones(n_same, dtype=np.int64),
                             np.zeros(n_same, dtype=np.int64)])

    auc = float(roc_auc_score(labels, scores))
    auc_lo, auc_hi = _bootstrap_auc(scores, labels, seed=seed)

    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    eer_thresh = float(thresholds[idx])

    result = VerificationResult(
        n_train_subjects=int(len(np.unique(subj_train))),
        n_test_subjects=int(len(test_subjects)),
        n_train_windows=int(len(X_train)),
        n_test_windows=int(len(X_test)),
        n_pairs=int(len(scores)),
        auc=auc, auc_ci_low=auc_lo, auc_ci_high=auc_hi,
        eer=eer, eer_threshold=eer_thresh,
    )
    return result, scores, labels
