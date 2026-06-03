"""A5 — per-subject membership inference (Shokri et al. 2017 -style).

Threat model: an adversary holds black-box access to a deployed BCI
decoder plus labeled EEG from a candidate subject. They want to know
whether that subject's data was used to train the decoder.

Standard shadow-model methodology, adapted to the BCI setting:

    1. Train N "shadow" EEGNets, each on a random 50% subset of the
       enrolled cohort. For every shadow we know which subjects were in
       its training set (members) and which were not (non-members).

    2. For each (shadow, subject) pair, compute two features that
       summarize how the shadow responds to the subject's EEG:
         - mean per-window cross-entropy loss against the true motor-
           imagery class labels
         - mean per-window max softmax probability ("confidence")
       Member subjects produce lower loss / higher confidence than
       non-members, and the gap is what the attack exploits.

    3. Train a small attack classifier on the shadow data: features ->
       is_member (binary).

    4. Train a TARGET EEGNet on its own random 50% split. Apply the
       attack classifier to the target's (subject, features) pairs.
       Report AUC and the (TPR - FPR) advantage at the optimal threshold.

This complements A1-A4 (which test whether RELEASED EMBEDDINGS leak
identity for re-identification) by testing the orthogonal threat: even
with only the trained-model API, can an adversary tell who was in the
training cohort?
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from models.eegnet import EEGNetVictim


@dataclass
class MembershipInferenceResult:
    n_shadows: int
    n_subjects: int
    n_target_members: int
    n_target_nonmembers: int
    auc: float
    auc_ci_low: float
    auc_ci_high: float
    advantage: float
    advantage_threshold: float
    n_attack_features: int


def _train_shadow(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_chans: int,
    n_times: int,
    n_classes: int,
    n_epochs: int,
    seed: int,
) -> EEGNetVictim:
    victim = EEGNetVictim(
        n_channels=n_chans, n_times=n_times, n_classes=n_classes,
        n_epochs=n_epochs, seed=seed, verbose=False,
    )
    victim.fit(X_train, y_train)
    return victim


@torch.no_grad()
def _per_subject_features(
    victim: EEGNetVictim,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    *,
    batch_size: int = 256,
) -> dict[int, tuple[float, float]]:
    """For each subject in `subject_ids`, return (mean_loss, mean_max_prob)
    of `victim` on that subject's windows."""
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)
    assert victim.model_ is not None
    device = victim.device
    victim.model_.eval()

    losses = np.empty(len(X), dtype=np.float32)
    confs = np.empty(len(X), dtype=np.float32)
    for start in range(0, len(X), batch_size):
        sl = slice(start, start + batch_size)
        # Apply the same input_scale fix the victim uses internally
        xb = torch.from_numpy(X[sl] * victim.input_scale).to(device)
        yb = torch.from_numpy(y[sl]).long().to(device)
        logits = victim.model_(xb)
        loss = F.cross_entropy(logits, yb, reduction="none")
        prob = torch.softmax(logits, dim=1).max(dim=1).values
        losses[sl] = loss.cpu().numpy().astype(np.float32)
        confs[sl] = prob.cpu().numpy().astype(np.float32)

    out: dict[int, tuple[float, float]] = {}
    for s in np.unique(subject_ids):
        mask = subject_ids == s
        out[int(s)] = (float(losses[mask].mean()), float(confs[mask].mean()))
    return out


def _bootstrap_auc(scores: np.ndarray, labels: np.ndarray,
                   *, n_resamples: int = 1000, seed: int = 0) -> tuple[float, float]:
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


def membership_inference(
    X_train: np.ndarray,
    y_train: np.ndarray,
    subject_ids_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    subject_ids_eval: np.ndarray,
    *,
    all_subjects: np.ndarray,
    n_chans: int,
    n_times: int,
    n_classes: int,
    n_shadows: int = 20,
    n_epochs: int = 30,
    member_frac: float = 0.5,
    seed: int = 0,
    verbose: bool = True,
) -> MembershipInferenceResult:
    """Run the per-subject membership-inference attack.

    `X_train, y_train, subject_ids_train` are the windows victims train on
    (across the full cohort). `X_eval, y_eval, subject_ids_eval` are the
    windows the attack reads to score membership per subject -- typically
    the same windows the victim was trained on (the standard MIA threat
    model: the attacker has labeled data and asks "was THIS subject's data
    used to train the model?").
    """
    rng = np.random.default_rng(seed)
    all_subjects = np.asarray(sorted(int(s) for s in np.unique(all_subjects)))
    n_per_member = int(round(member_frac * len(all_subjects)))

    # ---- 1. Train shadow models ----
    shadow_attack_rows: list[tuple[float, float, int]] = []
    if verbose:
        print(f"Training {n_shadows} shadow EEGNets ({n_epochs} epochs each) ...",
              flush=True)
    for i in range(n_shadows):
        members = sorted(rng.choice(all_subjects, size=n_per_member,
                                    replace=False).tolist())
        member_set = set(members)
        train_mask = np.isin(subject_ids_train, members)
        shadow = _train_shadow(
            X_train[train_mask], y_train[train_mask],
            n_chans=n_chans, n_times=n_times, n_classes=n_classes,
            n_epochs=n_epochs, seed=seed + i,
        )
        feats = _per_subject_features(
            shadow, X_eval, y_eval, subject_ids_eval,
        )
        for s, (lo, co) in feats.items():
            shadow_attack_rows.append((lo, co, int(s in member_set)))
        if verbose:
            print(f"  shadow {i + 1}/{n_shadows} | members={n_per_member}",
                  flush=True)

    # ---- 2. Train attack classifier on shadow features ----
    Z = np.array([(lo, co) for (lo, co, _) in shadow_attack_rows], dtype=np.float32)
    y = np.array([m for (_, _, m) in shadow_attack_rows], dtype=np.int64)
    attack = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0).fit(Z, y)

    # ---- 3. Train the TARGET model and score membership on its split ----
    target_members = sorted(rng.choice(all_subjects, size=n_per_member,
                                       replace=False).tolist())
    target_member_set = set(target_members)
    train_mask = np.isin(subject_ids_train, target_members)
    if verbose:
        print(f"Training the target EEGNet on {len(target_members)} subjects ...",
              flush=True)
    target = _train_shadow(
        X_train[train_mask], y_train[train_mask],
        n_chans=n_chans, n_times=n_times, n_classes=n_classes,
        n_epochs=n_epochs, seed=seed + n_shadows,
    )
    target_feats = _per_subject_features(target, X_eval, y_eval, subject_ids_eval)

    Z_target = np.array([target_feats[int(s)] for s in all_subjects], dtype=np.float32)
    y_target = np.array([int(int(s) in target_member_set) for s in all_subjects],
                        dtype=np.int64)
    proba = attack.predict_proba(Z_target)[:, 1]

    auc = float(roc_auc_score(y_target, proba))
    auc_lo, auc_hi = _bootstrap_auc(proba, y_target, seed=seed)

    # Attack advantage = max_t (TPR(t) - FPR(t))
    fpr, tpr, thresholds = roc_curve(y_target, proba)
    diffs = tpr - fpr
    idx = int(np.argmax(diffs))
    advantage = float(diffs[idx])
    adv_thresh = float(thresholds[idx])

    return MembershipInferenceResult(
        n_shadows=int(n_shadows),
        n_subjects=int(len(all_subjects)),
        n_target_members=int(y_target.sum()),
        n_target_nonmembers=int((1 - y_target).sum()),
        auc=auc, auc_ci_low=auc_lo, auc_ci_high=auc_hi,
        advantage=advantage, advantage_threshold=adv_thresh,
        n_attack_features=int(Z.shape[1]),
    )

