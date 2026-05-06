"""A1b — closed-set re-identification with per-subject (within-subject) victims.

Threat model: a BCI service ships a *personal* motor-imagery decoder for
each enrolled user (the actual deployment pattern for clinical BCIs that
need within-subject calibration to perform). An adversary holds all N
personal models plus the labeled training windows used to fit them, and
must attribute an unlabeled EEG window to a subject.

Two attack variants:

  Naive argmax-confidence:
    Run x through every personal model. Pick the subject whose model is
    most confident in any class. This is biased: a model whose training
    data was easily separable produces uniformly peaky softmaxes and
    "wins" the comparison regardless of who x came from. Reported here
    as the dumb baseline.

  Softmax-fingerprint probe:
    For every window, build a 4*N-dim "fingerprint" vector by stacking
    each model's softmax output. Train a logistic-regression probe on
    (training-window fingerprint -> subject_id), evaluate on test
    fingerprints. Calibration-invariant because the LR learns to weight
    each model's confidences appropriately. This is the actual attack.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import top_k_accuracy_score

from eval.bootstrap import grouped_bootstrap_ci
from models.base import VictimModel


@dataclass
class PerSubjectAttackResult:
    victim_family: str
    attack: str  # "argmax_conf" | "softmax_probe"
    n_subjects: int
    n_train_windows: int
    n_test_windows: int
    top1: float
    top1_ci_low: float
    top1_ci_high: float
    top5: float
    top10: float
    chance_top1: float


def _model_class_probs(model: VictimModel, X: np.ndarray) -> np.ndarray:
    """Return per-window full softmax / class-probability vector (n, n_classes)."""
    # EEGNet — softmax the network logits
    if hasattr(model, "model_") and model.model_ is not None and hasattr(model, "_iter_batches"):
        confs: list[np.ndarray] = []
        model.model_.eval()
        with torch.no_grad():
            for xb, _ in model._iter_batches(X.astype(np.float32, copy=False), None,
                                             shuffle=False):
                logits = model.model_(xb)
                p = torch.softmax(logits, dim=1)
                confs.append(p.cpu().numpy())
        return np.concatenate(confs, axis=0)

    # FBCSP — explicit pipeline (band features -> scaler -> LDA)
    if hasattr(model, "lda_") and hasattr(model, "scaler_") and hasattr(model, "_band_features"):
        F = model._band_features(X)
        F_std = model.scaler_.transform(F)
        return model.lda_.predict_proba(F_std)

    # Riemann — sklearn pipeline
    if hasattr(model, "_pipeline") and model._pipeline is not None:
        return model._pipeline.predict_proba(X.astype(np.float64, copy=False))

    raise NotImplementedError(f"Don't know how to get class probs from {type(model).__name__}")


def _fingerprint_matrix(
    victims_by_subject: dict[int, VictimModel],
    X: np.ndarray,
    *,
    sids_order: list[int],
) -> np.ndarray:
    """Return (n_windows, n_subjects * n_classes) softmax fingerprint matrix.

    Column block j corresponds to victim j, columns within the block are
    that victim's per-class softmax. We pre-allocate the full matrix and
    fill block-by-block to amortize per-victim setup overhead (BatchNorm
    eval mode, GPU transfers).
    """
    blocks: list[np.ndarray] = []
    for s in sids_order:
        probs = _model_class_probs(victims_by_subject[s], X)
        blocks.append(probs.astype(np.float32, copy=False))
    return np.concatenate(blocks, axis=1)


def per_subject_closed_set_reid(
    victims_by_subject: dict[int, VictimModel],
    train_X_by_subject: dict[int, np.ndarray],
    test_X_by_subject: dict[int, np.ndarray],
    test_trial_ids_by_subject: dict[int, np.ndarray],
    *,
    bootstrap_n: int = 1000,
    seed: int = 0,
) -> list[PerSubjectAttackResult]:
    """Run both within-subject re-ID attacks against the per-subject victims."""
    sids = sorted(victims_by_subject)
    n_subjects = len(sids)
    sid_to_idx = {s: j for j, s in enumerate(sids)}

    # Stack train + test windows with their subject ids and trial ids
    X_train = np.concatenate([train_X_by_subject[s] for s in sids], axis=0)
    y_train = np.concatenate([np.full(len(train_X_by_subject[s]), s) for s in sids])
    X_test = np.concatenate([test_X_by_subject[s] for s in sids], axis=0)
    y_test = np.concatenate([np.full(len(test_X_by_subject[s]), s) for s in sids])
    trials_test = np.concatenate([test_trial_ids_by_subject[s] for s in sids])

    # Compute the fingerprints. Each window is run through every personal model.
    Z_train = _fingerprint_matrix(victims_by_subject, X_train, sids_order=sids)
    Z_test = _fingerprint_matrix(victims_by_subject, X_test, sids_order=sids)

    family = next(iter(victims_by_subject.values())).name
    results: list[PerSubjectAttackResult] = []

    # ---- Naive argmax-confidence ----
    # Per window, take each victim's max-class probability, take the subject
    # whose model is most confident.
    n_classes = Z_train.shape[1] // n_subjects
    # Reshape (n, n_subjects, n_classes), then max over class -> (n, n_subjects)
    M_test = Z_test.reshape(-1, n_subjects, n_classes).max(axis=2)
    order = np.argsort(-M_test, axis=1)  # descending
    truth_idx = np.array([sid_to_idx[y] for y in y_test])
    correct = (order[:, 0] == truth_idx).astype(np.float64)
    ci = grouped_bootstrap_ci(correct, groups=trials_test, statistic=np.mean,
                              n_resamples=bootstrap_n, seed=seed)
    top5 = float((order[:, :5] == truth_idx[:, None]).any(axis=1).mean()) \
        if n_subjects > 5 else float("nan")
    top10 = float((order[:, :10] == truth_idx[:, None]).any(axis=1).mean()) \
        if n_subjects > 10 else float("nan")
    results.append(PerSubjectAttackResult(
        victim_family=family, attack="argmax_conf",
        n_subjects=n_subjects, n_train_windows=int(len(Z_train)),
        n_test_windows=int(len(Z_test)),
        top1=ci.point, top1_ci_low=ci.low, top1_ci_high=ci.high,
        top5=top5, top10=top10, chance_top1=1.0 / n_subjects,
    ))

    # ---- Softmax-fingerprint trained probe ----
    # Logistic regression on the 4*N-d fingerprint vector. Calibration-
    # invariant because the LR learns its own per-feature scaling.
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0)
    clf.fit(Z_train, y_train)
    proba = clf.predict_proba(Z_test)  # (n_test, n_classes_lr)
    preds = clf.predict(Z_test)
    correct = (preds == y_test).astype(np.float64)
    ci = grouped_bootstrap_ci(correct, groups=trials_test, statistic=np.mean,
                              n_resamples=bootstrap_n, seed=seed)
    top5_p = float(top_k_accuracy_score(y_test, proba, k=5, labels=clf.classes_)) \
        if n_subjects > 5 else float("nan")
    top10_p = float(top_k_accuracy_score(y_test, proba, k=10, labels=clf.classes_)) \
        if n_subjects > 10 else float("nan")
    results.append(PerSubjectAttackResult(
        victim_family=family, attack="softmax_probe",
        n_subjects=n_subjects, n_train_windows=int(len(Z_train)),
        n_test_windows=int(len(Z_test)),
        top1=ci.point, top1_ci_low=ci.low, top1_ci_high=ci.high,
        top5=top5_p, top10=top10_p, chance_top1=1.0 / n_subjects,
    ))

    return results
