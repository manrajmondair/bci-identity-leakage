"""A1 — closed-set subject re-identification.

Threat model: an adversary has white-box access to a motor-imagery decoder
trained on a known cohort of N subjects, and access to some labeled EEG
recordings from those same subjects (for instance, from the consent
records the company kept). The adversary wants to predict which of the
N enrolled subjects produced a new EEG window.

This is a closed-set 1-of-N classification problem. It is the weakest of
the five attacks (the others test cross-task, cross-session, open-set
verification on unseen subjects, and membership inference); it remains
as the reference column on the privacy-utility plots.

Pipeline
--------
    Train task-decoder victim   on (X_train_runs, y_motor_imagery)
    Extract victim embeddings   on X_train_runs and X_test_runs
    Fit subject-ID probe        on (Z_train, subject_id)
    Predict subject ID          on Z_test
    Report top-1 / top-5 / top-10 accuracy with bootstrap CIs
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import top_k_accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from eval.bootstrap import grouped_bootstrap_ci
from models.base import VictimModel
from preprocess.windows import WindowedDataset


@dataclass
class ClosedSetResult:
    victim: str
    probe: str
    n_subjects: int
    n_train_windows: int
    n_test_windows: int
    top1: float
    top1_ci_low: float
    top1_ci_high: float
    top5: float
    top10: float
    chance_top1: float


def _fit_probe(kind: str, Z_train: np.ndarray, y_train: np.ndarray, n_classes: int):
    if kind == "knn":
        # Use cosine kNN with k=5, weighted by similarity. For ~10k train windows
        # and 100s of dims, brute-force is fine and avoids ANN approximations.
        clf = KNeighborsClassifier(
            n_neighbors=5, metric="cosine", weights="distance", algorithm="brute",
        )
    elif kind == "logreg":
        clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0)
    else:
        raise ValueError(kind)
    return clf.fit(Z_train, y_train)


def _topk_proba(clf, Z_test: np.ndarray, classes: np.ndarray, k: int) -> float:
    """Top-k accuracy from a fitted probabilistic classifier."""
    proba = clf.predict_proba(Z_test)  # (n_test, n_classes)
    return float(top_k_accuracy_score(classes, proba, k=k, labels=clf.classes_))


def closed_set_reid(
    victim: VictimModel,
    train_set: WindowedDataset,
    test_set: WindowedDataset,
    *,
    probes: tuple[str, ...] = ("knn", "logreg"),
    bootstrap_n: int = 1000,
    seed: int = 0,
) -> list[ClosedSetResult]:
    """Run the closed-set re-ID attack against a fitted victim model.

    Parameters
    ----------
    victim : an already-fit VictimModel (we don't re-train it here; the
        caller chooses whether the victim was trained per-subject or
        cross-subject).
    train_set, test_set : the same train/test windows the victim was
        trained on (subject ids must overlap).
    probes : which probe classifiers to evaluate.
    """
    Z_train = victim.embed(train_set.X)
    Z_test = victim.embed(test_set.X)
    y_train = train_set.subject_ids
    y_test = test_set.subject_ids
    classes = np.unique(np.concatenate([y_train, y_test]))
    n_subjects = len(classes)

    out: list[ClosedSetResult] = []
    for probe_kind in probes:
        clf = _fit_probe(probe_kind, Z_train, y_train, n_classes=n_subjects)

        # Top-k from predicted probabilities. top-k for k >= n_classes is
        # trivially 1.0, so we report NaN there to avoid presenting a
        # meaningless number alongside the real ones.
        top5 = _topk_proba(clf, Z_test, y_test, k=5) if n_subjects > 5 else float("nan")
        top10 = _topk_proba(clf, Z_test, y_test, k=10) if n_subjects > 10 else float("nan")

        # Bootstrap CI on top-1, resampling whole TRIALS so within-trial
        # correlated windows do not give us spuriously tight bounds.
        preds = clf.predict(Z_test)
        correct = (preds == y_test).astype(np.float64)
        ci = grouped_bootstrap_ci(
            correct, groups=test_set.trial_ids,
            statistic=np.mean, n_resamples=bootstrap_n, seed=seed,
        )

        out.append(ClosedSetResult(
            victim=victim.name,
            probe=probe_kind,
            n_subjects=n_subjects,
            n_train_windows=len(Z_train),
            n_test_windows=len(Z_test),
            top1=ci.point,
            top1_ci_low=ci.low,
            top1_ci_high=ci.high,
            top5=top5,
            top10=top10,
            chance_top1=1.0 / n_subjects,
        ))
    return out

