"""A5 on the classical victims (FBCSP+LDA and Riemann tangent-space).

The original A5 result covers EEGNet only. This extends membership
inference to the two classical pipelines so we can compare MI
vulnerability across the three victim families.

Same Shokri-style methodology as A5: N shadow models on random 50%
subject splits + 1 target; per-(subject, model) features = (mean per-
window cross-entropy loss against true MI labels, mean max class
probability); attack classifier = logreg.

We use --victim {fbcsp,riemann} and reduced shadow counts so each victim
fits in the 1-hour budget independently. Riemann is fast (~1 min per
shadow); FBCSP is slow (~4 min per shadow on L4 / Mac CPU) so we cap
N_shadows there.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from config import RESULTS_DIR
from data.physionet_loader import valid_subjects
from models.fbcsp import FBCSPVictim
from models.riemannian import RiemannianVictim
from preprocess.windows import windowed_subjects

VICTIM_TRAIN_RUNS = (4, 6, 8, 10)


@dataclass
class ClassicalMIResult:
    victim_family: str
    n_shadows: int
    n_subjects: int
    n_target_members: int
    n_target_nonmembers: int
    auc: float
    auc_ci_low: float
    auc_ci_high: float
    advantage: float
    advantage_threshold: float


def _per_subject_features(victim, X: np.ndarray, y: np.ndarray,
                          subject_ids: np.ndarray) -> dict[int, tuple[float, float]]:
    """Return {subject_id: (mean per-window CE loss, mean max-prob)}."""
    # Both FBCSP and Riemann expose predict_proba on their inner classifier.
    # Re-extract embeddings + run their respective predict_proba paths.
    if isinstance(victim, FBCSPVictim):
        F_feat = victim._band_features(X)
        F_std = victim.scaler_.transform(F_feat)
        proba = victim.lda_.predict_proba(F_std)
    elif isinstance(victim, RiemannianVictim):
        proba = victim._pipeline.predict_proba(X.astype(np.float64, copy=False))
    else:
        raise NotImplementedError(type(victim).__name__)

    # Map proba columns to canonical class index using clf.classes_
    if isinstance(victim, FBCSPVictim):
        classes_ = victim.lda_.classes_
    else:
        classes_ = victim._pipeline.named_steps["clf"].classes_
    class_to_col = {int(c): i for i, c in enumerate(classes_)}

    # Per-window CE loss against the true class
    eps = 1e-8
    losses = np.empty(len(X), dtype=np.float64)
    for i in range(len(X)):
        true_col = class_to_col[int(y[i])]
        losses[i] = -np.log(proba[i, true_col] + eps)
    confs = proba.max(axis=1)

    out: dict[int, tuple[float, float]] = {}
    for s in np.unique(subject_ids):
        mask = subject_ids == s
        out[int(s)] = (float(losses[mask].mean()), float(confs[mask].mean()))
    return out


def _train_shadow(victim_kind: str, X_train, y_train, *, sfreq, seed):
    if victim_kind == "fbcsp":
        v = FBCSPVictim(sfreq=sfreq, n_classes=4)
    elif victim_kind == "riemann":
        v = RiemannianVictim(n_classes=4, seed=seed)
    else:
        raise ValueError(victim_kind)
    v.fit(X_train, y_train)
    return v


def _bootstrap_auc(scores, labels, *, n_resamples=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(labels)
    samples = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        try:
            samples[i] = roc_auc_score(labels[idx], scores[idx])
        except ValueError:
            samples[i] = float("nan")
    return (float(np.nanpercentile(samples, 2.5)),
            float(np.nanpercentile(samples, 97.5)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--victim", choices=["fbcsp", "riemann"], required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--n-shadows", type=int, default=None,
                   help="Defaults: 12 for FBCSP, 20 for Riemann.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.n_shadows is None:
        args.n_shadows = 12 if args.victim == "fbcsp" else 20

    if args.smoke:
        subjects = valid_subjects()[:20]
        args.n_shadows = 4
    elif args.all:
        subjects = valid_subjects()
    else:
        p.error("Provide --smoke or --all")

    rng = np.random.default_rng(args.seed)
    print(f"Victim family: {args.victim}")
    print(f"Subjects: {len(subjects)}  shadows: {args.n_shadows}\n", flush=True)

    print("Loading windowed data ...", flush=True)
    full = windowed_subjects(subjects, runs="imagery")
    train = full.filter_runs(list(VICTIM_TRAIN_RUNS))
    print(f"  loaded | windows={train.n_windows}\n", flush=True)

    member_frac = 0.5
    n_per_member = int(round(member_frac * len(subjects)))
    all_subjects = np.array(sorted(int(s) for s in subjects))

    # ---- Train shadow models, collect attack-training rows ----
    shadow_rows: list[tuple[float, float, int]] = []
    for i in range(args.n_shadows):
        members = sorted(rng.choice(all_subjects, size=n_per_member,
                                    replace=False).tolist())
        member_set = set(members)
        mask = np.isin(train.subject_ids, members)
        t0 = time.time()
        v = _train_shadow(args.victim, train.X[mask], train.y[mask],
                          sfreq=train.sfreq, seed=args.seed + i)
        feats = _per_subject_features(v, train.X, train.y, train.subject_ids)
        for s, (lo, co) in feats.items():
            shadow_rows.append((lo, co, int(s in member_set)))
        print(f"  shadow {i + 1}/{args.n_shadows}  {time.time() - t0:.0f}s  "
              f"members={len(members)}", flush=True)

    Z = np.array([(lo, co) for (lo, co, _) in shadow_rows], dtype=np.float32)
    y = np.array([m for (_, _, m) in shadow_rows], dtype=np.int64)
    attack = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0).fit(Z, y)

    # ---- Train target ----
    target_members = sorted(rng.choice(all_subjects, size=n_per_member,
                                       replace=False).tolist())
    target_set = set(target_members)
    mask = np.isin(train.subject_ids, target_members)
    print(f"\nTraining target on {len(target_members)} subjects ...",
          flush=True)
    target = _train_shadow(args.victim, train.X[mask], train.y[mask],
                           sfreq=train.sfreq, seed=args.seed + args.n_shadows)
    target_feats = _per_subject_features(target, train.X, train.y,
                                          train.subject_ids)
    Z_t = np.array([target_feats[int(s)] for s in all_subjects], dtype=np.float32)
    y_t = np.array([int(int(s) in target_set) for s in all_subjects], dtype=np.int64)
    proba = attack.predict_proba(Z_t)[:, 1]

    auc = float(roc_auc_score(y_t, proba))
    auc_lo, auc_hi = _bootstrap_auc(proba, y_t, seed=args.seed)
    fpr, tpr, thresholds = roc_curve(y_t, proba)
    diffs = tpr - fpr
    idx = int(np.argmax(diffs))
    advantage = float(diffs[idx])
    adv_thresh = float(thresholds[idx])

    result = ClassicalMIResult(
        victim_family=args.victim,
        n_shadows=int(args.n_shadows),
        n_subjects=int(len(all_subjects)),
        n_target_members=int(y_t.sum()),
        n_target_nonmembers=int((1 - y_t).sum()),
        auc=auc, auc_ci_low=auc_lo, auc_ci_high=auc_hi,
        advantage=advantage, advantage_threshold=adv_thresh,
    )

    print()
    print(f"  AUC = {result.auc:.4f} [{result.auc_ci_low:.4f}, "
          f"{result.auc_ci_high:.4f}]")
    print(f"  Advantage (TPR - FPR) = {result.advantage:.4f}  @ "
          f"threshold {result.advantage_threshold:.3f}")

    out_path = RESULTS_DIR / f"16_a5_{args.victim}_mi.json"
    out_path.write_text(json.dumps(asdict(result), indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
