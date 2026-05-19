"""D3 stronger MIA — DP-aware shadow attacker against a DP-SGD-trained victim.

Experiment 08 / 16 run the Shokri shadow methodology on undefended
EEGNet, FBCSP, and Riemann victims. The published MIA tradition tests
the defended setting by training shadows IN THE SAME DEFENDED PIPELINE
as the target — the attacker's shadow distribution then matches the
defender's noise distribution, and the resulting attack is uniformly
stronger than running undefended shadows against a defended target.

This experiment runs that protocol explicitly for D3:

    Target  — EEGNet, DP-SGD ε=3, δ=1e-5, max_grad_norm=1
    Shadows — n_shadows EEGNets, each DP-SGD ε=3, δ=1e-5,
              random 50% subject split per shadow
    Attack  — per-(shadow, subject) features
              (mean per-window CE loss, mean max-softmax),
              logistic regression on the shadow rows,
              evaluated on the target's split.

If the formal (ε, δ) bound holds empirically in the DP-aware setting,
the MIA AUC stays near 0.5 + the (small) DP slack. If AUC -> the
no-defense 0.878 figure even when shadows are DP-trained, the empirical
gap between "DP holds vs encoder fine-tune" and "DP holds vs DP-aware
MIA" tells us something about what attack the bound actually covers.

Reuses `attacks.membership_inference.membership_inference` after
re-pointing the shadow / target trainers from `EEGNetVictim` to
`DPSGDVictim`. The function takes an `optional` victim_factory hook
via the local `_dp_mi` adapter below.

Usage
-----
    python -m experiments.27_d3_membership_aware_attacker --smoke
    python -m experiments.27_d3_membership_aware_attacker --all
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from config import RESULTS_DIR
from data.physionet_loader import valid_subjects
from defenses.dp_sgd import DPSGDVictim
from preprocess.windows import windowed_subjects

VICTIM_TRAIN_RUNS = (4, 6, 8, 10)


@dataclass
class DPAwareMIAResult:
    n_shadows: int
    n_subjects: int
    n_target_members: int
    n_target_nonmembers: int
    target_epsilon: float
    target_final_epsilon: float | None
    target_delta: float
    max_grad_norm: float
    auc: float
    auc_ci_low: float
    auc_ci_high: float
    advantage: float
    advantage_threshold: float
    baseline_auc_undefended_eegnet: float


def _train_dp_shadow(X, y, *,
                     n_chans, n_times, n_classes,
                     n_epochs, target_eps, target_delta,
                     max_grad_norm, seed) -> DPSGDVictim:
    v = DPSGDVictim(
        n_channels=n_chans, n_times=n_times, n_classes=n_classes,
        n_epochs=n_epochs, batch_size=256, lr=1.0,
        target_epsilon=float(target_eps), target_delta=float(target_delta),
        max_grad_norm=float(max_grad_norm),
        seed=seed, verbose=False,
    )
    v.fit(X, y)
    return v


@torch.no_grad()
def _per_subject_features(victim: DPSGDVictim,
                          X: np.ndarray, y: np.ndarray, subject_ids: np.ndarray,
                          *, batch_size: int = 256) -> dict[int, tuple[float, float]]:
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)
    assert victim.model_ is not None
    device = victim.device
    victim.model_.eval()
    losses = np.empty(len(X), dtype=np.float32)
    confs = np.empty(len(X), dtype=np.float32)
    for start in range(0, len(X), batch_size):
        sl = slice(start, start + batch_size)
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


def _bootstrap_auc(scores, labels, n_resamples=1000, seed=0):
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--n-shadows", type=int, default=8)
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--target-epsilon", type=float, default=3.0)
    p.add_argument("--target-delta", type=float, default=1e-5)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--member-frac", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.smoke:
        subjects = valid_subjects()[:20]
        args.n_shadows = 3
        args.n_epochs = 12
    elif args.all:
        subjects = valid_subjects()
    else:
        p.error("Provide --smoke or --all")

    print(f"Subjects: {len(subjects)}")
    print(f"DP shadows: {args.n_shadows} | epochs/shadow: {args.n_epochs}")
    print(f"ε={args.target_epsilon} δ={args.target_delta:.0e} "
          f"max_grad_norm={args.max_grad_norm}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n", flush=True)

    print("Loading imagery windows ...", flush=True)
    t0 = time.time()
    full = windowed_subjects(subjects, runs="imagery")
    train = full.filter_runs(list(VICTIM_TRAIN_RUNS))
    print(f"  loaded in {time.time() - t0:.1f}s | "
          f"train_windows={train.n_windows} chans={train.n_channels}\n",
          flush=True)

    rng = np.random.default_rng(args.seed)
    all_subj = np.asarray(sorted(int(s) for s in np.unique(train.subject_ids)))
    n_per = int(round(args.member_frac * len(all_subj)))

    shadow_rows: list[tuple[float, float, int]] = []
    final_eps_history: list[float] = []
    for i in range(args.n_shadows):
        members = sorted(rng.choice(all_subj, size=n_per, replace=False).tolist())
        member_set = set(members)
        mask = np.isin(train.subject_ids, members)
        t = time.time()
        shadow = _train_dp_shadow(
            train.X[mask], train.y[mask],
            n_chans=train.n_channels, n_times=train.n_times, n_classes=4,
            n_epochs=args.n_epochs, target_eps=args.target_epsilon,
            target_delta=args.target_delta, max_grad_norm=args.max_grad_norm,
            seed=args.seed + i,
        )
        if shadow.final_epsilon_ is not None:
            final_eps_history.append(float(shadow.final_epsilon_))
        feats = _per_subject_features(shadow, train.X, train.y, train.subject_ids)
        for s, (lo, co) in feats.items():
            shadow_rows.append((lo, co, int(s in member_set)))
        print(f"  shadow {i + 1}/{args.n_shadows}  "
              f"ε_final={(shadow.final_epsilon_ or float('nan')):.2f}  "
              f"members={n_per}  {time.time() - t:.0f}s", flush=True)
        # Free GPU between shadows
        del shadow
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    Z = np.array([(lo, co) for (lo, co, _) in shadow_rows], dtype=np.float32)
    y_attack = np.array([m for (_, _, m) in shadow_rows], dtype=np.int64)
    attack = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0).fit(Z, y_attack)

    # Target = one fresh DP-SGD model on its own random split
    target_members = sorted(rng.choice(all_subj, size=n_per, replace=False).tolist())
    target_set = set(target_members)
    print(f"\nTraining target DP-EEGNet ({len(target_members)} subjects) ...",
          flush=True)
    t = time.time()
    target = _train_dp_shadow(
        train.X[np.isin(train.subject_ids, target_members)],
        train.y[np.isin(train.subject_ids, target_members)],
        n_chans=train.n_channels, n_times=train.n_times, n_classes=4,
        n_epochs=args.n_epochs, target_eps=args.target_epsilon,
        target_delta=args.target_delta, max_grad_norm=args.max_grad_norm,
        seed=args.seed + args.n_shadows,
    )
    print(f"  target trained in {time.time() - t:.0f}s  "
          f"ε_final={(target.final_epsilon_ or float('nan')):.2f}",
          flush=True)
    target_feats = _per_subject_features(target, train.X, train.y, train.subject_ids)

    Z_target = np.array([target_feats[int(s)] for s in all_subj], dtype=np.float32)
    y_target = np.array([int(int(s) in target_set) for s in all_subj], dtype=np.int64)
    proba = attack.predict_proba(Z_target)[:, 1]

    auc = float(roc_auc_score(y_target, proba))
    auc_lo, auc_hi = _bootstrap_auc(proba, y_target, seed=args.seed)
    fpr, tpr, thresh = roc_curve(y_target, proba)
    idx = int(np.argmax(tpr - fpr))
    advantage = float(tpr[idx] - fpr[idx])
    adv_thresh = float(thresh[idx])

    BASELINE_UNDEFENDED = 0.878  # from experiments/08_a5_membership_inference.json

    result = DPAwareMIAResult(
        n_shadows=int(args.n_shadows),
        n_subjects=int(len(all_subj)),
        n_target_members=int(y_target.sum()),
        n_target_nonmembers=int((1 - y_target).sum()),
        target_epsilon=float(args.target_epsilon),
        target_final_epsilon=(float(target.final_epsilon_)
                              if target.final_epsilon_ is not None else None),
        target_delta=float(args.target_delta),
        max_grad_norm=float(args.max_grad_norm),
        auc=auc, auc_ci_low=auc_lo, auc_ci_high=auc_hi,
        advantage=advantage, advantage_threshold=adv_thresh,
        baseline_auc_undefended_eegnet=BASELINE_UNDEFENDED,
    )

    print(f"\n  DP-aware MIA AUC = {auc:.4f} [{auc_lo:.4f}, {auc_hi:.4f}]")
    print(f"  Advantage (TPR - FPR) = {advantage:.4f}")
    print(f"  Undefended-EEGNet baseline AUC (exp 08): {BASELINE_UNDEFENDED:.4f}\n")

    # Filename keys off target eps so a sweep across {0.5, 1.0, 3.0, ...}
    # leaves each run's JSON in place. The original eps=3 file used the
    # un-suffixed name; we keep that for backwards compatibility.
    if abs(args.target_epsilon - 3.0) < 1e-9:
        out_path = RESULTS_DIR / "27_d3_membership_aware_attacker.json"
    else:
        out_path = (RESULTS_DIR
                    / f"27_d3_membership_aware_attacker_eps{args.target_epsilon}.json")
    out_path.write_text(json.dumps({**asdict(result),
                                    "shadow_epsilon_final_history": final_eps_history,
                                    "seed": args.seed}, indent=2))
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
