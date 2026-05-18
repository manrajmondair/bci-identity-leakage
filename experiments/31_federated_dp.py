"""D4 — Federated DP-FedAvg on PhysioNet motor imagery.

Trains an EEGNet via central-DP FedAvg with 104 clients (each subject
is one client). Each round: random 50% subset of clients trains locally
for 3 epochs at SGD lr=5e-3, server clips per-client model deltas to
||Δ||₂ ≤ 1, sums, adds Gaussian noise with σ = 0.4, divides by
participant count, applies update.

After 30 rounds we run the same closed-set A1 + encoder-fine-tune
adaptive attacks used against centralised DP-SGD in experiments 10 /
18 / 29. The headline comparison: does federated DP-FedAvg buy the
same empirical privacy as centralised DP-SGD at a comparable ε
budget? Or does the participant-level (vs sample-level) DP weaken the
protection materially?

The deployment story this experiment proves out: a federated BCI
service where raw EEG never leaves the user device is the cleanest
defense story for §1.1 of the milestone — and we want to know
whether it actually delivers privacy under the same attack suite.

Usage
-----
    python -m experiments.31_federated_dp --smoke
    python -m experiments.31_federated_dp --all
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import top_k_accuracy_score

from attacks.closed_set import closed_set_reid
from config import RESULTS_DIR
from data.physionet_loader import valid_subjects
from defenses.federated import FederatedDPVictim
from eval.bootstrap import grouped_bootstrap_ci
from experiments.eegnet_helpers import finetune_to_reid_head
from preprocess.windows import windowed_subjects

VICTIM_TRAIN_RUNS = (4, 6, 8, 10)
VICTIM_TEST_RUNS = (12, 14)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--n-rounds", type=int, default=30)
    p.add_argument("--participant-fraction", type=float, default=0.5)
    p.add_argument("--local-epochs", type=int, default=3)
    p.add_argument("--local-lr", type=float, default=5e-3)
    p.add_argument("--clip-norm", type=float, default=1.0)
    p.add_argument("--noise-sigma", type=float, default=0.4)
    p.add_argument("--ft-epochs", type=int, default=15)
    p.add_argument("--bootstrap-n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.smoke:
        subjects = valid_subjects()[:10]
        args.n_rounds = 6
        args.local_epochs = 2
        args.ft_epochs = 5
    elif args.all:
        subjects = valid_subjects()
    else:
        p.error("Provide --smoke or --all")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(args.seed)
    print(f"Clients: {len(subjects)}  Rounds: {args.n_rounds}")
    print(f"Participant fraction: {args.participant_fraction}  "
          f"Local epochs: {args.local_epochs}")
    print(f"Clip norm: {args.clip_norm}  σ: {args.noise_sigma}")
    print(f"Device: {device}\n", flush=True)

    print("Loading imagery windows ...", flush=True)
    t0 = time.time()
    full = windowed_subjects(subjects, runs="imagery")
    train = full.filter_runs(list(VICTIM_TRAIN_RUNS))
    test = full.filter_runs(list(VICTIM_TEST_RUNS))
    print(f"  loaded in {time.time() - t0:.1f}s | "
          f"train={train.n_windows}  test={test.n_windows}\n", flush=True)

    # ---- Train federated victim ----
    print(f"=== training federated DP victim ({args.n_rounds} rounds) ===",
          flush=True)
    t0 = time.time()
    victim = FederatedDPVictim(
        n_channels=train.n_channels, n_times=train.n_times, n_classes=4,
        n_rounds=args.n_rounds,
        participant_fraction=args.participant_fraction,
        local_epochs=args.local_epochs, local_lr=args.local_lr,
        clip_norm=args.clip_norm, noise_sigma=args.noise_sigma,
        seed=args.seed, verbose=True,
    )
    victim.fit(train.X, train.y, client_ids=train.subject_ids)
    task_acc = victim.score(test.X, test.y)
    eps_informal = victim.informal_epsilon_estimate()
    print(f"\n  trained in {time.time() - t0:.0f}s  task_acc={task_acc:.3f}  "
          f"informal_ε≈{eps_informal:.2f} (Gaussian-mechanism composition)\n",
          flush=True)

    # ---- Attack 1: logreg probe ----
    print(f"=== attack 1: logreg probe ===", flush=True)
    t0 = time.time()
    a1 = closed_set_reid(victim, train, test, probes=("logreg",),
                        bootstrap_n=args.bootstrap_n, seed=args.seed)[0]
    print(f"  top1 = {a1.top1:.3f} [{a1.top1_ci_low:.3f}, {a1.top1_ci_high:.3f}]  "
          f"({time.time() - t0:.0f}s)\n", flush=True)

    # ---- Attack 2: encoder fine-tune ----
    print(f"=== attack 2: encoder fine-tune ===", flush=True)
    t0 = time.time()
    ft_model, sid_to_idx = finetune_to_reid_head(
        victim, train.X, train.subject_ids,
        device=device, n_epochs=args.ft_epochs, seed=args.seed,
        source="vanilla",   # Federated victim has no Opacus hooks
    )

    @torch.no_grad()
    def _predict_with_proba(model, X, sid_to_idx, n_classes):
        inv = {v: k for k, v in sid_to_idx.items()}
        model.eval()
        chunks = []
        for i in range(0, len(X), 256):
            xb = torch.from_numpy(X[i:i + 256].astype(np.float32, copy=False)).to(device)
            chunks.append(F.softmax(model(xb), dim=1).cpu().numpy())
        proba = np.concatenate(chunks, axis=0)
        pred_idx = proba.argmax(axis=1)
        sorted_classes = np.array([inv[i] for i in range(n_classes)])
        preds = np.array([inv[int(i)] for i in pred_idx])
        return preds, proba, sorted_classes

    n_subj = len(np.unique(train.subject_ids))
    preds, proba, sorted_classes = _predict_with_proba(
        ft_model, test.X, sid_to_idx, n_classes=n_subj,
    )
    correct = (preds == test.subject_ids).astype(np.float64)
    ci = grouped_bootstrap_ci(correct, groups=test.trial_ids,
                              statistic=np.mean,
                              n_resamples=args.bootstrap_n, seed=args.seed)
    top5 = float(top_k_accuracy_score(test.subject_ids, proba, k=5,
                                      labels=sorted_classes))
    top10 = float(top_k_accuracy_score(test.subject_ids, proba, k=10,
                                       labels=sorted_classes))
    print(f"  top1 = {ci.point:.3f} [{ci.low:.3f}, {ci.high:.3f}]  "
          f"({time.time() - t0:.0f}s)", flush=True)

    payload = {
        "n_subjects": int(len(subjects)),
        "chance_top1": float(1.0 / len(subjects)),
        "task_acc": float(task_acc),
        "n_rounds": int(args.n_rounds),
        "participant_fraction": float(args.participant_fraction),
        "local_epochs": int(args.local_epochs),
        "local_lr": float(args.local_lr),
        "clip_norm": float(args.clip_norm),
        "noise_sigma": float(args.noise_sigma),
        "informal_epsilon_participant_level": float(eps_informal),
        "round_log": [vars(r) for r in victim.round_log_],
        "attack_logreg": asdict(a1),
        "attack_finetune": {
            "top1": float(ci.point),
            "top1_ci_low": float(ci.low),
            "top1_ci_high": float(ci.high),
            "top5": float(top5),
            "top10": float(top10),
            "n_test_windows": int(test.n_windows),
            "n_subjects": int(n_subj),
            "chance_top1": float(1.0 / n_subj),
        },
        "seed": int(args.seed),
    }
    out_path = RESULTS_DIR / "31_federated_dp.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
