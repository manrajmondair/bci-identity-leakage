"""D3 ε sweep — full privacy-utility Pareto under generic and adaptive attack.

The original D3 result was reported at a single point ε=3. A reviewer
can fairly ask whether the picture changes in the strong-privacy regime
ε ∈ [0.5, 1] (where task accuracy is more pressured) or in the loose
regime ε ≥ 10 (where the formal guarantee buys very little). This
experiment sweeps five ε values and runs both attacks that matter in
practice (logreg probe and encoder fine-tune) against each:

    ε ∈ { 0.5, 1.0, 3.0, 10.0, ∞ (= no DP) }

For every ε we report:
  - final ε achieved by the RDP accountant (target ε is requested)
  - task accuracy on the held-out runs
  - A1 closed-set re-ID top-1 under the generic logreg probe
  - A1 closed-set re-ID top-1 under encoder fine-tune adaptive attacker

The result JSON is the canonical input for the privacy-utility Pareto
figure across the formal-DP frontier.

Wall budget on L4: ~140 min total (5 DP trainings × ~18 min + 5 fine-
tunes × ~5 min + 5 logreg attacks × ~1 min).

Usage
-----
    python -m experiments.29_d3_eps_sweep --smoke
    python -m experiments.29_d3_eps_sweep --all
    python -m experiments.29_d3_eps_sweep --all --epsilons 0.5 1.0 3.0
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
from defenses.dp_sgd import DPSGDVictim
from eval.bootstrap import grouped_bootstrap_ci
from experiments.eegnet_helpers import finetune_to_reid_head
from preprocess.windows import windowed_subjects

VICTIM_TRAIN_RUNS = (4, 6, 8, 10)
VICTIM_TEST_RUNS = (12, 14)


def _parse_epsilons(values: list[str]) -> list[float | None]:
    out: list[float | None] = []
    for s in values:
        s = s.lower()
        if s in ("none", "inf", "no_dp", "infty", "infinity"):
            out.append(None)
        else:
            out.append(float(s))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--epsilons", nargs="+",
                   default=["0.5", "1.0", "3.0", "10.0", "none"])
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--n-epochs", type=int, default=40)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1.0)
    p.add_argument("--ft-epochs", type=int, default=15)
    p.add_argument("--ft-lr", type=float, default=5e-4)
    p.add_argument("--bootstrap-n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.smoke:
        subjects = valid_subjects()[:10]
        args.epsilons = ["1.0", "none"]
        args.n_epochs = 10
        args.ft_epochs = 5
    elif args.all:
        subjects = valid_subjects()
    else:
        p.error("Provide --smoke or --all")

    epsilons = _parse_epsilons(args.epsilons)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Subjects: {len(subjects)}  chance top-1 = {100/len(subjects):.2f}%")
    print(f"ε sweep: {epsilons}  δ={args.delta:.0e}  epochs/condition={args.n_epochs}")
    print(f"Device: {device}\n", flush=True)

    print("Loading windowed imagery ...", flush=True)
    t0 = time.time()
    full = windowed_subjects(subjects, runs="imagery")
    train = full.filter_runs(list(VICTIM_TRAIN_RUNS))
    test = full.filter_runs(list(VICTIM_TEST_RUNS))
    print(f"  loaded in {time.time() - t0:.1f}s | "
          f"train={train.n_windows} test={test.n_windows} "
          f"chans={train.n_channels}\n", flush=True)

    pareto = []
    for eps in epsilons:
        eps_label = "no_dp" if eps is None else f"eps_{eps}"
        print(f"=== ε target = {eps if eps is not None else '∞ (no DP)'} ===",
              flush=True)
        t0 = time.time()
        victim = DPSGDVictim(
            n_channels=train.n_channels, n_times=train.n_times, n_classes=4,
            n_epochs=args.n_epochs, batch_size=args.batch_size, lr=args.lr,
            target_epsilon=eps, target_delta=args.delta,
            max_grad_norm=args.max_grad_norm,
            seed=args.seed, verbose=False,
        )
        victim.fit(train.X, train.y)
        task_acc = victim.score(test.X, test.y)
        eps_final = victim.final_epsilon_
        print(f"  train+task: {time.time() - t0:.0f}s  "
              f"task_acc={task_acc:.3f}  "
              f"ε_final={(eps_final if eps_final is not None else float('inf')):.2f}",
              flush=True)

        # Attack 1: logreg probe
        t0 = time.time()
        logreg_res = closed_set_reid(
            victim, train, test, probes=("logreg",),
            bootstrap_n=args.bootstrap_n, seed=args.seed,
        )[0]
        print(f"  logreg probe   top1={logreg_res.top1:.3f} "
              f"[{logreg_res.top1_ci_low:.3f}, {logreg_res.top1_ci_high:.3f}]  "
              f"({time.time() - t0:.0f}s)", flush=True)

        # Attack 2: encoder fine-tune (adaptive)
        t0 = time.time()
        ft_model, sid_to_idx = finetune_to_reid_head(
            victim, train.X, train.subject_ids,
            device=device, n_epochs=args.ft_epochs, lr=args.ft_lr,
            seed=args.seed, source="dp" if eps is not None else "dp",
        )

        @torch.no_grad()
        def _predict_with_proba(model, X, sid_to_idx, n_classes):
            inv = {v: k for k, v in sid_to_idx.items()}
            model.eval()
            chunks = []
            for i in range(0, len(X), 256):
                xb = torch.from_numpy(
                    X[i:i + 256].astype(np.float32, copy=False)
                ).to(device)
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
        print(f"  fine-tune       top1={ci.point:.3f} "
              f"[{ci.low:.3f}, {ci.high:.3f}]  "
              f"({time.time() - t0:.0f}s)", flush=True)

        pareto.append({
            "defense": eps_label,
            "target_epsilon": eps,
            "final_epsilon": eps_final,
            "delta": float(args.delta) if eps is not None else None,
            "task_acc": float(task_acc),
            "attack_logreg": {
                **asdict(logreg_res),
            },
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
        })

        # Free GPU between conditions
        del victim, ft_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print()

    out = {
        "n_subjects": int(len(subjects)),
        "chance_top1": float(1.0 / len(subjects)),
        "n_epochs_per_condition": int(args.n_epochs),
        "max_grad_norm": float(args.max_grad_norm),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "ft_epochs": int(args.ft_epochs),
        "delta": float(args.delta),
        "pareto": pareto,
        "seed": int(args.seed),
    }
    out_path = RESULTS_DIR / "29_d3_eps_sweep.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Results written to {out_path}\n")

    print("| ε (target) | ε (final) | Task acc | Logreg top-1 | Fine-tune top-1 |")
    print("|---|---|---|---|---|")
    for row in pareto:
        eps_t = "—" if row["target_epsilon"] is None else f"{row['target_epsilon']:.1f}"
        eps_f = "—" if row["final_epsilon"] is None else f"{row['final_epsilon']:.2f}"
        ta = row["task_acc"]
        lr_top1 = row["attack_logreg"]["top1"]
        ft_top1 = row["attack_finetune"]["top1"]
        print(f"| {eps_t} | {eps_f} | {ta:.3f} | {lr_top1:.3f} | {ft_top1:.3f} |")


if __name__ == "__main__":
    main()

