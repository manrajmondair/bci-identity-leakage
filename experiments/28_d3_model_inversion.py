"""D3 model inversion — does DP-SGD prevent input reconstruction?

Fredrikson et al. 2015 -style model inversion: given white-box access to
a trained subject-id classifier, the attacker reconstructs a synthetic
EEG window x*_k that the model assigns to subject k with high
probability. The reconstruction is scored in a held-out reference
embedding (a contrastive EEGNet trained on subjects disjoint from the
training pool) by cosine similarity to subject k's real EEG.

We run the same protocol on two victims:

    No-defense EEGNet (vanilla, AdamW + BN) — the baseline.
    DP-SGD EEGNet ε=3 (Opacus, GN + SGD)   — the defended target.

Pipeline:
    1. Split 104 subjects into 80 victim + 24 reference subjects.
    2. Train both victims (task: motor imagery 4-class) on the 80 subjects.
    3. Fine-tune each victim into an 80-way re-ID head (encoder fine-tune,
       same protocol as experiments 15 / 18).
    4. Train a fresh contrastive EEGNet on the 24 reference subjects.
       This embedder has not seen the target subjects, so its similarity
       scores are an unbiased "neutral judge" of reconstruction quality.
    5. Pick K target subjects from the 80 victim subjects. For each
       target k: invert the re-ID head to produce x*_k. Embed x*_k via
       the reference embedder. Rank cosine similarity to each victim
       subject's real EEG (averaged over windows). Report rank-1 and
       rank-5 recovery.

If the defense is real, the DP-SGD-target rank-1 recovery should drop
to ~1/n_targets while the no-DP baseline should recover the target's
identity above chance.

Usage
-----
    python -m experiments.28_d3_model_inversion --smoke
    python -m experiments.28_d3_model_inversion --all
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import numpy as np
import torch

from attacks.model_inversion import (
    _per_channel_std_bounds,
    evaluate_reconstructions,
    invert_subject,
)
from attacks.verification import _train_contrastive
from config import RESULTS_DIR
from data.physionet_loader import valid_subjects
from defenses.dp_sgd import DPSGDVictim
from experiments.eegnet_helpers import (  # noqa: F401  — created below
    finetune_to_reid_head,
)
from models.contrastive import ContrastiveEEGNet
from preprocess.windows import windowed_subjects

VICTIM_TRAIN_RUNS = (4, 6, 8, 10)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--n-victim-subjects", type=int, default=80)
    p.add_argument("--n-reference-subjects", type=int, default=24)
    p.add_argument("--n-targets", type=int, default=10)
    p.add_argument("--n-inversion-steps", type=int, default=600)
    p.add_argument("--dp-epochs", type=int, default=40)
    p.add_argument("--no-dp-epochs", type=int, default=80)
    p.add_argument("--ft-epochs", type=int, default=15)
    p.add_argument("--contrastive-epochs", type=int, default=22)
    p.add_argument("--target-epsilon", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.smoke:
        all_subjects = valid_subjects()[:30]
        args.n_victim_subjects = 20
        args.n_reference_subjects = 8
        args.n_targets = 4
        args.n_inversion_steps = 200
        args.dp_epochs = 12
        args.no_dp_epochs = 20
        args.ft_epochs = 6
        args.contrastive_epochs = 8
    elif args.all:
        all_subjects = valid_subjects()
    else:
        p.error("Provide --smoke or --all")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(np.asarray(all_subjects))
    victim_subjects = sorted(int(s) for s in perm[: args.n_victim_subjects])
    reference_subjects = sorted(int(s) for s in perm[args.n_victim_subjects:
                                                     args.n_victim_subjects
                                                     + args.n_reference_subjects])
    target_subjects = sorted(int(s) for s in
                             rng.choice(victim_subjects, size=args.n_targets,
                                        replace=False))

    print(f"Victim cohort:    {len(victim_subjects)} subjects")
    print(f"Reference cohort: {len(reference_subjects)} subjects (held out)")
    print(f"Target subjects:  {target_subjects}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n", flush=True)

    print("Loading windowed imagery for victim + reference cohorts ...", flush=True)
    t0 = time.time()
    victim_train = windowed_subjects(victim_subjects,
                                     runs="imagery").filter_runs(list(VICTIM_TRAIN_RUNS))
    reference_ds = windowed_subjects(reference_subjects,
                                     runs="imagery").filter_runs(list(VICTIM_TRAIN_RUNS))
    print(f"  loaded in {time.time() - t0:.1f}s | "
          f"victim={victim_train.n_windows}  reference={reference_ds.n_windows}\n",
          flush=True)

    # ---- 1. Train DP-SGD victim ----
    print(f"=== training DP-SGD victim (ε={args.target_epsilon}) ===", flush=True)
    t0 = time.time()
    dp = DPSGDVictim(
        n_channels=victim_train.n_channels, n_times=victim_train.n_times,
        n_classes=4, n_epochs=args.dp_epochs, batch_size=256, lr=1.0,
        target_epsilon=float(args.target_epsilon),
        target_delta=1e-5, max_grad_norm=1.0,
        seed=args.seed, verbose=False,
    )
    dp.fit(victim_train.X, victim_train.y)
    print(f"  trained in {time.time() - t0:.1f}s  "
          f"ε_final={dp.final_epsilon_:.2f}", flush=True)

    # ---- 2. Train no-DP victim ----
    print(f"\n=== training no-DP EEGNet victim (vanilla) ===", flush=True)
    from models.eegnet import EEGNetVictim
    t0 = time.time()
    nodp = EEGNetVictim(
        n_channels=victim_train.n_channels, n_times=victim_train.n_times,
        n_classes=4, n_epochs=args.no_dp_epochs, seed=args.seed, verbose=False,
    )
    nodp.fit(victim_train.X, victim_train.y)
    print(f"  trained in {time.time() - t0:.1f}s", flush=True)

    # ---- 3. Fine-tune both into 80-way re-ID heads ----
    print(f"\n=== fine-tuning DP victim into re-ID head ===", flush=True)
    dp_head, dp_sid_to_idx = finetune_to_reid_head(
        dp, victim_train.X, victim_train.subject_ids,
        device=device, n_epochs=args.ft_epochs, seed=args.seed, source="dp",
    )
    print(f"=== fine-tuning no-DP victim into re-ID head ===", flush=True)
    nodp_head, nodp_sid_to_idx = finetune_to_reid_head(
        nodp, victim_train.X, victim_train.subject_ids,
        device=device, n_epochs=args.ft_epochs, seed=args.seed, source="vanilla",
    )

    # ---- 4. Train contrastive reference embedder ----
    print(f"\n=== training reference contrastive embedder on {len(reference_subjects)} held-out subjects ===",
          flush=True)
    t0 = time.time()
    ref = _train_contrastive(
        reference_ds.X, reference_ds.subject_ids,
        n_chans=reference_ds.n_channels, n_times=reference_ds.n_times,
        embed_dim=64, n_epochs=args.contrastive_epochs, device=device,
        seed=args.seed, verbose=False,
    )
    print(f"  trained in {time.time() - t0:.1f}s", flush=True)

    # ---- 5. Per-channel std envelope from victim data (feasibility projection) ----
    std_lo, std_hi = _per_channel_std_bounds(victim_train.X)

    def _run_inversion_and_score(head: torch.nn.Module, sid_to_idx, label: str) -> dict:
        print(f"\n=== inverting {label} ({len(target_subjects)} target subjects) ===",
              flush=True)
        t0 = time.time()
        reconstructions: dict[int, torch.Tensor] = {}
        for k in target_subjects:
            if int(k) not in sid_to_idx:
                continue
            x_star = invert_subject(
                head,
                target_subject_logit_idx=int(sid_to_idx[int(k)]),
                n_chans=victim_train.n_channels, n_times=victim_train.n_times,
                std_lo=std_lo, std_hi=std_hi,
                input_scale=1e6,
                n_steps=args.n_inversion_steps, lr=0.05,
                weight_decay=0.0, device=device, seed=args.seed + int(k),
            )
            reconstructions[int(k)] = x_star.squeeze(0).cpu()
        # Eval: how well does the reference embedder identify each x*_k?
        # Restrict the "real" pool to the victim's subjects so the ranking
        # task is "argmax over the 80 trained subjects."
        real_X = victim_train.X
        real_sid = victim_train.subject_ids
        result = evaluate_reconstructions(
            reconstructions=reconstructions,
            real_windows=real_X, real_subject_ids=real_sid,
            reference_embedder=ref, device=device,
        )
        dt = time.time() - t0
        print(f"  {label}: rank1={result.rank1_acc:.3f}  rank5={result.rank5_acc:.3f}  "
              f"med_sim_self={result.median_sim_to_target:.3f}  "
              f"med_sim_other={result.median_sim_to_other:.3f}  "
              f"({dt:.0f}s)",
              flush=True)
        return {**asdict(result), "wall_seconds": dt, "victim": label,
                "target_subjects": target_subjects}

    out_nodp = _run_inversion_and_score(nodp_head, nodp_sid_to_idx, "no_defense")
    out_dp = _run_inversion_and_score(dp_head, dp_sid_to_idx, f"dp_eps={args.target_epsilon}")

    # ---- Persist ----
    payload = {
        "target_epsilon": float(args.target_epsilon),
        "dp_final_epsilon": float(dp.final_epsilon_) if dp.final_epsilon_ else None,
        "n_victim_subjects": int(len(victim_subjects)),
        "n_reference_subjects": int(len(reference_subjects)),
        "n_targets": int(len(target_subjects)),
        "n_inversion_steps": int(args.n_inversion_steps),
        "results": {"no_defense": out_nodp,
                    f"dp_eps={args.target_epsilon}": out_dp},
    }
    out_path = RESULTS_DIR / "28_d3_model_inversion.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nResults written to {out_path}")
    print(f"\n  Comparison:")
    print(f"    no defense   rank1={out_nodp['rank1_acc']:.3f}  rank5={out_nodp['rank5_acc']:.3f}")
    print(f"    DP ε={args.target_epsilon}      rank1={out_dp['rank1_acc']:.3f}  rank5={out_dp['rank5_acc']:.3f}")
    print(f"    chance       rank1={1.0/len(target_subjects):.3f}  "
          f"rank5={5.0/len(target_subjects):.3f}")


if __name__ == "__main__":
    main()
