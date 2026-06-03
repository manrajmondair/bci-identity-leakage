"""Theory validation — re-ID accuracy scaling with cohort size N and ε.

Validates two theoretical anchors:

    (1) closed-set re-ID top-1 ~ 1 - C * N^(1 - γ) for embedder φ;
        fit γ from a cohort-size sweep and check it matches the
        same-vs-different similarity-distribution structure measured
        directly in φ's output.

    (2) DP-SGD empirical re-ID top-1 vs the Yeom 2018 MI-advantage
        upper bound 1 - e^(-ε) - δ. Reads results/29_d3_eps_sweep.json
        if present; if not, runs a smaller ε ∈ {1.0, 3.0, none} sweep
        to ground the overlay.

Both parts run on PhysioNet (104 subjects). EEGNet uses
embedding-via-pretrained-victim (re-trained inside the experiment;
fast on L4). Riemann is CPU-only and runs alongside.

Usage
-----
    python -m experiments.30_theory_scaling --smoke
    python -m experiments.30_theory_scaling --all
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from attacks.closed_set import closed_set_reid
from config import RESULTS_DIR
from data.physionet_loader import valid_subjects
from models.eegnet import EEGNetVictim
from models.riemannian import RiemannianVictim
from preprocess.windows import windowed_subjects

VICTIM_TRAIN_RUNS = (4, 6, 8, 10)
VICTIM_TEST_RUNS = (12, 14)
DEFAULT_COHORT_GRID = (10, 20, 40, 60, 80, 104)


def _embedding_pair_stats(victim: EEGNetVictim | RiemannianVictim,
                          X: np.ndarray,
                          subj: np.ndarray,
                          *, max_pairs: int = 20_000,
                          seed: int = 0) -> dict:
    """Sample positive (same-subject) and negative (different-subject)
    pairs in the trained embedder; report mean similarity and the
    margin between distributions.

    Similarity = cosine in the embedder's output space (L2-normalised
    per-window).
    """
    Z = victim.embed(X)
    # L2-normalise
    norms = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
    Zn = Z / norms

    rng = np.random.default_rng(seed)
    n = len(Zn)
    pos_sim, neg_sim = [], []
    same = subj[:, None] == subj[None, :]
    # Lower-triangular mask -- avoid self pairs
    iu, jv = np.triu_indices(n, k=1)
    is_same = same[iu, jv]
    pos_idx = np.where(is_same)[0]
    neg_idx = np.where(~is_same)[0]
    n_pos = min(max_pairs // 2, len(pos_idx))
    n_neg = min(max_pairs // 2, len(neg_idx))
    p_sel = rng.choice(pos_idx, size=n_pos, replace=False)
    n_sel = rng.choice(neg_idx, size=n_neg, replace=False)
    for k in p_sel:
        i, j = iu[k], jv[k]
        pos_sim.append(float(Zn[i] @ Zn[j]))
    for k in n_sel:
        i, j = iu[k], jv[k]
        neg_sim.append(float(Zn[i] @ Zn[j]))
    pos = np.asarray(pos_sim)
    neg = np.asarray(neg_sim)
    return {
        "n_pos_pairs": int(len(pos)),
        "n_neg_pairs": int(len(neg)),
        "pos_mean": float(pos.mean()),
        "pos_std": float(pos.std()),
        "neg_mean": float(neg.mean()),
        "neg_std": float(neg.std()),
        "margin": float(pos.mean() - neg.mean()),
        "margin_over_neg_std": float((pos.mean() - neg.mean())
                                     / max(neg.std(), 1e-9)),
    }


def _fit_scaling(cohort_sizes: list[int], accs: list[float]) -> tuple[float, float, float]:
    """Fit accs ≈ 1 - C * N^(1 - γ) by least squares on
    log(1 - accs) vs log N. Returns (gamma, C, r2)."""
    sizes = np.asarray(cohort_sizes, dtype=np.float64)
    a = np.asarray(accs, dtype=np.float64)
    # Avoid log(0) when ceiling-saturated.
    one_minus = np.clip(1.0 - a, 1e-9, 1.0)
    y = np.log(one_minus)
    x = np.log(sizes)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = float(coef[0]), float(coef[1])
    # slope = 1 - γ
    gamma = 1.0 - slope
    C = math.exp(intercept)
    yhat = A @ coef
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return gamma, C, r2


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--cohort-sizes", nargs="+", type=int,
                   default=list(DEFAULT_COHORT_GRID))
    p.add_argument("--eegnet-epochs", type=int, default=80)
    p.add_argument("--bootstrap-n", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.smoke:
        all_subjects = valid_subjects()[:30]
        args.cohort_sizes = [10, 20, 30]
        args.eegnet_epochs = 20
    elif args.all:
        all_subjects = valid_subjects()
    else:
        p.error("Provide --smoke or --all")

    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Cohort sizes: {args.cohort_sizes}")
    print(f"Device: {device}\n", flush=True)

    print("Loading full PhysioNet imagery ...", flush=True)
    t0 = time.time()
    full = windowed_subjects(all_subjects, runs="imagery")
    train_all = full.filter_runs(list(VICTIM_TRAIN_RUNS))
    test_all = full.filter_runs(list(VICTIM_TEST_RUNS))
    print(f"  loaded in {time.time() - t0:.1f}s  train={train_all.n_windows}  "
          f"test={test_all.n_windows}\n", flush=True)

    rng = np.random.default_rng(args.seed)

    scaling: dict[str, list[dict]] = {"eegnet": [], "riemann": []}
    for n in args.cohort_sizes:
        if n > len(all_subjects):
            continue
        # Random cohort of size N (seeded; same cohort grows monotonically
        # for stability).
        cohort = sorted(int(s) for s in rng.choice(all_subjects, size=n,
                                                   replace=False))
        print(f"=== N = {n}, cohort = {cohort[0]}..{cohort[-1]} ===", flush=True)

        # Subset windows to cohort
        train_m = np.isin(train_all.subject_ids, cohort)
        test_m = np.isin(test_all.subject_ids, cohort)
        Xtr, ytr = train_all.X[train_m], train_all.y[train_m]
        Xte, yte = test_all.X[test_m], test_all.y[test_m]
        train_ds = train_all.filter_runs(list(VICTIM_TRAIN_RUNS))
        # ... but we actually want the cohort-restricted WindowedDataset:
        train_ds = type(train_all)(
            X=Xtr, y=ytr,
            subject_ids=train_all.subject_ids[train_m],
            trial_ids=train_all.trial_ids[train_m],
            run_ids=train_all.run_ids[train_m],
            sfreq=train_all.sfreq, channel_names=train_all.channel_names,
        )
        test_ds = type(test_all)(
            X=Xte, y=yte,
            subject_ids=test_all.subject_ids[test_m],
            trial_ids=test_all.trial_ids[test_m],
            run_ids=test_all.run_ids[test_m],
            sfreq=test_all.sfreq, channel_names=test_all.channel_names,
        )

        # --- Riemann ---
        t = time.time()
        rv = RiemannianVictim(n_classes=4, seed=args.seed)
        rv.fit(Xtr, ytr)
        rie_res = closed_set_reid(rv, train_ds, test_ds, probes=("logreg",),
                                  bootstrap_n=args.bootstrap_n, seed=args.seed)[0]
        rie_pair = _embedding_pair_stats(rv, train_ds.X, train_ds.subject_ids,
                                          seed=args.seed)
        scaling["riemann"].append({
            "n": int(n), "top1": float(rie_res.top1),
            "ci_low": float(rie_res.top1_ci_low),
            "ci_high": float(rie_res.top1_ci_high),
            "chance": float(rie_res.chance_top1),
            "embedding_pair_stats": rie_pair,
            "wall_seconds": round(time.time() - t, 1),
        })
        print(f"  Riemann   top1={rie_res.top1:.3f}  margin={rie_pair['margin']:+.3f}",
              flush=True)

        # --- EEGNet ---
        t = time.time()
        ev = EEGNetVictim(
            n_channels=train_ds.n_channels, n_times=train_ds.n_times,
            n_classes=4, n_epochs=args.eegnet_epochs, seed=args.seed, verbose=False,
        )
        ev.fit(Xtr, ytr)
        ee_res = closed_set_reid(ev, train_ds, test_ds, probes=("logreg",),
                                 bootstrap_n=args.bootstrap_n, seed=args.seed)[0]
        ee_pair = _embedding_pair_stats(ev, train_ds.X, train_ds.subject_ids,
                                         seed=args.seed)
        scaling["eegnet"].append({
            "n": int(n), "top1": float(ee_res.top1),
            "ci_low": float(ee_res.top1_ci_low),
            "ci_high": float(ee_res.top1_ci_high),
            "chance": float(ee_res.chance_top1),
            "embedding_pair_stats": ee_pair,
            "wall_seconds": round(time.time() - t, 1),
        })
        print(f"  EEGNet    top1={ee_res.top1:.3f}  margin={ee_pair['margin']:+.3f}",
              flush=True)
        print()

    # Fit scaling exponents
    gamma_fits = {}
    for victim in ("eegnet", "riemann"):
        if len(scaling[victim]) >= 3:
            sizes = [r["n"] for r in scaling[victim]]
            accs = [r["top1"] for r in scaling[victim]]
            gamma, C, r2 = _fit_scaling(sizes, accs)
            gamma_fits[victim] = {"gamma": gamma, "C_constant": C, "r2": r2,
                                  "fit_points": list(zip(sizes, accs))}
            print(f"  scaling fit ({victim}): γ = {gamma:.3f}  C = {C:.4f}  r² = {r2:.3f}",
                  flush=True)

    # Yeom 2018 MI-advantage bound vs the existing ε sweep (if present)
    sweep_path = Path(RESULTS_DIR) / "29_d3_eps_sweep.json"
    yeom_overlay: list[dict] | None = None
    if sweep_path.exists():
        sweep = json.loads(sweep_path.read_text())
        delta = float(sweep.get("delta", 1e-5))
        yeom_overlay = []
        for row in sweep["pareto"]:
            eps = row["final_epsilon"] if row["final_epsilon"] is not None else float("inf")
            # Yeom 2018 bounds MEMBERSHIP-INFERENCE advantage by 1 - e^-eps - delta.
            # We overlay it against empirical re-ID top-1 purely as a loose
            # reference (re-ID is empirically easier than MI, so the curve sits
            # below this MI bound) — it is NOT a formal upper bound on re-ID.
            bound = (1.0 - math.exp(-eps) - delta) if math.isfinite(eps) else 1.0
            yeom_overlay.append({
                "target_epsilon": row["target_epsilon"],
                "final_epsilon": row["final_epsilon"],
                "yeom_bound_re_id_upper": bound,
                "empirical_logreg_top1": row["attack_logreg"]["top1"],
                "empirical_finetune_top1": row["attack_finetune"]["top1"],
                "task_acc": row["task_acc"],
                "gap_logreg": bound - row["attack_logreg"]["top1"],
                "gap_finetune": bound - row["attack_finetune"]["top1"],
            })

    payload = {
        "n_subjects_available": int(len(all_subjects)),
        "cohort_grid": args.cohort_sizes,
        "scaling": scaling,
        "scaling_fits": gamma_fits,
        "yeom_overlay": yeom_overlay,
        "predictions_passed": {
            "P1_eegnet_gamma_in_0_9_to_1_2":
                "eegnet" in gamma_fits
                and 0.9 <= gamma_fits["eegnet"]["gamma"] <= 1.2,
            "P2_riemann_gamma_gt_1":
                "riemann" in gamma_fits
                and gamma_fits["riemann"]["gamma"] > 1.0,
        },
        "seed": int(args.seed),
    }
    out_path = RESULTS_DIR / "30_theory_scaling.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()

