"""D1 ad-hoc defenses (general) — Gaussian noise on channels OR channel-drop.

Mirrors 07_d1_pca.py but accepts a `--transform` flag picking between two
non-PCA ad-hoc defenses:

  --transform noise           additive zero-mean Gaussian per channel,
                              std = σ × channel_std on the train set.
                              Sweeps σ ∈ {0, 0.5, 1.0, 2.0}.
                              σ=0 reproduces the no-defense baseline.

  --transform channel_drop    keep top-k channels by training-set
                              variance, drop the rest. Sweeps
                              k ∈ {64, 32, 16, 8}. k=64 is the
                              no-defense baseline.

Each strength × victim combination retrains the victim on the
transformed data, then runs the A1 closed-set re-ID attack. EEGNet uses
40 epochs (vs 80 in A1) so the four-condition sweep stays under the
1-hour budget.

Output: results/11_d1_<transform>.json + figures/11_d1_<transform>.pdf
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import numpy as np

from attacks.closed_set import closed_set_reid
from config import FIGURES_DIR, RESULTS_DIR
from data.physionet_loader import valid_subjects
from defenses.adhoc import ChannelDrop, ChannelGaussianNoise
from eval.plots import _setup_axes
from models.eegnet import EEGNetVictim
from models.fbcsp import FBCSPVictim
from models.riemannian import RiemannianVictim
from preprocess.windows import WindowedDataset, windowed_subjects

VICTIM_TRAIN_RUNS = (4, 6, 8, 10)
VICTIM_TEST_RUNS = (12, 14)


def _build_victim(name: str, *, n_channels: int, n_times: int, sfreq: float,
                  eegnet_epochs: int, seed: int):
    if name == "eegnet":
        return EEGNetVictim(
            n_channels=n_channels, n_times=n_times, n_classes=4,
            n_epochs=eegnet_epochs, seed=seed, verbose=False,
        )
    if name == "fbcsp":
        return FBCSPVictim(sfreq=sfreq, n_classes=4)
    if name == "riemann":
        return RiemannianVictim(n_classes=4, seed=seed)
    raise ValueError(name)


def _wrap_windowed(orig: WindowedDataset, X_new: np.ndarray) -> WindowedDataset:
    new_chs = (tuple(orig.channel_names[:X_new.shape[1]])
               if X_new.shape[1] <= len(orig.channel_names)
               else tuple(f"ch{i:02d}" for i in range(X_new.shape[1])))
    return WindowedDataset(
        X=X_new, y=orig.y, subject_ids=orig.subject_ids,
        trial_ids=orig.trial_ids, run_ids=orig.run_ids,
        sfreq=orig.sfreq, channel_names=new_chs,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--transform", choices=["noise", "channel_drop"], required=True)
    p.add_argument("--smoke", action="store_true",
                   help="10 subjects, 2 strengths, EEGNet 15 epochs.")
    p.add_argument("--all", action="store_true",
                   help="All 104 PhysioNet subjects.")
    p.add_argument("--strengths", type=float, nargs="+", default=None,
                   help="Defaults to a sensible sweep for the chosen transform.")
    p.add_argument("--models", nargs="+",
                   default=["eegnet", "fbcsp", "riemann"],
                   choices=["eegnet", "fbcsp", "riemann"])
    p.add_argument("--eegnet-epochs", type=int, default=40)
    p.add_argument("--bootstrap-n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.smoke:
        subjects = valid_subjects()[:10]
        args.eegnet_epochs = min(args.eegnet_epochs, 15)
        if args.transform == "noise":
            args.strengths = [0.0, 1.0]
        else:
            args.strengths = [64, 16]
    elif args.all:
        subjects = valid_subjects()
    else:
        p.error("Provide --smoke or --all")

    if args.strengths is None:
        args.strengths = [0.0, 0.5, 1.0, 2.0] if args.transform == "noise" \
                         else [64, 32, 16, 8]

    np.random.seed(args.seed)
    print(f"Subjects: {len(subjects)} (chance top-1 = {100/len(subjects):.2f}%)")
    print(f"transform={args.transform}  strengths={args.strengths}")
    print(f"victims={args.models}\n", flush=True)

    print("Loading windowed data ...", flush=True)
    t0 = time.time()
    full = windowed_subjects(subjects, runs="imagery")
    train = full.filter_runs(list(VICTIM_TRAIN_RUNS))
    test = full.filter_runs(list(VICTIM_TEST_RUNS))
    print(f"  loaded in {time.time() - t0:.1f}s | "
          f"train={train.n_windows} test={test.n_windows} chans={train.n_channels}\n",
          flush=True)

    all_results = []
    for s in args.strengths:
        if args.transform == "noise":
            label = "no_defense" if float(s) == 0.0 else f"noise_sigma{s}"
            if float(s) == 0.0:
                X_train_t = train.X
                X_test_t = test.X
            else:
                tr = ChannelGaussianNoise(sigma=float(s), seed=args.seed).fit(train.X)
                X_train_t = tr.transform(train.X)
                X_test_t = tr.transform(test.X)
        else:  # channel_drop
            label = ("no_defense"
                     if int(s) >= train.n_channels
                     else f"channel_drop_k{int(s)}")
            if int(s) >= train.n_channels:
                X_train_t = train.X
                X_test_t = test.X
            else:
                tr = ChannelDrop(k=int(s)).fit(train.X)
                X_train_t = tr.transform(train.X)
                X_test_t = tr.transform(test.X)

        train_t = _wrap_windowed(train, X_train_t)
        test_t = _wrap_windowed(test, X_test_t)
        print(f"--- {label} ---  shape={X_train_t.shape}", flush=True)

        for victim_name in args.models:
            victim = _build_victim(
                victim_name, n_channels=train_t.n_channels,
                n_times=train_t.n_times, sfreq=train_t.sfreq,
                eegnet_epochs=args.eegnet_epochs, seed=args.seed,
            )
            t0 = time.time()
            victim.fit(train_t.X, train_t.y)
            task_acc = victim.score(test_t.X, test_t.y)
            print(f"  {victim_name:8s}  victim+score: {time.time() - t0:.1f}s | "
                  f"task_acc={task_acc:.3f}", flush=True)
            t0 = time.time()
            results = closed_set_reid(
                victim, train_t, test_t, probes=("knn", "logreg"),
                bootstrap_n=args.bootstrap_n, seed=args.seed,
            )
            for r in results:
                row = {**asdict(r), "strength": float(s),
                       "defense": label, "transform": args.transform,
                       "task_acc": float(task_acc)}
                all_results.append(row)
                if r.probe == "logreg":
                    print(f"    logreg  top1={r.top1:.3f} "
                          f"[{r.top1_ci_low:.3f}, {r.top1_ci_high:.3f}]",
                          flush=True)
        print()

    out_path = RESULTS_DIR / f"11_d1_{args.transform}.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"Results written to {out_path}")

    fig_path = FIGURES_DIR / f"11_d1_{args.transform}.pdf"
    _plot(all_results, fig_path,
          transform=args.transform,
          title=f"D1 {args.transform} defense  ({len(subjects)} subj, "
                f"chance top-1 = {100/len(subjects):.1f}%)")
    print(f"Figure written to {fig_path}\n")
    _print_table(all_results, transform=args.transform)


def _plot(results: list[dict], out_path, *, transform: str, title: str) -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update(_setup_axes())
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(7.8, 3.6), sharex=True)

    families = sorted({r["victim"] for r in results})
    colors = {"eegnet": "#2c3e50", "fbcsp_lda": "#7f8c8d", "riemann_ts_lr": "#34495e"}
    markers = {"eegnet": "o", "fbcsp_lda": "s", "riemann_ts_lr": "^"}

    for fam in families:
        rows = [r for r in results if r["victim"] == fam and r["probe"] == "logreg"]
        rows.sort(key=lambda r: r["strength"]
                  if transform == "noise" else -r["strength"])
        xs = [r["strength"] for r in rows]
        ys_priv = [r["top1"] for r in rows]
        lo = [r["top1"] - r["top1_ci_low"] for r in rows]
        hi = [r["top1_ci_high"] - r["top1"] for r in rows]
        ys_util = [r["task_acc"] for r in rows]
        ax_l.errorbar(xs, ys_priv, yerr=[lo, hi], color=colors.get(fam, "#000"),
                      marker=markers.get(fam, "o"), linewidth=1.0, capsize=2, label=fam)
        ax_r.plot(xs, ys_util, color=colors.get(fam, "#000"),
                  marker=markers.get(fam, "o"), linewidth=1.0, label=fam)

    chance = results[0]["chance_top1"]
    ax_l.axhline(chance, color="#c0392b", linestyle="--", linewidth=0.8,
                 label=f"chance = {chance:.3f}")
    xlabel = ("Noise σ (× channel std)" if transform == "noise"
              else "Channels retained (k)")
    ax_l.set_xlabel(xlabel)
    ax_l.set_ylabel("Re-ID top-1")
    ax_l.set_title("Privacy: re-ID vs strength")
    ax_l.set_ylim(0, 1.05)
    if transform == "channel_drop":
        ax_l.invert_xaxis()
    ax_l.grid(linestyle=":", linewidth=0.4, alpha=0.5)
    ax_l.legend(frameon=False, fontsize=7, loc="upper right")

    ax_r.set_xlabel(xlabel)
    ax_r.set_ylabel("BCI task accuracy")
    ax_r.set_title("Utility: task acc vs strength")
    ax_r.set_ylim(0, 1.05)
    if transform == "channel_drop":
        ax_r.invert_xaxis()
    ax_r.grid(linestyle=":", linewidth=0.4, alpha=0.5)
    ax_r.axhline(0.25, color="#c0392b", linestyle="--", linewidth=0.8,
                 label="chance task = 0.25")
    ax_r.legend(frameon=False, fontsize=7, loc="lower right")

    fig.suptitle(title, y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")


def _print_table(results: list[dict], *, transform: str) -> None:
    print(f"\n| Victim | Probe | Strength | Defense | Top-1 (95% CI) | Task acc |")
    print(f"|---|---|---|---|---|---|")
    sort_key = (lambda r: (r["victim"], r["probe"], r["strength"])) \
        if transform == "noise" \
        else (lambda r: (r["victim"], r["probe"], -r["strength"]))
    for r in sorted(results, key=sort_key):
        ci = f"{r['top1']:.3f} [{r['top1_ci_low']:.3f}, {r['top1_ci_high']:.3f}]"
        print(f"| {r['victim']} | {r['probe']} | {r['strength']} | "
              f"{r['defense']} | {ci} | {r['task_acc']:.3f} |")


if __name__ == "__main__":
    main()
