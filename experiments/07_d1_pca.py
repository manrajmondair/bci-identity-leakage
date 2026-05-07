"""D1 PCA — defender compresses the raw EEG channel dimension to top-k.

Sweeps k ∈ {64 (no defense), 32, 16, 8} on PhysioNet motor imagery.
For each k, fits channel-mode PCA on the training windows, projects both
train and test windows to (k, n_times), then trains all three victims
(EEGNet at 40 epochs, FBCSP+LDA, Riemann tangent-space) on the projected
data and runs the A1 closed-set re-ID attack on each.

Compares the privacy-utility trade-off: as k decreases, BCI task accuracy
should drop AND re-ID top-1 should drop. The shape of the joint trajectory
is the defense's quality.

Designed to fit the 1-hour-per-run budget on a Colab L4: EEGNet uses 40
epochs (vs 80 for the unconditional A1) and the sweep is limited to four
k values. Other transforms (Gaussian noise, channel-drop) live in their
own scripts so each notebook stays under the budget.
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
from defenses.adhoc import ChannelPCA
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


def _projected_windowed(ds: WindowedDataset, X_new: np.ndarray) -> WindowedDataset:
    """Wrap projected arrays back into WindowedDataset preserving labels."""
    new_chs = tuple(f"pc{i:02d}" for i in range(X_new.shape[1]))
    return WindowedDataset(
        X=X_new, y=ds.y, subject_ids=ds.subject_ids,
        trial_ids=ds.trial_ids, run_ids=ds.run_ids,
        sfreq=ds.sfreq, channel_names=new_chs,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="10 subjects, EEGNet capped at 20 epochs.")
    p.add_argument("--all", action="store_true",
                   help="All 104 PhysioNet subjects.")
    p.add_argument("--ks", type=int, nargs="+", default=[64, 32, 16, 8],
                   help="k values to sweep. 64 = no compression.")
    p.add_argument("--models", nargs="+",
                   default=["eegnet", "fbcsp", "riemann"],
                   choices=["eegnet", "fbcsp", "riemann"])
    p.add_argument("--eegnet-epochs", type=int, default=40,
                   help="Reduced from 80 to fit 4-condition sweep in 1 hour.")
    p.add_argument("--bootstrap-n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.smoke:
        subjects = valid_subjects()[:10]
        args.eegnet_epochs = min(args.eegnet_epochs, 15)
        args.ks = [64, 16]
    elif args.all:
        subjects = valid_subjects()
    else:
        p.error("Provide --smoke or --all")

    np.random.seed(args.seed)
    print(f"Subjects: {len(subjects)} (chance top-1 = {100/len(subjects):.2f}%)")
    print(f"k sweep: {args.ks}")
    print(f"Victims: {args.models}\n", flush=True)

    print("Loading windowed data ...", flush=True)
    t0 = time.time()
    full = windowed_subjects(subjects, runs="imagery")
    train = full.filter_runs(list(VICTIM_TRAIN_RUNS))
    test = full.filter_runs(list(VICTIM_TEST_RUNS))
    print(f"  loaded in {time.time() - t0:.1f}s | "
          f"train={train.n_windows} test={test.n_windows} chans={train.n_channels}\n",
          flush=True)

    all_results = []
    for k in args.ks:
        print(f"--- k = {k} ---", flush=True)
        t = time.time()
        if k >= train.n_channels:
            X_train_k = train.X
            X_test_k = test.X
            label = "no_defense"
        else:
            pca = ChannelPCA(k=k).fit(train.X)
            X_train_k = pca.transform(train.X)
            X_test_k = pca.transform(test.X)
            label = f"pca_k{k}"
        train_k = _projected_windowed(train, X_train_k)
        test_k = _projected_windowed(test, X_test_k)
        print(f"  PCA fit+apply: {time.time() - t:.1f}s | "
              f"shape={X_train_k.shape}", flush=True)

        for victim_name in args.models:
            victim = _build_victim(
                victim_name, n_channels=train_k.n_channels,
                n_times=train_k.n_times, sfreq=train_k.sfreq,
                eegnet_epochs=args.eegnet_epochs, seed=args.seed,
            )
            t = time.time()
            victim.fit(train_k.X, train_k.y)
            task_acc = victim.score(test_k.X, test_k.y)
            print(f"  {victim_name:8s}  victim+score: {time.time() - t:.1f}s | "
                  f"task_acc={task_acc:.3f}", flush=True)

            t = time.time()
            results = closed_set_reid(
                victim, train_k, test_k,
                probes=("knn", "logreg"),
                bootstrap_n=args.bootstrap_n, seed=args.seed,
            )
            for r in results:
                row = {**asdict(r), "k": k, "defense": label,
                       "task_acc": float(task_acc)}
                all_results.append(row)
                if r.probe == "logreg":
                    print(f"    {r.probe:7s}  top1={r.top1:.3f} "
                          f"[{r.top1_ci_low:.3f}, {r.top1_ci_high:.3f}]",
                          flush=True)
        print()

    out_path = RESULTS_DIR / "07_d1_pca.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"Results written to {out_path}")

    fig_path = FIGURES_DIR / "07_d1_pca.pdf"
    _plot_d1(all_results, fig_path,
             title=f"D1 PCA defense  ({len(subjects)} subj, "
                   f"chance top-1 = {100/len(subjects):.1f}%)")
    print(f"Figure written to {fig_path}\n")
    _print_d1_table(all_results)


def _plot_d1(results: list[dict], out_path, *, title: str) -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update(_setup_axes())
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(7.8, 3.6), sharex=True)

    families = sorted({r["victim"] for r in results})
    colors = {"eegnet": "#2c3e50", "fbcsp_lda": "#7f8c8d", "riemann_ts_lr": "#34495e"}
    markers = {"eegnet": "o", "fbcsp_lda": "s", "riemann_ts_lr": "^"}

    ks = sorted({r["k"] for r in results}, reverse=True)
    for fam in families:
        rows = [r for r in results if r["victim"] == fam and r["probe"] == "logreg"]
        rows.sort(key=lambda r: -r["k"])
        xs = [r["k"] for r in rows]
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
    ax_l.set_xlabel("PCA components retained (k)")
    ax_l.set_ylabel("Re-ID top-1")
    ax_l.set_title("Privacy: re-ID vs k")
    ax_l.set_ylim(0, 1.05)
    ax_l.invert_xaxis()
    ax_l.grid(linestyle=":", linewidth=0.4, alpha=0.5)
    ax_l.legend(frameon=False, fontsize=7, loc="upper right")

    ax_r.set_xlabel("PCA components retained (k)")
    ax_r.set_ylabel("BCI task accuracy")
    ax_r.set_title("Utility: task acc vs k")
    ax_r.set_ylim(0, 1.05)
    ax_r.invert_xaxis()
    ax_r.grid(linestyle=":", linewidth=0.4, alpha=0.5)
    ax_r.axhline(0.25, color="#c0392b", linestyle="--", linewidth=0.8,
                 label="chance task = 0.25")
    ax_r.legend(frameon=False, fontsize=7, loc="lower right")

    fig.suptitle(title, y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")


def _print_d1_table(results: list[dict]) -> None:
    print("\n| Victim | Probe | k | Defense | Top-1 (95% CI) | Task acc | Chance |")
    print("|---|---|---|---|---|---|---|")
    for r in sorted(results, key=lambda x: (x["victim"], x["probe"], -x["k"])):
        ci = f"{r['top1']:.3f} [{r['top1_ci_low']:.3f}, {r['top1_ci_high']:.3f}]"
        print(f"| {r['victim']} | {r['probe']} | {r['k']} | {r['defense']} | "
              f"{ci} | {r['task_acc']:.3f} | {r['chance_top1']:.3f} |")


if __name__ == "__main__":
    main()
