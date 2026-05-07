"""Reusable plotting helpers used by experiment scripts and the report.

Style choices:
  - matplotlib only (no seaborn dependence in figures so they reproduce
    cleanly on any machine);
  - hairline black baseline + neutral fills, no decorative color;
  - error bars are bootstrap CI ranges, never SEM.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _setup_axes() -> dict:
    return {
        "figure.figsize": (6.5, 3.6),
        "font.size": 9,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.2,
    }


def closed_set_bar_chart(
    results: list[dict],
    out_path: str | Path,
    *,
    title: str = "Closed-set subject re-identification (A1)",
) -> None:
    """One bar per (victim, probe) cell. Bootstrap-CI error bars; chance line.

    `results` is the JSON from experiments/02_closed_set_reid.py — a list
    of dicts with keys victim, probe, top1, top1_ci_low, top1_ci_high,
    chance_top1, task_acc.
    """
    plt.rcParams.update(_setup_axes())
    fig, ax = plt.subplots()

    victims = sorted({r["victim"] for r in results})
    probes = sorted({r["probe"] for r in results})

    n_v, n_p = len(victims), len(probes)
    bar_w = 0.8 / n_p
    x_centers = np.arange(n_v)

    fills = ["#2c3e50", "#7f8c8d"]  # logreg darker, knn lighter
    for j, probe in enumerate(probes):
        ys = []
        ylo = []
        yhi = []
        for v in victims:
            row = next((r for r in results if r["victim"] == v and r["probe"] == probe), None)
            ys.append(row["top1"] if row else np.nan)
            ylo.append(row["top1"] - row["top1_ci_low"] if row else 0)
            yhi.append(row["top1_ci_high"] - row["top1"] if row else 0)
        offsets = (j - (n_p - 1) / 2) * bar_w
        ax.bar(x_centers + offsets, ys, bar_w,
               yerr=[ylo, yhi], color=fills[j % len(fills)],
               edgecolor="white", linewidth=0.8, label=probe,
               error_kw=dict(elinewidth=0.6, capsize=2, capthick=0.6))

    chance = results[0]["chance_top1"]
    ax.axhline(chance, color="#c0392b", linewidth=0.8, linestyle="--",
               label=f"chance = {chance:.3f}")

    ax.set_xticks(x_centers)
    ax.set_xticklabels(victims)
    ax.set_ylabel("Re-identification top-1 accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def verification_panel(
    scores: np.ndarray,
    labels: np.ndarray,
    auc: float,
    eer: float,
    out_path: str | Path,
    *,
    title: str = "Open-set verification (A4)",
) -> None:
    """Two-panel plot for verification results.

    Left: similarity histogram, same vs different pairs.
    Right: ROC curve with AUC and EER annotated.
    """
    plt.rcParams.update(_setup_axes())
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(7.5, 3.6))

    same = scores[labels == 1]
    diff = scores[labels == 0]
    bins = np.linspace(min(scores.min(), -1), max(scores.max(), 1), 50)
    ax_l.hist(diff, bins=bins, alpha=0.6, color="#7f8c8d", label="different subject")
    ax_l.hist(same, bins=bins, alpha=0.6, color="#2c3e50", label="same subject")
    ax_l.set_xlabel("Cosine similarity")
    ax_l.set_ylabel("Count")
    ax_l.set_title("Pair-similarity distribution")
    ax_l.legend(frameon=False, fontsize=8)
    ax_l.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)

    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    ax_r.plot(fpr, tpr, color="#2c3e50", linewidth=1.2)
    ax_r.plot([0, 1], [0, 1], color="#c0392b", linewidth=0.6, linestyle="--")
    ax_r.set_xlabel("False-positive rate")
    ax_r.set_ylabel("True-positive rate")
    ax_r.set_title(f"ROC  (AUC = {auc:.3f},  EER = {eer:.3f})")
    ax_r.set_xlim(0, 1)
    ax_r.set_ylim(0, 1.02)
    ax_r.grid(linestyle=":", linewidth=0.4, alpha=0.5)

    fig.suptitle(title, y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def verification_summary_card(
    auc: float,
    auc_ci_low: float,
    auc_ci_high: float,
    eer: float,
    n_train_subjects: int,
    n_test_subjects: int,
    n_pairs: int,
    out_path: str | Path,
    *,
    title: str = "A4 open-set verification — unseen subjects",
) -> None:
    """Compact card rendering AUC and EER without per-pair scores.

    Used when only the summary stats from the experiment JSON are available
    (e.g., when re-generating figures from a Colab run that did not ship the
    raw score arrays). The richer two-panel `verification_panel` is preferred
    when scores are on hand.
    """
    plt.rcParams.update(_setup_axes())
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    ax.axis("off")

    ax.text(0.02, 0.85, title, fontsize=11, fontweight="bold")
    ax.text(0.02, 0.65,
            f"Trained on {n_train_subjects} subjects   ·   "
            f"Evaluated on {n_test_subjects} unseen subjects   ·   "
            f"{n_pairs:,} verification pairs",
            fontsize=8)

    ax.text(0.02, 0.30, f"AUC = {auc:.3f}",
            fontsize=22, fontweight="bold", color="#2c3e50")
    ax.text(0.32, 0.30,
            f"95% CI [{auc_ci_low:.3f}, {auc_ci_high:.3f}]",
            fontsize=9, color="#7f8c8d", verticalalignment="bottom")
    ax.text(0.02, 0.07, f"Equal Error Rate = {eer:.3f}",
            fontsize=11, color="#34495e")

    # Tiny chance reference
    ax.text(0.55, 0.30, "Random = 0.500",
            fontsize=9, color="#c0392b", verticalalignment="bottom",
            fontstyle="italic")

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def closed_set_table(results: list[dict]) -> str:
    """Render the same JSON as a markdown table — used in milestone draft."""
    lines = [
        "| Victim | Probe | Top-1 (95% CI) | Top-5 | Top-10 | Task acc | Chance top-1 |",
        "|---|---|---|---|---|---|---|",
    ]
    def _fmt(v):
        # Handle both legacy NaN floats and the JSON-strict null we now write.
        if v is None:
            return "—"
        try:
            return "—" if np.isnan(v) else f"{v:.3f}"
        except TypeError:
            return "—"

    for r in sorted(results, key=lambda x: (x["victim"], x["probe"])):
        ci = f"{r['top1']:.3f} [{r['top1_ci_low']:.3f}, {r['top1_ci_high']:.3f}]"
        task_acc = r.get("task_acc")
        task_acc_s = f"{task_acc:.3f}" if isinstance(task_acc, (int, float)) else "—"
        lines.append(
            f"| {r['victim']} | {r['probe']} | {ci} | {_fmt(r.get('top5'))} | "
            f"{_fmt(r.get('top10'))} | {task_acc_s} | {r['chance_top1']:.3f} |"
        )
    return "\n".join(lines)
