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


def closed_set_table(results: list[dict]) -> str:
    """Render the same JSON as a markdown table — used in milestone draft."""
    lines = [
        "| Victim | Probe | Top-1 (95% CI) | Top-5 | Top-10 | Task acc | Chance top-1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda x: (x["victim"], x["probe"])):
        ci = f"{r['top1']:.3f} [{r['top1_ci_low']:.3f}, {r['top1_ci_high']:.3f}]"
        top5 = "—" if np.isnan(r.get("top5", float("nan"))) else f"{r['top5']:.3f}"
        top10 = "—" if np.isnan(r.get("top10", float("nan"))) else f"{r['top10']:.3f}"
        lines.append(
            f"| {r['victim']} | {r['probe']} | {ci} | {top5} | {top10} | "
            f"{r.get('task_acc', float('nan')):.3f} | {r['chance_top1']:.3f} |"
        )
    return "\n".join(lines)
