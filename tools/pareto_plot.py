"""Privacy-utility Pareto plot — combines every defense result into one figure.

Reads result JSONs from results/ and renders, per victim family, a scatter
of (task accuracy, A1 closed-set re-ID top-1) points. Lower-right corner =
"high utility, low privacy leak" = ideal defense.

Sources merged:
  02_closed_set_reid.json     A1 baseline (no defense)
  07_d1_pca.json              D1 PCA sweep (k ∈ {64, 32, 16, 8})
  11_d1_noise.json            D1 noise sweep (σ ∈ {0, 0.5, 1, 2})
  11_d1_channel_drop.json     D1 channel-drop sweep (k ∈ {64, 32, 16, 8})
  09_d2_dann.json             D2 DANN sweep (λ ∈ {0, 0.1, 0.5, 1.0}) — EEGNet only
  10_d3_dp_sgd.json           D3 DP-SGD sweep (when present)

Three-panel layout (one per victim family). Each panel shows:
  - all defense points as colored markers
  - the empirical Pareto frontier as a black step line
  - the chance reference and the no-defense baseline as crosshairs
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from config import FIGURES_DIR, RESULTS_DIR
from eval.plots import _setup_axes


_FAMILY_COLORS = {
    "no_defense":         "#2c3e50",
    "d1_pca":             "#1f78b4",
    "d1_noise":           "#33a02c",
    "d1_channel_drop":    "#ff7f00",
    "d2_dann":            "#e31a1c",
    "d3_dp_sgd":          "#6a3d9a",
}
_FAMILY_MARKERS = {
    "no_defense":         "*",
    "d1_pca":             "o",
    "d1_noise":           "s",
    "d1_channel_drop":    "^",
    "d2_dann":            "D",
    "d3_dp_sgd":          "P",
}


def _load_rows(path: Path, family: str, victim_name_map: dict | None = None) -> list[dict]:
    if not path.exists():
        return []
    rows = json.loads(path.read_text())
    out = []
    for r in rows:
        if r.get("probe") not in (None, "logreg"):
            continue  # logreg is the strongest probe — keep that one
        v = r.get("victim", "unknown")
        if victim_name_map:
            v = victim_name_map.get(v, v)
        out.append({
            "family": family,
            "victim": v,
            "label": r.get("defense", family),
            "task_acc": float(r.get("task_acc", float("nan"))),
            "top1": float(r.get("top1", float("nan"))),
            "top1_ci_low": float(r.get("top1_ci_low", float("nan"))),
            "top1_ci_high": float(r.get("top1_ci_high", float("nan"))),
        })
    return out


def _pareto_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return the lower-right Pareto frontier of (task_acc, top1) points.

    A point dominates another if it has higher task_acc AND lower top1.
    Frontier sorted by task_acc ascending.
    """
    pts = sorted(points, key=lambda p: (p[0], -p[1]))
    out: list[tuple[float, float]] = []
    best_top1 = float("inf")
    for x, y in pts:
        if y < best_top1:
            out.append((x, y))
            best_top1 = y
    return out


def main() -> None:
    families: list[tuple[str, Path]] = [
        ("no_defense",      RESULTS_DIR / "02_closed_set_reid.json"),
        ("d1_pca",          RESULTS_DIR / "07_d1_pca.json"),
        ("d1_noise",        RESULTS_DIR / "11_d1_noise.json"),
        ("d1_channel_drop", RESULTS_DIR / "11_d1_channel_drop.json"),
        ("d2_dann",         RESULTS_DIR / "09_d2_dann.json"),
        ("d2_dann",         RESULTS_DIR / "09_d2_dann_extended.json"),
        ("d3_dp_sgd",       RESULTS_DIR / "10_d3_dp_sgd.json"),
    ]

    rows: list[dict] = []
    # DANN's victim is named "eegnet_dann" — fold into "eegnet" for the plot.
    name_map = {"eegnet_dann": "eegnet", "eegnet_dpsgd": "eegnet"}
    for family, path in families:
        rows.extend(_load_rows(path, family, name_map))

    plt.rcParams.update(_setup_axes())
    victims = ["eegnet", "fbcsp_lda", "riemann_ts_lr"]
    titles = {
        "eegnet": "EEGNet",
        "fbcsp_lda": "FBCSP + LDA",
        "riemann_ts_lr": "Riemann tangent-space",
    }
    fig, axes = plt.subplots(1, len(victims), figsize=(11.0, 3.8), sharey=True)
    if len(victims) == 1:
        axes = [axes]

    for ax, victim in zip(axes, victims):
        vrows = [r for r in rows if r["victim"] == victim]
        if not vrows:
            ax.set_visible(False)
            continue

        # Plot points, family by family
        for family in _FAMILY_COLORS:
            fam_rows = [r for r in vrows if r["family"] == family]
            if not fam_rows:
                continue
            xs = [r["task_acc"] for r in fam_rows]
            ys = [r["top1"] for r in fam_rows]
            ax.scatter(xs, ys, s=46, color=_FAMILY_COLORS[family],
                       marker=_FAMILY_MARKERS[family], edgecolor="white",
                       linewidth=0.5, label=family, zorder=3)

        # Pareto frontier
        all_pts = [(r["task_acc"], r["top1"]) for r in vrows]
        front = _pareto_frontier(all_pts)
        if len(front) > 1:
            xs = [p[0] for p in front]
            ys = [p[1] for p in front]
            ax.step(xs + [max(xs) + 0.005], ys + [ys[-1]],
                    where="post", color="#000", linewidth=0.7,
                    linestyle="--", alpha=0.7, zorder=2,
                    label="Pareto frontier")

        # Chance reference
        chance = 1.0 / 104  # PhysioNet 104-subject chance
        ax.axhline(chance, color="#c0392b", linestyle=":", linewidth=0.7,
                   label=f"chance = {chance:.3f}", zorder=1)
        ax.axvline(0.25, color="#7f8c8d", linestyle=":", linewidth=0.5,
                   alpha=0.6, zorder=1)

        ax.set_title(titles.get(victim, victim))
        ax.set_xlabel("Task accuracy (utility)")
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(0.20, 0.46)
        ax.grid(linestyle=":", linewidth=0.4, alpha=0.5)
        if ax is axes[0]:
            ax.set_ylabel("Re-ID top-1 (privacy threat)")
            ax.legend(loc="upper left", frameon=False, fontsize=6.5,
                      ncol=2, columnspacing=0.6, handletextpad=0.3)

    fig.suptitle(
        "Privacy-utility Pareto on PhysioNet 104-subject motor imagery\n"
        "Lower-right = high utility AND low identity leak",
        y=1.02, fontsize=10,
    )
    fig.tight_layout()

    out = FIGURES_DIR / "pareto_privacy_utility.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")

    # Also write a quick markdown table of every (family, victim, label)
    print()
    print("| Victim | Family | Defense | Task acc | Re-ID top-1 |")
    print("|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["victim"], r["family"], r["label"])):
        print(f"| {r['victim']} | {r['family']} | {r['label']} | "
              f"{r['task_acc']:.3f} | {r['top1']:.3f} |")


if __name__ == "__main__":
    main()
