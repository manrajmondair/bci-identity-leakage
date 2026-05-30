"""Privacy-utility Pareto — markdown table + delegated figure render.

The figure itself (figures/pareto_privacy_utility.pdf) is rendered by the
canonical `tools.regenerate_figures.render_pareto`; this module just prints a
quick markdown table of every (family, victim, label) point and delegates the
plot so the PDF has a single writer. (It previously duplicated the plotting
logic and overwrote the same path, so whichever ran last won.)

Sources merged:
  02_closed_set_reid.json     A1 baseline (no defense)
  07_d1_pca.json              D1 PCA sweep (k ∈ {64, 32, 16, 8})
  11_d1_noise.json            D1 noise sweep (σ ∈ {0, 0.5, 1, 2})
  11_d1_channel_drop.json     D1 channel-drop sweep (k ∈ {64, 32, 16, 8})
  09_d2_dann.json             D2 DANN sweep (λ ∈ {0, 0.1, 0.5, 1.0}) — EEGNet only
  10_d3_dp_sgd.json           D3 DP-SGD sweep (when present)
"""
from __future__ import annotations

import json
from pathlib import Path

from config import RESULTS_DIR


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
    # Single source of truth for the figure — delegate to the canonical renderer.
    from tools.regenerate_figures import render_pareto
    render_pareto()

    families: list[tuple[str, Path]] = [
        ("no_defense",      RESULTS_DIR / "02_closed_set_reid.json"),
        ("d1_pca",          RESULTS_DIR / "07_d1_pca.json"),
        ("d1_noise",        RESULTS_DIR / "11_d1_noise.json"),
        ("d1_channel_drop", RESULTS_DIR / "11_d1_channel_drop.json"),
        ("d2_dann",         RESULTS_DIR / "09_d2_dann.json"),
        ("d2_dann",         RESULTS_DIR / "09_d2_dann_extended.json"),
        ("d3_dp_sgd",       RESULTS_DIR / "10_d3_dp_sgd.json"),
    ]
    # DANN's victim is named "eegnet_dann" — fold into "eegnet" for the table.
    name_map = {"eegnet_dann": "eegnet", "eegnet_dpsgd": "eegnet"}
    rows: list[dict] = []
    for family, path in families:
        rows.extend(_load_rows(path, family, name_map))

    print()
    print("| Victim | Family | Defense | Task acc | Re-ID top-1 |")
    print("|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["victim"], r["family"], r["label"])):
        print(f"| {r['victim']} | {r['family']} | {r['label']} | "
              f"{r['task_acc']:.3f} | {r['top1']:.3f} |")


if __name__ == "__main__":
    main()
