"""Reusable plotting helpers used by experiment scripts and the report.

Style philosophy: journal-grade matplotlib. Serif math, sans-serif text,
neutral palette with two accent colors for binary contrasts, no
grid by default, top and right spines hidden, tight constrained
layout. Vector PDF output throughout.

Single-column figures are 3.4 inches wide; two-column / full-width
figures are 6.5 -- 7.2 inches wide. All sizing is consistent across
renderers so the report's figures align cleanly when laid out side
by side.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Color palette -- Wong (2011) color-blind-safe set, plus neutrals.
# ---------------------------------------------------------------------------
PALETTE = {
    "ink":        "#1f1f1f",   # near-black, primary text / axis
    "neutral":    "#6e6e6e",   # secondary text, chance lines
    "muted":      "#a8a8a8",   # tertiary, light grid
    "accent":     "#0072B2",   # blue (primary data)
    "contrast":   "#D55E00",   # vermillion (secondary data)
    "ok":         "#009E73",   # green (success / above-threshold)
    "warn":       "#E69F00",   # orange (warning band)
    "fail":       "#CC2A36",   # deep red (collapse / failure)
    "purple":     "#882E72",   # purple (third-line, used sparingly)
    "skyblue":    "#56B4E9",   # sky-blue (auxiliary line)
}
WONG = [
    PALETTE["ink"], PALETTE["accent"], PALETTE["contrast"],
    PALETTE["ok"], PALETTE["warn"], PALETTE["purple"], PALETTE["skyblue"],
]


# ---------------------------------------------------------------------------
# Canonical rcParams. Apply by `plt.rcParams.update(JOURNAL_STYLE)` before
# every figure; the regenerate_figures pipeline calls this at the top of
# each renderer so style is stable across the whole pipeline.
# ---------------------------------------------------------------------------
JOURNAL_STYLE: dict = {
    # text — clean sans-serif (Nature/Science figure convention), math in CM
    "font.family":         "sans-serif",
    "font.sans-serif":     ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":           9.5,
    "axes.titlesize":      10.5,
    "axes.titleweight":    "bold",
    "axes.titlepad":       7.0,
    "axes.labelsize":      10.0,
    "axes.labelweight":    "regular",
    "axes.labelpad":       4.0,
    "axes.labelcolor":     PALETTE["ink"],
    "xtick.labelsize":     9.0,
    "ytick.labelsize":     9.0,
    "legend.fontsize":     8.5,
    "legend.frameon":      False,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.55,
    "legend.labelspacing": 0.35,
    "legend.columnspacing": 1.4,
    # math
    "mathtext.fontset":    "dejavusans",
    "mathtext.default":    "regular",
    # axes
    "axes.edgecolor":      PALETTE["ink"],
    "axes.linewidth":      0.7,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.titlecolor":     PALETTE["ink"],
    "axes.grid":           False,
    "grid.color":          PALETTE["muted"],
    "grid.linestyle":      ":",
    "grid.linewidth":      0.45,
    "grid.alpha":          0.55,
    # ticks
    "xtick.color":         PALETTE["ink"],
    "ytick.color":         PALETTE["ink"],
    "xtick.major.size":    3.0,
    "ytick.major.size":    3.0,
    "xtick.major.width":   0.7,
    "ytick.major.width":   0.7,
    "xtick.major.pad":     2.5,
    "ytick.major.pad":     2.5,
    "xtick.minor.size":    1.6,
    "ytick.minor.size":    1.6,
    # lines / patches
    "lines.linewidth":     1.3,
    "lines.markersize":    4.5,
    "patch.linewidth":     0.7,
    # figure
    "figure.dpi":          150,
    "figure.constrained_layout.use":   True,
    "figure.constrained_layout.h_pad": 0.04,
    "figure.constrained_layout.w_pad": 0.04,
    "figure.constrained_layout.hspace": 0.06,
    "figure.constrained_layout.wspace": 0.06,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.03,
    "savefig.transparent": False,
    "pdf.fonttype":        42,
    "ps.fonttype":         42,
}


def journal_style() -> dict:
    """Return a copy of the canonical rcParams. Callers do
    `plt.rcParams.update(journal_style())` at the top of every renderer."""
    return dict(JOURNAL_STYLE)


# Backwards-compatible alias for existing call sites that imported `_setup_axes`.
def _setup_axes() -> dict:
    return journal_style()


# Common figure sizes (inches)
FIG_SINGLE = (3.4, 2.4)        # single column, compact
FIG_SINGLE_TALL = (3.4, 3.0)   # single column, more vertical
FIG_DOUBLE = (7.0, 2.6)        # double column, two-panel
FIG_DOUBLE_TALL = (7.0, 3.4)   # double column, taller
FIG_WIDE = (7.0, 4.0)          # double column, full
FIG_TRIPLE = (7.2, 2.6)        # double column, three panels side by side


# ---------------------------------------------------------------------------
# Re-usable plot helpers
# ---------------------------------------------------------------------------
def _annotate_bar_value(ax, x: float, value: float, *,
                        text: str | None = None,
                        offset: float = 0.012,
                        color: str = PALETTE["ink"],
                        fontsize: float = 7.5,
                        ha: str = "center") -> None:
    """Place a numeric label just above a bar."""
    ax.text(x, value + offset, text if text is not None else f"{value:.3f}",
            ha=ha, va="bottom", color=color, fontsize=fontsize)


def auc_dotplot(
    rows: Sequence[dict],
    *,
    out_path: str | Path,
    title: str | None = None,
    chance: float = 0.5,
    threshold: float = 0.6,
    figsize: tuple[float, float] = FIG_DOUBLE,
    sort_by_value: bool = False,
) -> None:
    """Horizontal dot-plot of AUCs with bootstrap CIs.

    Each row needs `label`, `value`, `ci_low`, `ci_high`, and optional
    `annotation` (a short string appended next to the dot).

    `threshold` colours rows: green if value >= threshold, red if
    value <= chance + 0.05, otherwise neutral.
    """
    plt.rcParams.update(journal_style())
    rows = list(rows)
    if sort_by_value:
        rows = sorted(rows, key=lambda r: r["value"])

    n = len(rows)
    fig, ax = plt.subplots(figsize=figsize)

    y_positions = np.arange(n)
    values = np.array([r["value"] for r in rows])
    lows = np.array([r["ci_low"] for r in rows])
    highs = np.array([r["ci_high"] for r in rows])
    err_lo = values - lows
    err_hi = highs - values

    # Colour by status against the chance / threshold band
    colors = []
    for v in values:
        if v >= threshold:
            colors.append(PALETTE["accent"])
        elif v <= chance + 0.03:
            colors.append(PALETTE["fail"])
        else:
            colors.append(PALETTE["warn"])

    ax.axvline(chance, color=PALETTE["neutral"], lw=0.7, ls=(0, (4, 3)),
               label=f"chance ({chance:.3f})")
    ax.errorbar(values, y_positions, xerr=[err_lo, err_hi], fmt="o",
                color=PALETTE["ink"], ecolor=PALETTE["ink"],
                elinewidth=0.7, capsize=2.5, capthick=0.7,
                markersize=5.5, markerfacecolor="white",
                markeredgewidth=1.2, zorder=3)
    # Color overlay on the markers themselves
    for x, y, c in zip(values, y_positions, colors):
        ax.plot([x], [y], "o", color=c, markersize=4.0, zorder=4)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_xlabel("AUC")
    ax.set_xlim(min(0.35, chance - 0.1), 1.02)
    ax.set_ylim(-0.5, n - 0.5)
    if title:
        ax.set_title(title)
    ax.legend(loc="lower right", fontsize=7.5)

    # Annotations to the right of each dot
    for x, y, r in zip(values, y_positions, rows):
        ann = r.get("annotation")
        if ann is None:
            ann = f"{x:.3f}"
        ax.text(min(1.005, max(x + 0.015, highs[int(y)] + 0.02)), y, ann,
                va="center", fontsize=7.5, color=PALETTE["ink"])
    ax.grid(axis="x", linestyle=":", linewidth=0.4,
            alpha=0.35, color=PALETTE["muted"])
    ax.set_axisbelow(True)

    fig.savefig(out_path)
    plt.close(fig)


def value_with_ci_bar(
    *,
    out_path: str | Path,
    label: str,
    value: float,
    ci_low: float,
    ci_high: float,
    chance: float = 0.5,
    yaxis_label: str = "value",
    title: str | None = None,
    reference_lines: Sequence[tuple[str, float, str]] = (),
    figsize: tuple[float, float] = FIG_SINGLE,
) -> None:
    """Single bar with CI, optional baseline / reference lines.

    `reference_lines` is an iterable of (label, value, color) triples
    drawn as dashed horizontal lines.
    """
    plt.rcParams.update(journal_style())
    fig, ax = plt.subplots(figsize=figsize)

    ax.bar([label], [value],
                 yerr=[[value - ci_low], [ci_high - value]],
                 color=PALETTE["accent"], edgecolor=PALETTE["ink"],
                 linewidth=0.6, width=0.45,
                 error_kw=dict(ecolor=PALETTE["ink"], elinewidth=0.9,
                               capsize=3, capthick=0.9))
    ax.axhline(chance, color=PALETTE["neutral"], lw=0.7, ls=(0, (4, 3)),
               label=f"chance ({chance:.3f})")
    for ref_label, ref_value, ref_color in reference_lines:
        ax.axhline(ref_value, color=ref_color, lw=0.7, ls=":",
                   label=f"{ref_label} ({ref_value:.3f})")

    ax.set_ylabel(yaxis_label)
    ax.set_ylim(min(0.0, ci_low - 0.05), 1.02)
    if title:
        ax.set_title(title)
    ax.text(0, value + (ci_high - value) + 0.025, f"{value:.3f}",
            ha="center", va="bottom", fontsize=9, color=PALETTE["ink"])
    ax.legend(loc="upper right", fontsize=7.5)
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public renderers used by experiment scripts and tools/regenerate_figures
# ---------------------------------------------------------------------------
_VICTIM_LABEL = {
    "eegnet":         "EEGNet",
    "eegnet_dpsgd":   "EEGNet (DP-SGD)",
    "fbcsp_lda":      "FBCSP + LDA",
    "fbcsp":          "FBCSP + LDA",
    "riemann_ts_lr":  "Riemann tang.-space",
    "riemann":        "Riemann tang.-space",
}
_PROBE_LABEL = {
    "knn":     "kNN (cosine)",
    "logreg":  "logistic regression",
}


def _pretty_victim(name: str) -> str:
    return _VICTIM_LABEL.get(name, name)


def _pretty_probe(name: str) -> str:
    return _PROBE_LABEL.get(name, name)


def closed_set_bar_chart(
    results: list[dict],
    out_path: str | Path,
    *,
    title: str | None = "Closed-set subject re-identification (A1)",
    figsize: tuple[float, float] | None = None,
) -> None:
    """Grouped bar chart, one bar per (victim, probe) cell. Bootstrap CI
    error bars + chance line.

    `results` is the JSON from `experiments/02_closed_set_reid.py` and the
    analogous Lee 2019 / IV-2a versions: a list of dicts with keys
    victim, probe, top1, top1_ci_low, top1_ci_high, chance_top1.
    """
    plt.rcParams.update(journal_style())
    fig, ax = plt.subplots(figsize=figsize or FIG_DOUBLE)

    victims = sorted({r["victim"] for r in results})
    probes = sorted({r["probe"] for r in results})

    n_v, n_p = len(victims), len(probes)
    bar_w = 0.78 / max(n_p, 1)
    x_centers = np.arange(n_v)

    probe_colors = {probes[0]: PALETTE["accent"]}
    if n_p > 1:
        probe_colors[probes[1]] = PALETTE["contrast"]
    if n_p > 2:
        for k, p in enumerate(probes[2:], start=2):
            probe_colors[p] = WONG[k % len(WONG)]

    max_y = 0.0
    for j, probe in enumerate(probes):
        ys, ylo, yhi = [], [], []
        for v in victims:
            row = next((r for r in results
                        if r["victim"] == v and r["probe"] == probe), None)
            ys.append(row["top1"] if row else np.nan)
            ylo.append((row["top1"] - row["top1_ci_low"]) if row else 0)
            yhi.append((row["top1_ci_high"] - row["top1"]) if row else 0)
        offsets = (j - (n_p - 1) / 2) * bar_w
        ax.bar(x_centers + offsets, ys, bar_w,
               yerr=[ylo, yhi],
               color=probe_colors[probe],
               edgecolor=PALETTE["ink"], linewidth=0.5,
               label=_pretty_probe(probe),
               error_kw=dict(ecolor=PALETTE["ink"], elinewidth=0.7,
                             capsize=2.5, capthick=0.7))
        for x, y, hi_v in zip(x_centers + offsets, ys, yhi):
            if y is not np.nan and not np.isnan(y):
                ax.text(x, y + hi_v + 0.018, f"{y:.2f}",
                        ha="center", va="bottom",
                        fontsize=7.0, color=PALETTE["ink"])
                max_y = max(max_y, y + hi_v + 0.05)

    chance = results[0]["chance_top1"]
    ax.axhline(chance, color=PALETTE["neutral"], lw=0.7, ls=(0, (4, 3)),
               label=f"chance ({chance:.3f})")

    ax.set_xticks(x_centers)
    ax.set_xticklabels([_pretty_victim(v) for v in victims])
    ax.set_ylabel("Re-identification top-1 accuracy")
    upper = max(1.10, max_y + 0.06)
    # Drop the lower edge just below 0 so a chance line at very small
    # values (e.g. 1/104 ≈ 0.010) sits clearly inside the plot box and
    # the dashed pattern is not clipped against the bottom spine.
    ax.set_ylim(-0.02, upper)
    if title:
        ax.set_title(title)
    # Anchor legend below the axes to guarantee it never collides with
    # bar value labels at the top of the plot.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
              fontsize=8.0, ncol=min(n_p + 1, 4),
              frameon=False, columnspacing=1.6, handletextpad=0.7)
    ax.grid(axis="y", which="major", linestyle=":", linewidth=0.4,
            alpha=0.35, color=PALETTE["muted"])
    ax.set_axisbelow(True)
    fig.savefig(out_path)
    plt.close(fig)


def verification_panel(
    scores: np.ndarray,
    labels: np.ndarray,
    auc: float,
    eer: float,
    out_path: str | Path,
    *,
    title: str | None = "Open-set verification (A4)",
    figsize: tuple[float, float] = FIG_DOUBLE_TALL,
) -> None:
    """Two-panel ROC + score histogram for verification results.

    Used when per-pair score arrays were persisted. When only the
    summary stats are available, use `verification_dotplot` instead.
    """
    from sklearn.metrics import roc_curve

    plt.rcParams.update(journal_style())
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=figsize)

    same = scores[labels == 1]
    diff = scores[labels == 0]
    bins = np.linspace(min(scores.min(), -1.0), max(scores.max(), 1.0), 42)
    ax_l.hist(diff, bins=bins, color=PALETTE["neutral"], alpha=0.85,
              edgecolor="white", linewidth=0.3, label="different subject")
    ax_l.hist(same, bins=bins, color=PALETTE["accent"], alpha=0.75,
              edgecolor="white", linewidth=0.3, label="same subject")
    ax_l.set_xlabel("Cosine similarity")
    ax_l.set_ylabel("Pair count")
    ax_l.legend(loc="upper left", fontsize=7.5)
    ax_l.grid(axis="y", linestyle=":", linewidth=0.4,
              alpha=0.35, color=PALETTE["muted"])
    ax_l.set_axisbelow(True)

    fpr, tpr, _ = roc_curve(labels, scores)
    ax_r.plot([0, 1], [0, 1], color=PALETTE["neutral"], lw=0.7,
              ls=(0, (4, 3)), label="chance")
    ax_r.plot(fpr, tpr, color=PALETTE["accent"], lw=1.5,
              label=f"AUC $=$ {auc:.3f}")
    ax_r.set_xlabel("False-positive rate")
    ax_r.set_ylabel("True-positive rate")
    ax_r.set_xlim(0, 1)
    ax_r.set_ylim(0, 1.02)
    ax_r.legend(loc="lower right", fontsize=8.0)
    ax_r.set_aspect("equal", adjustable="box")
    ax_r.text(0.96, 0.05, f"EER $=$ {eer:.3f}",
              ha="right", va="bottom", fontsize=8.0,
              color=PALETTE["ink"])

    if title:
        fig.suptitle(title, fontsize=10.5)
    fig.savefig(out_path)
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
    title: str | None = "A4 open-set verification on unseen subjects",
    chance: float = 0.5,
    extra_seeds: Sequence[float] | None = None,
    seed_mean: float | None = None,
    seed_std: float | None = None,
) -> None:
    """Horizontal forest plot of an AUC measurement plus per-seed
    replications. One row per seed; the seed-0 measurement carries its
    bootstrap CI as a horizontal whisker; the multi-seed extension
    contributes one row per replicate. A vertical mean rule and a
    shaded ±std band summarise the multi-seed cluster; a chance
    reference line on the far left anchors the AUC scale.
    """
    plt.rcParams.update(journal_style())

    err_lo = auc - auc_ci_low
    err_hi = auc_ci_high - auc

    has_multi = extra_seeds is not None and len(extra_seeds) > 0
    seeds_arr = (np.asarray(extra_seeds, dtype=float)
                 if has_multi else np.array([]))
    n_rows = 1 + (len(seeds_arr) if has_multi else 0)

    # Vertical size scales with row count. Single-row cards get a
    # taller-than-needed canvas so the title, axis label and legend
    # don't crash into each other when there's only one dot to show.
    height = (3.0 if n_rows == 1
              else 2.6 + 0.35 * max(0, n_rows - 1))
    fig, ax = plt.subplots(figsize=(FIG_DOUBLE[0], height))

    # Row layout: seed 0 sits at the TOP (y=n_rows-1) so the eye reads
    # top-to-bottom from headline measurement → replications. The
    # replicates appear in input order beneath: replicate 1 just below
    # the headline, replicate N at the bottom of the panel.
    seed0_y = n_rows - 1
    multi_ys = (np.arange(len(seeds_arr) - 1, -1, -1)
                if has_multi else np.array([], dtype=int))

    # Decide x range: tight around the data with chance pulled into view
    # as a left-edge reference; this keeps the dots away from the spines
    # and yields a high data-ink ratio for AUCs that cluster well above
    # chance.
    data_lo = min([auc - err_lo] +
                  ([float(seeds_arr.min())] if has_multi else []))
    data_hi = max([auc + err_hi] +
                  ([float(seeds_arr.max())] if has_multi else []))
    span = max(0.06, data_hi - data_lo)
    x_lo = min(chance - 0.02, data_lo - 0.30 * span)
    x_hi = min(1.005, data_hi + 0.40 * span)

    # Multi-seed mean ± std shown as a thin vertical band + rule.
    if has_multi:
        smean = (seed_mean if seed_mean is not None
                 else float(seeds_arr.mean()))
        sstd = (seed_std if seed_std is not None
                else float(seeds_arr.std(ddof=1)))
        ax.axvspan(smean - sstd, smean + sstd,
                   color=PALETTE["warn"], alpha=0.14, zorder=1,
                   label=f"multi-seed mean ± std  ({smean:.3f} ± {sstd:.3f})")
        ax.axvline(smean, color=PALETTE["contrast"], lw=1.2,
                   ls=(0, (5, 2)), zorder=2)

        # Per-seed rows: dots with the AUC label to their right.
        for y_pos, seed_val in zip(multi_ys, seeds_arr):
            ax.plot([seed_val], [y_pos], "o",
                    color=PALETTE["contrast"],
                    markerfacecolor=PALETTE["contrast"],
                    markeredgecolor=PALETTE["ink"], markeredgewidth=0.7,
                    markersize=6.5, zorder=3)
            ax.text(min(x_hi - 0.005, seed_val + 0.008), y_pos,
                    f"{seed_val:.3f}",
                    va="center", ha="left", fontsize=7.5,
                    color=PALETTE["ink"])

    # Seed-0 row with bootstrap CI whisker.
    ax.errorbar([auc], [seed0_y], xerr=[[err_lo], [err_hi]],
                fmt="o", color=PALETTE["accent"],
                ecolor=PALETTE["ink"], elinewidth=1.0,
                capsize=4.5, capthick=1.0,
                markersize=8.5,
                markerfacecolor=PALETTE["accent"],
                markeredgecolor=PALETTE["ink"], markeredgewidth=0.9,
                zorder=5)
    ci_text = f"  {auc:.3f}  [{auc_ci_low:.3f}, {auc_ci_high:.3f}]"
    ax.text(min(x_hi - 0.003, auc + err_hi + 0.006), seed0_y, ci_text,
            va="center", ha="left", fontsize=8.5,
            fontweight="bold", color=PALETTE["ink"])

    # Chance reference at the far left of the AUC axis.
    ax.axvline(chance, color=PALETTE["neutral"], lw=0.9,
               ls=(0, (4, 3)),
               label=f"chance ({chance:.3f})")

    # Row labels (use "replicate i" because the seed-extension JSONs
    # don't carry seed numbers in this code path — these are independent
    # 80/24 random splits, not the literal seeds 1..N).
    yticks = list(multi_ys) + [seed0_y]
    yticklabels = ([f"replicate {i + 1}" for i in range(len(seeds_arr))]
                   if has_multi else []) + ["seed 0\n(headline)"]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylim(-0.55, n_rows - 0.45)
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel("AUC")

    # Title carries the full cohort line so the subtitle never duplicates
    # the "X unseen subjects" string from the calling renderer.
    if title:
        ax.set_title(
            f"{title}\n"
            f"train n={n_train_subjects} · unseen n={n_test_subjects} · "
            f"{n_pairs:,} pairs · EER = {eer:.3f}",
            fontsize=10.0,
        )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
              fontsize=7.5, ncol=2, frameon=False,
              columnspacing=1.6, handletextpad=0.6)
    ax.grid(axis="x", which="major", linestyle=":", linewidth=0.4,
            alpha=0.35, color=PALETTE["muted"])
    ax.set_axisbelow(True)
    fig.savefig(out_path)
    plt.close(fig)


def closed_set_table(results: list[dict]) -> str:
    """Render the closed-set JSON as a markdown table."""
    lines = [
        "| Victim | Probe | Top-1 (95% CI) | Top-5 | Top-10 | Task acc | Chance top-1 |",
        "|---|---|---|---|---|---|---|",
    ]

    def _fmt(v):
        if v is None:
            return "—"
        try:
            return "—" if np.isnan(v) else f"{v:.3f}"
        except TypeError:
            return "—"

    for r in sorted(results, key=lambda x: (x["victim"], x["probe"])):
        ci = f"{r['top1']:.3f} [{r['top1_ci_low']:.3f}, {r['top1_ci_high']:.3f}]"
        task_acc = r.get("task_acc")
        task_acc_s = (f"{task_acc:.3f}"
                      if isinstance(task_acc, (int, float)) else "—")
        lines.append(
            f"| {r['victim']} | {r['probe']} | {ci} | "
            f"{_fmt(r.get('top5'))} | {_fmt(r.get('top10'))} | "
            f"{task_acc_s} | {r['chance_top1']:.3f} |"
        )
    return "\n".join(lines)

