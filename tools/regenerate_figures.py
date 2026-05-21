"""Rebuild every figure under figures/ from the canonical result JSONs.

Single-command reproducibility: a reviewer who clones the repo and runs

    python -m tools.regenerate_figures

ends up with every PDF in figures/ regenerated from results/*.json, with
a unified journal-grade style throughout. Renderers here delegate the
common helpers to `eval.plots`; per-experiment specifics (axis ticks,
annotations, comparison lines) are kept inline so the renderers double
as a reference for which JSON drives which figure.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import FIGURES_DIR, RESULTS_DIR
from eval.plots import (
    FIG_DOUBLE,
    FIG_DOUBLE_TALL,
    FIG_SINGLE,
    FIG_SINGLE_TALL,
    FIG_TRIPLE,
    FIG_WIDE,
    PALETTE,
    closed_set_bar_chart,
    journal_style,
    verification_panel,
    verification_summary_card,
)


# ---------------------------------------------------------------------------
# Generic primitives -- single source of truth for grouped bars, dot-plots,
# attack-vs-defense comparison plots, and the like. Inlined here so the
# experiment modules don't need to be loaded at figure-regeneration time.
# ---------------------------------------------------------------------------
def _maybe_grid(ax, axis: str = "y") -> None:
    ax.grid(axis=axis, which="major", linestyle=":", linewidth=0.4,
            alpha=0.35, color=PALETTE["muted"])
    ax.set_axisbelow(True)


def _annotate_bar(ax, x: float, y: float, *, text: str | None = None,
                  fontsize: float = 7.0, dy: float = 0.012) -> None:
    ax.text(x, y + dy, text if text is not None else f"{y:.3f}",
            ha="center", va="bottom", fontsize=fontsize, color=PALETTE["ink"])


# ---------------------------------------------------------------------------
# A1 -- A3 + Lee 2019 A3 (closed-set bar charts)
# ---------------------------------------------------------------------------
def render_within_subject_reid() -> None:
    """A1 within-subject (experiment 03): per-subject task accuracy +
    cross-subject re-ID via the all-models attack."""
    p = RESULTS_DIR / "03_within_subject_reid.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    task_rows = d.get("task_rows", [])
    attack_rows = d.get("attack_rows", [])
    if not task_rows or not attack_rows:
        return

    plt.rcParams.update(journal_style())
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(8.4, 3.8))

    # Left panel: per-subject task accuracy as a strip plot per victim
    pretty_v = {"eegnet": "EEGNet", "fbcsp": "FBCSP+LDA",
                "riemann": "Riemann TS"}
    victims = list(pretty_v.keys())
    rng = np.random.default_rng(0)
    for i, v in enumerate(victims):
        accs = [r["task_acc"] for r in task_rows
                if r["victim_family"] == v]
        if not accs:
            continue
        jit = rng.uniform(-0.10, 0.10, size=len(accs))
        ax_l.scatter(np.full(len(accs), i) + jit, accs,
                     color=PALETTE["accent"], edgecolor=PALETTE["ink"],
                     linewidth=0.4, s=34, alpha=0.85, zorder=3)
        ax_l.hlines(np.mean(accs), i - 0.22, i + 0.22,
                    color=PALETTE["ink"], lw=1.2, zorder=4)
        ax_l.text(i, max(accs) + 0.03, f"mean {np.mean(accs):.2f}",
                  ha="center", va="bottom", fontsize=7.5,
                  color=PALETTE["neutral"])
    ax_l.axhline(0.25, color=PALETTE["neutral"], lw=0.7, ls=(0, (4, 3)),
                 label="chance (4-class = 0.250)")
    ax_l.set_xticks(np.arange(len(victims)))
    ax_l.set_xticklabels([pretty_v[v] for v in victims])
    ax_l.set_ylabel("Per-subject motor-imagery task accuracy")
    ax_l.set_ylim(0, 1.0)
    ax_l.set_title("Within-subject task performance (n=10)")
    ax_l.legend(loc="lower right", fontsize=7.5)
    _maybe_grid(ax_l, "y")

    # Right panel: all-models attack re-ID across victim families
    victim_map = {"eegnet": "EEGNet", "fbcsp_lda": "FBCSP+LDA",
                  "riemann_ts_lr": "Riemann TS"}
    families = list(victim_map.keys())
    attacks = sorted({r["attack"] for r in attack_rows})
    width = 0.38
    x = np.arange(len(families))
    for j, atk in enumerate(attacks):
        ys, ylo, yhi = [], [], []
        for fam in families:
            row = next((r for r in attack_rows
                        if r["victim_family"] == fam and r["attack"] == atk),
                       None)
            ys.append(row["top1"] if row else np.nan)
            ylo.append((row["top1"] - row["top1_ci_low"]) if row else 0)
            yhi.append((row["top1_ci_high"] - row["top1"]) if row else 0)
        ax_r.bar(x + (j - (len(attacks) - 1) / 2) * width, ys, width,
                 yerr=[ylo, yhi],
                 color=PALETTE["accent"] if atk == "argmax_conf"
                                            else PALETTE["contrast"],
                 edgecolor=PALETTE["ink"], linewidth=0.5,
                 label=atk.replace("_", " "),
                 error_kw=dict(ecolor=PALETTE["ink"], elinewidth=0.7,
                               capsize=2.5, capthick=0.7))
    ax_r.axhline(attack_rows[0]["chance_top1"],
                 color=PALETTE["neutral"], lw=0.7, ls=(0, (4, 3)),
                 label=f"chance ({attack_rows[0]['chance_top1']:.3f})")
    ax_r.set_xticks(x)
    ax_r.set_xticklabels([victim_map[f] for f in families], fontsize=8.0)
    ax_r.set_ylabel("Re-ID top-1 (all-models attack)")
    ax_r.set_ylim(0, 1.05)
    ax_r.set_title("Within-subject re-identification (10 personal victims)")
    ax_r.legend(loc="upper left", fontsize=7.5)
    _maybe_grid(ax_r, "y")

    fig.suptitle(
        "Within-subject re-ID baseline (experiment 03, PhysioNet, n=10)",
        fontsize=10.5,
    )
    fig.savefig(FIGURES_DIR / "03_within_subject_reid.pdf")
    plt.close(fig)
    print("regenerated figures/03_within_subject_reid.pdf")


def render_a1() -> None:
    rows = json.loads((RESULTS_DIR / "02_closed_set_reid.json").read_text())
    n = rows[0]["n_subjects"]
    closed_set_bar_chart(
        rows, FIGURES_DIR / "02_closed_set_reid.pdf",
        title=f"A1 closed-set re-identification (PhysioNet, n={n})",
    )
    print("regenerated figures/02_closed_set_reid.pdf")


def render_a2() -> None:
    rows = json.loads((RESULTS_DIR / "04_a2_cross_task.json").read_text())
    n = rows[0]["n_subjects"]
    closed_set_bar_chart(
        rows, FIGURES_DIR / "04_a2_cross_task.pdf",
        title=f"A2 cross-task re-ID (PhysioNet, n={n}, execution→imagery)",
    )
    print("regenerated figures/04_a2_cross_task.pdf")


def render_a3() -> None:
    rows = json.loads((RESULTS_DIR / "05_a3_cross_session.json").read_text())
    n = rows[0]["n_subjects"]
    closed_set_bar_chart(
        rows, FIGURES_DIR / "05_a3_cross_session.pdf",
        title=f"A3 cross-session re-ID (BCI IV-2a, n={n}, session-1→session-2)",
    )
    print("regenerated figures/05_a3_cross_session.pdf")


def render_a2_vs_rest() -> None:
    path = RESULTS_DIR / "21_a2_vs_rest.json"
    if not path.exists():
        return
    rows = json.loads(path.read_text())
    n = rows[0]["n_subjects"]
    closed_set_bar_chart(
        rows, FIGURES_DIR / "21_a2_vs_rest.pdf",
        title=(f"A2 cross-task re-ID with resting-state probe "
               f"(PhysioNet, n={n})"),
    )
    print("regenerated figures/21_a2_vs_rest.pdf")


def render_lee2019_a3() -> None:
    path = RESULTS_DIR / "20_a3_lee2019.json"
    if not path.exists():
        return
    rows = json.loads(path.read_text())
    n = rows[0]["n_subjects"]
    closed_set_bar_chart(
        rows, FIGURES_DIR / "20_a3_lee2019.pdf",
        title=(f"A3 cross-session re-ID (Lee 2019 OpenBMI, n={n}, "
               f"session-1→session-2)"),
    )
    print("regenerated figures/20_a3_lee2019.pdf")


# ---------------------------------------------------------------------------
# A4 verification panels / summary cards
# ---------------------------------------------------------------------------
def render_a4() -> None:
    a4 = json.loads((RESULTS_DIR / "06_a4_open_set.json").read_text())
    scores_path = RESULTS_DIR / "06_a4_open_set_scores.npz"
    if scores_path.exists():
        data = np.load(scores_path)
        verification_panel(
            scores=data["scores"], labels=data["labels"],
            auc=a4["auc"], eer=a4["eer"],
            out_path=FIGURES_DIR / "06_a4_open_set.pdf",
            title=(f"A4 open-set verification (PhysioNet, "
                   f"{a4['n_test_subjects']} unseen subjects, "
                   f"{a4['n_pairs']:,} pairs)"),
        )
    else:
        # Overlay the 5-seed extension from experiment 14 when available
        multi_seed_path = RESULTS_DIR / "14_a4_multi_seed.json"
        extra_seeds = None
        seed_mean = None
        seed_std = None
        if multi_seed_path.exists():
            ms = json.loads(multi_seed_path.read_text())
            extra_seeds = [r["auc"] for r in ms.get("per_seed", [])]
            agg = ms.get("aggregate", {})
            seed_mean = agg.get("auc_mean")
            seed_std = agg.get("auc_std")
        verification_summary_card(
            auc=a4["auc"], auc_ci_low=a4["auc_ci_low"],
            auc_ci_high=a4["auc_ci_high"], eer=a4["eer"],
            n_train_subjects=a4["n_train_subjects"],
            n_test_subjects=a4["n_test_subjects"],
            n_pairs=a4["n_pairs"],
            out_path=FIGURES_DIR / "06_a4_open_set.pdf",
            title="A4 open-set verification (PhysioNet)",
            extra_seeds=extra_seeds,
            seed_mean=seed_mean, seed_std=seed_std,
        )
    print("regenerated figures/06_a4_open_set.pdf")


def render_lee2019_a4() -> None:
    # Pull per-seed AUCs from experiment 34 when available so the
    # within-session variant gets a strip-plot overlay alongside the
    # seed-0 bar.
    multi_seed_path = RESULTS_DIR / "34_multi_seed.json"
    a4_seeds = None
    a4_mean = None
    a4_std = None
    if multi_seed_path.exists():
        ms = json.loads(multi_seed_path.read_text())
        agg = (ms.get("rows", {}).get("a4_lee2019", {})
                  .get("aggregated", {}).get("auc_within_session", {}))
        a4_seeds = agg.get("values")
        a4_mean = agg.get("mean")
        a4_std = agg.get("std")

    for variant, label in (("within_session", "within-session"),
                            ("cross_session", "cross-session")):
        meta_path = RESULTS_DIR / f"24_a4_lee2019_{variant}.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        verification_summary_card(
            auc=meta["auc"], auc_ci_low=meta["auc_ci_low"],
            auc_ci_high=meta["auc_ci_high"], eer=meta["eer"],
            n_train_subjects=meta["n_train_subjects"],
            n_test_subjects=meta["n_test_subjects"],
            n_pairs=meta["n_pairs"],
            out_path=FIGURES_DIR / f"24_a4_lee2019_{variant}.pdf",
            title=f"A4 open-set verification (Lee 2019, {label})",
            extra_seeds=a4_seeds if variant == "within_session" else None,
            seed_mean=a4_mean if variant == "within_session" else None,
            seed_std=a4_std if variant == "within_session" else None,
        )
        print(f"regenerated figures/24_a4_lee2019_{variant}.pdf")


# ---------------------------------------------------------------------------
# A5 membership inference (PhysioNet + Lee 2019 + classical victims)
# ---------------------------------------------------------------------------
def _render_mi_card(*, label: str, auc: float, lo: float, hi: float,
                    advantage: float, n_members: int, n_nonmembers: int,
                    title: str, out_path: Path) -> None:
    """Two-bar MI summary: AUC bar + TPR-FPR advantage bar.

    Both metrics share the [0, 1] axis; chance / random reference is at
    0.5 for AUC and 0 for advantage, indicated by the dashed lines.
    """
    plt.rcParams.update(journal_style())
    fig, ax = plt.subplots(figsize=FIG_DOUBLE)
    labels = ["MI AUC", "TPR − FPR advantage"]
    x = np.arange(len(labels))
    # Forest-plot points for both metrics. Neither AUC nor the
    # TPR-FPR advantage is a quantity that grows out of zero — chance
    # sits at 0.5 and 0 respectively — so bars from the axis would
    # overstate magnitude. Points + chance reference lines read honestly.
    ax.errorbar([x[0]], [auc], yerr=[[auc - lo], [hi - auc]],
                fmt="o", color=PALETTE["accent"],
                ecolor=PALETTE["ink"], elinewidth=1.0,
                capsize=4.5, capthick=1.0,
                markersize=9.0, markerfacecolor=PALETTE["accent"],
                markeredgecolor=PALETTE["ink"], markeredgewidth=0.9,
                zorder=4)
    ax.plot([x[1]], [advantage], "o", color=PALETTE["warn"],
            markersize=9.0, markerfacecolor=PALETTE["warn"],
            markeredgecolor=PALETTE["ink"], markeredgewidth=0.9,
            zorder=4)
    # Annotations: bold value above the point, CI bracket just beneath
    # for the AUC measurement. The advantage statistic in this experiment
    # family is reported without a CI so we annotate it as a point value.
    ci_y = auc + (hi - auc) + 0.028
    value_y = ci_y + 0.065
    ax.text(x[0], value_y, f"{auc:.3f}",
            ha="center", va="bottom", fontsize=11.0,
            fontweight="bold", color=PALETTE["ink"])
    ax.text(x[0], ci_y, f"[{lo:.3f}, {hi:.3f}]",
            ha="center", va="bottom", fontsize=7.0,
            color=PALETTE["neutral"])
    ax.text(x[1], advantage + 0.045, f"{advantage:.3f}",
            ha="center", va="bottom", fontsize=11.0,
            fontweight="bold", color=PALETTE["ink"])
    # Reference chance lines per column.
    ax.hlines(0.5, x[0] - 0.40, x[0] + 0.40, color=PALETTE["neutral"],
              lw=1.0, ls=(0, (4, 3)),
              label="MI chance (AUC = 0.5)")
    ax.hlines(0.0, x[1] - 0.40, x[1] + 0.40, color=PALETTE["neutral"],
              lw=1.0, ls=":",
              label="advantage chance (0)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlim(-0.55, len(labels) - 0.45)
    ax.set_ylim(-0.15, 1.18)
    ax.set_ylabel("metric value")
    # Cohort line as a subtitle below the axes title
    ax.set_title(
        f"{title}\n"
        f"{label} · {n_members} members vs {n_nonmembers} non-members",
        fontsize=10.0,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
              fontsize=7.5, ncol=2, frameon=False,
              columnspacing=1.6, handletextpad=0.6)
    ax.grid(axis="y", which="major", linestyle=":", linewidth=0.4,
            alpha=0.35, color=PALETTE["muted"])
    ax.set_axisbelow(True)
    fig.savefig(out_path)
    plt.close(fig)


def render_a5() -> None:
    path = RESULTS_DIR / "08_a5_membership_inference.json"
    if not path.exists():
        return
    d = json.loads(path.read_text())
    _render_mi_card(
        label=f"{d['n_shadows']} EEGNet shadows on PhysioNet imagery",
        auc=d["auc"], lo=d["auc_ci_low"], hi=d["auc_ci_high"],
        advantage=d["advantage"],
        n_members=d["n_target_members"],
        n_nonmembers=d["n_target_nonmembers"],
        title="A5 black-box membership inference (PhysioNet, EEGNet shadows)",
        out_path=FIGURES_DIR / "08_a5_membership_inference.pdf",
    )
    print("regenerated figures/08_a5_membership_inference.pdf")


def render_lee2019_a5() -> None:
    path = RESULTS_DIR / "25_a5_lee2019.json"
    if not path.exists():
        return
    d = json.loads(path.read_text())
    _render_mi_card(
        label=f"{d['n_shadows']} EEGNet shadows on Lee 2019 imagery",
        auc=d["auc"], lo=d["auc_ci_low"], hi=d["auc_ci_high"],
        advantage=d["advantage"],
        n_members=d["n_target_members"],
        n_nonmembers=d["n_target_nonmembers"],
        title="A5 black-box membership inference (Lee 2019)",
        out_path=FIGURES_DIR / "25_a5_lee2019.pdf",
    )
    print("regenerated figures/25_a5_lee2019.pdf")


def render_a5_classical() -> None:
    """Comparison across victim families for the classical MI experiment.
    One bar per victim with CI."""
    rows = []
    for vf, fname in (("EEGNet", "08_a5_membership_inference.json"),
                       ("FBCSP+LDA", "16_a5_fbcsp_mi.json"),
                       ("Riemann tangent-space", "16_a5_riemann_mi.json")):
        p = RESULTS_DIR / fname
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        rows.append((vf, d["auc"], d["auc_ci_low"], d["auc_ci_high"],
                     d["advantage"]))
    if not rows:
        return
    plt.rcParams.update(journal_style())
    fig, ax = plt.subplots(figsize=FIG_DOUBLE)
    labels = [r[0] for r in rows]
    values = np.array([r[1] for r in rows])
    lo = values - np.array([r[2] for r in rows])
    hi = np.array([r[3] for r in rows]) - values
    colors = [PALETTE["accent"], PALETTE["contrast"], PALETTE["fail"]][: len(rows)]
    bars = ax.bar(labels, values, yerr=[lo, hi], color=colors,
                  edgecolor=PALETTE["ink"], linewidth=0.6, width=0.55,
                  error_kw=dict(ecolor=PALETTE["ink"], elinewidth=0.9,
                                capsize=3, capthick=0.9))
    ax.axhline(0.5, color=PALETTE["neutral"], lw=0.8, ls=(0, (4, 3)),
               label="chance (AUC = 0.5)")
    for b, v, h in zip(bars, values, hi):
        ax.text(b.get_x() + b.get_width() / 2, v + h + 0.015, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9.0,
                fontweight="bold", color=PALETTE["ink"])
    ax.set_ylim(0.45, 1.10)
    ax.set_ylabel("MI AUC")
    ax.set_title("A5 black-box membership inference, victim comparison (PhysioNet)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              fontsize=7.5, ncol=2, frameon=False,
              columnspacing=1.6, handletextpad=0.6)
    _maybe_grid(ax, "y")
    fig.savefig(FIGURES_DIR / "16_a5_classical.pdf")
    plt.close(fig)
    print("regenerated figures/16_a5_classical.pdf")


# ---------------------------------------------------------------------------
# Defense sweeps -- D1 / D2 / D3 / multi-defense Pareto
# ---------------------------------------------------------------------------
def _defense_sweep_panel(rows, out_path: Path, *, title: str, x_label: str,
                          key: str = "lambda") -> None:
    """Generic two-panel: left = leakage vs sweep, right = task acc vs sweep."""
    if not rows:
        return
    logreg = sorted([r for r in rows if r.get("probe") == "logreg"],
                    key=lambda r: r[key])
    if not logreg:
        return
    plt.rcParams.update(journal_style())
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=FIG_DOUBLE_TALL,
                                       sharex=True)
    xs = [r[key] for r in logreg]
    leak = [r["top1"] for r in logreg]
    lo = [r["top1"] - r["top1_ci_low"] for r in logreg]
    hi = [r["top1_ci_high"] - r["top1"] for r in logreg]
    util = [r["task_acc"] for r in logreg]
    chance = logreg[0]["chance_top1"]

    ax_l.errorbar(xs, leak, yerr=[lo, hi], color=PALETTE["contrast"],
                  marker="o", lw=1.3, capsize=2.5, capthick=0.7, markersize=5,
                  markerfacecolor="white", markeredgewidth=1.2,
                  label="re-ID top-1 (logreg probe)")
    ax_l.axhline(chance, color=PALETTE["neutral"], lw=0.7, ls=(0, (4, 3)),
                 label=f"chance ({chance:.3f})")
    ax_l.set_xlabel(x_label)
    ax_l.set_ylabel("Re-ID top-1 accuracy")
    ax_l.set_ylim(0, max(leak) * 1.25 + 0.05)
    ax_l.legend(loc="upper right", fontsize=7.5)
    _maybe_grid(ax_l, "y")

    ax_r.plot(xs, util, color=PALETTE["ok"], marker="s",
              lw=1.3, markersize=5, markerfacecolor="white",
              markeredgewidth=1.2, label="motor-imagery task accuracy")
    ax_r.axhline(0.25, color=PALETTE["neutral"], lw=0.7, ls=(0, (4, 3)),
                 label="chance (4-class = 0.250)")
    ax_r.set_xlabel(x_label)
    ax_r.set_ylabel("Task accuracy")
    ax_r.set_ylim(0.20, max(0.45, max(util) + 0.03))
    ax_r.legend(loc="lower left", fontsize=7.5)
    _maybe_grid(ax_r, "y")

    fig.suptitle(title, fontsize=10.5)
    fig.savefig(out_path)
    plt.close(fig)


def render_d1_pca() -> None:
    p = RESULTS_DIR / "07_d1_pca.json"
    if not p.exists():
        return
    rows = json.loads(p.read_text())
    rows_eeg = [r for r in rows if r.get("victim") == "eegnet"]
    if not rows_eeg:
        rows_eeg = rows  # fallback for older schemas
    # The PCA sweep uses either `k` or `strength` depending on commit history.
    sweep_key = "k" if rows_eeg and "k" in rows_eeg[0] else "strength"
    _defense_sweep_panel(
        rows_eeg, FIGURES_DIR / "07_d1_pca.pdf",
        title=(f"D1 PCA channel-compression defense "
               f"(EEGNet victim, n={rows[0]['n_subjects']})"),
        x_label="PCA components (k)", key=sweep_key,
    )
    print("regenerated figures/07_d1_pca.pdf")


def render_d1_other(transform: str) -> None:
    p = RESULTS_DIR / f"11_d1_{transform}.json"
    if not p.exists():
        return
    rows = json.loads(p.read_text())
    # The D1 transforms all key their sweep variable as `strength` in the
    # canonical JSON; the eegnet rows additionally tag `defense` strings.
    # Restrict to EEGNet logreg rows for the sweep panel.
    rows_eeg = [r for r in rows if r.get("victim") == "eegnet"]
    if transform == "noise":
        _defense_sweep_panel(
            rows_eeg, FIGURES_DIR / "11_d1_noise.pdf",
            title=(f"D1 additive Gaussian noise defense "
                   f"(EEGNet victim, n={rows[0]['n_subjects']})"),
            x_label=r"noise $\sigma$ (× per-channel std)", key="strength",
        )
    elif transform == "channel_drop":
        _defense_sweep_panel(
            rows_eeg, FIGURES_DIR / "11_d1_channel_drop.pdf",
            title=(f"D1 channel-drop defense (top-k by variance, "
                   f"EEGNet victim, n={rows[0]['n_subjects']})"),
            x_label="channels kept (k)", key="strength",
        )
    print(f"regenerated figures/11_d1_{transform}.pdf")


def render_d2_dann() -> None:
    rows = []
    for fname in ("09_d2_dann.json", "09_d2_dann_extended.json"):
        p = RESULTS_DIR / fname
        if p.exists():
            rows.extend(json.loads(p.read_text()))
    if not rows:
        return
    _defense_sweep_panel(
        rows, FIGURES_DIR / "09_d2_dann.pdf",
        title=(f"D2 DANN adversarial subject-invariance "
               f"(EEGNet victim, n={rows[0]['n_subjects']})"),
        x_label="DANN gradient-reversal strength λ",
        key="lambda",
    )
    print("regenerated figures/09_d2_dann.pdf")


def render_d3() -> None:
    p = RESULTS_DIR / "10_d3_dp_sgd.json"
    if not p.exists():
        return
    rows = json.loads(p.read_text())
    logreg = [r for r in rows if r["probe"] == "logreg"]
    if not logreg:
        return
    # Build a synthetic numeric key so the sort works for the {None, 10, 3} sweep
    for r in logreg:
        r["_eps_key"] = (r["target_epsilon"]
                         if r["target_epsilon"] is not None else 1e9)
    logreg = sorted(logreg, key=lambda r: r["_eps_key"])
    labels = ["no DP" if r["target_epsilon"] is None
              else f"ε={r['target_epsilon']:g}" for r in logreg]

    plt.rcParams.update(journal_style())
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=FIG_DOUBLE_TALL)
    x = np.arange(len(logreg))
    leak = [r["top1"] for r in logreg]
    lo = [r["top1"] - r["top1_ci_low"] for r in logreg]
    hi = [r["top1_ci_high"] - r["top1"] for r in logreg]
    util = [r["task_acc"] for r in logreg]

    ax_l.errorbar(x, leak, yerr=[lo, hi], color=PALETTE["contrast"],
                  marker="o", lw=1.3, capsize=2.5, capthick=0.7,
                  markersize=5, markerfacecolor="white",
                  markeredgewidth=1.2, label="re-ID top-1 (logreg probe)")
    ax_l.axhline(logreg[0]["chance_top1"], color=PALETTE["neutral"],
                 lw=0.7, ls=(0, (4, 3)),
                 label=f"chance ({logreg[0]['chance_top1']:.3f})")
    ax_l.set_xticks(x); ax_l.set_xticklabels(labels)
    ax_l.set_xlabel("DP-SGD target ε")
    ax_l.set_ylabel("Re-ID top-1 accuracy")
    ax_l.set_ylim(0, max(leak) * 1.6 + 0.05)
    ax_l.legend(loc="upper right", fontsize=7.5)
    _maybe_grid(ax_l, "y")

    ax_r.plot(x, util, color=PALETTE["ok"], marker="s",
              lw=1.3, markersize=5, markerfacecolor="white",
              markeredgewidth=1.2, label="motor-imagery task accuracy")
    ax_r.set_xticks(x); ax_r.set_xticklabels(labels)
    ax_r.set_xlabel("DP-SGD target ε")
    ax_r.set_ylabel("Task accuracy")
    ax_r.set_ylim(0.20, 0.40)
    ax_r.legend(loc="lower right", fontsize=7.5)
    _maybe_grid(ax_r, "y")

    fig.suptitle("D3 DP-SGD ε sweep, three-point grid",
                 fontsize=10.5)
    fig.savefig(FIGURES_DIR / "10_d3_dp_sgd.pdf")
    plt.close(fig)
    print("regenerated figures/10_d3_dp_sgd.pdf")


def render_pareto() -> None:
    """All-defense Pareto: re-ID vs task acc, color-coded by family.

    The defenses sweep only EEGNet (the victim used by the D1/D2/D3
    columns), so the no-defense reference is filtered to the EEGNet row
    of A1 — otherwise the two extra A1 stars (FBCSP, Riemann) would
    misleadingly suggest we evaluate defenses on classical victims too.
    """
    plt.rcParams.update(journal_style())
    points = []  # (family, defense_label, top1, task_acc)

    def _pull(path: Path, family: str):
        if not path.exists():
            return
        for r in json.loads(path.read_text()):
            if r.get("probe") != "logreg":
                continue
            # Accept the EEGNet variant tags used by each defense JSON:
            #   D1   → "eegnet"
            #   D2   → "eegnet_dann"
            #   D3   → "eegnet_dpsgd"
            # Rows lacking a `victim` field are legacy EEGNet runs.
            # Cross-victim rows (FBCSP, Riemann tangent-space) are
            # filtered out so the no-defense reference and the defense
            # scatter share the same victim baseline.
            row_victim = r.get("victim", "eegnet")
            if row_victim not in ("eegnet", "eegnet_dann", "eegnet_dpsgd"):
                continue
            label = r.get("defense") or r.get("transform") or ""
            points.append((family, label, r["top1"], r["task_acc"]))

    _pull(RESULTS_DIR / "02_closed_set_reid.json", "no defense")
    _pull(RESULTS_DIR / "07_d1_pca.json", "D1 PCA")
    _pull(RESULTS_DIR / "11_d1_noise.json", "D1 noise")
    _pull(RESULTS_DIR / "11_d1_channel_drop.json", "D1 channel-drop")
    for fname in ("09_d2_dann.json", "09_d2_dann_extended.json"):
        _pull(RESULTS_DIR / fname, "D2 DANN")
    _pull(RESULTS_DIR / "10_d3_dp_sgd.json", "D3 DP-SGD")

    if not points:
        return

    family_color = {
        "no defense":      PALETTE["ink"],
        "D1 PCA":          PALETTE["skyblue"],
        "D1 noise":        PALETTE["warn"],
        "D1 channel-drop": PALETTE["accent"],
        "D2 DANN":         PALETTE["fail"],
        "D3 DP-SGD":       PALETTE["purple"],
    }
    family_marker = {
        "no defense":      "*",
        "D1 PCA":          "o",
        "D1 noise":        "s",
        "D1 channel-drop": "^",
        "D2 DANN":         "v",
        "D3 DP-SGD":       "D",
    }

    fig, ax = plt.subplots(figsize=FIG_WIDE)
    for fam in family_color:
        xs = [p[3] for p in points if p[0] == fam]
        ys = [p[2] for p in points if p[0] == fam]
        if not xs:
            continue
        ax.scatter(xs, ys, color=family_color[fam],
                   marker=family_marker[fam],
                   s=46 if fam != "no defense" else 110,
                   edgecolor=PALETTE["ink"], linewidth=0.6,
                   label=fam, zorder=3 + (fam == "no defense"))
    ax.set_xlabel("Motor-imagery task accuracy")
    ax.set_ylabel("A1 closed-set re-ID top-1 (logreg)")
    max_x = max(p[3] for p in points)
    max_y = max(p[2] for p in points)
    ax.set_xlim(0.22, max_x + 0.012)
    ax.set_ylim(-0.02, max(0.55, max_y + 0.08))
    ax.set_title("Privacy–utility Pareto across defense families (EEGNet victim)")
    # Legend anchored beneath the axes so the upper-right scatter region
    # (where the no-defense star and the worst-leaking D1 / Riemann
    # points cluster) stays uncluttered.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              fontsize=8.0, ncol=6, frameon=False,
              columnspacing=1.4, handletextpad=0.6)
    _maybe_grid(ax, "both")
    fig.savefig(FIGURES_DIR / "pareto_privacy_utility.pdf")
    plt.close(fig)
    print("regenerated figures/pareto_privacy_utility.pdf")


# ---------------------------------------------------------------------------
# Cross-dataset, multi-seed, adaptive, MI on classical victims, EEGNet fairness
# ---------------------------------------------------------------------------
def render_a4_cross_dataset() -> None:
    p = RESULTS_DIR / "13_a4_cross_dataset.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    plt.rcParams.update(journal_style())
    fig, ax = plt.subplots(figsize=(FIG_DOUBLE[0], 3.0))
    err_lo = d["auc"] - d["auc_ci_low"]
    err_hi = d["auc_ci_high"] - d["auc"]
    # Horizontal forest-plot single row, matching the new
    # verification_summary_card layout so the cross-dataset card
    # composes alongside 06 / 24 with one consistent visual grammar.
    ax.errorbar([d["auc"]], [0],
                xerr=[[err_lo], [err_hi]],
                fmt="o", color=PALETTE["warn"],
                ecolor=PALETTE["ink"], elinewidth=1.0,
                capsize=4.5, capthick=1.0,
                markersize=9.0, markerfacecolor=PALETTE["warn"],
                markeredgecolor=PALETTE["ink"], markeredgewidth=0.9,
                zorder=4)
    label = (f"  {d['auc']:.3f}  "
             f"[{d['auc_ci_low']:.3f}, {d['auc_ci_high']:.3f}]")
    ax.text(d["auc"] + err_hi + 0.005, 0, label,
            va="center", ha="left", fontsize=8.5,
            fontweight="bold", color=PALETTE["ink"])
    ax.axvline(0.5, color=PALETTE["neutral"], lw=0.9, ls=(0, (4, 3)),
               label="chance (AUC = 0.5)")
    ax.set_yticks([0])
    ax.set_yticklabels(["PhysioNet → IV-2a"])
    ax.set_ylim(-0.55, 0.55)
    ax.set_xlim(0.45, 1.005)
    ax.set_xlabel("A4 AUC (open-set verification)")
    ax.set_title(
        "A4 cross-dataset verification (PhysioNet → IV-2a)\n"
        f"train n=80 PhysioNet · unseen n=9 IV-2a · EER = {d['eer']:.3f}",
        fontsize=10.0,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20),
              fontsize=7.5, frameon=False)
    ax.grid(axis="x", which="major", linestyle=":", linewidth=0.4,
            alpha=0.35, color=PALETTE["muted"])
    ax.set_axisbelow(True)
    fig.savefig(FIGURES_DIR / "13_a4_cross_dataset.pdf")
    plt.close(fig)
    print("regenerated figures/13_a4_cross_dataset.pdf")


def render_a4_multi_seed() -> None:
    p = RESULTS_DIR / "14_a4_multi_seed.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    per_seed = d.get("per_seed", [])
    agg = d.get("aggregate", {})
    if not per_seed:
        return
    plt.rcParams.update(journal_style())
    fig, ax = plt.subplots(figsize=FIG_DOUBLE)
    seeds = [r["seed"] for r in per_seed]
    aucs = np.asarray([r["auc"] for r in per_seed])
    bars = ax.bar(seeds, aucs, color=PALETTE["accent"],
                  edgecolor=PALETTE["ink"], linewidth=0.5, width=0.55)
    mean_auc = agg.get("auc_mean")
    std_auc = agg.get("auc_std")
    # Push each bar label above its own bar AND above the mean dashed line
    # so the two never visually intersect.
    mean_for_label = mean_auc if mean_auc is not None else aucs.max()
    for b, a in zip(bars, aucs):
        y = max(a, mean_for_label) + 0.018
        ax.text(b.get_x() + b.get_width() / 2, y, f"{a:.3f}",
                ha="center", va="bottom", fontsize=7.5,
                color=PALETTE["ink"])
    if mean_auc is not None:
        ax.axhline(mean_auc, color=PALETTE["contrast"], lw=1.0,
                   ls=(0, (4, 3)),
                   label=f"mean $=$ {mean_auc:.3f} ± {std_auc:.3f}")
    ax.axhline(0.5, color=PALETTE["neutral"], lw=0.7, ls=":",
               label="chance (AUC = 0.5)")
    ax.set_xticks(seeds)
    ax.set_xlabel("Random seed")
    ax.set_ylabel("A4 AUC")
    ax.set_ylim(0.45, 1.02)
    ax.set_title("A4 PhysioNet multi-seed replication "
                 "(5 random 80 / 24 splits, 24 unseen subjects each)")
    ax.legend(loc="lower right", fontsize=7.5)
    _maybe_grid(ax, "y")
    fig.savefig(FIGURES_DIR / "14_a4_multi_seed.pdf")
    plt.close(fig)
    print("regenerated figures/14_a4_multi_seed.pdf")


_ATTACK_LABEL = {
    "logreg_probe":      "logreg probe\n(generic)",
    "deep_mlp_probe":    "deep MLP probe\n(generic)",
    "encoder_finetune":  "encoder fine-tune\n(adaptive)",
}


def _adaptive_attacker_bar(*, json_path: Path, out_path: Path, title: str,
                            baseline: float = 0.411,
                            baseline_label: str = "no-defense baseline") -> None:
    if not json_path.exists():
        return
    d = json.loads(json_path.read_text())
    attacks = d["attacks"]
    plt.rcParams.update(journal_style())
    fig, ax = plt.subplots(figsize=FIG_DOUBLE_TALL)
    labels = [_ATTACK_LABEL.get(a["attack"], a["attack"].replace("_", " "))
              for a in attacks]
    top1 = np.array([a["top1"] for a in attacks])
    lo = top1 - np.array([a["top1_ci_low"] for a in attacks])
    hi = np.array([a["top1_ci_high"] for a in attacks]) - top1
    colors = [PALETTE["accent"], PALETTE["skyblue"], PALETTE["contrast"]]
    bars = ax.bar(labels, top1, yerr=[lo, hi],
                  color=[colors[i % len(colors)] for i in range(len(attacks))],
                  edgecolor=PALETTE["ink"], linewidth=0.5, width=0.55,
                  error_kw=dict(ecolor=PALETTE["ink"], elinewidth=0.9,
                                capsize=3, capthick=0.9))
    # Position bar value labels so they never collide with the dashed
    # `baseline` reference line. If a bar's natural label position lands
    # in the baseline's neighbourhood we lift it above the baseline.
    for b, v, h in zip(bars, top1, hi):
        nat_y = v + h + 0.022
        if abs(nat_y - baseline) < 0.04:
            nat_y = baseline + 0.045
        ax.text(b.get_x() + b.get_width() / 2, nat_y, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9.0,
                fontweight="bold", color=PALETTE["ink"])
    ax.axhline(baseline, color=PALETTE["neutral"], lw=1.0,
               ls=(0, (4, 3)), label=f"{baseline_label} ({baseline:.3f})")
    ax.axhline(attacks[0]["chance_top1"], color=PALETTE["fail"],
               lw=0.7, ls=":", label=f"chance ({attacks[0]['chance_top1']:.3f})")
    ax.set_ylabel("Re-ID top-1")
    ax.set_ylim(0, max(top1.max() + max(hi) + 0.12, baseline + 0.10))
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20),
              fontsize=7.5, ncol=2, frameon=False,
              columnspacing=1.6, handletextpad=0.6)
    _maybe_grid(ax, "y")
    fig.savefig(out_path)
    plt.close(fig)


def render_adaptive_attacker() -> None:
    _adaptive_attacker_bar(
        json_path=RESULTS_DIR / "15_d2_adaptive_attacker.json",
        out_path=FIGURES_DIR / "15_d2_adaptive_attacker.pdf",
        title="D2 DANN λ=0.2 under three adaptive attackers (PhysioNet, n=104)",
    )
    print("regenerated figures/15_d2_adaptive_attacker.pdf")


def render_d3_adaptive_attacker() -> None:
    _adaptive_attacker_bar(
        json_path=RESULTS_DIR / "18_d3_adaptive_attacker.json",
        out_path=FIGURES_DIR / "18_d3_adaptive_attacker.pdf",
        title="D3 DP-SGD ε=3 under three adaptive attackers (PhysioNet, n=104)",
    )
    print("regenerated figures/18_d3_adaptive_attacker.pdf")


def render_d1_adaptive_attacker() -> None:
    p = RESULTS_DIR / "23_d1_adaptive_attacker.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    # The JSON is a flat list keyed by (defense, attack).
    pivoted: dict[str, dict[str, dict]] = {}
    for row in d.get("defenses", []):
        pivoted.setdefault(row["defense"], {})[row["attack"]] = row
    defenses = list(pivoted.keys())
    if not defenses:
        return
    pretty_defense = {
        "pca_k8":            "D1 PCA  (k=8)",
        "noise_sigma1.0":    "D1 noise  (σ=1.0)",
        "channel_drop_k8":   "D1 channel-drop  (k=8)",
    }
    plt.rcParams.update(journal_style())
    fig, ax = plt.subplots(figsize=FIG_DOUBLE_TALL)
    width = 0.36
    x = np.arange(len(defenses))
    generic = np.array([pivoted[k].get("logreg_probe", {}).get("top1", np.nan)
                        for k in defenses])
    adaptive = np.array([pivoted[k].get("encoder_finetune", {}).get("top1", np.nan)
                         for k in defenses])
    g_lo = generic - np.array([pivoted[k].get("logreg_probe", {})
                                .get("top1_ci_low", np.nan) for k in defenses])
    g_hi = np.array([pivoted[k].get("logreg_probe", {})
                      .get("top1_ci_high", np.nan) for k in defenses]) - generic
    a_lo = adaptive - np.array([pivoted[k].get("encoder_finetune", {})
                                 .get("top1_ci_low", np.nan) for k in defenses])
    a_hi = np.array([pivoted[k].get("encoder_finetune", {})
                      .get("top1_ci_high", np.nan) for k in defenses]) - adaptive
    bars_g = ax.bar(x - width / 2, generic, width,
                    yerr=[g_lo, g_hi], color=PALETTE["accent"],
                    edgecolor=PALETTE["ink"], linewidth=0.5,
                    label="logreg probe (generic)",
                    error_kw=dict(ecolor=PALETTE["ink"], elinewidth=0.8,
                                  capsize=2.5, capthick=0.7))
    bars_a = ax.bar(x + width / 2, adaptive, width,
                    yerr=[a_lo, a_hi], color=PALETTE["contrast"],
                    edgecolor=PALETTE["ink"], linewidth=0.5,
                    label="encoder fine-tune (adaptive)",
                    error_kw=dict(ecolor=PALETTE["ink"], elinewidth=0.8,
                                  capsize=2.5, capthick=0.7))
    baseline = 0.411  # from A1 EEGNet logreg
    def _safe_label(b, v, h):
        nat_y = v + h + 0.012
        if abs(nat_y - baseline) < 0.04:
            nat_y = baseline + 0.045
        ax.text(b.get_x() + b.get_width() / 2, nat_y, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7.0,
                color=PALETTE["ink"])
    for b, v, h in zip(bars_g, generic, g_hi):
        _safe_label(b, v, h)
    for b, v, h in zip(bars_a, adaptive, a_hi):
        _safe_label(b, v, h)
    ax.axhline(baseline, color=PALETTE["neutral"], lw=1.0, ls=(0, (4, 3)),
               label=f"no-defense baseline ({baseline:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_defense.get(d, d) for d in defenses])
    ax.set_ylabel("Re-ID top-1")
    ax.set_ylim(0, 1.0)
    ax.set_title("D1 ad-hoc defenses under encoder fine-tune (PhysioNet, n=104)")
    ax.legend(loc="upper left", fontsize=7.5)
    _maybe_grid(ax, "y")
    fig.savefig(FIGURES_DIR / "23_d1_adaptive_attacker.pdf")
    plt.close(fig)
    print("regenerated figures/23_d1_adaptive_attacker.pdf")


def render_dp_sgd_arch_ablation() -> None:
    """D3 architecture-vs-noise decomposition. Combines three sources:

      1. AdamW + BatchNorm baseline = the A1 EEGNet logreg result
         (results/02_closed_set_reid.json, hard-coded as 0.411 here
         to match the canonical number).
      2. SGD + GroupNorm, no DP -- experiment 19, which retrains the
         architecture-equivalent Opacus stack with the PrivacyEngine
         bypassed (target_epsilon = None).
      3. SGD + GroupNorm + DP-SGD ε = 3 -- experiment 10's ε=3 logreg
         row (results/10_d3_dp_sgd.json).

    Plotting all three side by side surfaces the result that the
    architectural change (BN→GN, AdamW→SGD) accounts for ~89% of the
    empirical privacy and the formal DP mechanism adds only ~5% on top.
    """
    p19 = RESULTS_DIR / "19_dp_sgd_arch_ablation.json"
    p10 = RESULTS_DIR / "10_d3_dp_sgd.json"
    rows19 = json.loads(p19.read_text()) if p19.exists() else []
    rows10 = json.loads(p10.read_text()) if p10.exists() else []

    def _logreg_eps3():
        for r in rows10:
            if (r.get("probe") == "logreg"
                    and r.get("target_epsilon") == 3.0):
                return r
        return None

    no_dp = next((r for r in rows19
                  if r.get("probe") == "logreg"
                  and r.get("configuration") == "groupnorm_sgd_no_dp"),
                 None)
    dp_eps3 = _logreg_eps3()

    rows = [
        ("AdamW + BatchNorm\n(vanilla A1)", 0.411, 0.0, 0.0, PALETTE["fail"]),
    ]
    if no_dp is not None:
        rows.append((
            "SGD + GroupNorm\n(no DP)",
            no_dp["top1"],
            no_dp["top1"] - no_dp["top1_ci_low"],
            no_dp["top1_ci_high"] - no_dp["top1"],
            PALETTE["warn"],
        ))
    if dp_eps3 is not None:
        rows.append((
            "SGD + GroupNorm +\nDP-SGD ε = 3",
            dp_eps3["top1"],
            dp_eps3["top1"] - dp_eps3["top1_ci_low"],
            dp_eps3["top1_ci_high"] - dp_eps3["top1"],
            PALETTE["ok"],
        ))
    if len(rows) < 2:
        return

    labels = [r[0] for r in rows]
    values = np.array([r[1] for r in rows])
    lo = np.array([r[2] for r in rows])
    hi = np.array([r[3] for r in rows])
    colors = [r[4] for r in rows]

    plt.rcParams.update(journal_style())
    fig, ax = plt.subplots(figsize=FIG_DOUBLE)
    bars = ax.bar(labels, values, yerr=[lo, hi], color=colors,
                  edgecolor=PALETTE["ink"], linewidth=0.5, width=0.55,
                  error_kw=dict(ecolor=PALETTE["ink"], elinewidth=0.9,
                                capsize=3, capthick=0.9))
    for b, v, h in zip(bars, values, hi):
        ax.text(b.get_x() + b.get_width() / 2, v + h + 0.016, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9.0,
                fontweight="bold", color=PALETTE["ink"])

    # Δ annotations between consecutive bars, drawn high enough to clear
    # the bar value labels.
    for i in range(1, len(values)):
        delta = values[i] - values[i - 1]
        sign = "+" if delta >= 0 else ""
        midpoint = (bars[i - 1].get_x() + bars[i - 1].get_width() / 2
                    + bars[i].get_x() + bars[i].get_width() / 2) / 2
        y_anchor = max(values[i - 1] + hi[i - 1], values[i] + hi[i]) + 0.075
        ax.text(midpoint, y_anchor,
                f"Δ = {sign}{delta:.3f}", ha="center", va="bottom",
                fontsize=8.0, color=PALETTE["neutral"], style="italic")
    ax.axhline(0.411, color=PALETTE["neutral"], lw=0.7, ls=":",
               label="AdamW + BN A1 baseline")
    ax.set_ylabel("A1 re-ID top-1 (logreg)")
    ax.set_ylim(0, max(0.55, (values + hi).max() * 1.40))
    ax.set_title(
        "D3 architecture vs noise decomposition "
        "(EEGNet, PhysioNet n=104)"
    )
    ax.legend(loc="upper right", fontsize=7.5)
    _maybe_grid(ax, "y")
    fig.savefig(FIGURES_DIR / "19_dp_sgd_arch_ablation.pdf")
    plt.close(fig)
    print("regenerated figures/19_dp_sgd_arch_ablation.pdf")


def render_subgroup_fairness() -> None:
    p = RESULTS_DIR / "12_subgroup_fairness.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    summary = d.get("summary", {})
    victims = [v for v in ("fbcsp", "riemann") if v in summary]
    if not victims:
        return
    pretty = {"fbcsp": "FBCSP + LDA", "riemann": "Riemann tang.-space"}

    plt.rcParams.update(journal_style())
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6))
    width = 0.36
    x = np.arange(len(victims))

    def _ci_pair(v: str, key_ci: str):
        ci = summary[v][key_ci]
        return ci[0], ci[1]

    # ---- Sex panel ----
    male = np.array([summary[v]["sex_M_mean"] for v in victims])
    female = np.array([summary[v]["sex_F_mean"] for v in victims])
    pvals_sex = [summary[v]["sex_diff_p"] for v in victims]
    axes[0].bar(x - width / 2, male, width, color=PALETTE["accent"],
                edgecolor=PALETTE["ink"], linewidth=0.5, label="male")
    axes[0].bar(x + width / 2, female, width, color=PALETTE["contrast"],
                edgecolor=PALETTE["ink"], linewidth=0.5, label="female")
    for xi, m, f in zip(x, male, female):
        axes[0].text(xi - width / 2, m + 0.012, f"{m:.2f}",
                     ha="center", va="bottom", fontsize=7.0,
                     color=PALETTE["ink"])
        axes[0].text(xi + width / 2, f + 0.012, f"{f:.2f}",
                     ha="center", va="bottom", fontsize=7.0,
                     color=PALETTE["ink"])
    for xi, p_ in zip(x, pvals_sex):
        axes[0].text(xi, 1.10, f"p = {p_:.2f}",
                     ha="center", va="bottom", fontsize=7.5,
                     color=PALETTE["neutral"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([pretty[v] for v in victims], fontsize=8.0)
    axes[0].set_ylabel("Per-subject A1 attack accuracy")
    axes[0].set_ylim(0, 1.22)
    axes[0].set_title("Sex stratification")
    axes[0].legend(loc="lower left", fontsize=7.5)
    _maybe_grid(axes[0], "y")

    # ---- Age tertile panel ----
    low = np.array([summary[v]["age_low_mean"] for v in victims])
    high = np.array([summary[v]["age_high_mean"] for v in victims])
    pvals_age = [summary[v]["age_diff_p"] for v in victims]
    axes[1].bar(x - width / 2, low, width, color=PALETTE["accent"],
                edgecolor=PALETTE["ink"], linewidth=0.5, label="low tertile")
    axes[1].bar(x + width / 2, high, width, color=PALETTE["contrast"],
                edgecolor=PALETTE["ink"], linewidth=0.5, label="high tertile")
    for xi, lo_v, hi_v in zip(x, low, high):
        axes[1].text(xi - width / 2, lo_v + 0.012, f"{lo_v:.2f}",
                     ha="center", va="bottom", fontsize=7.0,
                     color=PALETTE["ink"])
        axes[1].text(xi + width / 2, hi_v + 0.012, f"{hi_v:.2f}",
                     ha="center", va="bottom", fontsize=7.0,
                     color=PALETTE["ink"])
    for xi, p_ in zip(x, pvals_age):
        axes[1].text(xi, 1.10, f"p = {p_:.2f}",
                     ha="center", va="bottom", fontsize=7.5,
                     color=PALETTE["neutral"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([pretty[v] for v in victims], fontsize=8.0)
    axes[1].set_ylim(0, 1.22)
    axes[1].set_title("Age tertile stratification")
    axes[1].legend(loc="lower left", fontsize=7.5)
    _maybe_grid(axes[1], "y")

    fig.suptitle("Subgroup fairness (PhysioNet, n=97 sex-known / 91 age-known)",
                 fontsize=10.5)
    fig.savefig(FIGURES_DIR / "12_subgroup_fairness.pdf")
    plt.close(fig)
    print("regenerated figures/12_subgroup_fairness.pdf")


def render_subgroup_fairness_eegnet() -> None:
    p = RESULTS_DIR / "17_subgroup_fairness_eegnet.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    per_subject = d.get("per_subject", [])
    if not per_subject:
        return
    values = np.array([r["attack_acc"] for r in per_subject])
    plt.rcParams.update(journal_style())
    fig, ax = plt.subplots(figsize=FIG_DOUBLE)
    ax.hist(values, bins=18, color=PALETTE["accent"],
            edgecolor=PALETTE["ink"], linewidth=0.5, alpha=0.85)
    ax.axvline(float(values.mean()), color=PALETTE["contrast"],
               lw=1.2, ls=(0, (4, 3)),
               label=f"mean ({values.mean():.3f})")
    sex_info = d.get("sex", {})
    age_info = d.get("age", {})
    annotation = []
    diff_sex = sex_info.get("diff_M_minus_F", {})
    diff_age = age_info.get("diff_low_minus_high", {})
    if diff_sex:
        annotation.append(
            f"Δ M−F = {diff_sex['point']:+.3f}, "
            f"p = {diff_sex.get('mannwhitneyu_p', float('nan')):.3f}"
        )
    if diff_age:
        annotation.append(
            f"Δ age low−high = {diff_age['point']:+.3f}, "
            f"p = {diff_age.get('mannwhitneyu_p', float('nan')):.3f}"
        )
    if annotation:
        ax.text(0.98, 0.95, "\n".join(annotation),
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, color=PALETTE["neutral"], style="italic")
    ax.set_xlabel("Per-subject A1 attack accuracy (EEGNet)")
    ax.set_ylabel("Subject count")
    ax.set_title(
        f"EEGNet within-cohort heterogeneity (PhysioNet, n={len(values)})"
    )
    ax.legend(loc="upper left", fontsize=7.5)
    _maybe_grid(ax, "y")
    fig.savefig(FIGURES_DIR / "17_subgroup_fairness_eegnet.pdf")
    plt.close(fig)
    print("regenerated figures/17_subgroup_fairness_eegnet.pdf")


def render_eegnet_age_seeds() -> None:
    p = RESULTS_DIR / "22_eegnet_age_seeds.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    per_seed = d.get("per_seed") or []
    if not per_seed:
        return
    plt.rcParams.update(journal_style())
    fig, ax = plt.subplots(figsize=FIG_DOUBLE)
    seeds = [r["seed"] for r in per_seed]
    deltas = [r.get("age_diff_low_minus_high", np.nan) for r in per_seed]
    pvals = [r.get("age_p", np.nan) for r in per_seed]
    colors = [PALETTE["contrast"] if (not np.isnan(p_) and p_ < 0.05)
              else PALETTE["accent"] for p_ in pvals]
    bars = ax.bar(seeds, deltas, color=colors,
                  edgecolor=PALETTE["ink"], linewidth=0.5, width=0.55)
    for b, p_, dlt in zip(bars, pvals, deltas):
        if p_ is not None and not np.isnan(p_):
            ax.text(b.get_x() + b.get_width() / 2,
                    max(dlt, 0) + 0.006, f"p = {p_:.3f}",
                    ha="center", va="bottom", fontsize=7.0,
                    color=PALETTE["neutral"])
    agg = d.get("aggregate", {})
    fisher_p = agg.get("fisher_age_p")
    age_diff_mean = agg.get("age_diff_mean")
    age_diff_std = agg.get("age_diff_std")
    if age_diff_mean is not None:
        ax.axhline(age_diff_mean, color=PALETTE["ink"], lw=1.0,
                   ls=(0, (4, 3)),
                   label=(f"5-seed mean Δ = "
                          f"{age_diff_mean:+.3f} ± {age_diff_std:.3f}"))
    title = "EEGNet age effect across 5 seeds (PhysioNet, n=91 age-known)"
    if fisher_p is not None:
        title += f" · Fisher combined p = {fisher_p:.4f}"
    ax.set_xticks(seeds)
    ax.set_xlabel("Random seed")
    ax.set_ylabel(r"$\Delta$ attack acc.  (low − high)")
    valid = [v for v in deltas if not np.isnan(v)]
    if valid:
        ax.set_ylim(min(0, min(valid) - 0.02), max(valid) + 0.05)
    ax.axhline(0, color=PALETTE["neutral"], lw=0.7)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7.5)
    _maybe_grid(ax, "y")
    fig.savefig(FIGURES_DIR / "22_eegnet_age_seeds.pdf")
    plt.close(fig)
    print("regenerated figures/22_eegnet_age_seeds.pdf")


# ---------------------------------------------------------------------------
# Second-corpus and extension renderers
# ---------------------------------------------------------------------------
def render_xds_symmetric() -> None:
    directions = [
        ("iv2a_to_physionet",      "IV-2a → PhysioNet",        9, 104),
        ("physionet_to_lee2019",   "PhysioNet → Lee 2019",     80, 54),
        ("lee2019_to_physionet",   "Lee 2019 → PhysioNet",     40, 104),
        ("iv2a_to_lee2019",        "IV-2a → Lee 2019",         9, 54),
    ]
    rows = []
    for tag, label, n_tr, n_te in directions:
        p = RESULTS_DIR / f"26_a4_xds_{tag}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        rows.append((label, d["auc"], d["auc_ci_low"], d["auc_ci_high"],
                     n_tr, n_te))
    p13 = RESULTS_DIR / "13_a4_cross_dataset.json"
    if p13.exists():
        d = json.loads(p13.read_text())
        rows.append(("PhysioNet → IV-2a",
                     d["auc"], d["auc_ci_low"], d["auc_ci_high"], 80, 9))
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r[1])

    plt.rcParams.update(journal_style())
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    y = np.arange(len(rows))
    values = np.array([r[1] for r in rows])
    err_lo = values - np.array([r[2] for r in rows])
    err_hi = np.array([r[3] for r in rows]) - values
    colors = [PALETTE["fail"] if v <= 0.55
              else (PALETTE["warn"] if v < 0.75 else PALETTE["ok"])
              for v in values]
    bars = ax.barh(y, values, color=colors,
                   xerr=[err_lo, err_hi],
                   edgecolor=PALETTE["ink"], linewidth=0.5, height=0.62,
                   error_kw=dict(ecolor=PALETTE["ink"], elinewidth=0.9,
                                 capsize=3, capthick=0.9))
    ax.axvline(0.5, color=PALETTE["neutral"], lw=0.8, ls=(0, (4, 3)),
               label="chance (AUC = 0.5)")
    # Value labels: AUC immediately after the CI right edge, cohort
    # annotation as a secondary line below the y-tick label (a separate
    # text added to the y-tick string, not over the bar).
    yticklabels = []
    for yi, r, v in zip(y, rows, values):
        yticklabels.append(f"{r[0]}\n(train n={r[4]} / unseen n={r[5]})")
        right_edge = v + (r[3] - r[1]) + 0.012
        ax.text(right_edge, yi, f"{v:.3f}",
                ha="left", va="center", fontsize=8.0,
                color=PALETTE["ink"], fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(yticklabels)
    ax.tick_params(axis="y", which="major", labelsize=8.5)
    ax.set_xlabel("A4 AUC (open-set verification on unseen subjects)")
    ax.set_xlim(0.43, 1.0)
    ax.set_title("A4 cross-dataset transfer in five directions")
    ax.legend(loc="lower right", fontsize=7.5)
    _maybe_grid(ax, "x")
    fig.savefig(FIGURES_DIR / "26_a4_xds_symmetric.pdf")
    plt.close(fig)
    print("regenerated figures/26_a4_xds_symmetric.pdf")


def render_dp_aware_mia() -> None:
    """Consolidate the eps=3 / eps=1 / eps=0.5 DP-aware MIA runs into a
    single dot-plot vs the Yeom bound. Replaces three individual cards."""
    candidates = [
        ("", 3.0),
        ("_eps1.0", 1.0),
        ("_eps0.5", 0.5),
    ]
    rows = []
    for tag, eps_target in candidates:
        p = RESULTS_DIR / f"27_d3_membership_aware_attacker{tag}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        rows.append({
            "eps_final": d.get("target_final_epsilon") or eps_target,
            "eps_target": eps_target,
            "auc": d["auc"], "lo": d["auc_ci_low"], "hi": d["auc_ci_high"],
            "delta": d["target_delta"],
        })
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r["eps_target"])

    plt.rcParams.update(journal_style())
    fig, ax = plt.subplots(figsize=FIG_DOUBLE_TALL)
    x = np.arange(len(rows))
    aucs = np.array([r["auc"] for r in rows])
    lo = aucs - np.array([r["lo"] for r in rows])
    hi = np.array([r["hi"] for r in rows]) - aucs
    yeom = np.array([1.0 - np.exp(-r["eps_final"]) - r["delta"] for r in rows])
    undefended = 0.878

    ax.bar(x, aucs, yerr=[lo, hi], color=PALETTE["accent"],
           edgecolor=PALETTE["ink"], linewidth=0.5, width=0.45,
           error_kw=dict(ecolor=PALETTE["ink"], elinewidth=0.9,
                         capsize=3, capthick=0.9),
           label="empirical DP-aware MIA AUC")
    ax.plot(x, yeom, color=PALETTE["contrast"], marker="D",
            lw=1.3, markersize=5, markerfacecolor="white",
            markeredgewidth=1.2,
            label=r"Yeom (2018) bound  $1 - e^{-\varepsilon} - \delta$")
    ax.axhline(undefended, color=PALETTE["neutral"], lw=0.8,
               ls=(0, (4, 3)),
               label=f"undefended baseline ({undefended:.3f})")
    ax.axhline(0.5, color=PALETTE["fail"], lw=0.7, ls=":",
               label="chance (AUC = 0.5)")
    # Bar value labels: always above whichever is higher — the upper CI
    # cap or the Yeom point at that ε — so the Yeom line never visually
    # cuts through the bold AUC number.
    hi_arr = np.array([r["hi"] for r in rows])
    for xi, v, h_v, y_v in zip(x, aucs, hi_arr, yeom):
        y = max(h_v, y_v) + 0.035
        ax.text(xi, y, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8.0,
                color=PALETTE["ink"], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"ε = {r['eps_target']:g}" for r in rows])
    ax.set_xlabel("DP-SGD target ε  (shadows and target both DP-trained)")
    ax.set_ylabel("MI AUC")
    ax.set_ylim(0, 1.18)
    ax.set_title(
        "DP-aware membership inference vs Yeom (2018) bound across ε "
        "(PhysioNet, EEGNet)"
    )
    ax.legend(loc="upper left", fontsize=7.5)
    _maybe_grid(ax, "y")
    fig.savefig(FIGURES_DIR / "27_d3_membership_aware_attacker.pdf")
    plt.close(fig)
    print("regenerated figures/27_d3_membership_aware_attacker.pdf")


def render_model_inversion() -> None:
    p = RESULTS_DIR / "28_d3_model_inversion.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    arms = list(d["results"].keys())

    plt.rcParams.update(journal_style())
    fig, ax = plt.subplots(figsize=FIG_DOUBLE_TALL)
    x = np.arange(len(arms))
    width = 0.36
    rank1 = [d["results"][a]["rank1_acc"] for a in arms]
    rank5 = [d["results"][a]["rank5_acc"] for a in arms]
    chance1 = 1.0 / d["results"][arms[0]]["n_subjects"]
    chance5 = 5.0 / d["results"][arms[0]]["n_subjects"]

    # Draw rank-1 bars; when the value is exactly zero a normal bar is
    # invisible, so leave a thin hatched stub above the axis so a reader
    # can tell the experiment ran and produced a null. The stub never
    # affects the y-scale because it sits inside the existing axis pad.
    rank1_visible = [max(v, 0.0) for v in rank1]
    ax.bar(x - width / 2, rank1_visible, width, color=PALETTE["accent"],
           edgecolor=PALETTE["ink"], linewidth=0.5, label="rank-1 recovery")
    ax.bar(x + width / 2, rank5, width, color=PALETTE["contrast"],
           edgecolor=PALETTE["ink"], linewidth=0.5, label="rank-5 recovery")
    for xi, v in zip(x - width / 2, rank1):
        if v == 0:
            ax.bar(xi, 0.004, width, bottom=0.0,
                   color=PALETTE["accent"], edgecolor=PALETTE["ink"],
                   linewidth=0.5, hatch="///", alpha=0.55)
    ax.axhline(chance1, color=PALETTE["neutral"], lw=0.7, ls=":",
               label=f"rank-1 chance ({chance1:.3f})")
    ax.axhline(chance5, color=PALETTE["neutral"], lw=0.7, ls=(0, (4, 3)),
               label=f"rank-5 chance ({chance5:.3f})")
    for xi, v in zip(x - width / 2, rank1):
        _annotate_bar(ax, xi, max(v, 0.004), fontsize=7.0,
                      text=f"{v:.3f}")
    for xi, v in zip(x + width / 2, rank5):
        _annotate_bar(ax, xi, v, fontsize=7.0)

    pretty = {"no_defense": "no defense",
              "dp_eps=3.0": "DP-SGD ε = 3"}
    ax.set_xticks(x)
    ax.set_xticklabels([pretty.get(a, a) for a in arms])
    ax.set_ylim(0, max(0.30, max(rank5) + 0.10))
    ax.set_ylabel("Reconstruction recovery rate")
    ax.set_title(
        f"Fredrikson model inversion null result "
        f"(n={d['n_targets']} target subjects, "
        f"{d['n_inversion_steps']} optimisation steps)"
    )
    ax.legend(loc="upper right", fontsize=7.5)
    _maybe_grid(ax, "y")
    fig.savefig(FIGURES_DIR / "28_d3_model_inversion.pdf")
    plt.close(fig)
    print("regenerated figures/28_d3_model_inversion.pdf")


def render_eps_sweep() -> None:
    p = RESULTS_DIR / "29_d3_eps_sweep.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    pareto = sorted(d["pareto"],
                    key=lambda r: (r["target_epsilon"] is None,
                                   r["target_epsilon"] or 0))
    labels = ["no DP" if r["target_epsilon"] is None
              else f"ε={r['target_epsilon']:g}" for r in pareto]
    task = [r["task_acc"] for r in pareto]
    lr_top1 = [r["attack_logreg"]["top1"] for r in pareto]
    ft_top1 = [r["attack_finetune"]["top1"] for r in pareto]
    chance = pareto[0]["attack_logreg"]["chance_top1"]
    no_def_baseline = next((r["attack_finetune"]["top1"]
                            for r in pareto if r["target_epsilon"] is None),
                           None)

    plt.rcParams.update(journal_style())
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=FIG_DOUBLE_TALL)
    x = np.arange(len(labels))

    ax_l.plot(x, lr_top1, color=PALETTE["accent"], marker="o", lw=1.4,
              markersize=5, markerfacecolor="white", markeredgewidth=1.2,
              label="generic logreg probe")
    ax_l.plot(x, ft_top1, color=PALETTE["contrast"], marker="s", lw=1.4,
              markersize=5, markerfacecolor="white", markeredgewidth=1.2,
              label="encoder fine-tune (adaptive)")
    ax_l.axhline(chance, color=PALETTE["neutral"], lw=0.7, ls=":",
                 label=f"chance ({chance:.4f})")
    if no_def_baseline is not None:
        ax_l.axhline(no_def_baseline, color=PALETTE["fail"], lw=0.7,
                     ls=(0, (4, 3)),
                     label=f"SGD+GN fine-tune ({no_def_baseline:.3f})")
    ax_l.set_xticks(x); ax_l.set_xticklabels(labels)
    ax_l.set_xlabel("DP-SGD target ε")
    ax_l.set_ylabel("Re-ID top-1 accuracy")
    ax_l.set_ylim(0, max(max(ft_top1), max(lr_top1)) * 1.25)
    ax_l.legend(loc="upper left", fontsize=7.5)
    _maybe_grid(ax_l, "y")

    ax_r.plot(x, task, color=PALETTE["ok"], marker="^", lw=1.4,
              markersize=5, markerfacecolor="white", markeredgewidth=1.2,
              label="motor-imagery task accuracy")
    ax_r.axhline(0.25, color=PALETTE["neutral"], lw=0.7, ls=(0, (4, 3)),
                 label="4-class chance (0.250)")
    ax_r.set_xticks(x); ax_r.set_xticklabels(labels)
    ax_r.set_xlabel("DP-SGD target ε")
    ax_r.set_ylabel("Motor-imagery task accuracy")
    ax_r.set_ylim(0.22, max(task) + 0.04)
    ax_r.legend(loc="lower right", fontsize=7.5)
    _maybe_grid(ax_r, "y")

    fig.suptitle(
        "D3 DP-SGD ε sweep on PhysioNet (EEGNet victim, n=104)\n"
        "generic and adaptive attackers; task accuracy",
        fontsize=10.5,
    )
    fig.savefig(FIGURES_DIR / "29_d3_eps_sweep.pdf")
    plt.close(fig)
    print("regenerated figures/29_d3_eps_sweep.pdf")


def render_theory_scaling() -> None:
    p = RESULTS_DIR / "30_theory_scaling.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    plt.rcParams.update(journal_style())
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(8.2, 3.6))

    # Left: cohort-size scaling, log-log
    pretty_victim = {"eegnet": "EEGNet", "riemann": "Riemann tangent-space"}
    riemann_at_ceiling = False
    for victim, color, marker in (("eegnet", PALETTE["accent"], "o"),
                                   ("riemann", PALETTE["contrast"], "s")):
        rows = d.get("scaling", {}).get(victim, [])
        if not rows:
            continue
        ns = np.array([r["n"] for r in rows])
        t1 = np.array([r["top1"] for r in rows])
        # If Riemann is essentially at ceiling for every tested N, skip the
        # log-scale dots (they degenerate to a single near-zero point) and
        # annotate the ceiling fact in a band along the bottom instead.
        if victim == "riemann" and (1.0 - t1).max() < 1e-3:
            riemann_at_ceiling = True
            continue
        ax_l.plot(ns, 1.0 - t1, marker=marker, color=color, lw=1.3,
                  markerfacecolor="white", markeredgewidth=1.2,
                  markersize=5, label=pretty_victim[victim])
    ax_l.plot([10, 104], [1 - 1 / 10, 1 - 1 / 104],
              color=PALETTE["neutral"], lw=0.7, ls=":",
              label="(1 − chance) reference  $= 1 - 1/N$")
    if riemann_at_ceiling:
        ax_l.axhspan(1e-4, 6e-4, color=PALETTE["contrast"], alpha=0.18,
                     zorder=1)
        ax_l.text(11, 3e-4,
                  "Riemann tangent-space at ceiling (top-1 ≥ 0.9996) "
                  "across every tested N",
                  color=PALETTE["contrast"], fontsize=7.5, va="center",
                  style="italic")
    ax_l.set_xscale("log"); ax_l.set_yscale("log")
    ax_l.set_xlabel("Cohort size  N")
    ax_l.set_ylabel("1 − A1 top-1  (log)")
    gamma = d.get("scaling_fits", {}).get("eegnet", {}).get("gamma")
    if isinstance(gamma, (int, float)):
        ax_l.text(0.55, 0.92, f"EEGNet γ = {gamma:.3f}",
                  transform=ax_l.transAxes, fontsize=9.0,
                  color=PALETTE["accent"], fontweight="bold")
    ax_l.set_title("Re-ID scaling vs cohort size")
    ax_l.legend(loc="lower left", fontsize=7.5)
    _maybe_grid(ax_l, "both")

    # Right: Yeom bound vs empirical
    overlay = d.get("yeom_overlay") or []
    if overlay:
        eps_x = []; bound = []; emp_lr = []; emp_ft = []
        for r in overlay:
            eps = (r["final_epsilon"] if r["final_epsilon"] is not None
                   else 12.0)
            eps_x.append(eps); bound.append(r["yeom_bound_re_id_upper"])
            emp_lr.append(r["empirical_logreg_top1"])
            emp_ft.append(r["empirical_finetune_top1"])
        idx = np.argsort(eps_x)
        eps_x = np.asarray(eps_x)[idx]
        bound = np.asarray(bound)[idx]
        emp_lr = np.asarray(emp_lr)[idx]
        emp_ft = np.asarray(emp_ft)[idx]
        ax_r.plot(eps_x, bound, color=PALETTE["fail"], marker="D",
                  lw=1.3, markersize=5, markerfacecolor="white",
                  markeredgewidth=1.2,
                  label="Yeom (2018) bound  $1 - e^{-\\varepsilon} - \\delta$")
        ax_r.plot(eps_x, emp_ft, color=PALETTE["contrast"], marker="s",
                  lw=1.3, markersize=5, markerfacecolor="white",
                  markeredgewidth=1.2,
                  label="encoder fine-tune (empirical)")
        ax_r.plot(eps_x, emp_lr, color=PALETTE["accent"], marker="o",
                  lw=1.3, markersize=5, markerfacecolor="white",
                  markeredgewidth=1.2,
                  label="logreg probe (empirical)")
        ax_r.set_xscale("log")
        ax_r.set_xlabel("DP-SGD final ε  (no-DP plotted at 12)")
        ax_r.set_ylabel("MI advantage upper bound  /  re-ID top-1")
        ax_r.legend(loc="upper left", fontsize=7.5)
        ax_r.set_title("Empirical re-ID vs Yeom bound")
        _maybe_grid(ax_r, "both")
    else:
        ax_r.text(0.5, 0.5, "Yeom overlay not generated\n"
                            "(run experiment 29 first)",
                  ha="center", va="center", transform=ax_r.transAxes,
                  fontsize=9, color=PALETTE["neutral"])
        ax_r.axis("off")

    fig.suptitle("Theoretical scaling validation (experiment 30)",
                 fontsize=10.5)
    fig.savefig(FIGURES_DIR / "30_theory_scaling.pdf")
    plt.close(fig)
    print("regenerated figures/30_theory_scaling.pdf")


def render_federated_dp() -> None:
    p = RESULTS_DIR / "31_federated_dp.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    plt.rcParams.update(journal_style())
    fig, ax = plt.subplots(figsize=FIG_DOUBLE)
    labels = ["logreg probe\n(generic)", "encoder fine-tune\n(adaptive)"]
    values = np.array([d["attack_logreg"]["top1"],
                       d["attack_finetune"]["top1"]])
    err_lo = values - np.array([d["attack_logreg"]["top1_ci_low"],
                                d["attack_finetune"]["top1_ci_low"]])
    err_hi = np.array([d["attack_logreg"]["top1_ci_high"],
                       d["attack_finetune"]["top1_ci_high"]]) - values

    bars = ax.bar(labels, values, yerr=[err_lo, err_hi],
                  color=[PALETTE["accent"], PALETTE["contrast"]],
                  edgecolor=PALETTE["ink"], linewidth=0.5, width=0.55,
                  error_kw=dict(ecolor=PALETTE["ink"], elinewidth=0.9,
                                capsize=3, capthick=0.9))
    for b, v in zip(bars, values):
        _annotate_bar(ax, b.get_x() + b.get_width() / 2, v, fontsize=8.0)
    ax.axhline(0.411, color=PALETTE["neutral"], lw=1.0, ls=(0, (4, 3)),
               label="no-defense A1 baseline (0.411)")
    ax.axhline(d["attack_logreg"]["chance_top1"], color=PALETTE["fail"],
               lw=0.7, ls=":",
               label=f"chance ({d['attack_logreg']['chance_top1']:.4f})")
    eps = d.get("epsilon_participant_level_rdp",
                d.get("informal_epsilon_participant_level"))
    ax.set_ylabel("Re-ID top-1")
    ax.set_ylim(0, 0.50)
    ax.set_title(
        f"D4 federated DP-FedAvg (104 clients, 30 rounds, q=0.5, σ=0.4)\n"
        f"participant-level ε (RDP) = {eps:.1f}"
    )
    # Anchor the legend beneath the axes so it never lands on top of the
    # horizontal no-defense baseline dashed line, which sweeps the upper
    # half of the plot.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              fontsize=7.5, ncol=2, frameon=False,
              columnspacing=1.6, handletextpad=0.6)
    _maybe_grid(ax, "y")
    fig.savefig(FIGURES_DIR / "31_federated_dp.pdf")
    plt.close(fig)
    print("regenerated figures/31_federated_dp.pdf")


def render_lee2019_fairness() -> None:
    p = RESULTS_DIR / "32_fairness_lee2019.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    plt.rcParams.update(journal_style())
    # Each panel gets its own y-axis so a ceiling-saturated Riemann
    # (54 subjects in a single bin) is not clipped by FBCSP / EEGNet's
    # ~10-bar max.
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.6), sharey=False)
    color_for = {"fbcsp": PALETTE["accent"], "riemann": PALETTE["contrast"],
                 "eegnet": PALETTE["ok"]}
    title_for = {"fbcsp": "FBCSP + LDA",
                 "riemann": "Riemann tang.-space",
                 "eegnet": "EEGNet"}
    for ax, victim in zip(axes, ("fbcsp", "riemann", "eegnet")):
        if victim not in d["victim_results"]:
            ax.axis("off"); continue
        v = d["victim_results"][victim]
        h = v["heterogeneity"]
        per_subj = v.get("per_subject_accuracy") or {}
        accs = (np.array(list(per_subj.values())) if per_subj
                else np.array([h["min"], h["mean"], h["max"]]))

        ceiling = bool(h["decile_gap"] < 1e-3 and h["max"] >= 0.99)
        if ceiling:
            # All subjects at top-1 ≈ 1 -- a histogram degenerates into a
            # single near-invisible spike. Show the cluster explicitly:
            # a strip plot of per-subject values along a horizontal axis
            # plus a textual ceiling annotation, which mirrors what the
            # theory panel does for Riemann at ceiling.
            rng = np.random.default_rng(0)
            jit = rng.uniform(-0.18, 0.18, size=len(accs))
            ax.scatter(accs, np.zeros_like(accs) + jit,
                       color=color_for[victim], edgecolor=PALETTE["ink"],
                       linewidth=0.4, s=36, alpha=0.85, zorder=3)
            ax.axvline(h["mean"], color=PALETTE["ink"], lw=1.1,
                       label=f"mean {h['mean']:.3f}")
            ax.set_yticks([])
            ax.set_ylim(-1.0, 1.0)
            ax.text(0.5, 0.92,
                    f"all {len(accs)} subjects at top-1 ≥ {h['min']:.3f}",
                    transform=ax.transAxes,
                    ha="center", va="top", fontsize=8.0,
                    color=PALETTE["neutral"], style="italic")
        else:
            ax.hist(accs, bins=14, color=color_for[victim],
                    edgecolor=PALETTE["ink"], linewidth=0.4, alpha=0.85)
            ax.axvline(h["mean"], color=PALETTE["ink"], lw=1.1,
                       label=f"mean {h['mean']:.3f}")
            ax.set_ylabel(("Subject count  (n=54)" if victim == "fbcsp"
                           else "Subject count"))
            # leave headroom so the mean line label doesn't clash with
            # the tallest bar
            _, top = ax.get_ylim()
            ax.set_ylim(0, top * 1.12)
        ax.set_title(
            f"{title_for[victim]}\n"
            f"task = {v['task_acc']:.3f}, "
            f"decile gap = {h['decile_gap']:.3f}",
            fontsize=9.5,
        )
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Per-subject A1 top-1")
        ax.legend(loc="upper left", fontsize=7.5)
        _maybe_grid(ax, "y" if not ceiling else "x")
    fig.suptitle(
        "Lee 2019 within-cohort heterogeneity "
        "(54 subjects, within-session protocol)",
        fontsize=10.5,
    )
    fig.savefig(FIGURES_DIR / "32_fairness_lee2019.pdf")
    plt.close(fig)
    print("regenerated figures/32_fairness_lee2019.pdf")


def render_asymmetry_mechanism() -> None:
    p = RESULTS_DIR / "33_a4_asymmetry_mechanism.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    plt.rcParams.update(journal_style())
    fig, ax = plt.subplots(figsize=FIG_DOUBLE)
    labels = ["binary L/R-hand\n(baseline)", "synthetic 4-class\n(hand × half-trial)"]
    values = np.array([d["binary_baseline_auc"], d["auc"]])
    err_lo = np.array([0, d["auc"] - d["auc_ci_low"]])
    err_hi = np.array([0, d["auc_ci_high"] - d["auc"]])
    bars = ax.bar(labels, values, yerr=[err_lo, err_hi],
                  color=[PALETTE["neutral"], PALETTE["contrast"]],
                  edgecolor=PALETTE["ink"], linewidth=0.5, width=0.55,
                  error_kw=dict(ecolor=PALETTE["ink"], elinewidth=0.9,
                                capsize=3, capthick=0.9))
    # Both arms sit on or just above chance — lift each annotation above
    # the chance dashed line so the number is never bisected by it.
    for b, v, h in zip(bars, values, err_hi):
        nat_y = v + h + 0.020
        if abs(nat_y - 0.5) < 0.04:
            nat_y = 0.5 + 0.045
        ax.text(b.get_x() + b.get_width() / 2, nat_y, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8.5,
                color=PALETTE["ink"])
    ax.axhline(0.5, color=PALETTE["neutral"], lw=0.8, ls=(0, (4, 3)),
               label="chance (AUC = 0.5)")
    ax.set_ylabel("Lee 2019 → PhysioNet A4 AUC")
    ax.set_ylim(0.30, 0.80)
    lift = d.get("auc_lift_over_binary_baseline", 0.0)
    hyp = d.get("hypothesis_supported", False)
    verdict = "supported" if hyp else "falsified"
    ax.set_title(
        f"Lee 2019 → PhysioNet asymmetry-mechanism test\n"
        f"task-complexity hypothesis: {verdict} · lift = {lift:+.3f}"
    )
    ax.legend(loc="upper right", fontsize=7.5)
    _maybe_grid(ax, "y")
    fig.savefig(FIGURES_DIR / "33_a4_asymmetry_mechanism.pdf")
    plt.close(fig)
    print("regenerated figures/33_a4_asymmetry_mechanism.pdf")


def render_multi_seed() -> None:
    p = RESULTS_DIR / "34_multi_seed.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    pretty_target = {
        "a3_lee2019":              "A3 Lee 2019",
        "a4_lee2019":              "A4 Lee 2019\n(within-session)",
        "xds_iv2a_to_physionet":   "A4 IV-2a → PhysioNet",
        "xds_physionet_to_lee2019": "A4 PhysioNet → Lee 2019",
        "xds_lee2019_to_physionet": "A4 Lee 2019 → PhysioNet",
        "xds_iv2a_to_lee2019":     "A4 IV-2a → Lee 2019",
    }
    pretty_metric = {
        "eegnet_logreg_top1":   "EEGNet top-1",
        "fbcsp_logreg_top1":    "FBCSP top-1",
        "riemann_logreg_top1":  "Riemann top-1",
        "auc_within_session":   "AUC",
        "auc":                  "AUC",
    }
    rows = []
    for target, t_data in d["rows"].items():
        for metric, agg in t_data["aggregated"].items():
            if metric == "eer_within_session":
                continue
            label = (f"{pretty_target.get(target, target)}\n"
                     f"{pretty_metric.get(metric, metric)}")
            rows.append((label, agg["mean"], agg["std"], agg["n"]))
    if not rows:
        return

    plt.rcParams.update(journal_style())
    height = max(3.4, 0.30 * len(rows) + 1.4)
    fig, ax = plt.subplots(figsize=(7.0, height))
    y = np.arange(len(rows))
    means = np.array([r[1] for r in rows])
    stds = np.array([r[2] for r in rows])
    colors = [PALETTE["fail"] if "Lee 2019 → PhysioNet" in r[0]
              else PALETTE["accent"]
              for r in rows]
    ax.barh(y, means, xerr=stds, color=colors,
            edgecolor=PALETTE["ink"], linewidth=0.5, height=0.6,
            error_kw=dict(ecolor=PALETTE["ink"], elinewidth=0.9,
                          capsize=3, capthick=0.9))
    # If a text label would land within ±0.02 of the chance line (x=0.5)
    # nudge it past the line so the dashed reference never bisects the
    # "mean ± std" string.
    for i, (m, s, n) in enumerate(zip(means, stds, [r[3] for r in rows])):
        x = m + s + 0.008
        if 0.48 < x < 0.54:
            x = 0.55
        ax.text(min(1.005, x), i,
                f"{m:.3f} ± {s:.3f}  (n={n})",
                va="center", fontsize=7.5, color=PALETTE["ink"])
    ax.axvline(0.5, color=PALETTE["neutral"], lw=0.7, ls=(0, (4, 3)),
               label="chance (verification AUC = 0.5)")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("metric value  (mean ± std across 5 seeds)")
    ax.set_xlim(0, max(1.0, (means + stds).max() + 0.08))
    ax.set_title(
        "Multi-seed replication (5 seeds per row, experiment 34)"
    )
    ax.legend(loc="lower right", fontsize=7.5)
    _maybe_grid(ax, "x")
    fig.savefig(FIGURES_DIR / "34_multi_seed.pdf")
    plt.close(fig)
    print("regenerated figures/34_multi_seed.pdf")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    print("Regenerating all figures from result JSONs ...\n")
    for fn in (
        # Core experiments
        render_a1, render_within_subject_reid,
        render_a2, render_a3, render_a4, render_a5,
        render_d1_pca,
        lambda: render_d1_other("noise"),
        lambda: render_d1_other("channel_drop"),
        render_d2_dann, render_d3,
        render_subgroup_fairness,
        render_a4_cross_dataset,
        render_a4_multi_seed,
        render_adaptive_attacker,
        render_a5_classical,
        render_subgroup_fairness_eegnet,
        render_d3_adaptive_attacker,
        render_dp_sgd_arch_ablation,
        render_eegnet_age_seeds,
        render_d1_adaptive_attacker,
        render_a2_vs_rest,
        # Second-corpus and extensions
        render_lee2019_a3,
        render_lee2019_a4,
        render_lee2019_a5,
        render_xds_symmetric,
        render_dp_aware_mia,
        render_model_inversion,
        render_eps_sweep,
        render_theory_scaling,
        render_federated_dp,
        render_lee2019_fairness,
        render_asymmetry_mechanism,
        render_multi_seed,
        # Pareto last so it sees every defense JSON if newly added
        render_pareto,
    ):
        try:
            fn()
        except FileNotFoundError as e:
            print(f"  skip ({e.filename or e}): result JSON not present yet")
        except Exception as exc:
            print(f"  FAILED in {getattr(fn, '__name__', repr(fn))}: "
                  f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
