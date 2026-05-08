"""Rebuild every figure under figures/ from the canonical result JSONs.

Single-command reproducibility: a reviewer who clones the repo and runs

    python -m tools.regenerate_figures

ends up with every PDF in figures/ regenerated locally from results/*.json.
This is also the reference for which JSON drives which figure (see also
docs/figures.md).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import FIGURES_DIR, RESULTS_DIR
from eval.plots import (
    _setup_axes,
    closed_set_bar_chart,
    verification_panel,
    verification_summary_card,
)


# ---------------------------------------------------------------------------
# Generic summary-card renderer, used for A5 EEGNet (no per-pair scores)
# ---------------------------------------------------------------------------
def _summary_card(*, title, lines, big_metric_label, big_metric_value,
                  big_metric_ci, footer, out_path):
    plt.rcParams.update(_setup_axes())
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ax.axis("off")
    ax.text(0.02, 0.85, title, fontsize=11, fontweight="bold")
    for i, line in enumerate(lines):
        ax.text(0.02, 0.65 - 0.08 * i, line, fontsize=8)
    ax.text(0.02, 0.30, f"{big_metric_label} = {big_metric_value:.3f}",
            fontsize=22, fontweight="bold", color="#2c3e50")
    if big_metric_ci is not None:
        ax.text(0.32, 0.30,
                f"95% CI [{big_metric_ci[0]:.3f}, {big_metric_ci[1]:.3f}]",
                fontsize=9, color="#7f8c8d", verticalalignment="bottom")
    ax.text(0.02, 0.07, footer, fontsize=11, color="#34495e")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure renderers — one per output PDF
# ---------------------------------------------------------------------------
def render_a1():
    rows = json.loads((RESULTS_DIR / "02_closed_set_reid.json").read_text())
    n = rows[0]["n_subjects"]
    closed_set_bar_chart(
        rows, FIGURES_DIR / "02_closed_set_reid.pdf",
        title=f"A1 closed-set re-ID on PhysioNet ({n} subjects)",
    )
    print("regenerated figures/02_closed_set_reid.pdf")


def render_a2():
    rows = json.loads((RESULTS_DIR / "04_a2_cross_task.json").read_text())
    n = rows[0]["n_subjects"]
    closed_set_bar_chart(
        rows, FIGURES_DIR / "04_a2_cross_task.pdf",
        title=(f"A2 cross-task re-ID  ({n} subj)\n"
               f"probe trained on execution-runs, tested on imagery-runs"),
    )
    print("regenerated figures/04_a2_cross_task.pdf")


def render_a3():
    rows = json.loads((RESULTS_DIR / "05_a3_cross_session.json").read_text())
    n = rows[0]["n_subjects"]
    closed_set_bar_chart(
        rows, FIGURES_DIR / "05_a3_cross_session.pdf",
        title=(f"A3 cross-session re-ID  (BCI IV-2a, {n} subj)\n"
               f"probe trained on session-1, tested on session-2"),
    )
    print("regenerated figures/05_a3_cross_session.pdf")


def render_a4():
    a4 = json.loads((RESULTS_DIR / "06_a4_open_set.json").read_text())
    scores_path = RESULTS_DIR / "06_a4_open_set_scores.npz"
    if scores_path.exists():
        data = np.load(scores_path)
        verification_panel(
            scores=data["scores"], labels=data["labels"],
            auc=a4["auc"], eer=a4["eer"],
            out_path=FIGURES_DIR / "06_a4_open_set.pdf",
            title=(f"A4 open-set verification on PhysioNet "
                   f"({a4['n_test_subjects']} unseen subjects, "
                   f"{a4['n_pairs']:,} pairs)"),
        )
    else:
        verification_summary_card(
            auc=a4["auc"], auc_ci_low=a4["auc_ci_low"],
            auc_ci_high=a4["auc_ci_high"], eer=a4["eer"],
            n_train_subjects=a4["n_train_subjects"],
            n_test_subjects=a4["n_test_subjects"],
            n_pairs=a4["n_pairs"],
            out_path=FIGURES_DIR / "06_a4_open_set.pdf",
            title="A4 open-set EEG verification on unseen subjects (PhysioNet)",
        )
    print("regenerated figures/06_a4_open_set.pdf")


def render_a5():
    path = RESULTS_DIR / "08_a5_membership_inference.json"
    if not path.exists():
        return
    a5 = json.loads(path.read_text())
    _summary_card(
        title="A5 — per-subject membership inference",
        lines=[
            f"{a5['n_shadows']} shadow EEGNets   ·   {a5['n_subjects']} subjects   ·   "
            f"{a5['n_target_members']} members vs {a5['n_target_nonmembers']} non-members",
        ],
        big_metric_label="AUC",
        big_metric_value=a5["auc"],
        big_metric_ci=(a5["auc_ci_low"], a5["auc_ci_high"]),
        footer=f"TPR – FPR advantage = {a5['advantage']:.3f}   "
               f"(at threshold {a5['advantage_threshold']:.3f})",
        out_path=FIGURES_DIR / "08_a5_membership_inference.pdf",
    )
    print("regenerated figures/08_a5_membership_inference.pdf")


def render_d1_pca():
    # The D1 PCA figure has bespoke logic in experiments/07_d1_pca.py;
    # importing it would trigger argparse if we're not careful, so we
    # construct it with the same plotting code inline.
    from experiments import __init__  # noqa
    import importlib
    mod = importlib.import_module("experiments.07_d1_pca")
    rows = json.loads((RESULTS_DIR / "07_d1_pca.json").read_text())
    n = rows[0]["n_subjects"]
    mod._plot_d1(
        rows, FIGURES_DIR / "07_d1_pca.pdf",
        title=f"D1 PCA defense  ({n} subj, chance top-1 = {100/n:.1f}%)",
    )
    print("regenerated figures/07_d1_pca.pdf")


def render_d1_other(transform: str):
    import importlib
    mod = importlib.import_module("experiments.11_d1_adhoc")
    src = RESULTS_DIR / f"11_d1_{transform}.json"
    if not src.exists():
        return
    rows = json.loads(src.read_text())
    n = rows[0]["n_subjects"]
    mod._plot(
        rows, FIGURES_DIR / f"11_d1_{transform}.pdf",
        transform=transform,
        title=f"D1 {transform} defense  ({n} subj, chance top-1 = {100/n:.1f}%)",
    )
    print(f"regenerated figures/11_d1_{transform}.pdf")


def render_d2_dann():
    rows = []
    for path in ["09_d2_dann.json", "09_d2_dann_extended.json"]:
        full = RESULTS_DIR / path
        if full.exists():
            rows.extend(json.loads(full.read_text()))
    if not rows:
        return
    logreg = sorted([r for r in rows if r["probe"] == "logreg"],
                    key=lambda r: r["lambda"])
    plt.rcParams.update(_setup_axes())
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(8.4, 3.8), sharex=True)

    xs = [r["lambda"] for r in logreg]
    ys_priv = [r["top1"] for r in logreg]
    lo = [r["top1"] - r["top1_ci_low"] for r in logreg]
    hi = [r["top1_ci_high"] - r["top1"] for r in logreg]
    ys_util = [r["task_acc"] for r in logreg]
    chance = logreg[0]["chance_top1"]

    ax_l.errorbar(xs, ys_priv, yerr=[lo, hi], color="#e31a1c", marker="o",
                  linewidth=1.0, capsize=2, markersize=4)
    ax_l.axhline(chance, color="#7f8c8d", linestyle="--", linewidth=0.6,
                 label=f"chance = {chance:.3f}")
    sweet = next((r for r in logreg if r["lambda"] == 0.2), None)
    if sweet is not None:
        ax_l.annotate(
            f'sweet spot λ=0.2:\nleak {sweet["top1"]:.3f}',
            xy=(0.2, sweet["top1"]), xytext=(0.45, 0.34),
            arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=0.6),
            fontsize=8, color="#2c3e50",
        )
    ax_l.set_xlabel(r"DANN adversary weight $\lambda$")
    ax_l.set_ylabel("Re-ID top-1 (logreg probe)")
    ax_l.set_ylim(0, 0.50)
    ax_l.set_title("Privacy: identity leakage vs adversary strength")
    ax_l.grid(linestyle=":", linewidth=0.4, alpha=0.5)
    ax_l.legend(frameon=False, fontsize=8, loc="upper right")

    ax_r.plot(xs, ys_util, color="#e31a1c", marker="s",
              linewidth=1.0, markersize=4)
    ax_r.axhline(0.25, color="#7f8c8d", linestyle="--", linewidth=0.6,
                 label="chance task = 0.25")
    ax_r.axvspan(0.35, 0.55, color="#c0392b", alpha=0.08, label="task collapses")
    ax_r.set_xlabel(r"DANN adversary weight $\lambda$")
    ax_r.set_ylabel("BCI task accuracy")
    ax_r.set_ylim(0, 0.50)
    ax_r.set_title("Utility: task accuracy vs adversary strength")
    ax_r.grid(linestyle=":", linewidth=0.4, alpha=0.5)
    ax_r.legend(frameon=False, fontsize=8, loc="upper right")

    fig.suptitle("D2 DANN — extended privacy-utility curve  "
                 "(8 λ values, 104 PhysioNet subjects)",
                 y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "09_d2_dann.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("regenerated figures/09_d2_dann.pdf")


def render_d3():
    path = RESULTS_DIR / "10_d3_dp_sgd.json"
    if not path.exists():
        return
    results = json.loads(path.read_text())
    plt.rcParams.update(_setup_axes())
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(7.8, 3.6), sharex=True)
    logreg = sorted(
        [r for r in results if r["probe"] == "logreg"],
        key=lambda r: (r["target_epsilon"] if r["target_epsilon"] else 999),
    )
    xs_label = ["∞ (no DP)" if r["target_epsilon"] is None
                else f"ε={r['target_epsilon']:.0f}" for r in logreg]
    xs = list(range(len(logreg)))
    ys_priv = [r["top1"] for r in logreg]
    lo = [r["top1"] - r["top1_ci_low"] for r in logreg]
    hi = [r["top1_ci_high"] - r["top1"] for r in logreg]
    ys_util = [r["task_acc"] for r in logreg]
    chance = logreg[0]["chance_top1"]

    ax_l.errorbar(xs, ys_priv, yerr=[lo, hi], color="#6a3d9a", marker="P",
                  linewidth=1.0, capsize=2, label="re-ID top-1")
    ax_l.axhline(chance, color="#c0392b", linestyle="--", linewidth=0.8,
                 label=f"chance = {chance:.3f}")
    ax_l.set_xticks(xs); ax_l.set_xticklabels(xs_label)
    ax_l.set_ylabel("Re-ID top-1 (logreg probe)")
    ax_l.set_ylim(0, 0.10)
    ax_l.set_title("Privacy: identity leakage vs ε")
    ax_l.grid(linestyle=":", linewidth=0.4, alpha=0.5)
    ax_l.legend(frameon=False, fontsize=8, loc="upper right")

    ax_r.plot(xs, ys_util, color="#6a3d9a", marker="s", linewidth=1.0,
              label="task accuracy")
    ax_r.axhline(0.25, color="#c0392b", linestyle="--", linewidth=0.8,
                 label="chance task = 0.25")
    ax_r.set_xticks(xs); ax_r.set_xticklabels(xs_label)
    ax_r.set_ylabel("BCI task accuracy")
    ax_r.set_ylim(0.20, 0.45)
    ax_r.set_title("Utility: task accuracy vs ε")
    ax_r.grid(linestyle=":", linewidth=0.4, alpha=0.5)
    ax_r.legend(frameon=False, fontsize=8, loc="upper right")

    fig.suptitle("D3 DP-SGD defense  (104 PhysioNet subjects, δ=1e-5)",
                 y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "10_d3_dp_sgd.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("regenerated figures/10_d3_dp_sgd.pdf")


def render_subgroup_fairness():
    path = RESULTS_DIR / "12_subgroup_fairness.json"
    if not path.exists():
        return
    import importlib
    mod = importlib.import_module("tools.subgroup_fairness")
    out = json.loads(path.read_text())
    mod.plot_subgroup(out["per_subject_by_victim"],
                       FIGURES_DIR / "12_subgroup_fairness.pdf")
    print("regenerated figures/12_subgroup_fairness.pdf")


def render_pareto():
    import importlib
    importlib.import_module("tools.pareto_plot").main()


# ---------------------------------------------------------------------------
# Extension figures (cross-dataset A4, multi-seed A4, adaptive attacker, MI
# on classical, EEGNet subgroup fairness)
# ---------------------------------------------------------------------------
def render_a4_cross_dataset():
    p_self = RESULTS_DIR / "06_a4_open_set.json"
    p_cross = RESULTS_DIR / "13_a4_cross_dataset.json"
    p_ms = RESULTS_DIR / "14_a4_multi_seed.json"
    if not (p_self.exists() and p_cross.exists()):
        return
    self_ = json.loads(p_self.read_text())
    cross = json.loads(p_cross.read_text())
    ms = json.loads(p_ms.read_text()) if p_ms.exists() else None

    plt.rcParams.update(_setup_axes())
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    labels = ["A4 within-dataset\n(PhysioNet, 24 unseen subj)"]
    aucs = [self_["auc"]]
    yerr_lo = [self_["auc"] - self_["auc_ci_low"]]
    yerr_hi = [self_["auc_ci_high"] - self_["auc"]]
    colors = ["#2c3e50"]
    if ms is not None:
        labels.append("A4 multi-seed mean\n(PhysioNet, 5 splits × 24 unseen)")
        aucs.append(ms["aggregate"]["auc_mean"])
        yerr_lo.append(ms["aggregate"]["auc_std"])
        yerr_hi.append(ms["aggregate"]["auc_std"])
        colors.append("#3498db")
    labels += ["A4 cross-dataset\n(PhysioNet → IV-2a, 9 unseen)",
               "Random\n(theoretical)"]
    aucs += [cross["auc"], 0.5]
    yerr_lo += [cross["auc"] - cross["auc_ci_low"], 0.0]
    yerr_hi += [cross["auc_ci_high"] - cross["auc"], 0.0]
    colors += ["#e67e22", "#7f8c8d"]
    bars = ax.bar(labels, aucs, yerr=[yerr_lo, yerr_hi], color=colors,
                  edgecolor="white", linewidth=0.8,
                  error_kw=dict(elinewidth=0.6, capsize=2, capthick=0.6))
    for b, v in zip(bars, aucs):
        ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.3f}",
                ha="center", fontsize=9)
    ax.axhline(0.5, color="#c0392b", linewidth=0.6, linestyle="--",
               label="random = 0.5")
    ax.set_ylabel("Verification ROC-AUC")
    ax.set_ylim(0.45, 1.05)
    ax.set_title("A4 verification AUC: within-dataset vs across-splits vs across-datasets")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "13_a4_cross_dataset.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("regenerated figures/13_a4_cross_dataset.pdf")


def render_a4_multi_seed():
    path = RESULTS_DIR / "14_a4_multi_seed.json"
    if not path.exists():
        return
    ms = json.loads(path.read_text())
    plt.rcParams.update(_setup_axes())
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    seeds = [r["seed"] for r in ms["per_seed"]]
    aucs = [r["auc"] for r in ms["per_seed"]]
    lo = [r["auc"] - r["auc_ci_low"] for r in ms["per_seed"]]
    hi = [r["auc_ci_high"] - r["auc"] for r in ms["per_seed"]]
    ax.errorbar(seeds, aucs, yerr=[lo, hi], fmt="o", color="#2c3e50",
                capsize=3, linewidth=1.0, markersize=6,
                label="per-seed AUC ± bootstrap CI")
    ax.axhline(ms["aggregate"]["auc_mean"], color="#e67e22", linewidth=1.2,
               label=f"5-seed mean = {ms['aggregate']['auc_mean']:.3f}")
    ax.fill_between([min(seeds) - 0.5, max(seeds) + 0.5],
                    ms["aggregate"]["auc_mean"] - ms["aggregate"]["auc_std"],
                    ms["aggregate"]["auc_mean"] + ms["aggregate"]["auc_std"],
                    alpha=0.15, color="#e67e22",
                    label=f"±1 std = {ms['aggregate']['auc_std']:.3f}")
    ax.set_xticks(seeds)
    ax.set_xlabel("Random seed (subject-split index)")
    ax.set_ylabel("Verification ROC-AUC")
    ax.set_ylim(0.85, 0.97)
    ax.set_title("A4 robustness: AUC across 5 random subject splits")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(linestyle=":", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "14_a4_multi_seed.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("regenerated figures/14_a4_multi_seed.pdf")


def render_adaptive_attacker():
    path = RESULTS_DIR / "15_d2_adaptive_attacker.json"
    if not path.exists():
        return
    adv = json.loads(path.read_text())
    plt.rcParams.update(_setup_axes())
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    labels = ["A1 baseline\n(no defense)\n0.411",
              "DANN λ=0.2\nlogreg probe\n(generic)",
              "DANN λ=0.2\ndeep MLP probe\n(stronger generic)",
              "DANN λ=0.2\nencoder fine-tune\n(adaptive)"]
    top1s = [0.411, adv["attacks"][0]["top1"],
             adv["attacks"][1]["top1"], adv["attacks"][2]["top1"]]
    lo = [0.0,
          adv["attacks"][0]["top1"] - adv["attacks"][0]["top1_ci_low"],
          adv["attacks"][1]["top1"] - adv["attacks"][1]["top1_ci_low"],
          adv["attacks"][2]["top1"] - adv["attacks"][2]["top1_ci_low"]]
    hi = [0.0,
          adv["attacks"][0]["top1_ci_high"] - adv["attacks"][0]["top1"],
          adv["attacks"][1]["top1_ci_high"] - adv["attacks"][1]["top1"],
          adv["attacks"][2]["top1_ci_high"] - adv["attacks"][2]["top1"]]
    colors = ["#7f8c8d", "#e31a1c", "#fb6a4a", "#67000d"]
    bars = ax.bar(labels, top1s, yerr=[lo, hi], color=colors,
                  edgecolor="white", linewidth=0.8,
                  error_kw=dict(elinewidth=0.6, capsize=2, capthick=0.6))
    for b, v in zip(bars, top1s):
        ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.3f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.axhline(0.0096, color="#c0392b", linewidth=0.6, linestyle="--",
               label="chance = 0.0096")
    ax.set_ylabel("Re-ID top-1 (104 PhysioNet subjects)")
    ax.set_ylim(0, 1.0)
    ax.set_title("DANN λ=0.2 collapses under an adaptive attacker who fine-tunes the encoder")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "15_d2_adaptive_attacker.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("regenerated figures/15_d2_adaptive_attacker.pdf")


def render_a5_classical():
    p_eegnet = RESULTS_DIR / "08_a5_membership_inference.json"
    p_riemann = RESULTS_DIR / "16_a5_riemann_mi.json"
    p_fbcsp = RESULTS_DIR / "16_a5_fbcsp_mi.json"
    if not (p_eegnet.exists() and p_riemann.exists() and p_fbcsp.exists()):
        return
    eegnet = json.loads(p_eegnet.read_text())
    riemann = json.loads(p_riemann.read_text())
    fbcsp = json.loads(p_fbcsp.read_text())
    plt.rcParams.update(_setup_axes())
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    victims = ["EEGNet\n(20 shadows)",
               "FBCSP+LDA\n(12 shadows)",
               "Riemann\ntangent-space\n(20 shadows)"]
    aucs = [eegnet["auc"], fbcsp["auc"], riemann["auc"]]
    lo = [eegnet["auc"] - eegnet["auc_ci_low"],
          fbcsp["auc"] - fbcsp["auc_ci_low"],
          riemann["auc"] - riemann["auc_ci_low"]]
    hi = [eegnet["auc_ci_high"] - eegnet["auc"],
          fbcsp["auc_ci_high"] - fbcsp["auc"],
          riemann["auc_ci_high"] - riemann["auc"]]
    bars = ax.bar(victims, aucs, yerr=[lo, hi],
                  color=["#2c3e50", "#7f8c8d", "#34495e"],
                  edgecolor="white", linewidth=0.8,
                  error_kw=dict(elinewidth=0.6, capsize=2, capthick=0.6))
    for b, v in zip(bars, aucs):
        ax.text(b.get_x() + b.get_width()/2, min(v + 0.02, 0.97),
                f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(0.5, color="#c0392b", linewidth=0.6, linestyle="--",
               label="random = 0.5")
    ax.set_ylabel("Membership-inference AUC")
    ax.set_ylim(0.4, 1.05)
    ax.set_title("A5 membership inference across victim families")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "16_a5_classical.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("regenerated figures/16_a5_classical.pdf")


def render_subgroup_fairness_eegnet():
    path = RESULTS_DIR / "17_subgroup_fairness_eegnet.json"
    if not path.exists():
        return
    sf = json.loads(path.read_text())
    plt.rcParams.update(_setup_axes())
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.6))
    rows = sf["per_subject"]
    attack = np.array([r["attack_acc"] for r in rows])
    task = np.array([r["task_acc"] for r in rows])

    ax = axes[0, 0]
    ax.hist(attack, bins=20, color="#2c3e50", alpha=0.7,
            edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Per-subject A1 attack accuracy")
    ax.set_ylabel("# subjects")
    ax.set_title(f"Per-subject heterogeneity "
                 f"(mean {attack.mean():.3f}, std {attack.std():.3f})")
    ax.grid(linestyle=":", linewidth=0.4, alpha=0.5)

    ax = axes[0, 1]
    ax.scatter(task, attack, s=20, color="#2c3e50", alpha=0.6,
               edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Per-subject task accuracy")
    ax.set_ylabel("Per-subject A1 attack accuracy")
    ax.set_xlim(0, 0.85); ax.set_ylim(0, 1.05)
    pearson = float(np.corrcoef(task, attack)[0, 1])
    ax.set_title(f"Task vs attack (Pearson r = {pearson:+.3f})")
    ax.grid(linestyle=":", linewidth=0.4, alpha=0.5)

    ax = axes[1, 0]
    m_acc = [r["attack_acc"] for r in rows if r["sex"] == "M"]
    f_acc = [r["attack_acc"] for r in rows if r["sex"] == "F"]
    bp = ax.boxplot([m_acc, f_acc],
                     tick_labels=[f"M (n={len(m_acc)})", f"F (n={len(f_acc)})"],
                     patch_artist=True, showfliers=True,
                     flierprops={"marker": ".", "markersize": 3, "alpha": 0.5})
    for patch, c in zip(bp["boxes"], ["#2c3e50", "#7f8c8d"]):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.set_ylabel("A1 attack accuracy")
    p_sex = sf["sex"]["diff_M_minus_F"]["mannwhitneyu_p"]
    ax.set_title(f"By sex (Δ = +{sf['sex']['diff_M_minus_F']['point']:.3f}, "
                 f"p={p_sex:.3f})")
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)

    ax = axes[1, 1]
    low_acc = [r["attack_acc"] for r in rows if r["age_bucket"] == "low"]
    mid_acc = [r["attack_acc"] for r in rows if r["age_bucket"] == "mid"]
    high_acc = [r["attack_acc"] for r in rows if r["age_bucket"] == "high"]
    bp = ax.boxplot([low_acc, mid_acc, high_acc],
                     tick_labels=[f"Low\n(n={len(low_acc)})",
                                  f"Mid\n(n={len(mid_acc)})",
                                  f"High\n(n={len(high_acc)})"],
                     patch_artist=True, showfliers=True,
                     flierprops={"marker": ".", "markersize": 3, "alpha": 0.5})
    for patch, c in zip(bp["boxes"], ["#3498db", "#7f8c8d", "#e67e22"]):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.set_ylabel("A1 attack accuracy"); ax.set_xlabel("Age tertile")
    p_age = sf["age"]["diff_low_minus_high"]["mannwhitneyu_p"]
    ax.set_title(f"By age tertile (Δ low−high = "
                 f"+{sf['age']['diff_low_minus_high']['point']:.3f}, "
                 f"p={p_age:.3f})")
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)

    fig.suptitle(f"EEGNet subgroup fairness — A1 attack accuracy across 104 "
                 f"subjects (decile gap {sf['decile_gap']:.3f})",
                 y=1.005, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "17_subgroup_fairness_eegnet.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("regenerated figures/17_subgroup_fairness_eegnet.pdf")


# ---------------------------------------------------------------------------
# Extension v2 figures (D3 adaptive, DP-SGD architecture ablation,
# EEGNet age 5-seed replication, D1 adaptive). 21_a2_vs_rest is rendered
# by experiments/21 directly.
# ---------------------------------------------------------------------------
def render_d3_adaptive_attacker():
    """experiments/18 — DP-SGD ε=3 against logreg, deep-MLP, encoder-fine-tune.

    Mirror of render_adaptive_attacker (DANN λ=0.2). Headline: where DANN
    collapsed under encoder fine-tune (0.21 generic → 0.80 fine-tune), DP
    holds (0.022 generic → 0.049 fine-tune; 5× chance vs DANN's 80×).
    """
    p = RESULTS_DIR / "18_d3_adaptive_attacker.json"
    if not p.exists():
        return
    adv = json.loads(p.read_text())
    plt.rcParams.update(_setup_axes())
    fig, ax = plt.subplots(figsize=(7.2, 3.6))

    # Format: A1 baseline (no defense) is point reference, then 3 attacks
    a1_no_def = 0.411  # from results/02_closed_set_reid.json EEGNet logreg
    bars_label = ["A1 baseline\n(no defense)\n0.411",
                  "DP-SGD ε=3\nlogreg probe\n(generic)",
                  "DP-SGD ε=3\ndeep MLP probe\n(stronger generic)",
                  "DP-SGD ε=3\nencoder fine-tune\n(adaptive)"]
    top1 = [a1_no_def,
            adv["attacks"][0]["top1"],
            adv["attacks"][1]["top1"],
            adv["attacks"][2]["top1"]]
    lo = [0.0,
          adv["attacks"][0]["top1"] - adv["attacks"][0]["top1_ci_low"],
          adv["attacks"][1]["top1"] - adv["attacks"][1]["top1_ci_low"],
          adv["attacks"][2]["top1"] - adv["attacks"][2]["top1_ci_low"]]
    hi = [0.0,
          adv["attacks"][0]["top1_ci_high"] - adv["attacks"][0]["top1"],
          adv["attacks"][1]["top1_ci_high"] - adv["attacks"][1]["top1"],
          adv["attacks"][2]["top1_ci_high"] - adv["attacks"][2]["top1"]]
    colors = ["#7f8c8d", "#6a51a3", "#9e9ac8", "#54278f"]
    bars = ax.bar(bars_label, top1, yerr=[lo, hi], color=colors,
                  edgecolor="white", linewidth=0.8,
                  error_kw=dict(elinewidth=0.6, capsize=2, capthick=0.6))
    for b, v in zip(bars, top1):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.axhline(0.0096, color="#c0392b", linewidth=0.6, linestyle="--",
               label="chance = 0.0096")
    ax.set_ylabel("Re-ID top-1 (104 PhysioNet subjects)")
    ax.set_ylim(0, 0.5)
    ax.set_title(f"DP-SGD (target ε=3, final ε={adv['final_epsilon']:.2f}) "
                 f"holds under adaptive attack — formal DP is attacker-agnostic")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "18_d3_adaptive_attacker.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("regenerated figures/18_d3_adaptive_attacker.pdf")


def render_dp_sgd_arch_ablation():
    """experiments/19 — three-bar breakdown of D3's privacy.

    AdamW + BatchNorm  → architecture/optimizer change → SGD + GroupNorm (no DP)
                                                       → SGD + GroupNorm + DP ε=3.
    Most of the privacy is from architecture; only ~2 pp from formal DP noise.
    """
    p_arch = RESULTS_DIR / "19_dp_sgd_arch_ablation.json"
    p_d3 = RESULTS_DIR / "10_d3_dp_sgd.json"
    p_a1 = RESULTS_DIR / "02_closed_set_reid.json"
    if not (p_arch.exists() and p_d3.exists() and p_a1.exists()):
        return
    arch_rows = json.loads(p_arch.read_text())
    arch = next(r for r in arch_rows if r["probe"] == "logreg")
    d3_rows = json.loads(p_d3.read_text())
    dp = next(r for r in d3_rows
              if r.get("target_epsilon") == 3.0 and r["probe"] == "logreg")
    a1_rows = json.loads(p_a1.read_text())
    a1 = next(r for r in a1_rows
              if r["victim"] == "eegnet" and r["probe"] == "logreg")

    plt.rcParams.update(_setup_axes())
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    labels = ["AdamW + BatchNorm\n(A1 baseline)",
              "SGD + GroupNorm\nno DP (this paper)",
              f"SGD + GroupNorm\nDP-SGD ε=3 (final {dp['final_epsilon']:.2f})"]
    top1 = [a1["top1"], arch["top1"], dp["top1"]]
    lo = [a1["top1"] - a1["top1_ci_low"],
          arch["top1"] - arch["top1_ci_low"],
          dp["top1"] - dp["top1_ci_low"]]
    hi = [a1["top1_ci_high"] - a1["top1"],
          arch["top1_ci_high"] - arch["top1"],
          dp["top1_ci_high"] - dp["top1"]]
    colors = ["#7f8c8d", "#bcbddc", "#54278f"]
    bars = ax.bar(labels, top1, yerr=[lo, hi], color=colors,
                  edgecolor="white", linewidth=0.8,
                  error_kw=dict(elinewidth=0.6, capsize=2, capthick=0.6))
    for b, v in zip(bars, top1):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.axhline(0.0096, color="#c0392b", linewidth=0.6, linestyle="--",
               label="chance = 0.0096")
    arch_delta = a1["top1"] - arch["top1"]
    noise_delta = arch["top1"] - dp["top1"]
    ax.set_ylabel("A1 closed-set re-ID top-1 (104 subjects)")
    ax.set_ylim(0, 0.5)
    ax.set_title(f"D3 privacy breakdown — architecture: -{arch_delta*100:.1f} pp,  "
                 f"DP noise: -{noise_delta*100:.1f} pp")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "19_dp_sgd_arch_ablation.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("regenerated figures/19_dp_sgd_arch_ablation.pdf")


def render_eegnet_age_seeds():
    """experiments/22 — 5-seed replication of the EEGNet age effect.

    Three panels:
      (a) per-seed age p-values + α=0.05 threshold + Fisher's combined p
      (b) per-seed age effect size (Δ low - high) — should be consistent
      (c) decile gap per seed — invariant to seed choice
    """
    p = RESULTS_DIR / "22_eegnet_age_seeds.json"
    if not p.exists():
        return
    sf = json.loads(p.read_text())
    per_seed = sf["per_seed"]
    agg = sf["aggregate"]

    seeds = [r["seed"] for r in per_seed]
    age_p = [r["age_p"] for r in per_seed]
    age_diff = [r["age_diff_low_minus_high"] for r in per_seed]
    decile = [r["decile_gap"] for r in per_seed]

    plt.rcParams.update(_setup_axes())
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))

    # Panel A: per-seed age p
    ax = axes[0]
    bars = ax.bar([f"s{s}" for s in seeds], age_p, color="#54278f", alpha=0.7,
                  edgecolor="white", linewidth=0.5)
    ax.axhline(0.05, color="#c0392b", linewidth=0.7, linestyle="--",
               label="α=0.05")
    for b, v in zip(bars, age_p):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}",
                ha="center", fontsize=8)
    ax.set_ylabel("age effect p (Mann-Whitney)")
    ax.set_xlabel("seed")
    ax.set_ylim(0, max(age_p) * 1.25)
    ax.set_title(f"Per-seed age p\nFisher combined = {agg['fisher_age_p']:.4f}")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)

    # Panel B: effect size (consistent direction)
    ax = axes[1]
    ax.bar([f"s{s}" for s in seeds], age_diff, color="#9e9ac8", alpha=0.85,
           edgecolor="white", linewidth=0.5)
    ax.axhline(agg["age_diff_mean"], color="#54278f", linewidth=1.2,
               label=f"mean = {agg['age_diff_mean']:.3f}")
    ax.axhline(0.0, color="#7f8c8d", linewidth=0.4)
    for s, d in zip(seeds, age_diff):
        ax.text(s, d + 0.005, f"{d:+.3f}", ha="center", fontsize=8)
    ax.set_ylabel("Δ (age low − age high)")
    ax.set_xlabel("seed")
    ax.set_title(f"Age-effect direction is consistent\n"
                 f"Δ = {agg['age_diff_mean']:+.3f} ± {agg['age_diff_std']:.3f}")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)

    # Panel C: decile gap per seed
    ax = axes[2]
    ax.bar([f"s{s}" for s in seeds], decile, color="#bcbddc", alpha=0.85,
           edgecolor="white", linewidth=0.5)
    ax.axhline(agg["decile_gap_mean"], color="#54278f", linewidth=1.2,
               label=f"mean = {agg['decile_gap_mean']:.3f}")
    ax.set_ylabel("decile gap (most − least leaked)")
    ax.set_xlabel("seed")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Decile gap is invariant to seed\n"
                 f"= {agg['decile_gap_mean']:.3f} ± {agg['decile_gap_std']:.3f}")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)

    fig.suptitle("EEGNet age effect across 5 seeds — direction consistent, "
                 "per-seed p-value noisy (Fisher's aggregate confirms)",
                 y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "22_eegnet_age_seeds.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("regenerated figures/22_eegnet_age_seeds.pdf")


def render_d1_adaptive_attacker():
    """experiments/23 — D1 ad-hoc defenses under encoder fine-tune.

    Grouped bar chart per defense: generic logreg vs encoder fine-tune.
    All three D1 defenses collapse under fine-tune (matched-negative to
    DANN, in contrast to D3 DP which holds).
    """
    p = RESULTS_DIR / "23_d1_adaptive_attacker.json"
    if not p.exists():
        return
    rows = json.loads(p.read_text())["defenses"]
    defenses = ["pca_k8", "noise_sigma1.0", "channel_drop_k8"]
    pretty = {"pca_k8": "PCA k=8", "noise_sigma1.0": "noise σ=1.0",
              "channel_drop_k8": "channel-drop k=8"}

    by = {}
    for r in rows:
        by[(r["defense"], r["attack"])] = r

    plt.rcParams.update(_setup_axes())
    fig, ax = plt.subplots(figsize=(7.6, 3.8))

    x = np.arange(len(defenses))
    w = 0.36
    generic = [by[(d, "logreg_probe")]["top1"] for d in defenses]
    finetune = [by[(d, "encoder_finetune")]["top1"] for d in defenses]
    g_lo = [by[(d, "logreg_probe")]["top1"] - by[(d, "logreg_probe")]["top1_ci_low"]
            for d in defenses]
    g_hi = [by[(d, "logreg_probe")]["top1_ci_high"] - by[(d, "logreg_probe")]["top1"]
            for d in defenses]
    f_lo = [by[(d, "encoder_finetune")]["top1"] - by[(d, "encoder_finetune")]["top1_ci_low"]
            for d in defenses]
    f_hi = [by[(d, "encoder_finetune")]["top1_ci_high"] - by[(d, "encoder_finetune")]["top1"]
            for d in defenses]

    b1 = ax.bar(x - w/2, generic, w, label="logreg probe (generic)",
                color="#bcbddc", edgecolor="white", linewidth=0.6,
                yerr=[g_lo, g_hi],
                error_kw=dict(elinewidth=0.5, capsize=2))
    b2 = ax.bar(x + w/2, finetune, w, label="encoder fine-tune (adaptive)",
                color="#54278f", edgecolor="white", linewidth=0.6,
                yerr=[f_lo, f_hi],
                error_kw=dict(elinewidth=0.5, capsize=2))
    for bs, vs in [(b1, generic), (b2, finetune)]:
        for b, v in zip(bs, vs):
            ax.text(b.get_x() + b.get_width()/2, v + 0.015, f"{v:.3f}",
                    ha="center", fontsize=8.5, fontweight="bold")
    ax.axhline(0.411, color="#34495e", linewidth=0.6, linestyle="-.",
               label="A1 no-defense baseline = 0.411")
    ax.axhline(0.0096, color="#c0392b", linewidth=0.6, linestyle="--",
               label="chance = 0.0096")
    ax.set_xticks(x)
    ax.set_xticklabels([pretty[d] for d in defenses])
    ax.set_ylabel("Re-ID top-1 (104 PhysioNet subjects)")
    ax.set_ylim(0, 0.9)
    ax.set_title("D1 ad-hoc defenses collapse under adaptive attacker "
                 "(all three above no-defense baseline)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "23_d1_adaptive_attacker.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("regenerated figures/23_d1_adaptive_attacker.pdf")


def render_a2_vs_rest():
    """experiments/21 — A2 with probe trained on resting-state EEG.

    Renders a closed-set bar chart from the canonical JSON; mirrors the
    A1/A2/A3 layout. (The original experiment also writes the figure
    directly, but having the regen path here keeps the one-command
    rebuild self-contained.)
    """
    p = RESULTS_DIR / "21_a2_vs_rest.json"
    if not p.exists():
        return
    rows = json.loads(p.read_text())
    n = rows[0]["n_subjects"]
    closed_set_bar_chart(
        rows, FIGURES_DIR / "21_a2_vs_rest.pdf",
        title=f"A2 cross-task re-ID  ({n} subj)\n"
              f"probe trained on RESTING-STATE, tested on motor-imagery",
    )
    print("regenerated figures/21_a2_vs_rest.pdf")


def main() -> None:
    print("Regenerating all figures from result JSONs ...\n")
    for fn in (
        render_a1, render_a2, render_a3, render_a4, render_a5,
        render_d1_pca,
        lambda: render_d1_other("noise"),
        lambda: render_d1_other("channel_drop"),
        render_d2_dann, render_d3,
        render_subgroup_fairness,
        # Extension batch (cross-dataset, multi-seed, adaptive, classical MI,
        # EEGNet fairness)
        render_a4_cross_dataset,
        render_a4_multi_seed,
        render_adaptive_attacker,
        render_a5_classical,
        render_subgroup_fairness_eegnet,
        # Extension batch v2 (D3 holds under fine-tune; arch ablation;
        # 5-seed age replication; D1 adaptive collapse; rest-state A2)
        render_d3_adaptive_attacker,
        render_dp_sgd_arch_ablation,
        render_eegnet_age_seeds,
        render_d1_adaptive_attacker,
        render_a2_vs_rest,
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
