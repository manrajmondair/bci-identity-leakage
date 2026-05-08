# Figure catalog

One row per figure: what it shows, what JSON drives it, what script
generates it. To rebuild all of them from the canonical result JSONs:

```bash
python -m tools.regenerate_figures
```

---

## Headline figures (likely to appear in the report)

### `figures/pareto_privacy_utility.pdf`
**The single most important figure.** Three-panel privacy-utility scatter
(one panel per victim family — EEGNet, FBCSP+LDA, Riemann tangent-space).
X-axis: motor-imagery task accuracy. Y-axis: A1 closed-set re-ID top-1.
Lower-right corner = high utility AND low identity leak. Colored by
defense family; each family's `no_defense` baseline is a star.

- **Source JSONs:** `results/02_closed_set_reid.json`, `07_d1_pca.json`, `11_d1_noise.json`, `11_d1_channel_drop.json`, `09_d2_dann.json`, `09_d2_dann_extended.json`, `10_d3_dp_sgd.json`
- **Script:** `tools/pareto_plot.py`
- **Color legend:** no_defense = navy, D1 PCA = blue, D1 noise = green, D1 channel-drop = orange, D2 DANN = red, D3 DP-SGD = purple

### `figures/06_a4_open_set.pdf`
**A4 headline summary card.** Compact text card showing AUC = 0.925 [0.923, 0.928] on 24 unseen subjects from a contrastive embedding trained on 80 disjoint subjects, with EER = 0.133.

- **Source JSON:** `results/06_a4_open_set.json`
- **Script:** `eval.plots.verification_summary_card` (call site in `experiments/06_a4_open_set.py`'s output)

### `figures/02_closed_set_reid.pdf`
**A1 closed-set bar chart.** Per-victim, per-probe re-ID top-1 with bootstrap CIs and a chance-line annotation.

- **Source JSON:** `results/02_closed_set_reid.json`
- **Script:** `eval.plots.closed_set_bar_chart`

### `figures/09_d2_dann.pdf`
**D2 DANN privacy-utility curve.** 8-point λ sweep (λ ∈ {0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}). Two panels: leakage vs λ (with annotated sweet spot at λ=0.2) and task accuracy vs λ (with collapse zone shaded).

- **Source JSONs:** `results/09_d2_dann.json` + `results/09_d2_dann_extended.json` merged
- **Script:** rendered inline by `tools/regenerate_figures.py` (combines the two JSONs)

---

## Per-experiment figures

### `figures/04_a2_cross_task.pdf`
A2 bar chart, same format as A1 but title indicates the cross-task probe protocol.

- **Source JSON:** `results/04_a2_cross_task.json`
- **Script:** `eval.plots.closed_set_bar_chart`

### `figures/05_a3_cross_session.pdf`
A3 bar chart on BCI IV-2a (9 subjects, chance 11%). Same format.

- **Source JSON:** `results/05_a3_cross_session.json`
- **Script:** `eval.plots.closed_set_bar_chart`

### `figures/08_a5_membership_inference.pdf`
A5 EEGNet membership-inference summary card showing AUC = 0.878 [0.803, 0.943] and advantage 0.635.

- **Source JSON:** `results/08_a5_membership_inference.json`
- **Script:** rendered inline by `tools/regenerate_figures.py` (a generic summary-card layout)

### `figures/07_d1_pca.pdf`
D1 PCA two-panel (privacy vs k, utility vs k) for the 4-point sweep, 3 victim families.

- **Source JSON:** `results/07_d1_pca.json`
- **Script:** `experiments.07_d1_pca._plot_d1`

### `figures/11_d1_noise.pdf`, `figures/11_d1_channel_drop.pdf`
Same two-panel layout as D1 PCA but for the σ-sweep (noise) and k-sweep (channel-drop).

- **Source JSONs:** `results/11_d1_noise.json`, `results/11_d1_channel_drop.json`
- **Script:** `experiments.11_d1_adhoc._plot`

### `figures/10_d3_dp_sgd.pdf`
D3 DP-SGD two-panel (privacy vs ε, utility vs ε) across the {None, 10, 3} sweep.

- **Source JSON:** `results/10_d3_dp_sgd.json`
- **Script:** rendered inline by `tools/regenerate_figures.py`

### `figures/12_subgroup_fairness.pdf`
W5.1 subgroup fairness — four panels: per-subject A1 attack-acc histogram (FBCSP and Riemann), per-subject task-vs-attack scatter, sex boxplot, age-tertile boxplot.

- **Source JSON:** `results/12_subgroup_fairness.json`
- **Script:** `tools.subgroup_fairness.plot_subgroup`

### `figures/03_within_subject_reid.pdf` (deferred experiment)
Within-subject A1b figure. Only present because the script that produced it (`experiments/03_within_subject_reid.py`) runs end-to-end on smoke data; the experiment itself is deferred from the milestone (see `runs/` audit).

### `figures/13_a4_cross_dataset.pdf`
Bar chart comparing A4 within-dataset AUC (0.925) to multi-seed mean (0.934) and cross-dataset (0.694, PhysioNet → IV-2a). Visualizes the headline-vs-cross-protocol gap.

- **Source JSONs:** `06_a4_open_set.json`, `13_a4_cross_dataset.json`, `14_a4_multi_seed.json`
- **Script:** `tools/regenerate_figures.py::render_a4_cross_dataset`

### `figures/14_a4_multi_seed.pdf`
A4 AUC across 5 random subject splits with per-seed bootstrap CI and the cross-seed mean ± std.

- **Source JSON:** `14_a4_multi_seed.json`
- **Script:** `tools/regenerate_figures.py::render_a4_multi_seed`

### `figures/15_d2_adaptive_attacker.pdf` ⭐
**The "DANN collapses under adaptive attacker" figure.** Bar chart comparing 4 attacker strengths against the DANN λ=0.2 victim: A1 baseline (0.41), logreg probe (0.25), deep MLP probe (0.36), end-to-end encoder fine-tune (**0.80**).

- **Source JSON:** `15_d2_adaptive_attacker.json`
- **Script:** `tools/regenerate_figures.py::render_adaptive_attacker`

### `figures/16_a5_classical.pdf`
MI AUC across the three victim families (EEGNet, FBCSP, Riemann). Riemann at ceiling (1.0).

- **Source JSONs:** `08_a5_membership_inference.json`, `16_a5_riemann_mi.json`, `16_a5_fbcsp_mi.json`
- **Script:** `tools/regenerate_figures.py::render_a5_classical`

### `figures/17_subgroup_fairness_eegnet.pdf`
4-panel EEGNet subgroup fairness — same layout as the FBCSP/Riemann fairness figure. Hist of per-subject leakage; task vs attack scatter; sex boxplot; age tertile boxplot. Decile gap 0.78, age p=0.044.

- **Source JSON:** `17_subgroup_fairness_eegnet.json`
- **Script:** `tools/regenerate_figures.py::render_subgroup_fairness_eegnet`

### `figures/18_d3_adaptive_attacker.pdf` ⭐
**The matched positive to figure 15.** DP-SGD ε=3 holds under the same encoder-fine-tune attack that broke DANN. Bar chart: A1 baseline (0.411), DP logreg (0.022), DP deep MLP (0.056), DP fine-tune (**0.049**). Formal DP is attacker-agnostic by construction; this empirically confirms it.

- **Source JSON:** `18_d3_adaptive_attacker.json`
- **Script:** `tools/regenerate_figures.py::render_d3_adaptive_attacker`

### `figures/19_dp_sgd_arch_ablation.pdf`
Three-bar breakdown of D3's privacy contribution: AdamW+BN baseline (0.411), SGD+GN no-DP (0.044), SGD+GN+DP ε=3 (0.022). 36.7 pp from architecture/optimizer, 2.2 pp from formal DP noise. Settles the milestone-time architectural confound.

- **Source JSONs:** `19_dp_sgd_arch_ablation.json`, `10_d3_dp_sgd.json`, `02_closed_set_reid.json`
- **Script:** `tools/regenerate_figures.py::render_dp_sgd_arch_ablation`

### `figures/21_a2_vs_rest.pdf`
A2 cross-task re-ID with the probe trained on **resting-state** EEG (PhysioNet R01 + R02) instead of motor execution. Riemann logreg recovers 94.1% from rest alone — the cleanest negative for the "task-shared component" objection.

- **Source JSON:** `21_a2_vs_rest.json`
- **Script:** `tools/regenerate_figures.py::render_a2_vs_rest` (also written by `experiments/21` directly)

### `figures/22_eegnet_age_seeds.pdf`
3-panel 5-seed replication of the EEGNet age effect. Left: per-seed age p (Fisher's combined = 0.008). Middle: per-seed effect size (Δ = +0.093 ± 0.030, consistent direction). Right: decile gap per seed (0.78 ± 0.025, invariant).

- **Source JSON:** `22_eegnet_age_seeds.json`
- **Script:** `tools/regenerate_figures.py::render_eegnet_age_seeds`

### `figures/23_d1_adaptive_attacker.pdf`
D1 ad-hoc defenses under encoder fine-tune. Grouped bar (3 defenses × 2 attackers). PCA k=8 (0.357 → 0.660), noise σ=1.0 (0.180 → 0.640), channel-drop k=8 (0.391 → 0.758). All three end up *above* the no-defense baseline of 0.411 under fine-tune.

- **Source JSON:** `23_d1_adaptive_attacker.json`
- **Script:** `tools/regenerate_figures.py::render_d1_adaptive_attacker`

---

## Style conventions

- **Fonts:** matplotlib default (DejaVu Sans), 9 pt body / 10 pt suptitle.
- **Bars/lines:** 1.0 pt linewidth, 0.6 pt error bars with capsize 2.
- **Colors:** consistent across figures via `tools.pareto_plot._FAMILY_COLORS` and per-victim {EEGNet=navy, FBCSP=gray, Riemann=dark blue-gray}.
- **Chance reference:** brick-red dashed line at the chance level on every privacy axis; gray dotted at chance task accuracy where shown.
- **No emojis or decorative shading** — only functional shading (e.g., the DANN "task collapses" zone in 09_d2_dann.pdf).

The shared style helper is `eval.plots._setup_axes`. Every figure-generation
script imports it; that's the contract enforcement.
