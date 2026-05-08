# Colab notebooks

Heavy compute (anything that trains EEGNet, DANN variants, DP-SGD, or shadow models for membership inference) runs here. Mac is for development, classical baselines, and attack code.

## How the workflow works

1. **Open a notebook in Colab** — File → Upload Notebook, or just open the `.ipynb` directly from the GitHub repo URL.
2. **Set runtime to L4 GPU** — Runtime → Change runtime type → L4. Default; only switch to A100 if a notebook explicitly says so.
3. **Run all cells.** Each notebook is self-contained: it clones `main`, installs deps, prefetches data, runs the experiment, prints results.
4. **Send the two JSON files back.** The final cell auto-downloads `results/<exp>.json` (the canonical scientific output) and `runs/<run_id>/meta.json` (the provenance / audit-trail entry — git SHA, hardware, runtime, timestamps). Drag both into the chat.
5. **I commit both** to the repo so the canonical state stays in `main`. The figure is regenerated locally from the result JSON; we don't round-trip the binary.

The notebooks intentionally don't push back to the repo themselves — that would require putting a GitHub PAT in Colab. Easier to keep that boundary clean.

### Why `runs/`?

Treats this like a "traditional repo": every reported number traces back to a specific git SHA + hardware + timestamp. See [`runs/README.md`](../runs/README.md) for the schema.

### One Colab quirk: do not "Save a copy in GitHub"

Colab's `File → Save a copy in GitHub` overwrites the source notebook with one that has cell IDs renamed, an `id` field added to every cell, output cells embedded, and any partial run state baked in. The result is a 200+ line diff against the canonical source for nothing. Just run the cells; the download cell at the bottom returns everything I need. The notebook source stays clean.

## Notebooks

| Notebook | What it runs | Wall time | Output |
|---|---|---|---|
| `A1_eegnet_rerun.ipynb` | `experiments/02_closed_set_reid --all` (rebuilds A1 with the volts → microvolts EEGNet fix) | ~25 min | `results/02_closed_set_reid.json`, `figures/02_closed_set_reid.pdf` |
| `A2_cross_task.ipynb` | `experiments/04_a2_cross_task --all` | ~30 min | `results/04_a2_cross_task.json` |
| `A3_cross_session.ipynb` | `experiments/05_a3_cross_session --all` (BCI IV-2a 9 subj × 2 sessions) | ~25 min | `results/05_a3_cross_session.json` |
| `A4_open_set.ipynb` | `experiments/06_a4_open_set --all` | ~45 min | `results/06_a4_open_set.json` |
| `A4_cross_dataset.ipynb` | `experiments/13_a4_cross_dataset --all` (PhysioNet → IV-2a transfer) | ~40 min | `results/13_a4_cross_dataset.json` |
| `A4_multi_seed.ipynb` | `experiments/14_a4_multi_seed --all` (5 splits × 22 epochs) | ~50 min | `results/14_a4_multi_seed.json` |
| `A5_membership_inference.ipynb` | `experiments/08_a5_membership_inference --all` (20 EEGNet shadows) | ~55 min | `results/08_a5_membership_inference.json` |
| `A5_riemann_mi.ipynb` | `experiments/16_a5_classical --victim riemann --all` | ~30 min | `results/16_a5_riemann_mi.json` |
| `A5_fbcsp_mi.ipynb` | `experiments/16_a5_classical --victim fbcsp --all` | ~25 min | `results/16_a5_fbcsp_mi.json` |
| `D1_pca.ipynb` | `experiments/07_d1_pca --all` | ~30 min | `results/07_d1_pca.json` |
| `D1_noise.ipynb` | `experiments/11_d1_adhoc --defense noise --all` | ~30 min | `results/11_d1_noise.json` |
| `D1_channel_drop.ipynb` | `experiments/11_d1_adhoc --defense channel_drop --all` | ~30 min | `results/11_d1_channel_drop.json` |
| `D2_dann.ipynb` | `experiments/09_d2_dann --all` (initial 4-point λ sweep) | ~45 min | `results/09_d2_dann.json` |
| `D2_dann_extended.ipynb` | `experiments/09_d2_dann --all` (extended λ sweep at fine grid) | ~50 min | `results/09_d2_dann_extended.json` |
| `D2_adaptive_attacker.ipynb` | `experiments/15_d2_adaptive_attacker --all` (DANN λ=0.2 vs 3 attackers) | ~35 min | `results/15_d2_adaptive_attacker.json` |
| `D3_dp_sgd.ipynb` | `experiments/10_d3_dp_sgd --all` (3-point ε sweep) | ~50 min | `results/10_d3_dp_sgd.json` |
| `SF_eegnet.ipynb` | `experiments/17_subgroup_fairness_eegnet --all` (single-seed) | ~30 min | `results/17_subgroup_fairness_eegnet.json` |

### Pending — extension batch v2 (post-milestone)

These six notebooks were generated programmatically from a shared template; each follows the same clone-deps-prefetch-run-emit-download pattern. Run on L4 GPU.

| Notebook | What it runs | Wall time | Output |
|---|---|---|---|
| `D3_adaptive_attacker.ipynb` | `experiments/18_d3_adaptive_attacker --all` (DP-SGD ε=3 vs 3 attackers — symmetry to D2 adaptive) | ~40-50 min | `results/18_d3_adaptive_attacker.json` |
| `DP_SGD_arch_ablation.ipynb` | `experiments/19_dp_sgd_arch_ablation --all` (GroupNorm-EEGNet + SGD with no DP, isolates arch contribution) | ~25-35 min | `results/19_dp_sgd_arch_ablation.json` |
| `A3_lee2019.ipynb` | `experiments/20_a3_lee2019 --all` (cross-session A3 on Lee2019, 54 subj × 2 sessions) | ~50-60 min (incl. ~25 min moabb download) | `results/20_a3_lee2019.json` |
| `A2_vs_rest.ipynb` | `experiments/21_a2_vs_rest --all` (probe trained on PhysioNet resting state, tested on motor imagery) | ~30-40 min | `results/21_a2_vs_rest.json` |
| `SF_eegnet_seeds.ipynb` | `experiments/22_eegnet_age_seeds --all` (5-seed replication of the EEGNet age fairness p=0.044) | ~50-55 min | `results/22_eegnet_age_seeds.json` |
| `D1_adaptive_attacker.ipynb` | `experiments/23_d1_adaptive_attacker --all` (PCA / noise / channel-drop vs encoder fine-tune) | ~50-55 min | `results/23_d1_adaptive_attacker.json` |

## Hardware notes

- **L4 default.** EEGNet is ~5K params; data transfer dominates over compute. L4 ≈ A100 for our workloads, costs fewer Pro compute units, and starts faster from cold.
- **A100 only if a notebook explicitly requests it.** Reserved for if/when we add a transformer-scale EEG model (we don't have one in scope right now).
- **Disk:** PhysioNet imagery cache is 1.7 GB, windowed-array cache is 2.3 GB; well under Colab's free disk.
