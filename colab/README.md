# Colab notebooks

Heavy compute (anything that trains EEGNet, DANN, DP-SGD, contrastive
embeddings, or shadow models for membership inference) runs in Colab on
an L4 GPU. Local development on Mac is for classical baselines (FBCSP,
Riemann tangent-space), attack code, and the audit / regen tooling.

## Workflow

1. **Open a notebook in Colab.** The `Open In Colab` URLs in the
   project's README point at this directory's notebooks on `main`. Each
   notebook is self-contained.
2. **Set runtime to L4 GPU.** Runtime → Change runtime type → L4. The
   notebooks fit the per-run wall-time budget (under one hour) on L4;
   A100 does not help any of them because the bottleneck is either data
   transfer or small-model gradient compute that L4 already saturates.
3. **Run all cells.** Each notebook clones `main`, installs deps,
   prefetches dataset shards, runs the experiment, and writes
   `results/<exp>.json` plus a per-run `runs/<run_id>/meta.json`
   provenance entry.
4. **Send the two artifacts back.** The final cell auto-downloads the
   result JSON and the run metadata. Drop both into the project chat.
5. **The artifacts are committed to canonical paths.** Figures
   regenerate locally from the JSON via `python -m tools.regenerate_figures`,
   so we never round-trip the binary PDFs through Colab.

The notebooks intentionally do not push to the repo themselves; that
would require a GitHub PAT in Colab. Keeping that boundary clean.

### One Colab quirk: do not "Save a copy in GitHub"

Colab's `File → Save a copy in GitHub` overwrites the source notebook
with one that has cell IDs renamed, an `id` field added to every cell,
output cells embedded, and any partial run state baked in. The result
is a 200+ line diff against the canonical source for nothing. Just run
the cells; the download cell at the bottom returns everything that
needs to be committed.

## Notebook index

### Attacks

| Notebook | Experiment script | Wall (L4) | Output |
|---|---|---|---|
| `A1_eegnet_rerun.ipynb` | `experiments.02_closed_set_reid` | ~25 min | `results/02_closed_set_reid.json`, `figures/02_closed_set_reid.pdf` |
| `A2_cross_task.ipynb` | `experiments.04_a2_cross_task` | ~30 min | `results/04_a2_cross_task.json` |
| `A2_vs_rest.ipynb` | `experiments.21_a2_vs_rest` | ~35 min | `results/21_a2_vs_rest.json` |
| `A3_cross_session.ipynb` | `experiments.05_a3_cross_session` | ~25 min | `results/05_a3_cross_session.json` |
| `A4_open_set.ipynb` | `experiments.06_a4_open_set` | ~45 min | `results/06_a4_open_set.json` |
| `A4_cross_dataset.ipynb` | `experiments.13_a4_cross_dataset` | ~40 min | `results/13_a4_cross_dataset.json` |
| `A4_multi_seed.ipynb` | `experiments.14_a4_multi_seed` | ~50 min | `results/14_a4_multi_seed.json` |
| `A5_membership_inference.ipynb` | `experiments.08_a5_membership_inference` | ~55 min | `results/08_a5_membership_inference.json` |
| `A5_riemann_mi.ipynb` | `experiments.16_a5_classical --victim riemann` | ~30 min | `results/16_a5_riemann_mi.json` |
| `A5_fbcsp_mi.ipynb` | `experiments.16_a5_classical --victim fbcsp` | ~25 min | `results/16_a5_fbcsp_mi.json` |

### Defenses

| Notebook | Experiment script | Wall (L4) | Output |
|---|---|---|---|
| `D1_pca.ipynb` | `experiments.07_d1_pca` | ~30 min | `results/07_d1_pca.json` |
| `D1_noise.ipynb` | `experiments.11_d1_adhoc --defense noise` | ~30 min | `results/11_d1_noise.json` |
| `D1_channel_drop.ipynb` | `experiments.11_d1_adhoc --defense channel_drop` | ~30 min | `results/11_d1_channel_drop.json` |
| `D2_dann.ipynb` | `experiments.09_d2_dann` (4-point λ sweep) | ~45 min | `results/09_d2_dann.json` |
| `D2_dann_extended.ipynb` | `experiments.09_d2_dann` (extended fine grid) | ~50 min | `results/09_d2_dann_extended.json` |
| `D3_dp_sgd.ipynb` | `experiments.10_d3_dp_sgd` (3-point ε sweep) | ~50 min | `results/10_d3_dp_sgd.json` |

### Adaptive attackers and ablations

| Notebook | Experiment script | Wall (L4) | Output |
|---|---|---|---|
| `D2_adaptive_attacker.ipynb` | `experiments.15_d2_adaptive_attacker` (DANN λ=0.2 vs 3 attackers) | ~35 min | `results/15_d2_adaptive_attacker.json` |
| `D3_adaptive_attacker.ipynb` | `experiments.18_d3_adaptive_attacker` (DP-SGD ε=3 vs 3 attackers) | ~45 min | `results/18_d3_adaptive_attacker.json` |
| `D1_adaptive_attacker.ipynb` | `experiments.23_d1_adaptive_attacker` (PCA / noise / channel-drop vs fine-tune) | ~55 min | `results/23_d1_adaptive_attacker.json` |
| `DP_SGD_arch_ablation.ipynb` | `experiments.19_dp_sgd_arch_ablation` (GroupNorm-EEGNet, no DP) | ~30 min | `results/19_dp_sgd_arch_ablation.json` |

### Fairness

| Notebook | Experiment script | Wall (L4) | Output |
|---|---|---|---|
| `SF_eegnet.ipynb` | `experiments.17_subgroup_fairness_eegnet` (single seed) | ~30 min | `results/17_subgroup_fairness_eegnet.json` |
| `SF_eegnet_seeds.ipynb` | `experiments.22_eegnet_age_seeds` (5 seeds + Fisher) | ~50 min | `results/22_eegnet_age_seeds.json` |

### Scaffolded but not run

| Notebook | Experiment script | Status |
|---|---|---|
| `A3_lee2019.ipynb` | `experiments.20_a3_lee2019` (cross-session A3 on 54-subject Lee 2019 OpenBMI) | scaffolded; full Colab run did not complete within the project's compute budget owing to ~3 MB/s sustained throughput from the Tokyo OpenBMI mirror. The reported A3 result is on BCI IV-2a (n=9) only; this notebook is preserved for future replication. |

## Hardware notes

- **L4 default.** EEGNet is ~5K parameters; for every notebook here the
  bottleneck is data transfer or small-model gradient compute, not GPU
  FLOPs. L4 saturates the workload and starts faster from cold than
  A100.
- **A100 does not help.** Verified empirically on the abandoned
  `A3_lee2019` run: with a Tokyo S3 mirror serving at ~3 MB/s, the
  GPU is idle for the entire download phase regardless of whether it
  is L4 or A100.
- **Disk.** PhysioNet imagery cache is 1.7 GB, windowed-array cache is
  2.3 GB, BCI IV-2a is 150 MB; comfortably under Colab's allocated
  disk.
