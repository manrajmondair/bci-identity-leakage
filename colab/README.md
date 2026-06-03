# Colab notebooks

Heavy compute (any experiment that trains EEGNet, DANN, DP-SGD, the
contrastive embedder, shadow models for membership inference, or
the federated server-aggregation loop) runs in Colab on an L4 or A100
GPU. Local development on Mac is for classical baselines (FBCSP,
Riemannian tangent-space), attack code, and the audit / regen tooling.

## Workflow

1. **Open a notebook in Colab.** The `Open In Colab` URLs in the
   project's README point at this directory's notebooks on `main`.
   Each notebook is self-contained.
2. **Set runtime to L4 GPU** (the default for almost everything), or
   **A100** for the three big sweeps: `D3_eps_sweep`,
   `D3_membership_aware_eps_sweep`, and `Multi_seed`. The
   per-notebook headers state the recommended runtime explicitly.
3. **Mount Drive when the notebook asks for it.** Lee 2019 experiments
   read a 4 GB compact float16 cache produced by
   `A3_lee2019_download.ipynb`; that cache lives at
   `/content/drive/MyDrive/bci_cache/` and survives runtime restarts.
4. **Run all cells.** Each notebook clones `main`, installs deps,
   prefetches dataset shards, runs the experiment, and writes
   `results/<exp>.json` plus a per-run `runs/<run_id>/meta.json`
   provenance entry.
5. **Send the result blocks back.** The final cell prints the result
   JSON between `--- BEGIN <name>.json ---` / `--- END <name>.json ---`
   markers and the run metadata between matching markers. Copy both
   blocks into the project chat.
6. **The artifacts are committed to canonical paths.** Figures
   regenerate locally from the JSON via
   `python -m tools.regenerate_figures`, so we never round-trip the
   binary PDFs through Colab.

The notebooks intentionally do not push to the repo themselves; that
would require a GitHub PAT in Colab. Keeping that boundary clean.

### One Colab quirk: do not "Save a copy in GitHub"

Colab's `File → Save a copy in GitHub` overwrites the source notebook
with one that has cell IDs renamed, an `id` field added to every cell,
output cells embedded, and any partial run state baked in. The result
is a 200+ line diff against the canonical source for nothing. Just run
the cells; the paste-back block at the bottom returns everything that
needs to be committed.

## Notebook index — core experiments (01–22)

### Attacks on PhysioNet

| Notebook | Experiment script | Wall (L4) |
|---|---|---|
| `A1_eegnet_rerun.ipynb` | `experiments.02_closed_set_reid` | ~25 min |
| `A2_cross_task.ipynb` | `experiments.04_a2_cross_task` | ~30 min |
| `A2_vs_rest.ipynb` | `experiments.21_a2_vs_rest` | ~35 min |
| `A3_cross_session.ipynb` | `experiments.05_a3_cross_session` | ~25 min |
| `A4_open_set.ipynb` | `experiments.06_a4_open_set` | ~45 min |
| `A4_cross_dataset.ipynb` | `experiments.13_a4_cross_dataset` | ~40 min |
| `A4_multi_seed.ipynb` | `experiments.14_a4_multi_seed` | ~50 min |
| `A5_membership_inference.ipynb` | `experiments.08_a5_membership_inference` | ~55 min |
| `A5_riemann_mi.ipynb` | `experiments.16_a5_classical --victim riemann` | ~30 min |
| `A5_fbcsp_mi.ipynb` | `experiments.16_a5_classical --victim fbcsp` | ~25 min |

### Defenses + adaptive attackers

| Notebook | Experiment script | Wall (L4) |
|---|---|---|
| `D1_pca.ipynb` | `experiments.07_d1_pca` | ~30 min |
| `D1_noise.ipynb` | `experiments.11_d1_adhoc --transform noise` | ~30 min |
| `D1_channel_drop.ipynb` | `experiments.11_d1_adhoc --transform channel_drop` | ~30 min |
| `D2_dann.ipynb` | `experiments.09_d2_dann` (4-point λ sweep) | ~45 min |
| `D2_dann_extended.ipynb` | `experiments.09_d2_dann` (extended grid) | ~50 min |
| `D3_dp_sgd.ipynb` | `experiments.10_d3_dp_sgd` (3-point ε sweep) | ~50 min |
| `D1_adaptive_attacker.ipynb` | `experiments.23_d1_adaptive_attacker` | ~55 min |
| `D2_adaptive_attacker.ipynb` | `experiments.15_d2_adaptive_attacker` | ~35 min |
| `D3_adaptive_attacker.ipynb` | `experiments.18_d3_adaptive_attacker` | ~45 min |
| `DP_SGD_arch_ablation.ipynb` | `experiments.19_dp_sgd_arch_ablation` | ~30 min |

### Fairness

| Notebook | Experiment script | Wall (L4) |
|---|---|---|
| `SF_eegnet.ipynb` | `experiments.17_subgroup_fairness_eegnet` (single seed) | ~30 min |
| `SF_eegnet_seeds.ipynb` | `experiments.22_eegnet_age_seeds` (5 seeds) | ~50 min |

## Notebook index — second-corpus and extension experiments (20, 24–34)

### Lee 2019 OpenBMI second-corpus replication

| Notebook | Purpose | Hardware | Wall |
|---|---|---|---|
| `A3_lee2019_download.ipynb` | Stream-and-compact Lee 2019 (54 × 2 sessions) into a 4 GB float16 cache on Drive. Restart-safe per (subject, session). | **CPU** | 2.5–4 h spread across one or more sessions |
| `A3_lee2019.ipynb` | A3 cross-session re-ID on the 54-subject cohort (chance 1.85%). | L4 | ~50 min |
| `A4_lee2019.ipynb` | A4 open-set verification (40 train / 14 unseen subjects), within- and cross-session protocols. | L4 | ~40 min |
| `A5_lee2019.ipynb` | Shokri-style MI with 12 EEGNet shadows on Lee 2019. | L4 | ~45 min |
| `SF_lee2019.ipynb` | Within-cohort heterogeneity (mean / decile gap / IQR) per victim family. | L4 | ~35 min |

### Cross-dataset

| Notebook | Purpose | Hardware | Wall |
|---|---|---|---|
| `A4_cross_dataset_symmetric.ipynb` | A4 verification in all four cross-corpus directions: IV-2a→PhysioNet, PhysioNet→Lee 2019, Lee 2019→PhysioNet, IV-2a→Lee 2019. | L4 | ~100 min |
| `A4_asymmetry_mechanism.ipynb` | Re-run Lee 2019→PhysioNet contrastive with a synthetic 4-class label (hand × early/late half) to test the task-complexity hypothesis for the observed asymmetry. | L4 | ~40 min |

### Defense frontier + stronger adaptive attackers

| Notebook | Purpose | Hardware | Wall |
|---|---|---|---|
| `D3_eps_sweep.ipynb` | DP-SGD ε ∈ {0.5, 1, 3, 10, ∞} × {logreg probe, encoder fine-tune}. | A100 (or L4 with longer wall) | ~150 min |
| `D3_membership_aware.ipynb` | DP-aware MIA: 8 DP-SGD shadows + 1 DP-SGD target at ε=3. | A100 | ~3 h |
| `D3_membership_aware_eps_sweep.ipynb` | DP-aware MIA at ε ∈ {0.5, 1.0}. | A100 | ~5–6 h |
| `D3_model_inversion.ipynb` | Fredrikson-style model inversion against fine-tuned re-ID head, no-defense vs DP-SGD ε=3, scored in a reference contrastive embedder. | L4 | ~50 min |
| `D4_federated_dp.ipynb` | FedAvg with central-DP server-side Gaussian noise. 104 clients, 30 rounds, 50% participation. | L4 | ~75 min |

### Theory + multi-seed

| Notebook | Purpose | Hardware | Wall |
|---|---|---|---|
| `theory_scaling.ipynb` | Closed-set re-ID accuracy vs cohort size N for EEGNet and Riemann; γ scaling fit; Yeom (ε, δ)-bound overlay against the ε sweep. | L4 | ~60 min |
| `Multi_seed.ipynb` | 5-seed sweep over experiments 20, 24, and all four directions of 26; aggregates mean/std per metric. | A100 (or L4 across sessions) | 5–7 h |

## Hardware notes

- **L4 default.** EEGNet is ~5K parameters; for most notebooks the
  bottleneck is data transfer or small-model gradient compute, not GPU
  FLOPs. L4 saturates the workload and starts faster from cold than
  A100.
- **A100 for the four big sweeps.** `D3_eps_sweep` (five DP-SGD
  trainings), `D3_membership_aware*` (eight shadow DP-SGD trainings),
  and `Multi_seed` (nine experiment runs × five seeds each) see ~2×
  wall-clock speedup on A100. The other notebooks do not benefit.
- **Disk.** PhysioNet imagery cache is 1.7 GB on local Colab disk;
  Lee 2019 raw .mat files are 600 MB per (subject, session) but the
  prefetcher deletes them as it goes, so peak local-disk usage stays
  bounded. The Lee 2019 compact cache on Drive is ~4 GB.
- **Mount Drive only for Lee 2019 notebooks.** Other notebooks pull
  data through MNE + moabb's local cache and need no Drive mount.

