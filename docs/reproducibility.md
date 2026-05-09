# Reproducibility

How to recreate every result in this repo from a fresh checkout. Each
result JSON in `results/` traces back to a specific commit + hardware +
timestamp via `runs/<run_id>/meta.json`.

---

## Local environment (Mac)

```bash
git clone https://github.com/manrajmondair/bci-identity-leakage
cd bci-identity-leakage

# Python 3.11 (tested on Apple Silicon, Linux, and Colab Linux)
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
pip install -e ".[dev]"

# Sanity-check the env
python -c "import mne, torch, braindecode, pyriemann, opacus, moabb"
python -m tools.audit                      # 76 invariants over the cached results
```

Hard pin: `torch==2.5.1` + `torchaudio==2.5.1`. The pip resolver otherwise pulls a `torchaudio` (e.g., 2.11) whose ABI mismatches torch and produces an obscure dlopen failure at import — see commit history for the original debugging story.

## Data prefetch

PhysioNet imagery (1.35 GB):

```bash
python -m data.prefetch_direct --runs imagery --workers 8
```

Downloads from `archive.physionet.org` (~10× faster than the main `physionet.org` host on a residential link). Resumes already-cached files, retries on network glitches with exponential backoff. Total wall: ~2 minutes on a typical link.

Add `execution` for A2:

```bash
python -m data.prefetch_direct --runs imagery execution --workers 8
```

BCI IV-2a (~150 MB) downloads on first call to `data.bciiv2a_loader.load_subject_session(...)`. The wrapper has a built-in retry-with-backoff (`data.bciiv2a_loader._get_subject_data_with_retry`) for moabb/pooch's no-retry behavior — required at least once when an `IncompleteRead` mid-stream killed an earlier Colab run.

## Reproducing each result

Every result JSON is keyed by an `experiments/NN_*.py` script. Below is the canonical command and target file for each.

### Attacks

| Result | Script | Command | Hardware | Wall |
|---|---|---|---|---|
| A1 closed-set | `experiments.02_closed_set_reid` | `--all` | L4 (EEGNet) / Mac (FBCSP, Riemann) | ~50 min |
| A2 cross-task | `experiments.04_a2_cross_task` | `--all` | L4 | ~25 min |
| A3 cross-session | `experiments.05_a3_cross_session` | `--all` | L4 | ~10 min |
| A4 open-set | `experiments.06_a4_open_set` | `--all` | L4 | ~35 min |
| A5 MI (EEGNet) | `experiments.08_a5_membership_inference` | `--all` | L4 | ~30 min |
| Cross-dataset A4 | `experiments.13_a4_cross_dataset` | `--all` | L4 | ~30 min |
| Multi-seed A4 | `experiments.14_a4_multi_seed` | `--all` | L4 | ~55 min |

### Defenses

| Result | Script | Command | Wall |
|---|---|---|---|
| D1 PCA | `experiments.07_d1_pca` | `--all` | ~50 min L4 |
| D1 noise | `experiments.11_d1_adhoc` | `--transform noise --all` | ~50 min L4 |
| D1 channel-drop | `experiments.11_d1_adhoc` | `--transform channel_drop --all` | ~50 min L4 |
| D2 DANN (4 points) | `experiments.09_d2_dann` | `--all` | ~50 min L4 |
| D2 DANN extended | `experiments.09_d2_dann` | `--all --lambdas 0.05 0.2 0.3 0.7` | ~50 min L4 |
| D3 DP-SGD | `experiments.10_d3_dp_sgd` | `--all` | ~55 min L4 |

### Adaptive analyses

| Result | Script | Wall |
|---|---|---|
| Adaptive attacker vs DANN λ=0.2 | `experiments.15_d2_adaptive_attacker --all` | ~35 min L4 |
| Adaptive attacker vs DP-SGD ε=3 | `experiments.18_d3_adaptive_attacker --all` | ~45 min L4 |
| Adaptive attacker vs D1 (PCA / noise / channel-drop) | `experiments.23_d1_adaptive_attacker --all` | ~55 min L4 |
| MI on Riemann | `experiments.16_a5_classical --victim riemann --all` | ~30 min L4 |
| MI on FBCSP | `experiments.16_a5_classical --victim fbcsp --all` | ~55 min L4 |
| EEGNet subgroup fairness (single seed) | `experiments.17_subgroup_fairness_eegnet --all` | ~25 min L4 |
| EEGNet subgroup fairness (5 seeds) | `experiments.22_eegnet_age_seeds --all` | ~50 min L4 |
| DP-SGD architecture ablation | `experiments.19_dp_sgd_arch_ablation --all` | ~30 min L4 |
| A2 with resting-state probe | `experiments.21_a2_vs_rest --all` | ~35 min L4 |
| Local subgroup fairness (FBCSP+Riemann) | `tools.subgroup_fairness --models fbcsp riemann` | ~12 min Mac CPU |

### Pareto + audit

```bash
python -m tools.pareto_plot                # consolidates all defense JSONs
python -m tools.audit                      # 76 invariants
```

## Colab path

For GPU-bound experiments (anything that trains EEGNet from scratch):

1. Open `colab/<NAME>.ipynb` directly via `https://colab.research.google.com/github/manrajmondair/bci-identity-leakage/blob/main/colab/<NAME>.ipynb`
2. Runtime → Change runtime type → **L4 GPU**
3. Runtime → Run all
4. The final cell auto-downloads `results/<exp>.json` and `runs/<run_id>/meta.json`. Send both files back to the project owner; they get committed to canonical paths and the figure regenerates locally from the JSON.

Workflow rules embedded in the notebooks:
- Each notebook clones `main` cleanly so it always picks up the latest fixes.
- The metadata cell uses an explicit `cwd=PROJECT_DIR` for `git rev-parse` and `os.chdir(PROJECT_DIR)` at the top, so it survives Colab kernel restarts (a single `%cd` in cell 1 doesn't persist after disconnect).
- **Don't `Save a copy in GitHub`** from inside Colab — that overwrites the source notebook with cell-id and output churn (and we have a documented rebase-once-this-happens recovery from the early A1 run).
- Each notebook fits the 1-hour-per-run budget on L4 — see [`feedback_compute_budget`](../) for the rationale.

## Provenance trail

Each completed experiment writes a `runs/<run_id>/meta.json` of the form:

```json
{
  "run_id": "20260507T085103_a3_cross_session_e1260a0",
  "experiment": "experiments.05_a3_cross_session",
  "args": ["--all"],
  "git_sha": "e1260a0d3b42f186e290494a14f5829d624ad6af",
  "hardware": "Colab NVIDIA L4",
  "platform": "Linux-6.6.113+-x86_64-with-glibc2.35",
  "torch_version": "2.10.0+cu128",
  "completed_at_utc": "2026-05-07T08:51:03Z",
  "outputs": ["results/05_a3_cross_session.json"]
}
```

This is the audit trail. To verify a number in the report:

1. Open the result JSON the report cites
2. Find the matching `runs/<run_id>/meta.json` (timestamp + experiment match)
3. Check out that commit, install pinned deps, rerun the experiment
4. Compare to the JSON you find at runtime — should match within bootstrap noise

## What we don't promise about reproducibility

- **Bit-exact equality:** PyTorch + cuDNN are not bit-deterministic across hardware even with seeded RNGs. Numbers should match within ~0.5 percentage points across re-runs on different hardware (L4 vs A100 vs Mac MPS).
- **Hardware drift:** A1 was originally run on Mac MPS for EEGNet, later runs on Colab L4. Both produce equivalent numbers within bootstrap CIs.
- **Software upgrades:** if you `pip install -U` past our pinned versions and braindecode renames a class, the embedding hook may need to find the new class name. The `EEGNetVictim._find_head` fallback handles the cases we've seen.
- **External data drift:** PhysioNet and OpenNeuro-ds004362 are stable but not immutable. Hashes for the downloaded files are NOT pinned in this repo (mne's pooch handles them); upstream re-uploads would invalidate that.

## Audit acceptance criterion

The chain of evidence is OK iff `python -m tools.audit` returns:

```
=== SUMMARY: 76 OK, 0 WARN, 0 FAIL ===
```

Any FAIL on a pull request blocks merge. Latest passing audit run is in `runs/<latest>_audit_<sha>/audit.md`.
