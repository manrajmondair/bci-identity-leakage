# EEG-as-Biometric: Cross-Session Identity Leakage in BCI Models

CS 281 — Ethics of AI, Stanford, Spring 2026 · Manraj Singh Mondair

This project tests whether motor-imagery BCI models leak subject-identifying information under realistic attacker access, and benchmarks defenses with a privacy–utility–fairness Pareto.

## Quickstart

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"

# Smoke test — downloads ~50 MB of PhysioNet and trains EEGNet on one subject
python -m experiments.01_baseline_utility --smoke
```

## Datasets

| Dataset | Role | Loader |
|---|---|---|
| PhysioNet EEG-MMIDB (109 subj, 1 session) | Primary; scale + subgroups | `data.physionet_loader` (uses `mne.datasets.eegbci`) |
| BCI Competition IV-2a (9 subj, 2 sessions) | True cross-session re-ID | `data.bciiv2a_loader` (uses `moabb`) |

Cached under `cache/` (gitignored).

## Layout

```
data/         # dataset loaders
preprocess/   # bandpass, ICA, epoching, windowing
models/       # FBCSP+LDA, Riemannian MDM, EEGNet (victim BCI decoders)
attacks/      # A1 closed-set, A2 cross-task, A3 cross-session, A4 open-set verification, A5 membership inference
defenses/     # D1 ad-hoc transforms, D2 DANN, D3 DP-SGD
eval/         # privacy-utility Pareto, subgroup fairness, bootstrap CIs, LaTeX tables
experiments/  # numbered top-level entrypoints, one per claim in the report
report/       # LaTeX milestone (May 20) and final (June 8)
```

## Reproducibility

Every script accepts `--seed` (default 0) and writes outputs to `figures/` / `results/` keyed by config hash. Final-report figures regenerate from `notebooks/figures.ipynb`.
