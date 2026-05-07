# EEG-as-Biometric: Subject Re-Identification Leakage in BCI Models

[![CS 281 — Ethics of AI, Stanford, Spring 2026](https://img.shields.io/badge/CS%20281-Ethics%20of%20AI%20%C2%B7%20Stanford%20%C2%B7%20Spring%202026-2c3e50)]()

A reproducible, audited benchmark of how much subject-identifying information leaks from machine-learning brain-computer interface (BCI) models trained for motor-imagery decoding — and how well three families of defenses prevent it. Five attacks × three victim model families × three defense families × two datasets, with bootstrap CIs on every reported number, demographic-stratified fairness analysis, and a 76-invariant audit.

## TL;DR

Train a classical motor-imagery decoder, hand its features to a probe, and you can re-identify subjects across 104 unseen recordings at near-perfect rates — even when the decoder itself is at chance task accuracy. A contrastive EEG embedding generalizes to *people the network never trained on* at AUC 0.925 (voice-biometric grade). Adversarial subject-invariant training (DANN λ=0.2) buys 20 percentage points of privacy at *zero* task-accuracy cost, strictly dominating every ad-hoc transform we tested. Demographic-stratified fairness on the analysis cohort shows the threat is sex-neutral but with a marginal age effect and large within-cohort heterogeneity (49-pp decile gap).

## Headline results

### Attacks (chance top-1 = 1/N where N = cohort size)

| Attack | Dataset | N | Best victim | Re-ID top-1 (logreg, 95% CI) | Lift |
|---|---|---|---|---|---|
| **A1 closed-set** | PhysioNet | 104 | Riemann tangent-space | **100.0%** [100.0, 100.0] | 104× |
| **A2 cross-task** (execution → imagery) | PhysioNet | 104 | Riemann tangent-space | **100.0%** [99.9, 100.0] | 104× |
| **A3 cross-session** (day-to-day) | BCI IV-2a | 9 | Riemann tangent-space | **91.3%** [90.2, 92.3] | 8.2× |
| **A4 open-set verification** ⭐ | PhysioNet (24 unseen) | 24 | contrastive EEGNet | **AUC 0.925** [0.923, 0.928] | voice-biometric grade |
| **A5 membership inference** | PhysioNet | 104 | EEGNet shadow models | **AUC 0.878** [0.803, 0.943] | advantage 0.635 |

The original A4 result is the project's single strongest claim: a learned EEG embedding generalizes across people, not just across windows of the training cohort.

### Defenses (privacy-utility on the EEGNet victim, 104-subject cohort)

| Defense | Re-ID top-1 | Task acc | Δ leak vs A1 baseline |
|---|---|---|---|
| **A1 baseline (no defense)** | 0.411 | 0.391 | — |
| D1 PCA k=8 | 0.356 | 0.358 | −5.5 pp |
| D1 noise σ=1.0 | 0.189 | 0.342 | −22.2 pp (but task drops 5 pp) |
| D1 channel-drop k=8 | 0.371 | 0.304 | −4.0 pp |
| **D2 DANN λ=0.2** ⭐ | **0.215** | **0.398** | **−19.6 pp** *with task slightly above baseline* |
| D2 DANN λ=0.5 | 0.076 | 0.253 | task collapsed to chance |
| D3 DP-SGD ε=3 | 0.022 | 0.295 | (architectural changes dominate; see report) |

DANN λ=0.2 is the only configuration tested that buys real privacy without paying utility. The full Pareto across all three victim families is in `figures/pareto_privacy_utility.pdf`.

### Subgroup fairness (104-subject cohort, demographics from OpenNeuro ds004362)

| Victim | M (n=41) / F (n=56), Δ, p | Age low (n=34) / high (n=30), Δ, p | Decile gap |
|---|---|---|---|
| FBCSP+LDA | 0.898 / 0.883, **p=0.67** | 0.935 / 0.854, **p=0.08** | +0.490 |
| Riemann   | 1.000 / 1.000, p=1.0 | 1.000 / 1.000, p=1.0 | 0 (ceiling) |

Fairness-neutral by sex; marginal age effect (younger leak more); large within-cohort heterogeneity. EEGNet subgroup numbers pending.

## Quickstart

```bash
# Local dev environment (PyTorch 2.5.1 / mne / pyriemann / opacus / braindecode)
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
pip install -e ".[dev]"

# Sanity check — ~30 sec on Apple Silicon
python -m tools.audit                      # 76 invariants over the cached results

# Reproduce a result locally (FBCSP / Riemann are CPU-fast on Mac)
python -m experiments.02_closed_set_reid --smoke   # A1, 10 subjects, ~5 min
python -m tools.subgroup_fairness                  # demographic stratification, ~12 min
```

GPU-bound experiments (anything that trains EEGNet from scratch) run on Colab L4 — the `colab/` directory has self-contained notebooks for each. See [`colab/README.md`](colab/README.md) for the workflow (clone → install → prefetch → run → download two JSONs → I commit).

## Datasets

| Dataset | Role | Subjects | Channels | Sampling rate | Loader |
|---|---|---|---|---|---|
| PhysioNet EEG-MMIDB ([physionet.org/content/eegmmidb](https://www.physionet.org/content/eegmmidb/1.0.0/)) | Primary; closed-set, cross-task, open-set, MI | 109 (drop 88/89/92/100/104 → n=104) | 64, 10-10 montage | 160 Hz | [`data.physionet_loader`](data/physionet_loader.py) |
| BCI Competition IV-2a ([bnci-horizon-2020](https://www.bbci.de/competition/iv/)) | Cross-session, cross-dataset | 9 (× 2 sessions) | 22 motor-cortex | 250 Hz | [`data.bciiv2a_loader`](data/bciiv2a_loader.py) (via moabb) |

Demographics for PhysioNet come from [OpenNeuro ds004362](https://openneuro.org/datasets/ds004362) — the BIDS conversion of the same source data, which republishes Gender + Age + Handedness for 95 of the 109 subjects (PhysioNet's EDFs scrub them). See [`data/external/README.md`](data/external/README.md).

## Threat models

We benchmark five attacks under realistic adversary access:

1. **A1 closed-set re-identification.** Adversary has the deployed motor-imagery decoder + a labeled corpus of EEG from N enrolled subjects, asks "which of the N produced this window?" 1-of-N classification.
2. **A2 cross-task re-identification.** Probe is trained on EEG from one cognitive task (real movement) and asked to identify subjects from EEG of a different cognitive task (imagined movement). Tests whether identity rides on cognitive-task-orthogonal components.
3. **A3 cross-session re-identification.** Probe trained on session-1 embeddings, tested on session-2 (recorded a different day on different hardware). The realistic biometric-linkage threat — same person, different session.
4. **A4 open-set verification.** Embedding network trained on 80 subjects; verification (same vs different person) evaluated on 24 *held-out* subjects the network never saw. The strongest test of "EEG functions as a biometric template."
5. **A5 per-subject membership inference.** Black-box adversary with access only to the trained-model API asks "was this subject's data used in training?" Standard Shokri-style shadow-model attack adapted to per-subject membership.

Three defense families:

- **D1 ad-hoc transforms** (the original proposal's mitigations, kept as baselines): channel-mode PCA, additive Gaussian noise on channels, channel-subset reduction.
- **D2 DANN adversarial subject-invariance** (Ganin et al. 2016): EEGNet + Gradient Reversal Layer + 104-way subject-classification head. Encoder pressured to produce features useful for the task and useless for subject ID simultaneously.
- **D3 DP-SGD** (Abadi et al. 2016) via Opacus: per-sample gradient clipping + calibrated Gaussian noise, formal (ε, δ)-differential privacy.

See [`docs/methods.md`](docs/methods.md) for the formal definitions and citations.

## Repository layout

```
attacks/         A1–A5 attack code (closed-set, verification, MI)
defenses/        D1 ad-hoc, D2 DANN, D3 DP-SGD
models/          FBCSP+LDA, Riemann tangent-space, EEGNet, ContrastiveEEGNet
preprocess/      bandpass, epoching, sliding-window extraction (with disk cache)
data/            dataset loaders + cross-dataset channel utilities
data/external/   OpenNeuro participants.tsv (demographic provenance for PhysioNet)
eval/            bootstrap CIs (grouped by trial), plot helpers
experiments/     numbered, one-per-claim entry points (01–17)
tools/           audit + heterogeneity/fairness analysis + Pareto plotter
colab/           one notebook per GPU-bound experiment
results/         canonical JSON outputs of every experiment
figures/         publication-grade PDFs
runs/            execution provenance (one meta.json per Colab run + audits)
report/          milestone draft, final draft (TBD)
```

## Provenance and audit

Every reported number traces back to a specific commit + hardware + timestamp via `runs/<run_id>/meta.json`. The audit script (`tools/audit.py`) checks 76 invariants across data integrity (shapes, NaN, channel order, class balance), train/test split correctness (closed-set vs disjoint-subject), probe methodology (bootstrap grouped by trial, fit/predict separation, EEGNet input_scale fix), a shuffled-label negative control, and effect-size sanity vs published EEG re-ID literature. **Current status: 76 OK / 0 WARN / 0 FAIL** — see latest run under `runs/*_audit_*/audit.md`.

## Reproducing the milestone results

A single-command path from a fresh checkout:

```bash
git clone https://github.com/manrajmondair/bci-identity-leakage
cd bci-identity-leakage
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# CPU-fast experiments — reproduce A1's FBCSP/Riemann numbers locally
python -m data.prefetch_direct --runs imagery --workers 8   # ~2 min
python -m experiments.02_closed_set_reid --all              # ~50 min on Mac

# GPU-bound — see colab/A1_eegnet_rerun.ipynb for the L4 path
```

For details, see [`docs/reproducibility.md`](docs/reproducibility.md).

## Limitations (read this)

- **Demographics**: PhysioNet's 109-subject release is anonymized. We recover demographics for 95 of 109 from the OpenNeuro BIDS sibling, leaving 9 subjects with unknown sex and 13 with unknown age in the analysis cohort. BCI IV-2a publishes only the cohort age range (22-30); per-subject ages are not public, so demographic stratification is not possible on IV-2a.
- **Cohort size.** PhysioNet's 109 subjects and IV-2a's 9 subjects are small by modern biometric standards (fingerprint datasets exceed 10⁵). Open-set claims at TUH-EEG scale (10⁴+ subjects) would strengthen but require institutional access we don't have.
- **Cognitive-task generality.** All claims are scoped to motor-imagery decoding. Whether the same patterns hold for resting-state, ERP-based, or sleep EEG is an open question.
- **EEGNet hyperparameters.** D3 DP-SGD's privacy-utility numbers conflate the formal DP guarantee with the architectural side-effects of being Opacus-compatible (BatchNorm → GroupNorm, parametrize-stripped, SGD optimizer). The report decomposes this honestly.

See [`docs/limitations.md`](docs/limitations.md) for the full list.

## Citations

This project builds on standard methods. Key references:

- **EEGNet** — Lawhern et al. 2018 (arXiv:1611.08024).
- **FBCSP** — Ang et al. 2008, IJCNN.
- **Riemannian tangent-space** — Barachant et al. 2012, IEEE TBME; pyRiemann.
- **Triplet loss** — Schroff et al. 2015 (FaceNet).
- **DANN** — Ganin et al. 2016, JMLR.
- **DP-SGD** — Abadi et al. 2016; Opacus (Yousefpour et al. 2021).
- **Membership inference** — Shokri et al. 2017.
- **EEG biometric prior work** — Maiorana 2016; Yang & Deravi 2017.

Full bibliography in [`docs/methods.md`](docs/methods.md).

## License

MIT for code. Datasets retain their original licenses (PhysioNet ODC-By; BCI IV-2a CC-BY).
