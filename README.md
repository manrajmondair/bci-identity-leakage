# EEG-as-Biometric: Subject Re-Identification Leakage in BCI Models

**Author:** Manraj Singh Mondair · CS 281 (Ethics of AI), Stanford, Spring 2026

A reproducible, audited benchmark of how much subject-identifying information
leaks from machine-learning brain-computer interface (BCI) models trained for
motor-imagery decoding, and how three families of defenses perform under both
generic and adaptive attackers. Eleven experiments span five attack protocols,
three victim model families, three defense families, two datasets, demographic-
stratified fairness, multi-seed robustness, cross-dataset transfer, and matched
adaptive-attacker stress tests against every defense. Every reported metric
carries a 1000-resample bootstrap CI; every commit passes a 76-invariant audit.

## Abstract

Motor-imagery brain-computer interfaces decode neural recordings into control
signals for assistive devices and consumer neurotechnology. The same neural
features used to perform that decoding may carry stable, subject-specific
patterns that allow re-identification of the user from later recordings — i.e.,
EEG may function as a biometric even when collected and labeled for a non-
biometric purpose.

We test this hypothesis empirically. On PhysioNet's 109-subject EEG Motor
Movement / Imagery Database (n=104 after dropping known-bad recordings) and
BCI Competition IV-2a's 9-subject two-session corpus, we benchmark five
attack protocols (closed-set re-identification A1; cross-task re-identification
A2; cross-session re-identification A3; open-set verification on subjects
held out from the embedding's training set A4; black-box membership inference
A5) against three classes of victim model (FBCSP+LDA; Riemannian tangent-space
+ logistic regression; EEGNet). We then re-run the attack suite against each
of three defense families (D1 ad-hoc input transforms; D2 DANN adversarial
subject-invariant training; D3 DP-SGD with formal (ε, δ) guarantees) and a
matched encoder-fine-tune adaptive attacker.

Closed-set re-identification on PhysioNet reaches **100% top-1 accuracy on
104 subjects** with a Riemannian tangent-space classifier whose task accuracy
is 35%. A contrastive EEG embedding trained on 80 PhysioNet subjects verifies
the identity of recordings from the **24 unseen subjects** at AUC = 0.925
(multi-seed mean 0.934 ± 0.020 over 5 random splits); the same embedding
transfers to BCI IV-2a at AUC = 0.694. Black-box membership inference reaches
AUC = 0.878 on EEGNet shadow models and 1.000 on Riemann tangent-space.

Of three defense families benchmarked under matched adaptive attack, only
formal differential privacy (DP-SGD ε=3) holds: encoder fine-tune top-1
remains at 0.049 vs the no-defense baseline of 0.411. Adversarial subject-
invariance (DANN λ=0.2) and every ad-hoc input transform tested **collapse
to or above the no-defense baseline** under the same adaptive attacker, with
DANN reaching top-1 = 0.804 — *higher* than no defense. A direct ablation
isolates ~89% of D3's empirical privacy as coming from the BatchNorm →
GroupNorm architectural change Opacus requires, and ~5% from the formal
noise mechanism; the (ε, δ) bound is unaffected by this decomposition since
it is mathematical and attacker-agnostic.

The empirical case for treating EEG as biometric data under GDPR Article 9 /
neurorights frameworks rests on the unseen-subject A4 result, the persistence
of identity signal across cognitive tasks (Riemann recovers 94.1% from
resting-state EEG alone, with no shared task structure with motor imagery),
and the cross-session A3 result on IV-2a. The deployable-privacy lesson is
that only mechanism-level differential privacy survives an attacker who knows
the defense exists.

## Headline numbers

### Attacks (1000-resample bootstrap CIs, trial-grouped)

| Attack | Dataset | N | Best victim | Re-ID metric (95% CI) | vs. chance |
|---|---|---|---|---|---|
| A1 closed-set | PhysioNet | 104 | Riemann tangent-space | top-1 = **1.000** [1.000, 1.000] | 104× |
| A2 cross-task (execution → imagery) | PhysioNet | 104 | Riemann tangent-space | top-1 = **1.000** [0.999, 1.000] | 104× |
| A2 cross-task (resting-state → imagery) | PhysioNet | 104 | Riemann tangent-space | top-1 = **0.941** [0.933, 0.949] | 98× |
| A3 cross-session | BCI IV-2a | 9 | Riemann tangent-space | top-1 = **0.913** [0.902, 0.923] | 8.2× |
| A4 open-set verification (unseen subjects) | PhysioNet | 24 held-out | contrastive EEGNet | AUC = **0.925** [0.923, 0.928] | 0.500 |
| A4 multi-seed (5 splits × 24 unseen) | PhysioNet | 24 × 5 | contrastive EEGNet | AUC = **0.934 ± 0.020** | 0.500 |
| A4 cross-dataset (PhysioNet → IV-2a) | IV-2a | 9 unseen | contrastive EEGNet | AUC = **0.694** [0.690, 0.699] | 0.500 |
| A5 membership inference | PhysioNet | 104 | EEGNet shadow models | AUC = **0.878** [0.803, 0.943] | 0.500 |
| A5 membership inference | PhysioNet | 104 | Riemann tangent-space | AUC = **1.000** [1.000, 1.000] | 0.500 |

### Defenses (EEGNet victim, 104 PhysioNet subjects)

| Defense family | Generic logreg attacker | Adaptive (encoder fine-tune) | Verdict |
|---|---|---|---|
| no defense (A1 baseline)            | 0.411 | — | — |
| D1 PCA k=8                          | 0.357 | **0.660** | collapses, exceeds baseline |
| D1 noise σ=1.0                      | 0.180 | **0.640** | collapses, exceeds baseline |
| D1 channel-drop k=8                 | 0.391 | **0.758** | collapses, exceeds baseline |
| D2 DANN λ=0.2 (initial sweet spot)  | 0.252 | **0.804** | collapses, *exceeds baseline* |
| **D3 DP-SGD ε=3** (final ε=2.996)   | **0.022** | **0.049** | **holds at chance + 4 pp** |

### D3 architecture vs. noise breakdown

| Configuration | A1 top-1 (logreg) | Δ vs. previous |
|---|---|---|
| AdamW + BatchNorm (vanilla A1)        | 0.411 | (baseline) |
| SGD + GroupNorm (no DP)               | 0.044 | −36.7 pp |
| SGD + GroupNorm + DP-SGD ε=3          | 0.022 | −2.2 pp  |

~89% of the empirical privacy comes from the BatchNorm → GroupNorm
architectural change Opacus requires; ~5% comes from the formal noise
mechanism. The (ε, δ) guarantee itself is mathematical and unaffected.

### Subgroup fairness on the 104-subject analysis cohort

Demographics for PhysioNet are recovered from the OpenNeuro ds004362 BIDS
sibling (the source EDFs anonymize them); see [`data/external/MAPPING_VERIFICATION.md`](data/external/MAPPING_VERIFICATION.md).

| Victim | Sex M (n=41) / F (n=56), Δ, p | Age low (n=34) / high (n=30), Δ, p | Decile gap |
|---|---|---|---|
| FBCSP+LDA  | 0.898 / 0.883, p = 0.67 | 0.935 / 0.854, p = 0.08 | +0.490 |
| Riemann    | 1.000 / 1.000, p = 1.00 | 1.000 / 1.000, p = 1.00 | 0 (ceiling) |
| EEGNet (single seed)   | +0.060, p = 0.288       | +0.128, **p = 0.044**     | +0.783 |
| EEGNet (5-seed Fisher) | Δ = +0.047 ± 0.021, **Fisher p = 0.298** | Δ = +0.093 ± 0.030, **Fisher p = 0.0083** | 0.778 ± 0.025 |

The 5-seed replication shows the EEGNet age effect is real but underpowered
per seed (only 1/5 individual seeds passes α = 0.05); Fisher's combined test
across the 5 seeds gives p = 0.0083, with consistent effect-size direction.
Sex effect is consistently null across all victims and seeds.

## Quickstart

```bash
git clone https://github.com/manrajmondair/bci-identity-leakage
cd bci-identity-leakage
python3.11 -m venv .venv && source .venv/bin/activate
pip install -U pip wheel setuptools
pip install -e ".[dev]"

python -m tools.audit                           # 76 invariants over current results
python -m tools.regenerate_figures              # rebuild every figure from results/*.json
```

GPU-bound experiments run on Colab L4. The `colab/` directory has one self-
contained notebook per experiment; each clones, installs, prefetches data,
runs the experiment, and returns a results JSON plus a run-metadata JSON
that becomes a row in the `runs/` audit trail. See
[`colab/README.md`](colab/README.md) for the full notebook index.

## Datasets

| Dataset | Role | Subjects | Channels | Sampling rate | Loader |
|---|---|---|---|---|---|
| PhysioNet EEG-MMIDB ([physionet.org/content/eegmmidb](https://www.physionet.org/content/eegmmidb/1.0.0/)) | Primary corpus: A1, A2, A4, A5, all defenses | 109 (drop {88, 89, 92, 100, 104}) → n = 104 | 64 (10-10 montage) | 160 Hz | [`data.physionet_loader`](data/physionet_loader.py) |
| BCI Competition IV-2a ([BNCI Horizon 2020](https://www.bbci.de/competition/iv/)) | Cross-session A3, cross-dataset A4 | 9 × 2 sessions | 22 (motor-cortex subset) | 250 Hz | [`data.bciiv2a_loader`](data/bciiv2a_loader.py) (via moabb) |

A scaled-up cross-session replication on the Lee 2019 OpenBMI corpus
(54 subjects × 2 sessions) was scaffolded as `experiments/20_a3_lee2019.py`
and `colab/A3_lee2019.ipynb`, but the full Colab run did not complete within
the project's compute budget owing to network throughput from the OpenBMI
mirror in Tokyo (≥ 4 h aggregate download for the 64 GB raw corpus). The
A3 result reported in this repository is therefore on IV-2a (n = 9), with
the small-N status documented in the report's limitations.

Demographic metadata for the PhysioNet cohort is sourced from the
OpenNeuro ds004362 BIDS conversion of the same recordings (Gender, Age,
Handedness; 95 of 109 subjects recovered). See
[`data/external/`](data/external/) for the participants TSV, the
duration-fingerprint mapping verification, and provenance documentation.

## Threat models

We benchmark five concrete attack protocols, motivated by three deployment
scenarios in the BCI / neurotechnology supply chain:

1. **Embedding-grabbing third party** — black-box or grey-box access to
   a deployed motor-imagery decoder; attempts to attribute new EEG windows
   to a member of an enrolled cohort. (A1 closed-set, A2 cross-task,
   A3 cross-session.)
2. **Biometric linker** — has two EEG datasets from the same individual
   collected on different days, recording protocols, or services; tests
   same-vs-different at template level. The trained embedding has not seen
   either recording. (A4 open-set verification.)
3. **Membership-inference adversary** — black-box access to a trained
   victim and a candidate dataset; asks whether each candidate was in the
   training cohort. Sensitive when cohort membership itself is private
   information (e.g., clinical-research cohort). (A5.)

Three defense families are benchmarked under both generic and adaptive
attackers:

- **D1 ad-hoc input transforms** — channel-mode PCA, additive
  Gaussian noise scaled by per-channel standard deviation, top-k
  variance-based channel-drop. The transform is fitted on training
  windows and applied before the victim sees the data.
- **D2 DANN adversarial subject-invariance** ([Ganin et al. 2016](https://jmlr.org/papers/v17/15-239.html))
  — EEGNet plus a 104-way subject-classification head connected via a
  Gradient Reversal Layer. The encoder is pressured to produce features
  useful for the motor-imagery task and useless for subject identification
  simultaneously. λ ∈ {0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0} swept.
- **D3 DP-SGD** ([Abadi et al. 2016](https://arxiv.org/abs/1607.00133)) via
  [Opacus](https://opacus.ai/) — per-sample gradient clipping with calibrated
  Gaussian noise, formal (ε, δ)-differential privacy. ε ∈ {None, 10, 3},
  δ = 10⁻⁵.

Adaptive-attacker stress tests (encoder fine-tune from the trained victim
weights, end-to-end backprop on subject-id labels) are run against every
defense family. See `experiments/15_d2_adaptive_attacker.py` (DANN),
`experiments/18_d3_adaptive_attacker.py` (DP-SGD), and
`experiments/23_d1_adaptive_attacker.py` (D1).

For full method definitions and citations, see [`docs/methods.md`](docs/methods.md).

## Repository layout

```
attacks/                   A1–A5 attack code
  closed_set.py            knn + logreg probes for A1/A2/A3
  verification.py          A4 open-set evaluation
  membership_inference.py  A5 shadow-model methodology
  per_subject.py           per-subject attack accuracy + decile gap

defenses/                  three defense families
  adhoc.py                 D1 ChannelPCA, ChannelGaussianNoise, ChannelDrop
  dann.py                  D2 DANN with Gradient Reversal Layer
  dp_sgd.py                D3 Opacus-wrapped EEGNet at target (ε, δ)

models/                    victim model implementations
  fbcsp.py                 9-band filter-bank Common Spatial Patterns + LDA
  riemannian.py            tangent-space + logistic regression on covariances
  eegnet.py                braindecode EEGNet wrapper (input_scale = 1e6)
  contrastive.py           ContrastiveEEGNet for A4 (batch-hard triplet loss)
  base.py                  shared VictimModel ABC

preprocess/                EEG preprocessing pipeline
  filtering.py             4–40 Hz bandpass
  windows.py               2 s sliding windows w/ 1 s stride; disk-cached

data/                      dataset loaders and channel utilities
  physionet_loader.py      PhysioNet EEGMMIDB via mne.datasets.eegbci
  bciiv2a_loader.py        BCI Competition IV-2a via moabb
  lee2019_loader.py        Lee 2019 OpenBMI MI via moabb (scaffolded)
  channel_subset.py        cross-dataset channel intersection
  prefetch_direct.py       parallel HTTP prefetch of PhysioNet EDFs
  external/                OpenNeuro participants TSV + mapping verification

eval/                      metric & plotting helpers
  bootstrap.py             1000-resample bootstrap CIs grouped by trial
  plots.py                 closed_set_bar_chart, verification_panel,
                           verification_summary_card, _setup_axes

experiments/               numbered, one-per-claim entry points (24 total)
  01_baseline_utility.py   victim task accuracy on imagery
  02_closed_set_reid.py    A1 baseline (104 subjects × 3 victims)
  03_within_subject_reid.py  A1b within-subject (deferred from milestone)
  04_a2_cross_task.py      A2 execution → imagery
  05_a3_cross_session.py   A3 IV-2a 9-subject cross-session
  06_a4_open_set.py        A4 contrastive embedding + verification
  07_d1_pca.py             D1 PCA-k sweep
  08_a5_membership_inference.py  A5 EEGNet shadow models
  09_d2_dann.py            D2 DANN λ sweep
  10_d3_dp_sgd.py          D3 DP-SGD ε sweep
  11_d1_adhoc.py           D1 noise + channel-drop
  13_a4_cross_dataset.py   A4 PhysioNet → IV-2a transfer
  14_a4_multi_seed.py      A4 5-seed robustness
  15_d2_adaptive_attacker.py  D2 DANN adaptive (3 attackers)
  16_a5_classical.py       A5 on FBCSP and Riemann
  17_subgroup_fairness_eegnet.py  EEGNet subgroup fairness (single seed)
  18_d3_adaptive_attacker.py  D3 DP-SGD adaptive (3 attackers)
  19_dp_sgd_arch_ablation.py  isolates BN→GN contribution from formal DP
  20_a3_lee2019.py         A3 cross-session on Lee 2019 (scaffolded)
  21_a2_vs_rest.py         A2 with resting-state probe (clean cross-task)
  22_eegnet_age_seeds.py   EEGNet age effect 5-seed replication
  23_d1_adaptive_attacker.py  D1 adaptive (3 defenses × encoder fine-tune)

tools/
  audit.py                 76 invariants over results/, run on every commit
  regenerate_figures.py    rebuild every figure from results/*.json
  pareto_plot.py           privacy-utility Pareto across all 3 victims
  subgroup_fairness.py     stratification + Mann-Whitney + bootstrap CIs

colab/                     one notebook per GPU-bound experiment (24 total)
results/                   canonical JSON outputs (one per experiment)
figures/                   publication-grade PDFs
runs/                      execution provenance + audit history
report/                    milestone.md + literature_comparison.md
docs/                      methods.md, limitations.md, reproducibility.md, figures.md
```

## Provenance and audit

Every reported number is reproducible from the canonical JSON in
`results/`, the experiment script in `experiments/`, and the run-trail
metadata in `runs/<run_id>/meta.json` (git SHA, hardware, runtime,
timestamp).

The audit script `tools/audit.py` runs 76 invariants on every commit
covering data integrity (window shapes, NaN, channel order, class
balance), train/test split correctness (closed-set and disjoint-subject
modes), probe methodology (bootstrap grouped by trial, fit/predict
separation, EEGNet input_scale fix), a shuffled-label negative control,
and effect-size sanity bounds against the published EEG-biometric
literature. The current commit returns **76 OK / 0 WARN / 0 FAIL**;
audit logs are persisted under `runs/*_audit_*/audit.md`.

## Reproducing the results

A single-command path from a fresh checkout to the canonical figure
set:

```bash
git clone https://github.com/manrajmondair/bci-identity-leakage
cd bci-identity-leakage
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m tools.audit                  # confirms results/ JSONs are self-consistent
python -m tools.regenerate_figures     # rebuilds every figure from the JSONs
```

To re-run the experiments themselves:

- **CPU-fast** (FBCSP, Riemann tangent-space; A1, A2 partial, D1, fairness):
  ```bash
  python -m data.prefetch_direct --runs imagery --workers 8
  python -m experiments.02_closed_set_reid --all          # ~50 min
  python -m tools.subgroup_fairness                       # ~12 min
  ```
- **GPU-bound** (anything that trains EEGNet from scratch, A4, A5, D2, D3,
  every adaptive-attacker experiment): see `colab/` for the L4 notebooks.

Detailed reproducibility notes — including hyperparameters, random seeds,
and the exact run sequence each headline number depends on — are in
[`docs/reproducibility.md`](docs/reproducibility.md).

## Limitations

A non-exhaustive list of caveats every reader should hold in mind:

- **Cohort sizes are modest.** PhysioNet's 104 valid subjects and IV-2a's
  9 subjects are small by modern biometric standards (fingerprint
  benchmarks exceed 10⁵ identities). Open-set claims at TUH-EEG scale
  (10⁴+ subjects) would strengthen the headline AUC = 0.925 result.
- **Cross-session is on n = 9.** A larger cross-session replication on
  Lee 2019 OpenBMI was scaffolded but did not finish in the project's
  compute budget; A3 in this repository is on IV-2a only.
- **Cognitive-task generality.** All claims scope to motor imagery (and
  resting-state, via experiment 21). Whether the same patterns hold for
  ERP-based, sleep, or seizure EEG is open.
- **Adaptive attacker is one realization.** Encoder fine-tune is *one*
  defender-aware attacker. Stronger attacks (gradient leakage against
  DP-SGD, membership-aware adaptive) are out of scope for this report.
- **DP-SGD evaluated at a single (ε, δ) point.** ε = 3 is in the loose
  privacy regime; we do not sweep ε ∈ [0.5, 1] in the strong regime
  where task accuracy is more pressured.
- **Demographic metadata is partial.** PhysioNet anonymizes its EDF
  headers; the OpenNeuro BIDS sibling re-publishes Gender, Age, and
  Handedness for 95 of 109 subjects, leaving 9 with unknown sex and
  13 with unknown age. IV-2a publishes only the cohort age range
  (22–30); per-subject ages are not public, so demographic
  stratification is not possible on IV-2a.

For the full discussion, see [`docs/limitations.md`](docs/limitations.md)
and the report's §6.

## Citations

This project builds on standard methods. Key references:

- Schalk, McFarland, Hinterberger, Birbaumer & Wolpaw (2004). *BCI2000:
  A General-Purpose Brain–Computer Interface (BCI) System.* IEEE TBME
  51(6):1034–1043. (PhysioNet EEGMMIDB.)
- Tangermann et al. (2012). *Review of the BCI Competition IV.* Frontiers
  in Neuroscience 6:55. (BCI IV-2a dataset.)
- Lawhern et al. (2018). *EEGNet: A Compact CNN for EEG-based BCIs.*
  Journal of Neural Engineering 15(5):056013. arXiv:1611.08024.
- Ang et al. (2008). *Filter Bank Common Spatial Pattern (FBCSP) in
  Brain-Computer Interface.* IJCNN.
- Barachant et al. (2012). *Multiclass Brain–Computer Interface
  Classification by Riemannian Geometry.* IEEE TBME 59(4):920–928.
- Schroff, Kalenichenko, Philbin (2015). *FaceNet: A Unified Embedding
  for Face Recognition and Clustering.* CVPR. (Triplet loss for A4.)
- Ganin et al. (2016). *Domain-Adversarial Training of Neural Networks.*
  JMLR 17(59):1–35. (D2 DANN.)
- Abadi et al. (2016). *Deep Learning with Differential Privacy.*
  ACM CCS. arXiv:1607.00133. (D3 DP-SGD.)
- Yousefpour et al. (2021). *Opacus: User-Friendly Differential Privacy
  Library in PyTorch.* arXiv:2109.12298.
- Shokri, Stronati, Song & Shmatikov (2017). *Membership Inference Attacks
  against Machine Learning Models.* IEEE S&P. arXiv:1610.05820.
- Carlini & Tramèr et al. (2019). *On Evaluating Adversarial Robustness.*
  arXiv:1902.06705. (Adaptive-attacker methodology.)
- Maciel, Maiorana & Campisi (2021). *A deep descriptor for cross-tasking
  EEG-based recognition.* Pattern Recognition Letters. PMC8157223.
- Maiorana & Campisi (2018). *Longitudinal Evaluation of EEG-Based
  Biometric Recognition.* IEEE TIFS 13(5):1123–1138.

For the per-experiment comparison against published prior work, see
[`report/literature_comparison.md`](report/literature_comparison.md).

## License

MIT for code in this repository. Datasets retain their original licenses:
PhysioNet EEGMMIDB is published under the Open Data Commons Attribution
license; BCI Competition IV-2a is published under CC-BY. The OpenNeuro
ds004362 conversion of the PhysioNet recordings used for demographic
metadata is published under CC0.
