# EEG-as-Biometric: Subject Re-Identification Leakage in BCI Models

**Author:** Manraj Singh Mondair · CS 281 (Ethics of AI), Stanford, Spring 2026

A reproducible, audited benchmark of how much subject-identifying
information leaks from machine-learning brain–computer interface
(BCI) models trained for motor-imagery decoding, and how four
families of defenses perform under both generic and adaptive
attackers. **Twenty-six numbered experiments** span five attack
protocols, three victim model families, four defense families,
**three EEG corpora**, demographic-stratified fairness, multi-seed
robustness, cross-dataset transfer in four directions, and matched
adaptive-attacker stress tests (encoder fine-tune, DP-aware MIA,
Fredrikson-style model inversion) against every defense. Every
reported metric carries a 1000-resample trial-grouped bootstrap CI;
every commit passes a **240-invariant audit**.

## Abstract

Motor-imagery brain–computer interfaces decode neural recordings into
control signals for assistive devices and consumer neurotechnology.
The same neural features used to perform that decoding may carry
stable, subject-specific patterns that allow re-identification of the
user from later recordings — i.e., EEG may function as a biometric
even when collected and labelled for a non-biometric purpose.

We test this hypothesis empirically across three independent
motor-imagery corpora: PhysioNet EEG-MMIDB (n=104), BCI Competition
IV-2a (n=9, two sessions on different days), and Lee 2019 OpenBMI
(n=54, two sessions on different days). The benchmark evaluates five
attack protocols (closed-set re-identification A1; cross-task A2;
cross-session A3; open-set verification on subjects held out from the
embedding's training set A4; black-box membership inference A5)
against three classes of victim model (FBCSP+LDA; Riemannian
tangent-space + LR; EEGNet), then re-runs the attack suite against
four defense families (D1 ad-hoc input transforms; D2 DANN adversarial
subject-invariant training; D3 DP-SGD with formal (ε, δ) guarantees;
D4 federated DP-FedAvg with central-aggregation noise) and three
stronger adaptive attackers (encoder fine-tune, DP-aware membership
inference with DP-trained shadows, Fredrikson model inversion).

**Headline findings.**

- Closed-set re-identification on PhysioNet reaches **100% top-1 on
  104 subjects** with a Riemannian tangent-space classifier whose task
  accuracy is 35%.
- A contrastive EEG embedding verifies the identity of recordings from
  **24 unseen PhysioNet subjects at AUC = 0.925** (multi-seed mean
  0.934 ± 0.020 over 5 random splits); the same claim **replicates on
  Lee 2019** at AUC = 0.920 on 14 unseen subjects within-session and
  AUC = 0.868 cross-session.
- A3 cross-session re-identification on Lee 2019's 54 subjects reaches
  **Riemann top-1 = 0.749** at chance 1.85% (40× lift), resolving the
  milestone-era weakness that cross-session held only at n = 9.
- Cross-dataset A4 transfer is **direction-dependent**: PhysioNet →
  Lee 2019 = 0.826 and IV-2a → PhysioNet = 0.831 succeed, but Lee
  2019 → PhysioNet collapses to 0.496. Experiment 33 falsifies-or-
  confirms the task-complexity hypothesis for the asymmetry.
- A full DP-SGD ε sweep over {0.5, 1, 3, 10, ∞} shows the
  **strong-privacy regime ε ≤ 1** blocks the encoder fine-tune
  attacker to ≤ 7% top-1 at ~6 pp task cost; **ε = 10 leaks more
  than no defense** (0.189 vs 0.153).
- **DP-aware MIA** with shadows DP-trained at the target's ε defeats
  DP-SGD at ε = 3: MI AUC = 0.891, statistically indistinguishable
  from the undefended 0.878. The defense story is attack-specific.
- **Federated DP-FedAvg** with 104 clients holds the fine-tune attacker
  to top-1 = 0.096, within 4 pp of centralised DP-SGD, without ever
  pooling raw EEG.
- Fredrikson-style model inversion is a **null result for both
  defended and undefended re-ID heads**.
- EEGNet's empirical re-ID scaling exponent is **γ = 0.474**: the
  biometric threat decays sub-linearly with cohort size.

The empirical case for treating EEG as biometric data under GDPR
Article 9 / neurorights frameworks rests on the unseen-subject A4
result (now corroborated on two corpora), the persistence of identity
signal across cognitive tasks (Riemann recovers 94.1% from
resting-state EEG alone, with no shared task structure with motor
imagery), and the cross-session results on both IV-2a and Lee 2019.
The deployable-privacy lesson is that only mechanism-level
differential privacy survives a defender-aware attacker, that the
empirical defense story depends on which attack you measure against,
and that federated DP-FedAvg provides a viable on-device alternative
without pooling raw EEG.

## Headline numbers

### Attacks (1000-resample trial-grouped bootstrap CIs)

| Attack | Dataset | N | Best victim | Metric (95% CI) | vs. chance |
|---|---|---|---|---|---|
| A1 closed-set | PhysioNet | 104 | Riemann tangent-space | top-1 = **1.000** [1.000, 1.000] | 104× |
| A2 cross-task (execution → imagery) | PhysioNet | 104 | Riemann tangent-space | top-1 = **1.000** [0.999, 1.000] | 104× |
| A2 cross-task (resting-state → imagery) | PhysioNet | 104 | Riemann tangent-space | top-1 = **0.941** [0.933, 0.949] | 98× |
| A3 cross-session | BCI IV-2a | 9 | Riemann tangent-space | top-1 = **0.913** [0.902, 0.923] | 8.2× |
| **A3 cross-session** | **Lee 2019** | **54** | Riemann tangent-space | top-1 = **0.749** [0.741, 0.757] | **40×** |
| A4 open-set verification (unseen) | PhysioNet | 24 unseen | contrastive EEGNet | AUC = **0.925** [0.923, 0.928] | 0.500 |
| A4 multi-seed (5 splits × 24 unseen) | PhysioNet | 24 × 5 | contrastive EEGNet | AUC = **0.934 ± 0.020** | 0.500 |
| **A4 open-set verification** | **Lee 2019** | **14 unseen** | contrastive EEGNet | AUC = **0.920** [0.918, 0.922] | 0.500 |
| **A4 cross-session unseen** | **Lee 2019** | **14 unseen × 2 days** | contrastive EEGNet | AUC = **0.868** [0.865, 0.871] | 0.500 |
| A4 cross-dataset PhysioNet → IV-2a | IV-2a | 9 unseen | contrastive EEGNet | AUC = 0.694 [0.690, 0.699] | 0.500 |
| A4 cross-dataset IV-2a → PhysioNet | PhysioNet | 104 unseen | contrastive EEGNet | AUC = **0.831** [0.827, 0.834] | 0.500 |
| A4 cross-dataset PhysioNet → Lee 2019 | Lee 2019 | 54 unseen | contrastive EEGNet | AUC = **0.826** [0.822, 0.829] | 0.500 |
| A4 cross-dataset Lee 2019 → PhysioNet | PhysioNet | 104 unseen | contrastive EEGNet | AUC = **0.496** [0.492, 0.500] | 0.500 |
| A4 cross-dataset IV-2a → Lee 2019 | Lee 2019 | 54 unseen | contrastive EEGNet | AUC = 0.673 [0.668, 0.677] | 0.500 |
| A5 membership inference | PhysioNet | 104 | EEGNet shadow models | AUC = **0.878** [0.803, 0.943] | 0.500 |
| A5 membership inference | PhysioNet | 104 | Riemann tangent-space | AUC = **1.000** [1.000, 1.000] | 0.500 |
| **A5 membership inference** | **Lee 2019** | **54** | EEGNet shadow models | AUC = **0.787** [0.658, 0.896] | 0.500 |

### Defenses on PhysioNet (104 subjects, EEGNet victim)

| Defense family | Generic logreg attacker | Adaptive (encoder fine-tune) | Verdict |
|---|---|---|---|
| no defense (A1 baseline, AdamW+BN)             | 0.411 | — | — |
| D1 PCA k=8                                     | 0.357 | 0.660 | collapses, exceeds baseline |
| D1 noise σ=1.0                                 | 0.180 | 0.640 | collapses, exceeds baseline |
| D1 channel-drop k=8                            | 0.391 | 0.758 | collapses, exceeds baseline |
| D2 DANN λ=0.2                                  | 0.252 | 0.804 | collapses, exceeds baseline |
| **D3 DP-SGD ε=0.5** (final ε=0.495)            | **0.027** | **0.043** | strong-privacy regime |
| **D3 DP-SGD ε=1.0** (final ε=0.998)            | **0.025** | **0.070** | strong-privacy regime |
| D3 DP-SGD ε=3.0 (final ε=2.996)                | 0.030 | 0.136 | holds against fine-tune |
| D3 DP-SGD ε=10.0 (final ε=9.992)               | 0.033 | 0.189 | leaks more than no defense |
| D3 DP-SGD no DP (SGD + GroupNorm only)         | 0.032 | 0.153 | architectural baseline |
| **D4 federated DP-FedAvg**                     | **0.044** | **0.096** | holds w/o pooling raw EEG |

### Stronger adaptive attackers against DP-SGD ε=3

| Attacker | MI AUC / re-ID metric | vs no-defense baseline |
|---|---|---|
| Generic logreg probe (re-ID, frozen encoder)    | top-1 = 0.030 | well below 0.411 |
| Encoder fine-tune (re-ID, AdamW lr=5e-4 × 15ep) | top-1 = 0.136 | well below 0.411 |
| **DP-aware MIA** (DP-trained shadows + target)  | **AUC = 0.891** [0.826, 0.943] | ≈ undefended 0.878 |
| Fredrikson model inversion                      | rank-1 = 0.00, rank-5 = 0.10 | null result for both arms |

DP-SGD at ε=3 holds well against re-ID fine-tune; it does **not** hold against DP-aware MIA. The Yeom (2018) MI-advantage upper bound at ε=3 is 0.95, so the formal guarantee is not violated — but the empirical defense story splits by attack type. ε ≤ 1 is the predicted deployable point for MI protection (Yeom bound at ε=1 is 0.63).

### D3 architecture vs noise breakdown

| Configuration | A1 top-1 (logreg) | Δ vs. previous |
|---|---|---|
| AdamW + BatchNorm (vanilla A1) | 0.411 | (baseline) |
| SGD + GroupNorm (no DP)        | 0.044 | −36.7 pp |
| SGD + GroupNorm + DP-SGD ε=3   | 0.022 | −2.2 pp  |

~89% of D3's empirical privacy comes from the BatchNorm → GroupNorm
architectural change Opacus requires; ~5% comes from the formal noise
mechanism. The (ε, δ) guarantee itself is mathematical and unaffected.

### Theoretical scaling

| N | EEGNet top-1 | Riemann top-1 |
|---|---|---|
| 10  | 0.833 | 1.000 |
| 20  | 0.702 | 1.000 |
| 40  | 0.554 | 1.000 |
| 60  | 0.523 | 1.000 |
| 80  | 0.446 | 0.9996 |
| 104 | 0.410 | 1.000 |

Least-squares fit on `log(1 − top1) vs log(N)`: **EEGNet γ = 0.474** (r² = 0.956). The biometric threat decays sub-linearly with cohort size; at N = 104, EEGNet still identifies 41% of users (43× chance). Riemann saturates at ceiling across every tested N; its decay regime is beyond the tested cohort scale.

### Yeom (2018) (ε, δ)-MI-advantage bound vs empirical fine-tune top-1

| ε | Yeom bound `1 − e^(−ε) − δ` | Empirical fine-tune top-1 | Gap |
|---|---|---|---|
| 0.5 | 0.391 | 0.043 | 0.347 |
| 1.0 | 0.631 | 0.070 | 0.561 |
| 3.0 | 0.950 | 0.136 | 0.814 |
| 10.0 | ~1.000 | 0.189 | 0.811 |
| ∞ | 1.000 | 0.153 | 0.847 |

Empirical re-ID protection from DP-SGD is materially stronger than the Yeom bound certifies for every ε ≥ 1; the bound starts to bind near ε ≈ 0.5, where it matches the AdamW+BN no-defense baseline of 0.411.

### Within-cohort heterogeneity (Lee 2019, 54 subjects, within-session)

| Victim | Task acc | Mean A1 | Decile gap | Min – Max |
|---|---|---|---|---|
| FBCSP+LDA | 60.6% | 0.803 | 0.317 | 0.498 – 0.966 |
| Riemann tangent-space | 67.3% | 0.999 | 0.000 (ceiling) | 0.987 – 1.000 |
| EEGNet | 71.6% | 0.480 | 0.490 | 0.148 – 0.917 |

### Subgroup fairness on the 104-subject PhysioNet analysis cohort

Demographics for PhysioNet are recovered from the OpenNeuro ds004362
BIDS sibling (the source EDFs anonymize them); see
[`data/external/MAPPING_VERIFICATION.md`](data/external/MAPPING_VERIFICATION.md).
Lee 2019 publishes only cohort aggregates (25F/29M, ages 24–35) so
per-subject demographic stratification on Lee 2019 is not possible; we
report within-cohort heterogeneity instead (above).

| Victim | Sex M / F, Δ, p | Age low / high, Δ, p | Decile gap |
|---|---|---|---|
| FBCSP+LDA | 0.898 / 0.883, p = 0.67 | 0.935 / 0.854, p = 0.08 | +0.490 |
| Riemann | 1.000 / 1.000, p = 1.00 | 1.000 / 1.000, p = 1.00 | 0 (ceiling) |
| EEGNet (single seed) | +0.060, p = 0.288 | +0.128, **p = 0.044** | +0.783 |
| EEGNet (5-seed Fisher) | Δ = +0.047 ± 0.021, **Fisher p = 0.298** | Δ = +0.093 ± 0.030, **Fisher p = 0.0083** | 0.778 ± 0.025 |

The 5-seed EEGNet age effect is consistent in direction and magnitude
(Δ ≈ 9 pp), with Fisher combined p = 0.0083 across seeds; sex effect
is consistently null across all victims and seeds. See
[`docs/fairness_audit.md`](docs/fairness_audit.md) for the audit of
which demographic axes each dataset publishes.

## Quickstart

```bash
git clone https://github.com/manrajmondair/bci-identity-leakage
cd bci-identity-leakage
python3.11 -m venv .venv && source .venv/bin/activate
pip install -U pip wheel setuptools
pip install -e ".[dev]"

python -m tools.audit                           # 240 invariants over current results
python -m tools.regenerate_figures              # rebuild every figure from results/*.json
```

GPU-bound experiments run on Colab L4 or A100. The `colab/` directory
has one self-contained notebook per experiment; each clones, installs
deps, prefetches data, runs the experiment, and prints the result JSON
between paste-back markers. See [`colab/README.md`](colab/README.md) for
the full notebook index and per-notebook recommended hardware.

## Datasets

| Dataset | Role in benchmark | Subjects | Channels | Sampling rate | Loader |
|---|---|---|---|---|---|
| PhysioNet EEG-MMIDB ([physionet.org/content/eegmmidb](https://www.physionet.org/content/eegmmidb/1.0.0/)) | Primary corpus: A1, A2, A4, A5, all defenses | 109 (drop {88, 89, 92, 100, 104}) → n = 104 | 64 (10-10 montage) | 160 Hz | [`data.physionet_loader`](data/physionet_loader.py) |
| BCI Competition IV-2a ([BNCI Horizon 2020](https://www.bbci.de/competition/iv/)) | Cross-session A3, cross-dataset A4 | 9 × 2 sessions | 22 (motor-cortex subset) | 250 Hz | [`data.bciiv2a_loader`](data/bciiv2a_loader.py) (via moabb) |
| Lee 2019 OpenBMI ([GigaScience 8(5):giz002](https://doi.org/10.1093/gigascience/giz002)) | Second corpus: A3, A4, A5, fairness, cross-dataset A4 | 54 × 2 sessions | 62 (10-10 montage) | 1000 Hz native, 250 Hz cached | [`data.lee2019_loader`](data/lee2019_loader.py) + [`data.lee2019_prefetch`](data/lee2019_prefetch.py) |

Lee 2019 is ingested via a stream-and-compact prefetcher: each
(subject, session) .mat is downloaded in eight parallel HTTP Range
chunks from Wasabi Tokyo, bandpassed, windowed, and written as a
compact float16 .npz to the path set by `BCI_LEE2019_CACHE`. Total
Drive footprint after a full run is ~4 GB rather than ~65 GB.

Demographic metadata for the PhysioNet cohort is sourced from the
OpenNeuro ds004362 BIDS conversion of the same recordings (Gender,
Age, Handedness; 95 of 109 subjects recovered). See
[`data/external/`](data/external/) for the participants TSV, the
duration-fingerprint mapping verification, and provenance documentation.

## Threat models

Five concrete attack protocols, motivated by three deployment
scenarios in the BCI / neurotechnology supply chain:

1. **Embedding-grabbing third party** — black-box or grey-box access
   to a deployed motor-imagery decoder; attempts to attribute new EEG
   windows to a member of an enrolled cohort. (A1 closed-set, A2
   cross-task, A3 cross-session.)
2. **Biometric linker** — has two EEG datasets from the same
   individual collected on different days, recording protocols, or
   services; tests same-vs-different at template level. The trained
   embedding has not seen either recording. (A4 open-set verification.)
3. **Membership-inference adversary** — black-box access to a trained
   victim and a candidate dataset; asks whether each candidate was in
   the training cohort. Sensitive when cohort membership itself is
   private information (e.g., clinical-research cohort). (A5.)

Four defense families benchmarked under both generic and adaptive
attackers:

- **D1 ad-hoc input transforms** — channel-mode PCA, additive
  Gaussian noise scaled by per-channel standard deviation, top-k
  variance-based channel-drop.
- **D2 DANN adversarial subject-invariance** ([Ganin et al. 2016](https://jmlr.org/papers/v17/15-239.html))
  — EEGNet + a 104-way subject-classification head via a Gradient
  Reversal Layer. λ ∈ {0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0} swept.
- **D3 DP-SGD** ([Abadi et al. 2016](https://arxiv.org/abs/1607.00133))
  via [Opacus](https://opacus.ai/) — per-sample gradient clipping with
  calibrated Gaussian noise, formal (ε, δ)-differential privacy.
  ε ∈ {0.5, 1.0, 3.0, 10.0, ∞} swept under both generic and adaptive
  attackers (experiment 29).
- **D4 federated DP-FedAvg** ([Geyer et al. 2017](https://arxiv.org/abs/1712.07557);
  [McMahan et al. 2017](https://proceedings.mlr.press/v54/mcmahan17a.html))
  — one client per subject, server clips + adds noise per round;
  participant-level (ε, δ) accounted by Opacus's RDP accountant
  (experiment 31).

Five adaptive-attacker realisations are run against the defenses:

1. Generic logreg probe on the frozen encoder.
2. Deep-MLP probe on the frozen encoder.
3. **Encoder fine-tune** (Carlini–Tramèr 2019 style; experiments 15,
   18, 23, 29, 31).
4. **DP-aware MIA** with shadows DP-trained at the target's ε
   (experiment 27).
5. **Fredrikson-style model inversion** scored in a held-out
   contrastive reference embedder (experiment 28).

For full method definitions and citations, see [`docs/methods.md`](docs/methods.md).

## Repository layout

```
attacks/                     A1–A5 attack code + adaptive variants
defenses/                    D1 ad-hoc, D2 DANN, D3 DP-SGD, D4 federated DP-FedAvg
models/                      FBCSP+LDA, Riemann tangent-space + LR, EEGNet, contrastive EEGNet
preprocess/                  4–40 Hz bandpass, 2-s sliding windowing, disk-cached
data/                        PhysioNet / IV-2a / Lee 2019 loaders + parallel prefetchers
eval/                        trial-grouped bootstrap CIs, plotting helpers
experiments/                 26 numbered, one-per-claim entry points (see below)
tools/                       audit (240 invariants), figure regeneration, Pareto, fairness
colab/                       one notebook per experiment (37 total)
results/                     canonical JSON outputs
figures/                     publication-grade PDFs (regenerated from JSON)
runs/                        per-experiment execution provenance + audit history
docs/                        methods, limitations, reproducibility, theory, fairness audit, figures
report/                      report-bound notes and the comparison-to-literature document
```

The 26 experiments are organised as follows. Milestone-era
(experiments 01–22 plus 23): the original A1–A5 attack suite, the D1
/ D2 / D3 defense families with three-attacker adaptive evaluation,
within-PhysioNet cross-dataset transfer, multi-seed open-set
verification, fairness stratification. Tier 1 + Tier 2 extensions
(experiments 20, 24–34): Lee 2019 second-corpus replication across
A3/A4/A5/fairness, symmetric cross-dataset A4 in four directions,
asymmetry-mechanism falsifier, DP-SGD full ε sweep, DP-aware MIA,
Fredrikson model inversion, theoretical γ-scaling fit with Yeom
overlay, federated DP-FedAvg, and a multi-seed wrapper.

## Provenance and audit

Every reported number is reproducible from the canonical JSON in
`results/`, the experiment script in `experiments/`, and the run-trail
metadata in `runs/<run_id>/meta.json` (git SHA, hardware, runtime,
timestamp).

The audit script `tools/audit.py` runs **240 invariants** on every
commit covering data integrity (window shapes, NaN, channel order,
class balance), train/test split correctness (closed-set and
disjoint-subject modes), probe methodology (bootstrap grouped by
trial, fit/predict separation, EEGNet input_scale fix), a
shuffled-label negative control, effect-size sanity bounds against
the published EEG-biometric literature, and Tier 1+2 specifics
(cross-dataset common-channel intersection, DP-SGD final-ε
consistency with target, Yeom bound holds empirically, multi-seed
aggregation has 3+ seeds per metric). The current commit returns
**240 OK / 0 WARN / 0 FAIL**; audit logs are persisted under
`runs/*_audit_*/audit.md`.

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

- **CPU-fast** (FBCSP, Riemann; A1, A2 partial, D1, fairness):
  ```bash
  python -m data.prefetch_direct --runs imagery --workers 8
  python -m experiments.02_closed_set_reid --all          # ~50 min
  python -m tools.subgroup_fairness                       # ~12 min
  ```
- **GPU-bound** (anything that trains EEGNet from scratch, A4, A5,
  D2, D3, D4, every adaptive-attacker experiment, all Lee 2019
  experiments): see [`colab/`](colab/) for the L4 / A100 notebooks
  and `BCI_LEE2019_CACHE` for the Drive cache path.

Detailed reproducibility notes — including hyperparameters, random
seeds, and the exact run sequence each headline number depends on —
are in [`docs/reproducibility.md`](docs/reproducibility.md).

## Limitations

A non-exhaustive list of caveats every reader should hold in mind:

- **Cohort sizes are modest.** PhysioNet's 104 valid subjects + Lee
  2019's 54 + IV-2a's 9 are small by modern biometric standards
  (fingerprint benchmarks exceed 10⁵ identities). Open-set claims at
  TUH-EEG scale (10⁴+ subjects) would strengthen the headline AUC =
  0.925 / 0.920 results.
- **DP-aware MIA was tested at ε = 3 in this revision.** ε ≤ 1
  experiments are predicted (Yeom bound at ε = 1 is 0.63) and the
  matching notebook is wired (`colab/D3_membership_aware_eps_sweep.ipynb`)
  but the empirical numbers at ε ∈ {0.5, 1.0} land in the next batch.
- **Cognitive-task generality.** All claims scope to motor imagery
  (and resting-state, via experiment 21). Whether the same patterns
  hold for ERP-based, sleep, or seizure EEG is open.
- **Federated ε is loose at the chosen configuration.** RDP-accounted
  participant-level ε at σ=0.4, q=0.5, 30 rounds is 97.7; the
  empirical fine-tune protection is the substantive deliverable, and
  tightening the formal budget requires either larger σ (task cost)
  or fewer rounds (utility cost).
- **Adaptive attacker is one realization per defense.** Encoder
  fine-tune, DP-aware MIA, and model inversion are three concrete
  realisations of "the attacker knows the defense." Stronger attacks
  (DLG gradient leakage against DP-SGD, membership-aware adaptive on
  federated, full gradient-attribute inversion) remain out of scope.
- **Demographic metadata is partial.** PhysioNet anonymises its EDF
  headers; the OpenNeuro BIDS sibling re-publishes Gender, Age, and
  Handedness for 95 of 109 subjects, leaving 9 with unknown sex and
  13 with unknown age. Lee 2019 publishes only cohort aggregates; no
  motor-imagery EEG dataset we use publishes race / ethnicity.

For the full discussion, see [`docs/limitations.md`](docs/limitations.md)
and [`docs/fairness_audit.md`](docs/fairness_audit.md).

## Citations

This project builds on standard methods. Key references:

- Schalk, McFarland, Hinterberger, Birbaumer & Wolpaw (2004). *BCI2000:
  A General-Purpose Brain–Computer Interface (BCI) System.* IEEE TBME
  51(6):1034–1043.
- Brunner et al. (2008). *BCI Competition 2008 — Graz data set A.*
  Graz University of Technology technical report.
- Lee, Kwon, Kim, Kim, Lee, Williamson, Fazli, Lee (2019). *EEG
  dataset and OpenBMI toolbox for three BCI paradigms.* GigaScience
  8(5):giz002.
- Lawhern et al. (2018). *EEGNet: A Compact CNN for EEG-based BCIs.*
  Journal of Neural Engineering 15(5):056013. arXiv:1611.08024.
- Ang et al. (2008). *Filter Bank Common Spatial Pattern (FBCSP) in
  Brain-Computer Interface.* IJCNN.
- Barachant et al. (2012). *Multiclass Brain–Computer Interface
  Classification by Riemannian Geometry.* IEEE TBME 59(4):920–928.
- Schroff, Kalenichenko, Philbin (2015). *FaceNet: A Unified Embedding
  for Face Recognition and Clustering.* CVPR.
- Ganin et al. (2016). *Domain-Adversarial Training of Neural
  Networks.* JMLR 17(59):1–35.
- Abadi et al. (2016). *Deep Learning with Differential Privacy.* ACM
  CCS. arXiv:1607.00133.
- Yousefpour et al. (2021). *Opacus: User-Friendly Differential
  Privacy Library in PyTorch.* arXiv:2109.12298.
- Geyer, Klein, Nabi (2017). *Differentially Private Federated
  Learning: A Client Level Perspective.* arXiv:1712.07557.
- McMahan, Moore, Ramage, Hampson, y Arcas (2017).
  *Communication-Efficient Learning of Deep Networks from
  Decentralized Data.* AISTATS.
- Shokri, Stronati, Song & Shmatikov (2017). *Membership Inference
  Attacks against Machine Learning Models.* IEEE S&P.
  arXiv:1610.05820.
- Yeom, Giacomelli, Fredrikson, Jha (2018). *Privacy Risk in Machine
  Learning: Analyzing the Connection to Overfitting.* IEEE CSF.
- Fredrikson, Jha, Ristenpart (2015). *Model Inversion Attacks That
  Exploit Confidence Information.* ACM CCS.
- Carlini & Tramèr et al. (2019). *On Evaluating Adversarial
  Robustness.* arXiv:1902.06705.
- Maciel, Maiorana & Campisi (2021). *A deep descriptor for
  cross-tasking EEG-based recognition.* Pattern Recognition Letters.
  PMC8157223.
- Maiorana & Campisi (2018). *Longitudinal Evaluation of EEG-Based
  Biometric Recognition.* IEEE TIFS 13(5):1123–1138.
- Wang et al. (2022). *M3CV: A multi-subject, multi-session, and
  multi-task database for EEG-based biometrics challenge.*
  NeuroImage 264:119666.

For the per-experiment comparison against published prior work, see
[`report/literature_comparison.md`](report/literature_comparison.md).

## License

MIT for code in this repository. Datasets retain their original
licenses: PhysioNet EEGMMIDB is published under the Open Data Commons
Attribution License; BCI Competition IV-2a is published under CC-BY;
Lee 2019 OpenBMI is published under CC-BY 4.0. The OpenNeuro ds004362
conversion of the PhysioNet recordings used for demographic metadata
is published under CC0.
