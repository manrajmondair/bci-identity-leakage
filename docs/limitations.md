# Limitations

Honest accounting of what this benchmark *cannot* tell us, organized by
the part of the claim each limitation affects.

---

## On the privacy claim

### Cohort scale

PhysioNet's 109 subjects and BCI IV-2a's 9 subjects are small by modern biometric benchmarks. Fingerprint datasets routinely exceed 10⁵ identities; iris benchmarks similar; even voice biometrics commonly evaluate on 10⁴+. The TUH EEG Corpus has ~14,000 subjects but requires institutional access not available for this project. **Implication:** the open-set verification claim (A4: AUC 0.925 on 24 unseen subjects) is solid for *this* cohort scale; whether the same template-quality holds at 10⁴ unseen subjects is an open question. A single subject's EEG identifying them in a cohort of 10⁴ is empirically much harder than in a cohort of 24.

### Cognitive task generality

All claims are scoped to *motor-imagery* EEG (left/right hand, both fists, both feet). Whether the same biometric structure exists in resting-state, ERP-based, sleep, or other cognitive-task EEG is not tested here. Prior work (Maiorana 2016, Yang & Deravi 2017) suggests EEG biometric properties are reasonably task-agnostic, but our specific numbers are motor-imagery numbers.

### Recording rig generality

PhysioNet uses the BCI2000 64-channel system at 160 Hz; IV-2a uses the Graz 22-channel system at 250 Hz. The cross-dataset A4 result tests whether claims transfer between these specific rigs after channel-subsetting and resampling. Whether claims transfer to consumer-grade headsets (e.g., Muse 4-channel, OpenBCI 8-channel) at lower SNR is not tested.

### Adaptive attackers (now addressed)

The original A1–A5 attacks used *generic* probes (kNN cosine, L2 logreg,
2-feature MLP). All three defense families have since been re-evaluated
against an end-to-end encoder-fine-tune adaptive attacker: D2 in
`experiments/15`, D3 in `experiments/18`, and D1 in `experiments/23`.
Headline result: only formal DP (D3) holds; DANN and every D1 point
collapse *above* the no-defense A1 baseline of 0.411. Encoder fine-tune
is, however, only one realization of "the attacker knows the defense";
stronger attacks (gradient-leakage attacks against DP-SGD, membership-
aware adaptive attacks) remain out of scope.

---

## On the fairness claim

### Demographics on PhysioNet

The PhysioNet release of EEG-MMIDB anonymizes subject metadata (`his_id: X, sex: 0` in every EDF header). We recover Gender + Age + Handedness for 95 of 109 subjects from OpenNeuro's BIDS conversion (ds004362), leaving:

- **7 subjects with unknown sex** in our 104-subject analysis cohort
- **13 subjects with unknown age**
- **1 transcription error** (`sub-044` has Gender=n/a but Age=`'M'`)

We drop unknown-sex subjects from sex stratification and unknown-age subjects from age tertiles; this is honest but it does shrink the effective sample size for those tests (from 104 to 97 sex-known and 91 age-known).

### Demographics on BCI IV-2a

Brunner et al. (2008) publishes only the cohort age range (22–30) and handedness count (8 R / 1 L), not per-subject demographics. **Implication:** demographic stratification on the cross-session A3 numbers is *not possible*. The IV-2a cohort is too narrow age-wise (all 22–30) to support stratification anyway.

### Demographic axes we don't measure

- **Race / ethnicity** is not in either dataset. EEG signal characteristics may correlate with skull conductivity, which has been linked to ancestry — this is a fairness axis we genuinely cannot audit.
- **Socioeconomic status / handedness / education** — handedness is in OpenNeuro but with sparse coverage; we don't stratify on it.
- **Mental health / neurological conditions** — both datasets are "healthy adults"; clinical populations may behave very differently and that's not tested.

### Statistical power

With 97 sex-known subjects and a Mann-Whitney U test, we can reliably detect effect sizes of about d=0.4 at α=0.05. Smaller effects (e.g., a 5-pp sex difference) might be present but invisible to our test. We report exact p-values rather than reject/accept verdicts to be honest about this.

---

## On the defense claims

### EEGNet hyperparameter sensitivity

The vanilla cross-subject EEGNet result in A1 (task 38.8%, leak 41.1%) depends critically on the `input_scale=1e6` correction. Without it, gradients vanish and EEGNet learns nothing (chance task acc, chance re-ID). This was discovered and patched mid-project — the implication is that the "EEGNet doesn't leak" framing one might extract from a *broken* EEGNet is misleading. Other architectural choices we didn't sweep (optimizer, schedule, regularization) may produce different absolute numbers.

### D3 DP-SGD architectural confound (now isolated)

Opacus requires removing braindecode's `nn.utils.parametrize` (max-norm
constraint on the spatial conv) and replacing BatchNorm with GroupNorm.
Experiment 19 (`experiments/19_dp_sgd_arch_ablation.py`) trains
EEGNet with the same architectural surgery and the same SGD optimizer
DP-SGD uses, but with the PrivacyEngine bypassed. The decomposition is
direct:

- AdamW + BatchNorm (vanilla A1):           0.411 top-1.
- SGD + GroupNorm, no DP (experiment 19):   0.044 top-1 (−36.7 pp).
- SGD + GroupNorm + DP-SGD ε=3:             0.022 top-1 (−2.2 pp).

**~89% of D3's empirical privacy comes from the architecture/optimizer
change; ~5% from the formal noise mechanism.** This does NOT weaken
the formal (ε, δ) guarantee — that bound is mathematical and
attacker-agnostic regardless of which component contributes what
fraction of the empirical privacy. The right read: GroupNorm-EEGNet
is itself a much weaker EEG biometric than BatchNorm-EEGNet; DP-SGD
adds a small additional empirical margin and the formal mathematical
guarantee.

### Single-seed experiments (mostly addressed)

Most experiments report a single seed-0 run. Two extensions provide
robustness CIs:

- `experiments/14_a4_multi_seed.py` reports A4 across 5 random subject
  splits (AUC = 0.934 ± 0.020).
- `experiments/22_eegnet_age_seeds.py` reports the EEGNet subgroup
  fairness pipeline across 5 seeds. The age-effect p-value is unstable
  per seed (0.028 to 0.353; 1/5 below α=0.05) but the effect size is
  consistent (Δ = +0.093 ± 0.030); Fisher's combined p across the 5
  seeds is 0.0083.

Other experiments still report a single seed; bootstrap CIs are tight
but cross-seed sensitivity is not formally verified for them.

### DANN at the chosen λ values

DANN's privacy-utility curve has 8 λ points; the sweet spot at λ=0.2 dominates them all on the EEGNet-victim Pareto. We did not exhaustively sweep λ in (0.15, 0.25); a finer grid in that region might find an even better point. Similarly, λ × n_epochs interactions are not explored.

---

## On the cross-dataset claim (A4 cross-dataset)

### Channel intersection

PhysioNet's 64 channels and IV-2a's 22 channels happen to share an exact 22-channel subset — IV-2a is a strict subset of PhysioNet's montage. This is *fortunate* but it means our cross-dataset claim is conditioned on the channel intersection happening to be empty of IV-2a-only channels. A dataset with channels not present in PhysioNet (e.g., one with frontal AF7/AF8 that IV-2a lacks) would require a different cross-dataset experiment (channel imputation or cross-channel transfer).

### Sampling rate

We resample IV-2a from 250 Hz to 160 Hz to match PhysioNet. This is standard but introduces a small spectral artifact in the resampled IV-2a. We don't think it materially affects the result but didn't ablate.

---

## On reproducibility

### Stochastic GPU ops

PyTorch on CUDA is not bit-exact across runs even with seeded RNGs (cuDNN nondeterminism). Re-running our notebooks should produce numbers within ~0.5 percentage points of those reported but exact bit-equality is not guaranteed.

### Software pin

`pyproject.toml` pins `torch==2.5.1` and `torchaudio==2.5.1` for ABI compatibility. Major-version drift in mne, braindecode, or pyriemann could change behavior subtly. The audit script catches major regressions; subtle regressions might slip through.

### Hardware drift

A1 was originally run on Apple Silicon MPS for the EEGNet baseline (A1 cross-subject); subsequent runs are on Colab L4. The numbers are very similar across both, but cross-hardware nondeterminism is real.

---

## Out of scope (could but didn't)

- **Federated DP setup** where each user's data never leaves their device. The threat model would shift from "attacker reads the model" to "attacker reads federated updates." Different paper.
- **Transfer between EEG and other modalities** (e.g., does an attacker who has fNIRS or fMRI on the same subjects transfer to EEG?). Impossible to test — no published joint-modality cohort at this scale.
- **Continuous biometric authentication** ("does this user *stay* identified second-by-second during a BCI session?") rather than the snapshot threat we benchmark.
- **Adversarial-perturbation attacks** on the trained victims (do small EEG perturbations cause misclassification?). Different threat model from re-identifiability.
- **Theoretical bounds.** We report empirical numbers; we don't derive a theoretical bound on contrastive-embedding leakage as a function of training cohort size.

---

## A3 cross-session: small N

The cross-session re-identification result is on BCI IV-2a only, n=9.
A scaled-up replication on Lee 2019 OpenBMI (54 subjects × 2 sessions)
was scaffolded as `experiments/20_a3_lee2019.py` and
`colab/A3_lee2019.ipynb`, but the full Colab run did not complete in
the project's compute budget — the OpenBMI Tokyo S3 mirror serves at
~3 MB/s, and downloading the 64 GB raw corpus exceeded a single Colab
session's wall budget. The A3 result reported here is therefore on
n=9; cross-seed bootstrap CIs are wide. The structural finding
(≥80% top-1 cross-session re-ID against chance 1/9) replicates the
direction of published prior work but should not be cited as if it
were a population-scale claim.

---

## What this benchmark *does* establish

Despite the limitations above, the empirical record on this scope is
consistent and audit-clean:

1. Subject identity is recoverable from a deployed motor-imagery
   decoder's features at 41–100% top-1 across 104 subjects (A1),
   with the strongest classical pipeline (Riemann tangent-space)
   reaching ceiling.
2. The recovery survives task changes (A2 with motor-execution probe;
   A2 with resting-state probe — Riemann recovers 94.1% from rest
   alone), session changes (A3 IV-2a), unseen-subject generalization
   (A4 AUC = 0.925, multi-seed 0.934 ± 0.020), and partial cross-
   dataset transfer (A4 PhysioNet → IV-2a, AUC = 0.694).
3. Membership inference works in the black-box setting on every
   victim family (EEGNet AUC = 0.878, FBCSP = 0.819, Riemann = 1.000).
4. Of three defense families, only formal differential privacy
   (D3 DP-SGD ε=3) holds under matched adaptive attack:
   - D1 PCA / noise / channel-drop fine-tune top-1: 0.640–0.758.
   - D2 DANN λ=0.2 fine-tune top-1: 0.804.
   - D3 DP-SGD ε=3 fine-tune top-1: 0.049.
   - No-defense A1 baseline: 0.411.
5. ~89% of D3's empirical privacy is contributed by the BatchNorm →
   GroupNorm architectural change Opacus requires; ~5% by the formal
   noise mechanism. The formal (ε, δ) guarantee is unaffected by
   this decomposition.
6. Within-cohort heterogeneity is large: EEGNet decile gap = 0.778
   (5-seed mean), FBCSP = 0.490, Riemann = 0 (ceiling). Demographic
   axes we can test show no significant sex effect; the EEGNet age
   effect is real but underpowered per seed (Fisher combined
   p = 0.0083 across 5 seeds).

Numbered limitations above bound how far each claim can be
extrapolated.
