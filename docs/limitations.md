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

- ~~**Federated DP setup** where each user's data never leaves their device.~~ **Addressed in Tier 2 (experiment 31).** Central-DP FedAvg with 104 PhysioNet clients, 30 rounds, 50% participation. The participant-level (ε, δ) is RDP-accounted via Opacus; the empirical encoder-fine-tune attacker is held to top-1 = 0.096 — within 4 pp of centralised DP-SGD at sample-level ε=3, without ever pooling raw EEG.
- **Transfer between EEG and other modalities** (e.g., does an attacker who has fNIRS or fMRI on the same subjects transfer to EEG?). Impossible to test — no published joint-modality cohort at this scale.
- **Continuous biometric authentication** ("does this user *stay* identified second-by-second during a BCI session?") rather than the snapshot threat we benchmark.
- **Adversarial-perturbation attacks** on the trained victims (do small EEG perturbations cause misclassification?). Different threat model from re-identifiability.
- **Theoretical bounds.** We report empirical numbers; we don't derive a theoretical bound on contrastive-embedding leakage as a function of training cohort size.

---

## A3 cross-session: small N (resolved)

The milestone reported A3 cross-session on BCI IV-2a only at n=9.
**Resolved in Tier 1.** Experiment 20 runs the same A3 protocol on
Lee 2019 OpenBMI (54 subjects × 2 sessions on different days, binary
L/R hand motor imagery). The corpus is now ingested via the parallel
range-request prefetcher (`data/lee2019_prefetch.py`) which writes a
compact float16 .npz cache to Drive, bypassing the original Tokyo
mirror's ~3 MB/s sequential-download bottleneck. The Lee 2019 A3
result is Riemann top-1 = 0.749 at chance 1.85% (40× lift), with
top-5 / top-10 = 0.923 / 0.968. Cross-session biometric linkage no
longer rests on a 9-subject corpus.

---

## What this benchmark *does* establish (after Tier 1 + Tier 2)

The empirical record across 3 corpora × 5 attacks × 4 defense
families × 5 adaptive attackers is audit-clean and consistent.

1. **Closed-set re-ID is near-perfect on the training cohort.**
   Riemann tangent-space top-1 = 1.000 on 104 PhysioNet subjects;
   FBCSP+LDA = 0.891; EEGNet = 0.411.

2. **The identity signal survives every protocol change tested.**
   Cross-task (motor execution / resting state), cross-session
   (IV-2a + Lee 2019), unseen-subject open-set verification
   (PhysioNet AUC = 0.925; Lee 2019 AUC = 0.920 within-session,
   0.868 cross-session), and three of four cross-dataset directions
   (AUC ≥ 0.67). The exception, Lee 2019 → PhysioNet at AUC = 0.496,
   tests the task-complexity hypothesis directly via experiment 33.

3. **Membership inference works in the black-box setting on every
   victim family and on both corpora.** EEGNet MI AUC = 0.878 on
   PhysioNet, 0.787 on Lee 2019; Riemann = 1.000.

4. **The defense story is attack-specific.** Of four defense families
   benchmarked under matched adaptive attack:
   - D1 ad-hoc transforms: encoder fine-tune top-1 = 0.640–0.758
     (above the no-defense baseline; the defense reshapes features
     without removing identity-discriminative content).
   - D2 DANN λ=0.2: fine-tune top-1 = 0.804 (also above baseline).
   - D3 DP-SGD: holds well against encoder fine-tune at every ε
     swept (top-1 = 0.043 at ε=0.5; 0.136 at ε=3; 0.189 at ε=10
     which exceeds the no-defense 0.153). DP-aware MIA at ε=3
     defeats the defense (AUC = 0.891 ≈ undefended 0.878); ε ≤ 1
     is the predicted deployable point for MI protection.
   - D4 federated DP-FedAvg: fine-tune top-1 = 0.096 at
     participant-level RDP-accounted ε ≈ 97.7 (loose budget).

5. **~89% of D3's empirical privacy is contributed by the
   BatchNorm → GroupNorm architectural change** Opacus requires;
   ~5% by the formal noise mechanism. The formal (ε, δ) guarantee
   is unaffected by this decomposition.

6. **Within-cohort heterogeneity is large on both corpora.**
   PhysioNet EEGNet decile gap = 0.778 (5-seed mean); Lee 2019
   EEGNet decile gap = 0.490. Demographic axes we can test on
   PhysioNet show no significant sex effect; the EEGNet age effect
   is real but underpowered per seed (Fisher combined p = 0.0083
   across 5 seeds). Lee 2019 publishes only cohort aggregates, so
   per-subject demographic stratification is not possible there.

7. **Fredrikson-style model inversion is a null result for both
   defended and undefended EEGNet re-ID heads.** Rank-1 recovery
   stays at 0 / 10 reconstructions for both arms; input-space
   optimisation does not yield subject-discriminative EEG at the
   tested input dimensionality.

8. **Re-ID accuracy decays sub-linearly with cohort size.** EEGNet
   scaling exponent γ = 0.474 fit across N ∈ {10, ..., 104}; the
   biometric threat shrinks slowly as cohort grows, not 1/N. At
   N = 104, EEGNet still identifies 41% of users (43× chance).

Numbered limitations above bound how far each claim can be
extrapolated.
