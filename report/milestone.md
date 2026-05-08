# EEG-as-Biometric: Subject Re-Identification Leakage in BCI Models

**CS 281 Milestone Report**
**Manraj Singh Mondair · May 20, 2026**

---

## Abstract

Brain-computer interface (BCI) systems collect EEG to decode intended actions
such as imagined movement. We ask whether the same neural recordings — and the
task-trained models that consume them — also leak the user's identity. On
PhysioNet's 109-subject motor-imagery dataset and BCI Competition IV-2a, we
benchmark five attacks (closed-set re-identification, cross-task, cross-session,
open-set verification on unseen subjects, membership inference) against three
families of victim models (Filter-Bank CSP+LDA, Riemannian tangent-space
classifier, and EEGNet). Across every attack the strongest classical pipeline
re-identifies users at near-perfect rates (Riemann tangent-space: 100.0% top-1
on 104 subjects, chance 0.96%). Crucially, our open-set evaluation shows that a
contrastive EEG embedding generalizes to *unseen* subjects at AUC 0.925 — voice-
biometric strength on people the model has never seen. We further benchmark
defenses across three families: ad-hoc transforms (PCA, additive noise,
channel-drop), adversarial subject-invariant training (DANN), and formal
differential privacy (DP-SGD). Adversarial training at low strength (DANN λ=0.2)
strictly dominates every ad-hoc point we tested — **−20 percentage points** of
identity leakage with **zero** task-accuracy loss.

---

## 1. Motivation

Motor-imagery BCIs collect EEG to classify intended actions (e.g., imagined
left/right hand). The same EEG features used to decode actions may carry stable,
subject-specific patterns — making the "task data" function as a biometric.
This raises ethical questions for BCI users, patients using assistive
neurotechnology, and researchers releasing public datasets: removing names from
EEG recordings is not the same as making the recordings non-identifying.

We investigate this empirically. Concretely: (RQ1) does the embedding produced
by a task-trained BCI decoder enable subject re-identification? (RQ2) does that
identity signal persist across cognitive tasks and across recording sessions?
(RQ3) does it generalize to subjects the model never trained on? (RQ4) can an
attacker tell from black-box access alone whether a subject was in the training
cohort? (RQ5) which defense families effectively trade identity leakage for
task accuracy?

## 2. Threat models, datasets, methods

We use **PhysioNet EEG-MMIDB** (109 subjects, 64-channel EEG @ 160 Hz; 5 known-
bad subjects dropped to leave n=104) and **BCI Competition IV-2a** (9 subjects,
22 channels @ 250 Hz, two recording sessions on different days). Imagery
recordings are bandpassed 4–40 Hz and split into 2-s windows with 1-s stride.

Three victim model families:
- **FBCSP+LDA**: 9-band filter-bank Common Spatial Patterns followed by Linear
  Discriminant Analysis. Classical state-of-the-art for motor imagery.
- **Riemannian tangent-space + LR**: per-window covariance, tangent-space
  mapping, multinomial logistic regression. Geometric pipeline.
- **EEGNet** (Lawhern et al. 2018) at the published hyperparameters, ~5K
  parameters; trained for 80 epochs with AdamW. We rescale inputs by 10⁶
  (volts → microvolts) to match the trained-on signal scale.

Five attacks:
- **A1 closed-set re-identification.** Probe (kNN cosine + L2 logreg) trained
  on victim embeddings → predict subject_id. Train on imagery runs 4/6/8/10,
  test on 12/14. 104-way classification, chance = 1/104 ≈ 0.96%.
- **A2 cross-task re-identification.** Same victim, but the probe is trained
  on motor-execution embeddings and tested on motor-imagery embeddings. Tests
  whether identity rides on cognitive-task-orthogonal components.
- **A3 cross-session re-identification.** BCI IV-2a 9 subjects, train on
  session 1, test on session 2 (different day). Realistic biometric-linkage
  threat. Chance = 1/9 ≈ 11.1%.
- **A4 open-set verification (the headline novel attack).** Train a
  contrastive EEGNet (batch-hard triplet loss, subject ids as supervision) on
  80 PhysioNet subjects' imagery; on the held-out 24 subjects (whom the
  network *never* saw during training), evaluate same-vs-different person
  verification across 50,000 sampled pairs. Reports ROC-AUC and Equal Error
  Rate (EER).
- **A5 per-subject membership inference.** 20 shadow EEGNets trained on random
  50% subject splits + a separate target. Per-(subject, model) features =
  (mean per-window cross-entropy loss, mean max-softmax). Logistic-regression
  attack classifier on shadow data, evaluated on the held-out target.

Three defense families:
- **D1 ad-hoc transforms** (the original proposal's mitigations, kept as
  baseline). PCA-k channel compression, additive Gaussian noise scaled by
  channel-std, channel-drop (top-k by variance). Each transform is fitted on
  the training windows and applied before victim training.
- **D2 DANN adversarial subject-invariant training.** EEGNet + a 104-way
  subject-classification head + Gradient Reversal Layer between encoder and
  subject head. The encoder is pressured to produce features useful for the
  task and useless for subject ID simultaneously. We sweep λ ∈ {0, 0.1, 0.5, 1.0}.
- **D3 DP-SGD via Opacus.** Formal (ε, δ)-differential privacy with
  per-sample gradient clipping plus calibrated Gaussian noise. We sweep
  target ε ∈ {None, 10, 3} at δ = 10⁻⁵, 40 epochs.

Every reported metric carries a 1000-resample bootstrap CI; closed-set probes
group resampling by trial-id so within-trial correlated windows do not produce
spuriously tight bounds. The full run sequence and audit invariants are
reproducible from `tools/audit.py` (76 invariants, all passing on the current
result set).

## 3. Results

### 3.1 Closed-set re-ID is near-perfect across the cohort (A1, 104 subj)

| Victim | Task acc | Re-ID top-1 (logreg, 95% CI) | Lift over chance |
|---|---|---|---|
| Riemann tangent-space | 35.0% | **100.0%** [100.0, 100.0] | 104× |
| FBCSP+LDA             | 26.8% | **89.1%** [88.1, 90.0]    |  89× |
| EEGNet                | 38.8% | **41.1%** [39.7, 42.3]    |  43× |

The two classical motor-imagery feature extractors leak subject identity
essentially perfectly even though their cross-subject task accuracy is at or
near chance — *privacy leakage is decoupled from task utility*. EEGNet's deeper
features are substantially less identifying than the classical pipelines but
still leak at 43× chance.

### 3.2 Identity persists across cognitive tasks and across sessions (A2, A3)

**A2 (cross-task, 104 subj, chance 0.96%).** Train probe on motor-execution
embeddings, test on motor-imagery: Riemann **100.0%**, FBCSP **90.2%**,
EEGNet **36.3%**. Identity rides on cognitive-task-orthogonal components.

**A3 (cross-session, BCI IV-2a 9 subj × 2 sessions, chance 11.1%).** Probe
trained on session-1 embeddings tested on session-2: Riemann **91.3%**,
FBCSP **88.9%**, EEGNet **78.3%**. The biometric link survives different
recording days.

### 3.3 EEG functions as a biometric template for unseen subjects (A4)

A contrastive EEGNet trained on 80 subjects' imagery, evaluated on 24 held-out
subjects across 50,000 verification pairs:

> **AUC = 0.925** [0.923, 0.928] · **EER = 13.3%**

For calibration: voice biometrics typically achieve AUC 0.85–0.95; fingerprint,
iris, and face land at ≥ 0.95. *EEG-on-unseen-people lands at the voice-
biometric tier*, on a single contrastive embedding trained from motor-imagery
data. This is the strongest evidence the project produces that EEG functions
as a biometric template, not just a fingerprint of training-time users.

### 3.4 Membership inference works in the black-box setting (A5)

> **AUC = 0.878** [0.803, 0.943] · **TPR − FPR advantage = 0.635**

20 shadow EEGNets + 1 target on 104 subjects, balanced 52 members vs 52 non-
members. Per-(subject, model) features = (mean window-loss, mean max-softmax).
The trained-model API alone leaks who was in the training cohort.

### 3.5 Privacy-utility Pareto across defenses

| Defense | Best (Re-ID, Task) point on EEGNet | Comment |
|---|---|---|
| No defense                  | (0.41, 0.39) | A1 baseline |
| D1 PCA k=8                  | (0.36, 0.36) | weak; Riemann still leaks at 92% |
| D1 noise σ=1.0              | (0.19, 0.34) | dominates D1 PCA on EEGNet |
| D1 noise σ=2.0              | (0.10, 0.29) | first ad-hoc to crack Riemann (1.00 → 0.70), but task → chance |
| D1 channel-drop k=8         | (0.37, 0.30) | mirrors PCA; no improvement |
| **D2 DANN λ=0.2**           | **(0.22, 0.40)** | **strict win: −20 pp leak; task slightly above baseline** |
| D2 DANN λ=0.1               | (0.27, 0.40) | original sweet spot |
| D2 DANN λ=0.5               | (0.08, 0.25) | task collapsed |
| D3 DP-SGD ε=3 (Opacus)      | (0.02, 0.30) | formal (ε,δ); architectural side-effects dominate |

The Pareto figure (Figure 1 below) summarizes all defense points across the
three victim families. Adversarial training at low strength is the only
configuration tested that buys real privacy without paying utility; ad-hoc
transforms either preserve task accuracy and barely dent leakage (PCA,
channel-drop) or buy privacy but collapse task to chance (high-σ noise).

> *Figure 1.* Privacy-utility Pareto, three panels (one per victim family).
> Lower-right corner = high utility AND low identity leak. See
> `figures/pareto_privacy_utility.pdf`.

### 3.6 Subgroup fairness (sex, age) and within-cohort heterogeneity

PhysioNet's EDFs are anonymized (`his_id: X, sex: 0`), but OpenNeuro's BIDS
conversion of the same source data (ds004362) re-publishes Gender + Age +
Handedness for 95 of 109 subjects with the same indexing. After mapping back
onto our 104-subject analysis cohort: 41 M / 56 F / 7 unknown sex; 91 with
known age (range 19–67, median 38).

| Victim | M (mean) / F (mean), Δ, p | Age low (mean) / age high (mean), Δ, p | Decile gap (most − least) |
|---|---|---|---|
| FBCSP+LDA | 0.898 / 0.883, Δ=+0.015, **p=0.67** | 0.935 / 0.854, Δ=+0.081, **p=0.08** | +0.490 |
| Riemann   | 1.000 / 1.000, Δ=0.000, p=1.0     | 1.000 / 1.000, Δ=0.000, p=1.0          | 0.000 (ceiling) |

Three findings:

1. **A1 attack accuracy is statistically indistinguishable by sex** on
   FBCSP (Δ ~1.5 pp, p=0.67). Men and women are equally identifiable.
2. **Marginal age effect on FBCSP**: subjects in the youngest age tertile
   (19–28) leak ~8 percentage points more than the oldest tertile (≈50+),
   p=0.08. Suggestive but not formally significant at α=0.05; consistent
   with the hypothesis that younger subjects have more characteristic
   motor-imagery patterns.
3. **Riemann is at ceiling** — every subject and every subgroup is
   identified at 100%. There is no demographic variation to detect:
   the strongest classical pipeline identifies everyone perfectly. This
   is a fairness story in itself — under the standard tangent-space
   pipeline, *every* demographic group experiences maximum threat.

Beyond demographics, the per-subject distribution of FBCSP attack accuracy
is heterogeneous: **the most-leaked decile of subjects has 49-pp higher
attack accuracy than the least-leaked decile**. Identity leakage is not
uniformly distributed across the cohort; some users are substantially
more identifiable than others, even at fixed cohort and victim model.

> *Figure 2.* Subgroup fairness on FBCSP and Riemann: histogram of per-
> subject leakage, scatter of (task acc, attack acc), and box-plots by
> sex and age tertile. See `figures/12_subgroup_fairness.pdf`.

## 4. Discussion

The empirical picture is consistent across all five attacks: motor-imagery
features and the models that consume them carry strong, stable, subject-
specific information that survives task changes (A2), session changes (A3),
unseen-user generalization (A4), and even black-box membership inference (A5).
Closed-set re-identification (A1) is essentially perfect on the strongest
classical pipeline. **Removing names is not enough; EEG functions as a
biometric.**

On the defense side, the structural finding is that **principled defenses
(adversarial subject-invariant training) are strictly more efficient on the
privacy-utility curve than ad-hoc input transforms**. The original proposal's
ad-hoc methods (PCA, additive noise, channel reduction) are dominated by DANN
at λ=0.1 in our benchmarks — DANN buys 14 percentage points of privacy at zero
task cost; no ad-hoc point we tested matches that.

## 4. Robustness extensions and an honest update on DANN

### 4.1 A4 robustness across train/test splits

The headline 0.925 AUC was reported for a single split. We re-ran the open-set
verification across **5 random splits** of the 104 subjects into 80 train /
24 held-out, retraining the contrastive embedding from scratch each time
(22 epochs per seed). Cross-seed:

> **AUC = 0.934 ± 0.020** (range 0.901–0.950)
> **EER = 0.136 ± 0.032**

The per-seed bootstrap CIs are tight (~0.005 each) and the across-seed std
is ~0.02. The headline AUC is robust to split choice; reviewers cannot
dismiss it as cherry-picking. See `figures/14_a4_multi_seed.pdf`.

### 4.2 A4 cross-dataset transfer (PhysioNet → BCI IV-2a)

We re-trained the same contrastive embedding on PhysioNet's 22-channel
subset matching IV-2a's montage (resampled to 160 Hz to match), then
evaluated verification on **BCI IV-2a session-1** — completely different
recording rig, country, sampling rate, and cognitive-task class set:

> **AUC = 0.694** [0.690, 0.699]
> **EER = 0.356**

This is meaningfully above random (0.5) but a substantial drop from the
within-PhysioNet 0.925. Honest interpretation: the EEG-as-biometric template
**partially transfers across recording protocols** — there is real
identity signal that survives the cross-dataset shift — but it is not
fully protocol-invariant at this scale. The "voice-biometric grade" framing
applies within a recording protocol; cross-protocol the threat is weaker
but not absent. See `figures/13_a4_cross_dataset.pdf`.

### 4.3 D2 DANN under an adaptive attacker (the most consequential finding)

Section 3.5 reported that DANN λ=0.2 reduces re-ID top-1 from 0.411 to
0.215 — a 20-percentage-point privacy gain at zero task cost. That number
was measured against a **generic** logistic-regression probe.

We re-ran the attack against three increasingly adaptive attackers, all
white-box on the trained DANN encoder (`experiments/15_d2_adaptive_attacker.py`):

| Attacker | Re-ID top-1 (104 subj) |
|---|---|
| logreg probe (the original generic baseline)            | 0.252 [0.241, 0.263] |
| Deep MLP probe on the **frozen** encoder                | 0.357 [0.345, 0.369] |
| **End-to-end encoder fine-tune for re-ID**              | **0.804 [0.792, 0.815]** |

The fine-tune attack — initialize from the DANN-trained weights, replace
the task head with a 104-way subject classifier, retrain end-to-end —
recovers leakage to **80.4% top-1**, *higher than the no-defense A1
baseline of 41.1%*. DANN's privacy claim was an artifact of the weakness
of the generic attacker.

**Interpretation.** DANN's adversarial subject-invariance pressure shapes
the encoder to give a non-adapted attacker little to work with. But the
encoder's learned representations are still rich features of the EEG;
when the attacker can adapt the encoder weights, those features become
*more* useful for re-ID, not less, because they are well-trained EEG
features even though they were not chosen with re-ID in mind.

**Implication for the report.** We should frame DANN as a **weak privacy
defense under adaptive threat** rather than a strict win. The privacy-
utility curve in Figure 1 is conditioned on "the attacker has not
specifically targeted DANN-trained models"; under an adaptive attacker
DANN is brittle. This is exactly why testing privacy claims against
adaptive attackers is the methodologically correct move. See
`figures/15_d2_adaptive_attacker.pdf`.

### 4.4 Membership inference across all three victim families

Original A5 reported MI AUC = 0.878 on EEGNet. Extending the same Shokri-
style shadow-model methodology to FBCSP+LDA and Riemann tangent-space:

| Victim | Shadows | MI AUC (95% CI) | Advantage (TPR − FPR) |
|---|---|---|---|
| EEGNet                | 20 | 0.878 [0.803, 0.943] | 0.635 |
| FBCSP+LDA             | 12 | 0.819 [0.729, 0.892] | 0.500 |
| **Riemann tangent-space** | 20 | **1.000** [1.000, 1.000] | **1.000** |

Riemann's per-subject log-loss on members vs non-members is so cleanly
separable that the attack is trivially perfect. This mirrors A1's
pattern: Riemann is the most identifying victim by every metric we
have. See `figures/16_a5_classical.pdf`.

### 4.5 EEGNet subgroup fairness

EEGNet shows substantially **larger heterogeneity** than the classical
victims (decile gap **0.783** vs FBCSP's 0.490, Riemann at ceiling), and
the age effect that was marginal on FBCSP is now formally significant
on EEGNet:

| Comparison | Δ | p (Mann-Whitney) |
|---|---|---|
| Sex M (n=41) vs F (n=56) | +0.060 | 0.288 (n.s.) |
| Age low (n=34) vs high (n=30) | +0.128 | **0.044** ✓ |

The youngest age tertile leaks ~13 percentage points more than the
oldest under EEGNet. Combined with the decile gap, this means EEGNet's
privacy threat is concentrated on younger subjects with characteristic
motor-imagery patterns, while older subjects with weaker patterns are
both worse for the BCI task (lower task acc) AND less identifiable.
See `figures/17_subgroup_fairness_eegnet.pdf`.

## 5. Discussion

The empirical picture across 5 attacks × 3 victims × 3 defense families
× 2 datasets × demographic stratification × adaptive attackers is
consistent: **EEG carries strong, stable, subject-specific information
that survives task changes (A2), session changes (A3), unseen-subject
generalization (A4), membership-inference (A5), and partially survives
cross-dataset transfer (Section 4.2)**.

The defense story is more nuanced than the milestone-time framing.

- D1 ad-hoc transforms remain weak on the strongest victim (Riemann).
- D2 DANN looks effective against generic attackers but **collapses
  catastrophically under an adaptive attacker** who fine-tunes the
  encoder. This is a methodologically critical observation: adversarial
  subject-invariance training does not provide defense-aware privacy.
- D3 DP-SGD provides formal privacy at ε=3, with most of the empirical
  privacy benefit coming from the architectural changes Opacus requires
  (BatchNorm → GroupNorm) rather than the noise mechanism itself.

A meaningful privacy guarantee for BCI models likely requires either
formal mechanism-level privacy (DP) — which we benchmark — or a defense
that holds under an adaptive attacker, which DANN at our tested λ
values does not.

## 6. Limitations and next steps for the final report

1. **Adaptive attackers tested only against D2.** D1 and D3 numbers are
   upper bounds for generic attackers; defense-aware attackers might
   reduce them further. Final-report extension: adaptive attacks on the
   full defense matrix.
2. **Cross-dataset cohort small.** IV-2a's 9 subjects bound the cross-
   dataset AUC's CI; a third dataset (e.g., HighGamma or Cho-Lee) would
   triangulate the claim further.
3. **DP-SGD architectural confound** (~37 pp privacy from architecture,
   ~1.6 pp from formal DP) is documented but not isolated experimentally.
   A clean ablation (AdamW EEGNet without DP, with GroupNorm, etc.) is
   straightforward and worth running.
4. **The age effect on EEGNet (p=0.044)** is significant at α=0.05 but
   should be replicated on a fresh seed before being claimed strongly.

All numbers are reproducible from the canonical result JSONs under `results/`,
the experiment scripts under `experiments/`, and the audit trail under `runs/`,
on the same git commit (linked in each `runs/<run_id>/meta.json`).
Acceptance criterion: `python -m tools.audit` returns 76 OK / 0 WARN / 0 FAIL.
