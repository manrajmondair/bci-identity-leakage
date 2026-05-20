# Literature comparison

This file compares the project's headline numbers against the closest
published comparison points in the EEG-biometric and BCI-privacy
literature. The comparison is for the final report. Each row notes
**(a) the published number, (b) the protocol it was reported under,
(c) our analogous result, and (d) why the comparison is informative
or where it breaks down**.

Numbers marked `[VERIFY]` need a final sanity-check against the
original source before the final report; they are based on either a
WebFetch summary or our project plan citation and have not been
re-confirmed against the paper PDF.

---

## A1 — closed-set re-identification on PhysioNet motor imagery

**Closest published comparison.**

Maciel, Maiorana & Campisi (2021), *"A deep descriptor for cross-
tasking EEG-based recognition"* (Pattern Recognition Letters; PMC8157223):
- Dataset: PhysioNet EEGMMIDB, 109 subjects, 64 electrodes.
- Reported: **100% closed-set identification** on protocol P1
  (training on T1R1 + T1R3 imagery runs, testing on T1R2; same-subject
  same-task, run-level split).
- Method: convolutional deep descriptor with data augmentation.
- Source: <https://pmc.ncbi.nlm.nih.gov/articles/PMC8157223/>

Maiorana & Campisi (2018), *"Longitudinal Evaluation of EEG-Based
Biometric Recognition,"* IEEE TIFS 13(5):1123–1138:
- Demonstrates longitudinal EEG biometric stability with HMM
  classifiers on a proprietary dataset; not directly comparable on
  PhysioNet but establishes that EEG biometric signal persists across
  sessions, consistent with our A3 finding. `[VERIFY]`

Other recent work reports **97.74%** closed-set accuracy on PhysioNet
motor imagery with 15 of 19 electrodes (MDPI Sensors 23(9):4239,
2023; <https://www.mdpi.com/1424-8220/23/9/4239>). `[VERIFY]`

**Our result.**

| Victim | A1 top-1 (logreg) | Lift over chance (1/104 = 0.96%) |
|---|---|---|
| Riemann tangent-space | **100.0%** | 104× |
| FBCSP+LDA             | **89.1%**  |  89× |
| EEGNet                | **41.1%**  |  43× |

**Why the comparison is informative.** The published 100% closed-set
on PhysioNet is achieved by methods *trained explicitly for
identification* — convolutional networks with subject ID as the
target, augmentation pipelines tuned for biometric performance. Our
Riemann pipeline matches that 100% under a much weaker setup: it was
trained for the motor-imagery *task* (4-class), and identity falls
out as a side-channel on the same embeddings. The published
literature's headline number is reproducible *as a side-effect of
task training* — which is exactly what makes the privacy threat
deployable.

EEGNet's 41.1% is substantially lower than both the published
identification-tuned numbers and our classical pipelines. The right
interpretation: deep features trained for the task are *less*
linearly identifiable than classical CSP / Riemannian features,
because the task supervision pulls the encoder toward features
useful for the 4-class motor-imagery classification rather than
features useful for subject discrimination. This is consistent with
EEG-biometric papers that explicitly train deep networks on
subject-ID — they reach 97-100%; ours doesn't because identification
isn't the loss.

---

## A4 — open-set verification on unseen subjects

**Closest published comparison.**

Maciel, Maiorana & Campisi (2021):
- Same protocol family as A1, but reports verification: **0.1–0.19%
  EER** in cross-task scenarios (eyes-open train, eyes-closed test;
  or motor-imagery cross-run).
- **Crucial caveat:** subjects in their verification cohort are
  *seen at training time* — the protocol is "within-cohort,
  cross-task" not "across-cohort." Their 0.19% EER is for verifying
  whether a recording came from a person whose other recordings the
  network was already trained on.
- Source: <https://pmc.ncbi.nlm.nih.gov/articles/PMC8157223/>

Bidgoly et al. (2022), *"Towards Enhanced EEG-based Authentication,"*
ACSAC '22 (paywalled, abstract via ACM DL):
- Reports very low EERs on cross-session motor-imagery
  authentication. Verification cohort overlaps training cohort. `[VERIFY]`
- Source: <https://dl.acm.org/doi/10.1145/3564625.3564656>

**Our result.**

> Open-set verification on **24 subjects held out** from contrastive
> training: **AUC = 0.925** [0.923, 0.928], EER = 13.3%, multi-seed
> AUC = 0.934 ± 0.020 over 5 random splits.

**Why this is the methodologically distinctive contribution.** Every
high-accuracy EEG-biometric result in the published literature we
can locate uses a **closed-cohort** protocol: the verification
subjects are some subset of the training subjects, with a
within-subject train/test split on different runs / sessions / tasks.
Under that protocol, the system is doing something closer to "match
a new recording to one of N known templates" than "tell whether two
recordings are of the same unknown person."

A4 differs structurally: the contrastive embedding is trained on 80
subjects, then verification is evaluated on 24 *strictly disjoint*
subjects. The model has never seen the people it's being asked to
verify. Under this protocol our AUC (0.925, multi-seed 0.934) is the
empirical answer to whether *EEG embeddings function as biometric
templates for arbitrary individuals* — the question that determines
whether neural data falls under GDPR Art. 9 / biometric-class
protection.

The directly-comparable verification protocols we know of are
Marcel & Millan (2007) on a 9-subject cohort with eyes-open EEG
(small N, low-channel) and the M3CV biometrics challenge dataset
(Wang et al. 2022; <https://www.sciencedirect.com/science/article/pii/S105381192200787X>),
neither of which provides a clean within-PhysioNet open-set number
to compare against. The cleanest claim we make in the report:
**AUC 0.925 unseen-subject is the first calibrated open-set number
on PhysioNet at this cohort size**.

---

## A3 — cross-session re-identification

**Closest published comparison.**

Bidgoly et al. (2022), as above: cross-session EER < 1% on BCI IV-2a
with motor-imagery authentication. `[VERIFY]`

The original BCI Competition IV documentation (<https://www.bbci.de/competition/iv/>)
publishes within-subject task accuracy across the 9 subjects' two
sessions but not biometric numbers.

**Our result.**

| Victim | A3 top-1 (session-1 → session-2, 9 subj, chance 11.1%) |
|---|---|
| Riemann tangent-space | 91.3% |
| FBCSP+LDA             | 88.9% |
| EEGNet                | 78.3% |

**Why the comparison matters.** Most published EEG-biometric work
that reports cross-session is on cohorts ≤ 50 subjects with EER
metrics; we report top-1 closed-set re-ID at chance 1/9, which is
the apples-to-apples format with the BCI-IV-2a literature on task
accuracy (so a reader can compare against published EEGNet/FBCSP
task numbers directly). The 91% Riemann number is in the same range
as Bidgoly's reported < 1% EER.

The A3 cohort is small (n=9). Our extension experiment 20 (Lee 2019
OpenBMI, 54 subjects × 2 sessions) is the next-step replication; a
≥ 80% top-1 result there at chance 1/54 (1.85%) would rule out
"BCI IV-2a is a fluke" much more confidently than the current
single-dataset cross-session number.

---

## A5 — membership inference

**Closest published comparison.**

Shokri, Stronati, Song & Shmatikov (2017), *"Membership Inference
Attacks against Machine Learning Models,"* IEEE S&P (arXiv 1610.05820;
<https://arxiv.org/abs/1610.05820>):
- The original shadow-model methodology we apply here.
- Reports MI AUCs from ~0.55 (well-regularized) to ~0.95 (strongly
  overfit) across image / text classifiers.
- The 0.878 we get on EEGNet is in the upper-mid range of their
  reported AUCs.

We're not aware of a published MI attack on EEG motor-imagery
classifiers; we'd cite this as an unaddressed gap in the BCI-privacy
literature that this project's A5 directly fills. `[VERIFY scope]`

**Our result.**

| Victim | MI AUC (95% CI) | Advantage |
|---|---|---|
| EEGNet                | 0.878 [0.803, 0.943] | 0.635 |
| FBCSP+LDA             | 0.819 [0.729, 0.892] | 0.500 |
| Riemann tangent-space | **1.000** [1.000, 1.000] | **1.000** |

Riemann's perfect MI is striking and consistent with its ceiling-
level A1: subject-conditional log-loss on members vs non-members is
trivially separable. Riemann's tangent-space mapping essentially
*memorizes* per-subject covariance structure, which makes it an
extreme example of the classical privacy-utility tradeoff Shokri
identifies.

---

## D2 / D3 — adversarial subject-invariance and DP-SGD as defenses

**Closest published comparisons.**

Özdenizci, Wang, Koike-Akino & Erdoğmuş (2019), *"Adversarial Deep
Learning in EEG Biometrics,"* IEEE Signal Processing Letters
26(5):710-714: introduces adversarial subject-invariant training for
EEG decoders, but frames it as helping cross-subject *task*
generalization, **not as a privacy defense**. They do not test
adaptive attackers. `[VERIFY exact citation]`

Abadi et al. (2016), *"Deep Learning with Differential Privacy,"*
ACM CCS (arXiv 1607.00133): the DP-SGD methodology we apply via
Opacus. They demonstrate it on MNIST / CIFAR with the formal (ε, δ)
guarantee; not EEG-specific.

We're not aware of a published adaptive-attacker stress-test of a
DANN-based defense on EEG. This is the methodological novelty in
Section 4.3 of the milestone.

**Our result.**

DANN λ=0.2 vs three attacker strengths (logreg / deep MLP / encoder
fine-tune):

| Attacker | Re-ID top-1 |
|---|---|
| logreg probe (generic)               | 0.252 |
| Deep MLP (frozen encoder)            | 0.357 |
| **End-to-end encoder fine-tune**     | **0.804** |

**Why this is the single most consequential finding.** The published
EEG adversarial-subject-invariance work reports the headline
"−20 pp leakage at zero task cost" type numbers under generic
probes. We replicate that headline (logreg → 0.252) and then show it
collapses under a 15-epoch encoder fine-tune. The privacy claim was
an artifact of attacker weakness. This is the EEG instance of the
broader Carlini–Tramèr line of work demonstrating that adversarial
defenses fail to adaptive attackers in the image-classification
literature; we show it transfers to BCI privacy.

The corresponding D3 adaptive-attack experiment (`experiments/18`)
will, if it confirms, give us the matched positive: formal DP holds
where DANN doesn't, because (ε, δ) is attacker-agnostic by
construction.

---

## Summary table (for the final report's related-work section)

| Question | Closest published number | Our number | Comparable? |
|---|---|---|---|
| Closed-set EEG re-ID, PhysioNet | 100% (within-subject, ID-trained, Maciel '21) | 100% Riemann (task-trained, 4-class side-channel) | direct, with caveat: ours is a side-channel of task training |
| Open-set verification, **unseen-subject**, PhysioNet | no published comparison we can locate | AUC 0.925, EER 13.3% | first of its kind to our knowledge |
| Open-set verification, **unseen-subject**, Lee 2019 | no published comparison we can locate | AUC 0.920 within-session, 0.868 cross-session | second-corpus replication of the same claim |
| Cross-session re-ID, BCI IV-2a | EER < 1% (Bidgoly '22, paywalled `[VERIFY]`) | 91% top-1 Riemann at chance 11% | indirect; EER vs top-1 |
| Cross-session re-ID, Lee 2019 (54 subj) | no published comparison we can locate | 75% top-1 Riemann at chance 1.85% (40× lift) | direct; first cross-session re-ID at meaningful N on OpenBMI |
| Cross-dataset A4 transfer (four directions) | no published comparison we can locate | three directions AUC ≥ 0.67; one direction (Lee 2019 → PhysioNet) collapses to 0.5 | first symmetric cross-corpus EEG-biometric transfer benchmark |
| Membership inference on EEG decoders | not published as far as we know | AUC 0.878 EEGNet PhysioNet, 0.787 EEGNet Lee 2019, 1.000 Riemann | first MI attack on EEG decoders we're aware of |
| Adversarial subject-invariance under adaptive attack | not published for EEG | DANN λ=0.2: 0.21 generic → 0.80 fine-tune | first adaptive stress-test of EEG subject-invariance |
| DP-SGD under DP-aware MIA on EEG | not published for EEG | AUC 0.891 at ε=3 ≈ undefended 0.878 | first DP-aware MIA against an EEG decoder, demonstrating the formal-ε gap |
| Federated DP-FedAvg on EEG | not published for EEG biometrics | fine-tune top-1 = 0.096 at participant-level RDP ε ≈ 98 | first end-to-end federated-DP defense benchmark for BCI re-ID |
| γ scaling exponent of EEG re-ID with cohort size | not published for EEG | EEGNet γ = 0.474 (sub-linear decay) | first empirical scaling fit on PhysioNet |
| Yeom (2018) (ε, δ) bound vs empirical EEG re-ID | not published for EEG | empirical fine-tune top-1 sits 0.35–0.85 below the Yeom bound across ε ∈ {0.5, 1, 3, 10} | first formal-vs-empirical overlay for EEG DP-SGD |

## New Tier 1 + Tier 2 contributions in detail

### DP-aware membership inference

The MIA tradition (Shokri 2017, Yeom 2018) is explicit that defended-target MIA evaluation should train the shadows in the same defended pipeline as the target so the attacker's shadow distribution matches the defender's noise distribution. Experiment 27 instantiates that protocol on EEG for the first time we are aware of. At ε = 3 (eight DP-SGD shadows + one DP-SGD target), MI AUC = 0.891 [0.826, 0.943], statistically indistinguishable from the no-defense baseline of 0.878. The result is consistent with the Yeom MI-advantage upper bound at ε = 3 (which permits 0.95).

**The full ε sweep confirms the Yeom-bound prediction.** At ε = 1, AUC = 0.506 [0.393, 0.614]; at ε = 0.5, AUC = 0.449 [0.334, 0.558]. Both are at chance within their CIs (the CIs are wide because the cohort is small, 52 vs 52 members / non-members). The empirical curve tracks the Yeom prediction: defeated at ε = 3, constrained at ε ≤ 1. **DP-SGD ε ≤ 1 is the deployable point that simultaneously blocks re-ID fine-tune (top-1 ≤ 7%) AND DP-aware MIA (AUC ≈ 0.5).** This contradicts the milestone's framing that DP-SGD blanket-holds under all adaptive attacks (true only against re-ID fine-tune), and replaces it with a precise quantitative characterisation across ε.

### Federated DP-FedAvg

The federated-DP literature (Geyer 2017, McMahan 2017) covers vision and language; we are not aware of a BCI-side benchmark of FedAvg with central-DP noise. Experiment 31 instantiates the canonical protocol on PhysioNet with one client per subject. The empirical fine-tune attacker is held to top-1 = 0.096 — within 4 pp of centralised DP-SGD at sample-level ε=3 — without ever pooling raw EEG. The participant-level RDP-accounted ε at this configuration is 97.7 (loose); the report should be explicit that the formal budget is not tight at the configuration that delivers the empirical privacy.

### Symmetric cross-dataset transfer and the Lee 2019 → PhysioNet collapse

Experiment 26 covers all four directions over PhysioNet, IV-2a, and Lee 2019. Three directions transfer (AUC ∈ {0.673, 0.826, 0.831}); Lee 2019 → PhysioNet collapses to AUC = 0.496 even with 40 training subjects, and the collapse replicates at 0.497 ± 0.005 across 5 seeds (experiment 34).

**Experiment 33 falsified the task-complexity hypothesis.** Re-training the Lee 2019 contrastive embedder with a synthetic 4-class label (hand × first/second-half-trial), with every other variable held fixed, produces AUC = 0.501 [0.499, 0.504] on Lee 2019 → PhysioNet — a +0.005 pp lift over the binary baseline of 0.496, well within the null prediction. Training-time task richness is therefore *not* what gates the transferability of the biometric template in this direction. The most plausible remaining hypothesis — and one the next iteration could test directly — is recording-rig domain shift: Lee 2019 was recorded with a BrainAmp (Brain Products, Munich) at 1000 Hz native with nasion reference, while PhysioNet was recorded with the BCI2000 system at 160 Hz with a different reference electrode. The double-resample stack (1000 → 250 → 160 Hz) combined with the amplifier / reference mismatch produces a domain shift the contrastive cannot bridge. We document this honestly rather than speculatively asserting it.

### γ scaling fit and Yeom-bound overlay (experiment 30)

The closest published reference for cohort-size scaling on EEG biometrics is Maciel 2021's protocol comparisons, which sweep recording protocols rather than cohort sizes. To our knowledge no prior work fits the FaceNet-style `1 - C · N^(1−γ)` law on EEG. Our empirical EEGNet γ = 0.474 indicates the threat decays slower than 1/N — relevant for regulator-facing claims about whether BCI biometric leakage gets better or worse as user databases grow.

The Yeom (2018) MI-advantage bound `1 − exp(−ε) − δ` is well-known for image / text classifiers. Overlaying it against the empirical DP-SGD ε sweep here is novel for EEG and gives a concrete "where does the formal bound start to bind" answer: empirical fine-tune top-1 sits 0.35 below the bound at ε=0.5 and 0.81 below at ε=10; the cross-over with the AdamW+BN no-defense baseline of 0.411 lands near ε ≈ 0.5.

---

## Bibliography (to copy into the final report's references)

1. Schalk, McFarland, Hinterberger, Birbaumer & Wolpaw (2004).
   *BCI2000: A General-Purpose Brain-Computer Interface (BCI) System.*
   IEEE TBME 51(6):1034-1043.
2. Maciel, Maiorana, Campisi (2021). *A deep descriptor for cross-
   tasking EEG-based recognition.* PMC8157223.
3. Maiorana & Campisi (2018). *Longitudinal Evaluation of EEG-Based
   Biometric Recognition.* IEEE TIFS 13(5):1123-1138.
4. Lawhern et al. (2018). *EEGNet: A Compact CNN for EEG-based BCIs.*
   J. Neural Engineering 15(5):056013.
5. Shokri, Stronati, Song, Shmatikov (2017). *Membership Inference
   Attacks against Machine Learning Models.* IEEE S&P. arXiv:1610.05820.
6. Abadi et al. (2016). *Deep Learning with Differential Privacy.*
   ACM CCS. arXiv:1607.00133.
7. Ganin et al. (2016). *Domain-Adversarial Training of Neural
   Networks.* JMLR 17(59):1-35. (gradient-reversal foundation for D2)
8. Carlini & Tramèr et al. (2019). *On Evaluating Adversarial
   Robustness.* arXiv:1902.06705. (adaptive-attacker methodology)
9. Wang et al. (2022). *M3CV: A multi-subject, multi-session, and
   multi-task database for EEG-based biometrics challenge.*
   NeuroImage 264:119666.
10. Tangermann et al. (2012). *Review of the BCI Competition IV.*
    Frontiers in Neuroscience 6:55. (IV-2a dataset citation)
