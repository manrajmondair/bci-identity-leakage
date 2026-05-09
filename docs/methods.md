# Methods and citations

Formal definitions, hyperparameters, and citations for every technique
used in the project. Organized by component.

---

## 1. Datasets

### PhysioNet EEG Motor Movement/Imagery Database (EEG-MMIDB)

> Schalk, G., McFarland, D. J., Hinterberger, T., Birbaumer, N., & Wolpaw, J. R. (2004). **BCI2000: a general-purpose brain-computer interface (BCI) system.** *IEEE Transactions on Biomedical Engineering*, 51(6), 1034–1043.

> Goldberger, A. L., Amaral, L. A. N., Glass, L., Hausdorff, J. M., Ivanov, P. C., Mark, R. G., Mietus, J. E., Moody, G. B., Peng, C.-K., & Stanley, H. E. (2000). **PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals.** *Circulation*, 101(23), e215–e220.

109 subjects, 64-channel EEG at 160 Hz, 14 runs per subject (2 baseline + 12 motor task). Hosted at <https://www.physionet.org/content/eegmmidb/1.0.0/>. We drop subjects 88, 89, 92, 100, 104 due to known recording issues (the standard EEG-MMIDB drop-list in the BCI literature).

### BCI Competition IV dataset 2a

> Brunner, C., Leeb, R., Müller-Putz, G. R., Schlögl, A., & Pfurtscheller, G. (2008). **BCI Competition 2008 — Graz data set A.** Institute for Knowledge Discovery, Graz University of Technology.

9 subjects, 22 EEG channels at 250 Hz, two recording sessions on different days (training "0train" and evaluation "1test"). 4-class motor imagery: left hand, right hand, both feet, tongue. Accessed via `moabb.datasets.BNCI2014_001`.

### OpenNeuro ds004362 (demographics for PhysioNet)

> James, S. (2023). **EEG Motor Movement/Imagery Dataset.** OpenNeuro. <https://openneuro.org/datasets/ds004362/versions/1.0.0>

BIDS conversion of the same BCI2000 source recordings PhysioNet hosts; preserves the `sub-001` ↔ `S001` indexing and republishes Gender / Age / Handedness for 95 of the 109 subjects. We use it solely for demographic stratification of the W5.1 fairness analysis.

---

## 2. Preprocessing

### Bandpass filtering

Zero-phase Butterworth IIR (4th order) bandpass at 4–40 Hz, applied per-channel via `scipy.signal.sosfiltfilt`. The wide passband is required so the FBCSP filterbank's sub-bands have spectral headroom (a narrower 8–30 Hz pre-filter zeroed out FBCSP's 4–8 Hz and 28–40 Hz bands and tanked its accuracy below chance — this is documented in commit history as the early FBCSP debugging story).

### Epoching and windowing

Per-trial epoch from `t=0` to `t=4 s` after the cue event (T1/T2 annotations on PhysioNet, 769–772 codes on IV-2a). Each trial is then cut into 2-s sliding windows with 1-s stride (3 windows per trial). Windows inherit their parent trial's class label and trial id; bootstrap CIs group resampling by trial-id so within-trial correlation doesn't produce spuriously tight bounds.

### Per-channel scaling

EEGNet's published hyperparameters were tuned for input at the **microvolt** scale (std ≈ 10–100). `mne` returns EEG in **volts** (std ≈ 2.75 × 10⁻⁵). We multiply by 10⁶ inside `EEGNetVictim._iter_batches` before the forward pass so gradients don't vanish. FBCSP and Riemann normalize internally and don't need this (FBCSP standardizes log-variances; Riemannian methods are scale-equivariant on the SPD manifold).

---

## 3. Victim models

### EEGNet (Lawhern et al. 2018)

> Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018). **EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces.** *Journal of Neural Engineering*, 15(5), 056013. arXiv:[1611.08024](https://arxiv.org/abs/1611.08024).

Compact CNN: temporal convolution → depthwise spatial convolution → separable convolution → linear classification. ~5,000 parameters total. We use `braindecode.models.EEGNet` (formerly `EEGNetv4` before braindecode 1.12) with `n_chans=64, n_outputs=4, n_times=320`. AdamW optimizer, lr=1e-3, weight_decay=1e-4, 80 epochs at batch size 64. Embeddings extracted via a forward hook on `model.final_layer` (the head Sequential), capturing its 160-dim input.

### FBCSP+LDA (Ang et al. 2008)

> Ang, K. K., Chin, Z. Y., Zhang, H., & Guan, C. (2008). **Filter Bank Common Spatial Pattern (FBCSP) in Brain-Computer Interface.** *IJCNN 2008*.

9-band filter bank covering 4–40 Hz in 4 Hz overlapping subbands; CSP with 4 components per band fitted via `mne.decoding.CSP(transform_into="average_power", log=True)`; per-band log-variance features concatenated, standard-scaled, and classified by `sklearn.discriminant_analysis.LinearDiscriminantAnalysis`. Total feature dimension = 4 × 9 = 36. Embeddings (used as input to A1's probe) are the standardized log-variance vector — i.e., LDA's input.

### Riemannian tangent-space + Logistic Regression (Barachant et al. 2012)

> Barachant, A., Bonnet, S., Congedo, M., & Jutten, C. (2012). **Multiclass brain–computer interface classification by Riemannian geometry.** *IEEE TBME*, 59(4), 920–928.

Per-window covariance matrix (OAS shrinkage estimator via `pyriemann.estimation.Covariances`); tangent-space mapping at the geometric mean (`pyriemann.tangentspace.TangentSpace`); multinomial logistic regression on the tangent vectors. Embeddings are the tangent vectors themselves (n_channels × (n_channels+1)/2 dimensions = 2080 for 64-channel PhysioNet).

### Contrastive EEGNet (for A4 and cross-dataset A4)

EEGNet backbone (with the head replaced by `nn.Identity()`) + projection head `Linear(160, 64)` + L2-normalize. Trained with batch-hard triplet loss (Schroff et al. 2015 / FaceNet) using subject ids as the supervisory signal: anchors and positives drawn from the same subject; negatives from any other subject. Margin 0.2, AdamW lr=1e-3, batch size 8 subjects × 4 windows = 32, 30 epochs.

> Schroff, F., Kalenichenko, D., & Philbin, J. (2015). **FaceNet: a unified embedding for face recognition and clustering.** *CVPR 2015*. arXiv:[1503.03832](https://arxiv.org/abs/1503.03832).

---

## 4. Attacks

### A1 — closed-set re-identification

Probe: `KNeighborsClassifier(n_neighbors=5, metric="cosine", weights="distance")` and `LogisticRegression(C=1.0, max_iter=2000)` on the victim's embeddings. Train on `runs ∈ {4, 6, 8, 10}`, test on `runs ∈ {12, 14}`. Top-1 / top-5 / top-10 accuracy with grouped bootstrap (1000 resamples) on trial-id.

### A2 — cross-task re-identification

Same victim as A1. Probe trained on motor-execution-run embeddings (runs 3, 5, 7, 9, 11, 13), tested on motor-imagery test-run embeddings (12, 14). Tests cognitive-task generalization of the identity signal.

### A3 — cross-session re-identification

BCI IV-2a only. Victim trained on session 1 task labels; probe trained on session-1 embeddings, tested on session-2 embeddings (different recording day).

### A4 — open-set verification

Subject pool split 80 train / 24 held-out. Contrastive embedding trained on the 80 (~21,600 windows). Verification pairs sampled from the 24 held-out subjects (same-subject and different-subject, 25,000 each). Score = cosine similarity. ROC-AUC and Equal Error Rate; AUC bootstrap over the 50,000-pair set.

### A5 — per-subject membership inference

> Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2017). **Membership inference attacks against machine learning models.** *IEEE Symposium on Security and Privacy*.

20 shadow EEGNets on random 50% subject splits. Per-(shadow, subject) features = (mean per-window cross-entropy loss, mean max-softmax). `LogisticRegression` attack classifier on shadow data, evaluated on a held-out target.

### Cross-dataset A4

Train contrastive EEGNet on PhysioNet's 22-channel subset matching IV-2a (resampled to 160 Hz). Evaluate verification on IV-2a session-1 (9 subjects, also 22 channels at 160 Hz after resampling). Tests whether the EEG-as-biometric claim transcends the specific dataset.

### Multi-seed A4

5 random train/test subject splits (80/24). Robustness CI on the AUC across splits.

### Adaptive attackers (defender-aware re-identification)

Three attacker tiers run against each defended victim:

1. **Standard logreg probe** (= A1 baseline) — generic linear probe on
   the frozen encoder's penultimate features.
2. **Deep MLP probe** (3 hidden layers, BatchNorm + ReLU + dropout) on
   the **frozen** encoder — higher-capacity generic attacker.
3. **End-to-end encoder fine-tune** — encoder initialized from the
   defended victim's weights, retrained end-to-end with a fresh re-ID
   head on subject-id labels (15 epochs, AdamW lr=5e-4, batch 64).
   The realistic worst-case: white-box access + adaptive optimization.

Run against three defense families:

- `experiments/15_d2_adaptive_attacker.py` — DANN λ=0.2.
- `experiments/18_d3_adaptive_attacker.py` — DP-SGD ε=3.
- `experiments/23_d1_adaptive_attacker.py` — D1 PCA k=8, noise σ=1.0,
  channel-drop k=8.

Empirical finding: only formal differential privacy (D3) holds under
fine-tune; DANN and every D1 point collapse *above* the no-defense
A1 baseline. See report §4.3, §4.6, §4.10.

### DP-SGD architecture ablation (experiment 19)

Disentangles D3's empirical privacy into (a) the BatchNorm → GroupNorm
architectural change Opacus's `ModuleValidator.fix` performs and (b) the
formal noise mechanism. Trains a fresh GroupNorm-EEGNet with the same
SGD optimizer DP-SGD uses, but `target_epsilon=None` (no per-sample
gradient clipping, no Gaussian noise). A1 closed-set re-ID is then run
against this configuration's embeddings. The privacy attributable to
architecture is `(AdamW+BN top-1) − (SGD+GN top-1)`; the privacy
attributable to formal DP is `(SGD+GN top-1) − (SGD+GN+DP top-1)`.

### Resting-state cross-task A2 (experiment 21)

Stronger version of A2's task-orthogonality test. The probe is trained
on PhysioNet's resting-state runs (R01 eyes-open, R02 eyes-closed),
which have no shared task structure with motor imagery, and tested on
the same imagery test runs (12, 14) used by A1. Sliding 2 s windows
with 1 s stride applied directly to the continuous baseline EEG (no
event-locked epoching, since baseline runs have no T1/T2 annotations).

### Multi-seed EEGNet fairness (experiment 22)

Re-runs the cross-subject EEGNet → per-subject A1 attack-acc →
demographic stratification pipeline across 5 random training seeds.
Reports per-seed Mann-Whitney p-values, mean ± std effect sizes,
and **Fisher's combined p** across seeds (the appropriate aggregate
test for combining independent p-values). Distinguishes "real but
underpowered per seed" from "robust at the aggregate level".

---

## 5. Defenses

### D1 — ad-hoc transforms (the original proposal's mitigations)

- **PCA channel compression:** fit per-channel PCA on training windows, project to top-k components, reconstruct or use directly. Sweeps k ∈ {64, 32, 16, 8}. `sklearn.decomposition.PCA`.
- **Additive Gaussian noise:** zero-mean per-channel noise scaled by σ × training-set channel-std. σ ∈ {0, 0.5, 1.0, 2.0}.
- **Channel-drop:** keep only the top-k highest-variance channels of the training set. k ∈ {64, 32, 16, 8}.

### D2 — Domain-Adversarial Neural Network (Ganin et al. 2016)

> Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., Marchand, M., & Lempitsky, V. (2016). **Domain-adversarial training of neural networks.** *Journal of Machine Learning Research*, 17(59), 1–35. arXiv:[1505.07818](https://arxiv.org/abs/1505.07818).

EEGNet backbone + two heads (task: 4-way; subject: 104-way). Subject head connected via Gradient Reversal Layer (multiplies the backward gradient by `-λ`). Total loss = L_task + λ · L_subject; encoder receives `+grad_task − λ · grad_subject`. λ swept across {0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}; sweet spot at λ=0.2.

### D3 — Differentially Private SGD (Abadi et al. 2016)

> Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., & Zhang, L. (2016). **Deep learning with differential privacy.** *ACM CCS 2016*. arXiv:[1607.00133](https://arxiv.org/abs/1607.00133).

> Yousefpour, A., Shilov, I., Sablayrolles, A., Testuggine, D., Prasad, K., Malek, M., Nguyen, J., Ghosh, S., Bharadwaj, A., Zhao, J., Cormode, G., & Mironov, I. (2021). **Opacus: User-friendly Differential Privacy Library in PyTorch.** arXiv:[2109.12298](https://arxiv.org/abs/2109.12298).

Implemented via `opacus.PrivacyEngine.make_private_with_epsilon`. Per-sample gradient clipping at L2 norm 1.0, calibrated Gaussian noise. SGD optimizer (Opacus is incompatible with adaptive optimizers without modification), lr=1.0, batch 256, 40 epochs. Target ε ∈ {None (no DP), 10, 3} at δ = 1e-5; achieved ε reported by the RDP accountant. Required ModuleValidator.fix to swap BatchNorm → GroupNorm and remove the max-norm parametrization on EEGNet's spatial conv (parametrized modules don't pickle, blocking Opacus's clone).

---

## 6. Statistical methodology

### Bootstrap CIs

1000 resamples, percentile method. For per-window outcomes (top-1 accuracy), we use **trial-grouped bootstrap** (`eval.bootstrap.grouped_bootstrap_ci`): each resample draws whole trials with replacement and includes all of that trial's windows together. This handles within-trial correlation honestly; treating windows as independent samples would produce spuriously tight CIs.

For pair-level outcomes (A4 AUC), we resample pairs directly (within-subject correlations of EEG windows are absorbed by the verification-pair construction).

### Subgroup tests

Mann-Whitney U two-sided (`scipy.stats.mannwhitneyu`) for Δ between sex groups and between age tertiles; bootstrap CI on the difference of group means via a separate resampling of each group.

### Audit invariants

`tools/audit.py` runs 76 invariants over the cached results, including:

- shape and NaN/Inf checks on the windowed data
- per-subject window count consistency (must be exactly 270 for imagery)
- trial-id-to-subject uniqueness (no cross-subject collisions)
- A1/A2/A3/A4 train/test set disjointness (closed-set vs open-set semantics)
- bootstrap CI brackets the point estimate (sanity)
- effect-size sanity vs published EEG re-ID literature (≥0.95 closed-set top-1 expected per Maiorana 2016, Yang & Deravi 2017)
- a **shuffled-label negative control**: with random subject labels the same probe collapses to chance (0.0100 vs chance 0.0096)

> Maiorana, E. (2016). **Deep learning for EEG-based biometric recognition.** *Pattern Recognition Letters*, 90, 27–32.

> Yang, S., & Deravi, F. (2017). **On the usability of electroencephalographic signals for biometric recognition: A survey.** *IEEE Transactions on Human-Machine Systems*, 47(6), 958–969.

---

## 7. Software

| Component | Version pinned | Notes |
|---|---|---|
| Python | 3.11 | tested on Apple Silicon and Colab L4 |
| PyTorch | 2.5.1 | stronger pin: torchaudio==2.5.1 (ABI match) |
| mne | ≥ 1.7 | EDF I/O + filtering |
| moabb | ≥ 1.1 | BCI IV-2a access |
| pyriemann | ≥ 0.6 | tangent-space classifier |
| braindecode | 1.4 (`<1.5`) | EEGNet implementation |
| opacus | ≥ 1.5 | DP-SGD |
| scikit-learn | ≥ 1.4 | LDA, LogisticRegression, kNN |

Full pin list in [`pyproject.toml`](../pyproject.toml).
