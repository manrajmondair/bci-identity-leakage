# Theoretical scaling — what the empirical numbers should look like

The paper-track milestone reports purely empirical numbers: closed-set
re-ID top-1, open-set verification AUC, membership-inference AUC,
encoder-fine-tune top-1 under each defense. Top venues (NeurIPS,
USENIX Security, IEEE TIFS) typically also want at least a soft
theoretical anchor that makes the empirical numbers fall out of
something predictable. This document lays out two such anchors and
their experimental validation under `experiments/30_theory_scaling.py`.

---

## 1. Closed-set re-ID accuracy vs. cohort size N

### Claim

For a fixed embedder $\phi: \mathbb{R}^{C \times T} \to \mathbb{R}^d$
producing per-window embeddings, the expected closed-set top-1 accuracy
on $N$ enrolled subjects satisfies

$$
\Pr[\text{top-1 correct}] \approx 1 - (N - 1) \cdot
\Pr_{(x, x') \sim \text{neg}}\!\left[\langle\phi(x), \phi(x')\rangle
\geq \tau_*\right] \cdot \mathbb{1}\!\left[\text{embedding is calibrated}\right],
$$

where $\tau_*$ is the decision threshold that maximises true-positive
rate at the cohort scale. Under mild independence + Gaussian-tails
assumptions on the negative-pair similarity distribution, the second
factor scales as $N^{-\gamma}$ with $\gamma$ controlled by the
margin between same-subject and different-subject pair
distributions in the embedding (Schroff et al. 2015 §3.2; FaceNet
analysis). Whence:

$$
\Pr[\text{top-1 correct}] \approx 1 - C \cdot N^{1 - \gamma}.
$$

If $\gamma > 1$ (positive margin), re-ID accuracy approaches 1.0 as
cohort grows. If $\gamma = 1$, top-1 accuracy approaches a fixed
constant. If $\gamma < 1$, accuracy decays as cohort scales — this is
what a regulator wants to know: *does the threat get worse with more
users in the database, or does it saturate?*

### Empirical validation

`experiments/30_theory_scaling.py` measures:

  1. Same-subject and different-subject similarity distributions in
     the trained PhysioNet contrastive embedding (from experiment 06).
  2. Top-1 accuracy as a function of cohort size $N \in \{10, 20, 40,
     60, 80, 104\}$ for the Riemann tangent-space victim and the
     EEGNet victim, on PhysioNet.
  3. The scaling exponent $\gamma$ fit by least squares to
     $\log(1 - \text{top-1}) \sim (1 - \gamma) \log N$.

The prediction the data should validate (and that we report in the
paper): **the empirical $\gamma$ should be approximately 1.0 for
EEGNet and approximately $> 1.0$ for Riemann tangent-space, matching
the observed ceiling behaviour of Riemann at $N = 104$**.

---

## 2. Empirical re-ID accuracy vs. formal DP budget $\varepsilon$

### Claim (Yeom et al. 2018 -style)

A model satisfying $(\varepsilon, \delta)$-DP bounds the success
probability of any membership-inference adversary by

$$
\mathrm{Adv}_{\text{MI}} \leq 1 - e^{-\varepsilon} - \delta,
$$

and an analogous (but weaker) bound on attribute-inference success
follows by a reduction. Empirically the trained-victim's
re-identifiability top-1 should sit *strictly below* this MI bound
(re-ID is harder than MI in the per-subject setting).

### Empirical validation

`experiments/29_d3_eps_sweep.py` sweeps
$\varepsilon \in \{0.5, 1.0, 3.0, 10.0, \infty\}$ and reports both
the generic-logreg and encoder-fine-tune attack accuracies. We
overlay the empirical curve against the Yeom MI-advantage upper
bound:

$$
\Pr[\text{re-ID correct at }\varepsilon] \;\leq\; 1 - e^{-\varepsilon} - 10^{-5}.
$$

At $\varepsilon = 3$, $1 - e^{-3} - 10^{-5} \approx 0.950$ — i.e.,
the formal bound is almost vacuous in the loose-DP regime: the
attacker is theoretically permitted up to ~95% re-ID accuracy. The
empirical observation that fine-tune top-1 sits at $\sim 0.05$
under DP-SGD $\varepsilon = 3$ is **far below** the formal upper
bound; the gap is the project's "DP buys more in practice than the
mathematical bound certifies" finding.

At $\varepsilon = 0.5$, $1 - e^{-0.5} - 10^{-5} \approx 0.393$ —
formally tighter than the empirical no-defense baseline of 0.411
(though only marginally), so DP $\varepsilon \approx 0.5$ is the
regime where the formal bound starts to bind on the empirical
threat.

---

## 3. Pre-registered predictions for `experiments/30`

Locking in falsifiable predictions before the experiment runs is the
methodological commitment that prevents post-hoc story-fitting. We
register the following:

  P1. **Scaling exponent $\gamma$ for EEGNet** between 0.9 and 1.2.
       Wider than this fails the calibration check.

  P2. **Scaling exponent $\gamma$ for Riemann tangent-space** above
      1.0 (because Riemann hits ceiling 100% at $N = 104$).

  P3. **Fine-tune top-1 vs Yeom bound** — empirical fine-tune top-1
      stays $\geq 5\times$ below the formal MI advantage upper bound
      for every $\varepsilon \in \{3, 10\}$. If this fails, the
      "DP buys more in practice" framing is wrong.

  P4. **Cross-over $\varepsilon$ between formal and empirical**
      is in $(0.3, 1.0)$. Below $\varepsilon \approx 1$ the formal
      bound becomes the binding constraint; above it, the empirical
      defense is what protects users.

These predictions are recorded in `results/30_theory_scaling.json`
before the run, with the realised numbers in the same file after.

---

## References

- Schroff, Kalenichenko, Philbin (2015). *FaceNet: A Unified Embedding
  for Face Recognition and Clustering.* CVPR.
- Yeom, Giacomelli, Fredrikson, Jha (2018). *Privacy Risk in Machine
  Learning: Analyzing the Connection to Overfitting.* IEEE CSF.
- Dwork & Roth (2014). *The Algorithmic Foundations of Differential
  Privacy.* Foundations and Trends in TCS.
- Carlini, Tramèr et al. (2019). *On Evaluating Adversarial
  Robustness.* arXiv:1902.06705.
