# Evidence fusion for fuzzy membership: the theory, and how we arrived at it

*Consolidated theory note, 2026-07-08. This document collects the theoretical framework developed across the
two-judge posterior arc (PR #3517), the Sigma(hop) confirmatory arc, the Product-Kalman arc, and the fusion-heads
design discussions (user + Claude, with Codex infrastructure) — including the corrections that shaped it, because
the corrections carry as much information as the results. Detailed numbers live in the linked reports; this is
the map.*

Companion docs: `REPORT_two_judge_posterior.md`, `PAPER_sigma_hop_confirmatory.md`, `DESIGN_product_kalman_poe.md`,
`DESIGN_amortized_fusion_heads.md`, `REPORT_product_kalman_realdata.md`.

---

## 1. The problem

Estimate fuzzy relation memberships `mu(op(x,y))` between concept nodes (for a bookmark-filing assistant), given
several **evidence channels** of very different character:

- the **model** (mu-attention over frozen e5): amortized, cheap, always available;
- the **graph** (walk hit-probability, hop distance): free but only for already-filed structure;
- the **LLM judge**: rich but expensive, usually absent at inference.

The recurring question in every arc: **how should correlated, partially-available, differently-priced evidence be
combined — and how confident should the combination be?**

## 2. The combiner ladder (a moment expansion)

For a log-linear posterior, the gradient depends on features linearly (1st moment) and the Hessian is the feature
covariance (2nd moment). This yields a ladder of combiners, each keeping one more moment:

| rung | keeps | drops |
|---|---|---|
| linear superposition / blending | main effects | all covariance |
| confidence-weighted | + variances (diagonal) | correlations |
| joint / covariance-aware | + off-diagonal | — |

**Path.** We started at the bottom (blended targets, `dir-blend`), found the naive blend trades direction for
magnitude, and reframed: *condition, don't blend* — `P(op | d, LLM_op)` (the two-judge posterior). Two structural
insights arrived from the user en route:
- **Soft constraints:** the second-order structure acts like Lagrangian penalties; sampled correlations are SOFT
  constraints (churn crosses them; that is the mechanism, not a failure), while structural facts (`mu_rev = 0`,
  graph + LLM agreeing) are HARD (clamp, don't penalize).
- **Pseudo-judges:** second-order constraints can be represented as synthetic judges (products of real readouts) —
  k operators need k(k+1)/2 of them (the unique entries of a Gram matrix) — lifting the interaction `⊗` into a new
  linear `⊕` coordinate.

## 3. The statistical journey (and its three corrections)

The empirical path to the covariance result was shaped by three user-driven corrections, each of which REVERSED a
conclusion:

1. **Binarization destroyed the signal.** Per-hop correlations on binarized labels looked random ("those numbers
   look random"); bootstrap CIs confirmed nothing. Redone on CONTINUOUS fuzzy mu, Wikipedia showed a strong
   heteroscedastic trend (`corr(mu_D, mu_S)`: −0.83 at h1 → +0.25 at h5). *Lesson: measure fuzzy sets on the
   continuous membership, never thresholded.* (The same lesson later demoted the "joint beats product-of-marginals"
   headline: it is a discrete co-occurrence effect that does not replicate on continuous mu.)
2. **Confidence, not correlation.** Measuring only `rho(hop)` gave nothing significant. The user's redirection —
   "for low hop counts you can be much more confident the relationship is directional... if the learning rate is
   small enough, this difference gets washed out" — pointed at the DIAGONAL: `sigma(hop)` is the effective example
   weight. Neither `sigma(hop)` nor `rho(hop)` alone clears noise; the FULL `Sigma(hop)` does. The mechanism is
   geometric (user's prediction, verified): the covariance's condition number reduces with hop (11 → 5.3) and the
   decoupling rotation swings −40° → +8°, so a constant Sigma is a bad average of incompatible geometries.
3. **Calibrated inference.** Review (model council) found two real defects: pair-level splits leaked nodes, and
   `mean/√n` over correlated resamples is not a significance level. Fixed with descendant-disjoint splits and a
   hop-shuffle permutation test — the effect SURVIVED (p = 0.001), and then a **pre-registered confirmation on a
   fresh, zero-overlap corpus passed at the permutation floor** (gain +0.060, p < 0.001). The correlation-geometry
   trend even replicated qualitatively across the two independent corpora.

**Landed theory:** the residual covariance of directional/symmetric memberships is a *smooth function of graph
position*, `Sigma(hop)`; a 6-parameter head (log-linear sigmas, tanh rho) beats per-hop empirical bins because it
POOLS — the expected shrinkage result (Ledoit–Wolf lineage), not a surprise.

## 4. PoE vs covariance: a type distinction

`DESIGN_product_kalman_poe.md` fixed a type error we kept circling: **PoE is a mean mechanism; the joint covariance
is the error geometry around the mean.** A weighted sum `λ·L_PoE + (1−λ)·L_joint` conflates them; the clean form is
one likelihood, `L = ½(y−mu)ᵀV⁻¹(y−mu) + ½log|V|`, where PoE-style aggregation may propose `mu` and `Sigma(hop)`
supplies `V`. Corollaries:
- `λ` is never set from a p-value; a bounded `λ∈[0,1]` is a convex shrinkage gate (PoE is then the CEILING); an
  additive `λ>1` is a temperature needing calibration. If dependence should be as strong as data supports, don't
  make `λ` the limiter — learn `Sigma` directly, regularized toward diagonal/PoE.
- The **bound lattice** (user): `mu_lower = Π mu_i^w` (PoE, AND-like) ≤ `mu_mid = Σ w·mu` (mixture) ≤
  `mu_upper = 1 − Π(1−mu_i)^w` (noisy-OR). `[lower, upper]` is a disagreement diagnostic → routing/abstention.

## 5. The architecture: amortized fusion heads, two timescales

**Question:** does the model SUBSUME the fusion intelligence, or is it a HYBRID with an explicit filter? Answer
(user's three-way learn): **both, at different timescales.**

- Heads `mu_graph`, `mu_LLM`, `mu_PoE`: the model amortizes each judge separately (channels stay exposed) plus the
  fusion itself. The fused head is trained on measured-data fusions (anchor) and regularized toward the product of
  its own readouts (stop-gradient — else the tautology/feedback trap); its VALUE is where it deviates from the
  naive product: the learned correlated-PoE correction.
- **Fast timescale:** model readout = prior; graph/judge = measurement channels fused by an explicit Kalman update
  when present. **Slow timescale:** distill the fused posteriors back into the model.
- **Function-name embeddings** (user): condition readouts on `W·e5("descriptive function name")` — one open-vocabulary
  mechanism unifying operators, judges, and fusion functions.

**Division of labor with the filter (user question, sharpened):** the Kalman filter does not learn — it is the
closed-form algebra that USES what the model learned. Prior mean = mu heads; prior covariance P = Sigma(hop) head;
measurement noise R = residual calibration; gain K = COMPUTED, never learned. The bet: *learn the uncertainty,
derive the fusion weights for free* (vs learning gate weights directly). The filter's *recursion* lives in the
filing DB: stored `(mu, P)` per relation, updated incrementally as evidence arrives — RLS (`P ~ 1/n`) when static,
exponential forgetting under drift.

## 6. The epistemic limit — why the hybrid is necessary, not convenient

User: *"means and correlations are something the model could also learn — but perhaps not the means and
correlations of the error."* Confirmed and sharpened into the deepest constraint:

- The **error mean is unlearnable from inside**: any predictable component of a model's error gets absorbed into
  its mean estimate and ceases to be error; the in-distribution error mean is ~0 by construction, and the
  out-of-distribution bias is nonzero exactly where the model cannot know it (Kalman: bias states are unobservable
  without measurements).
- The **error covariance is learnable only from outside the training loop**: held-out residuals (training residuals
  are optimistically small; joint mean+variance NLL has the variance-eats-the-loss pathology), and in-distribution
  only (error statistics are the first casualty of shift — which is why the fresh-corpus confirmation mattered most
  for the ERROR model).

**Crisp form: signal statistics are statistics of the world; error statistics are statistics OF THE MODEL —
estimating them requires standing outside it.** Adaptive Kalman filtering does exactly this (R/Q from the
innovation sequence). Hence: distillation can absorb everything EXCEPT the role of standing outside; the
innovation loop is the only mechanism that can see the model's bias. Subsume-vs-hybrid closes epistemically.

**The online closure (user):** the error statistics themselves can be modeled as **metastable states with drift**
— promote them into the state vector with random-walk dynamics (bias augmentation; stochastic volatility for
log-sigma; rho likewise). Timescale hierarchy: L0 relation state (fast) / L1 error statistics (slow walk) /
L2 drift rates (quasi-fixed). `Sigma(hop, corpus)` is the ATTRACTOR MAP: the model learns the metastable
attractors from pooled history; the filter random-walks around the current one; distillation updates the map —
*the model holds the climate, the filter tracks the weather.* Jumps (judge version changes = step in R_LLM;
category reorganizations = step in corpus Sigma) exceed diffusion: heavy-tailed process noise, innovation
chi-square with adaptive fading, or a switching state-space model — and a switching Kalman filter IS a mixture of
experts over regimes (the user's MoE hunch, in its natural home).

## 7. Why Gaussianity, specifically (user)

Gaussianity matters to the FILTER because of **closure**: linear + Gaussian ⇒ the posterior stays Gaussian, two
moments suffice, and the covariance evolves by its own autonomous matrix difference/Riccati equation — in which
**the measurements never appear**. Error bars follow a deterministic, precomputable schedule; only the mean is
data-driven. Two consequences:

- **This is the license for the Sigma(hop) head**: covariance-as-a-function-of-structure is a precomputed Riccati
  trajectory along the hop axis — predictable without labels because covariance propagates autonomously. The
  confirmed result is an empirical instance.
- **The two-timescale split follows the mathematical fault line**: the precomputable covariance schedule went to
  the model; the data-dependent statistics tracking (adaptive R, metastable drift) stayed with the filter. The
  engineering split and the theory split coincide.

Precision: covariance propagation needs only linearity + second moments (Kalman = best LINEAR estimator for any
noise); Gaussianity upgrades LMMSE to EXACT BAYES, making P an honest credible region rather than second-moment
bookkeeping.

## 8. Products, means, and measures (the coordinate arc)

A three-step exchange resolved what "PoE" even averages:

1. **Two different products** (user question): a product of DENSITIES (Gaussian PoE — what `independent_kalman`
   computes) has a precision-weighted **arithmetic** mean; it interpolates (fusion-of-estimates semantics). A
   product of VALUES (`mu_lower`) is AND-like ≤ min unnormalized, a **geometric** mean normalized (conjunction
   semantics) — that one is the lower estimate.
2. **Fuse-space = noise-model choice** (user correction): exact Bayes is reparameterization-invariant; the
   GAUSSIAN ASSUMPTION is not. Fusing in mu-space assumes additive noise; in log space, multiplicative
   (geometric mean); in logit space, noise on odds (multiplying Bayes factors). "The filter runs in any of these
   spaces" (user) — the choice is where to spend the Gaussian approximation, and it is empirically decidable.
3. **Measured, then corrected** (user's prior beat the aggregate statistic): on aggregate the residuals looked far
   more Gaussian in mu-space — but that was driven ENTIRELY by boundary atoms (~7% of labels at/near exact 0/1,
   which the logit link blows into spikes). On the INTERIOR, logit is the more natural Gaussian home (3 of 4
   corpus×channel cells; one at Jarque–Bera 0.1 — textbook). Logit's a priori advantage — support on all of R,
   the only self-consistent Gaussian family of the three — is real once the atoms are handled.
4. **Boundary atoms via Jacobian weighting** (user): a reported 0.0 is an uncertain, quantized estimate, not
   certainty. Give the label mu-space noise `sigma_mu` (≥ the quantization half-step) and propagate through the
   link: logit-precision `∝ (mu(1−mu))²` — boundary points SELF-DOWNWEIGHT inside the plain Kalman update. The
   rigorous form is a classical identity: the Fisher information a Bernoulli-type observation carries about its
   own log-odds IS `mu(1−mu)` (the logistic GLM variance function). Simpler and more junk-robust than Tobit
   censoring; with boundary DE-QUANTIZATION (0.0 → 0.025 at large-but-finite variance) the labels keep a whisper
   of "low" and the likelihood stays continuous.

## 9. Empirical state (what is measured vs designed)

**Measured:**
- `Sigma(hop)` — exploratory +0.094 (permutation p=0.001), **pre-registered confirmation on a fresh zero-overlap
  corpus: +0.060, p < 0.001** (the one confirmatory result in the program).
- Fusion (model prior + graph channel, covariance-aware): +0.4 to +0.9 held-out NLL over the model alone, 40/40
  splits, both corpora; MSE halves.
- The correlation term (correlated vs independent fusion = "Kalman vs Gaussian PoE"): real, corpus-dependent
  (+0.32 vs +0.05), and it lives mostly in the ERROR BARS (Mahal/dim 1.18 vs 1.51 at similar MSE), as predicted.
- **Independent PoE anti-scales with evidence**: adding correlated channels under an independence assumption is
  worse than not adding them (both corpora; error bars ~√2 too small); the correlated update absorbs the same
  channels harmlessly. Correlated fusion is not just better — it is what makes adding evidence SAFE.
- Interior residuals are logit-Gaussian; boundary atoms are quantization artifacts.

**Designed, not yet built:** the three heads + bound lattice + name embeddings; hop-conditioned V(hop) inside the
Kalman gain (target: the measured Mahal/dim 1.39 residual overconfidence → 1); logit-space fusion with
Jacobian-weighted per-row R + de-quantization (target: the interior-Gaussianity advantage → held-out NLL); the
metastable/adaptive layer.

## 10. Method lessons (portable beyond this project)

1. Measure fuzzy quantities on the continuous membership; binarization can destroy or invert the signal.
2. The confidence axis (diagonal) can carry the effect when the correlation axis (off-diagonal) alone does not —
   check the FULL covariance before declaring a null.
3. Repeated-split `mean/√n` is stability, not significance; permutation tests calibrate; node/entity-disjoint
   splits are the floor for graph data; pre-register before touching fresh data.
4. Post-exploratory p-values are conditional on the search path — say so, then confirm fresh.
5. Smooth-parametric beating per-bin empirical is shrinkage, not magic — frame it as regularization value.
6. When an aggregate diagnostic and a strong prior disagree, decompose the data (interior vs boundary) before
   trusting either.
7. Error statistics must be estimated from outside the model (held-out residuals, innovations, fresh corpora) —
   and the parts of a system that CAN be precomputed (autonomous covariance) vs MUST be tracked online
   (innovations, drift) tell you where to put the model/filter boundary.
