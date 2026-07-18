# Emergent-timescale learning: why it requires a precision-weighted update (and not Adam or plain SGD)

*Theory note, 2026-07-17 (user + Claude). "Emergent-timescale learning": the separation between a
FAST inner estimator and a SLOW meta-parameter should EMERGE from the information content of each
batch, not be imposed by a hand-set learning-rate ratio. Discovered while building the two-timescale
CE-calibration meta-judge (DESIGN_meta_judge_calibration.md). The catch: emergence only happens under
an update rule whose effective step scales with the observation's precision (signal-to-noise). Adam
and plain SGD both break it, for opposite reasons. This is the same inverse-variance principle that
runs through the whole fusion program (THEORY_evidence_fusion.md), now applied to the OPTIMIZER
instead of to the measurements.*

Companion docs: `DESIGN_meta_judge_calibration.md`, `REPORT_meta_judge_calibration.md`,
`DESIGN_amortized_fusion_heads.md` (the two-timescale/metastable-drift design),
`THEORY_evidence_fusion.md` (the precision/inverse-variance framing this generalizes).

---

## 1. The claim (user)

When you couple a FAST inner estimator (here: the Kalman-fused μ, distilled per batch by MSE) with a
SLOW meta-parameter (here: the judge that calibrates μ by candidate-ranking cross-entropy), you
should NOT need to hand-set the slow timescale (`lr_slow ≪ lr_fast`, or "update the judge every K
batches"). The separation should fall out on its own, because **each batch simply carries less usable
information for the meta-parameter than for the inner estimate**. The ranking CE is quantization-
limited and higher-variance; the MSE distillation is dense and low-variance. Less information ⇒
slower update, automatically.

This is correct — but it is a statement about a BAYESIAN/precision-weighted update, and it is false
for the two update rules one reaches for by default.

## 2. Why "slower" is not automatic under a fixed-rate optimizer

Write the meta-parameter update as `θ ← θ − η · f(g_t)` where `g_t` is the per-batch gradient of the
meta-loss and `f` is the optimizer's transform. Decompose the gradient into signal + noise:
`g_t = s + ε_t`, `E[ε_t]=0`, with signal power `‖s‖²` and noise power `E‖ε_t‖²`. Define the
per-batch information as the signal-to-noise ratio `SNR = ‖s‖² / E‖ε_t‖²`. The user's claim is:
**effective step size should be a monotone increasing function of SNR** — low SNR ⇒ small step ⇒
slow timescale. Check the defaults:

- **Plain SGD**: `f(g) = g`, step `= η·(s + ε_t)`. The per-batch step magnitude is `η·‖g_t‖`,
  which is INDEPENDENT of SNR — it is set by the raw gradient norm and the fixed `η`. A noisy,
  low-information batch produces just as large a step as a clean one; the noise only averages out
  over MANY steps, so the NET drift is slow but each step is not. In practice, to make the net drift
  slow you must shrink `η` by hand — i.e. you are back to imposing the timescale. (Measured: with a
  fixed SGD `η` the judge drifted ~66× faster than the fast heads, the opposite of intended.)

- **Adam / RMSProp**: `f(g) = g / (√v + ε)` with `v` the running second moment. This is WORSE: it
  divides by the gradient's own RMS, driving every coordinate to a UNIT-scale step regardless of how
  noisy it is. Adam's design goal — scale-invariance — is exactly the destruction of the SNR
  signal. A pure-noise direction (`s≈0`, `‖g‖≈‖ε‖`) still gets a unit step; the optimizer CHASES the
  quantization noise. Adam makes `effective_step ≈ η` for all SNR, the flat function the user's
  claim explicitly rules out.

So under both defaults the timescale does not emerge; it is either constant (Adam) or set by `η`
(SGD). The information asymmetry the user is pointing at is real, but neither optimizer READS it.

## 3. The update that makes it emerge: precision (SNR) gating

The claim is realized by a precision-weighted step — a Kalman/natural-gradient update on the
meta-parameter. Maintain EMA estimates of the gradient mean `m_t` (signal) and second moment
`p_t = E‖g‖²` (power). The coherent-signal fraction is

    gain_t  =  ‖m_t‖² / p_t   ∈ [0, 1]        (≈1 pure signal, ≈0 pure noise)

and the step is

    θ ← θ − η₀ · gain_t · m̂_t                 (m̂ = m/‖m‖, the signal direction)

This is inverse-variance weighting in disguise. Treat the batch gradient as a noisy OBSERVATION of
the descent direction with observation-noise variance `R ≈ p − ‖m‖²` and prior signal variance
`P ≈ ‖m‖²`. The Kalman gain for that scalar problem is `K = P/(P+R) = ‖m‖²/p = gain_t`. So the
meta-parameter is being Kalman-filtered: a low-information (quantization-noisy) batch has large `R`,
small `K`, small posterior movement — the inverse-variance rule the fusion filter already uses for
its measurements (THEORY_evidence_fusion.md §"price every source by its precision"), applied to the
optimizer.

**Correction (external statistical audit, 2026-07-17): what is and is not emergent.** The gate makes
the SNR-PROPORTIONAL SCALING emergent — the step shrinks with the information, which neither SGD nor
Adam provides. It does NOT make the ABSOLUTE timescale emergent: the step is `η₀·gain·m̂`, and `η₀`
is still a free constant. In the first shipped run `η₀ = 0.5` (vs Adam 5e-4 on the fast heads) made
the "slow" judge row move **22.7× farther per step than the fast heads** — the empirical timescale
was REVERSED even though the gain (~0.10) behaved as predicted. Two lessons: (i) compare drifts in
per-element RMS, not raw norms of different-sized tensors; (ii) the gate is NECESSARY for emergence
but not SUFFICIENT — `η₀` must be commensurate with the fast rate, and in this half-Kalman form it is
calibrated, not derived. The fully-derived version has no free `η₀`: a proper Kalman update on the
meta-parameter carries a state prior `P_θ` whose posterior contraction sets the absolute step
(`Δθ = K·innovation` with `K` from `P_θ` and `R` in the same units) — that is the honest meaning of
"the timescale falls out", and it is the rigorous follow-up, not what the current code does.

Measured on the meta-judge (REPORT_meta_judge_calibration.md §1): `gain_t ≈ 0.10` over training — the
candidate-ranking CE gradient is ~90% noise. With `η₀` calibrated the judge's per-element drift sits
below the fast heads'; the SNR-scaling is the emergent part, the base rate is engineering.

## 4. Why the fast estimator can keep Adam

The inner μ estimate is distilled by dense MSE against the fused target: high SNR, low variance per
batch. There the precision is high and roughly uniform, so gain-gating would be ≈1 everywhere and
buys nothing; Adam's convergence speed is worth more than a scaling it would not exercise. The
asymmetry in the RIGHT optimizer for each role — precision-gated for the low-SNR meta-parameter,
Adam for the high-SNR estimator — is itself the two-timescale, and it is dictated by the information,
not chosen.

## 5. Consequences and scope

- **General principle.** Any coupled fast-estimator / slow-meta-parameter scheme (MAML-style outer
  loops, learned-optimizer meta-parameters, calibration heads, metastable-drift states in
  DESIGN_amortized_fusion_heads.md) where the meta-signal is noisier per batch should use a
  precision-weighted meta-update if the timescale is meant to be emergent rather than tuned. Reaching
  for Adam on the meta-parameter silently reintroduces the hand-tuned timescale (via `η` and the
  unit-step normalization) and can chase noise.
- **Connection to the cross-campaign drift design.** DESIGN_amortized_fusion_heads.md's metastable
  bias states are a random walk with process noise `Q`; the precision-gated meta-update is the
  SGD-time analog — the state moves in proportion to how much each batch actually informs it. Adding
  `Q` (a floor on movement) and the gain (a ceiling from information) are the two knobs of the same
  filter.
- **Caveat — this is the update rule, not a performance claim.** The gate makes the timescale
  emergent and well-behaved; it does NOT make the meta-objective useful. On the Pearltrees campaign
  the calibrated μ did not beat the untrained ranker (REPORT_meta_judge_calibration.md §4). The
  theory here is about HOW the two timescales relate, not WHETHER the slow objective pays — those are
  independent, and the second is a data-scale question.
- **Not yet derived rigorously.** The `gain = K` identification treats the gradient as a scalar
  Gaussian observation with diagonal precision; the full matrix (Fisher) precision and its
  interaction with the fast estimator's updates (a genuine two-timescale stochastic-approximation
  analysis, Borkar-style) is the rigorous version — Codex's lane if it matters. The claim here is the
  qualitative law (SNR-proportional scaling requires precision weighting; Adam/SGD destroy it) plus
  the measured gain, NOT a convergence proof and NOT a claim that the absolute timescale is
  hyperparameter-free in the current implementation (see the §3 correction: `η₀` remains free until
  the full state-prior Kalman form is implemented).
