# Amortized fusion heads: the three-way learn, bound heads, and function-name embeddings

*Design note, 2026-07-08 (user + Claude discussion). This is the WHAT for the Product-Kalman arc: what the model
is that we are building, and how the Kalman machinery relates to it. Builds on `DESIGN_product_kalman_poe.md`
(the HOW of fusion math) and the confirmed `Sigma(hop)` result (the predicted-error ingredient). Design; not built.*

## The question this answers

The Kalman/PoE machinery raised an architecture question: does the model *subsume* the filter's intelligence
(train end-to-end, no explicit fusion at inference), or is it a *hybrid* (model is one expert; an explicit filter
fuses it with graph/judge evidence)? Answer: **both, as a two-timescale system**, and the user's three-way-learn
proposal is the architecture that holds both.

## Two-timescale hybrid

- **Fast timescale (inference):** the model readout is the amortized *prior* — always available. The graph walk
  and the LLM judge are *measurement channels* — free-but-partial and expensive-but-rich respectively — fused by
  the explicit Kalman/Gaussian update **when present**. `Sigma(hop)`/`V` is the predicted-error model that sets
  the gain. Channels can be added or dropped (judge absent at inference) without retraining.
- **Slow timescale (training):** periodically distill the filter's fused posteriors back into the model (the
  existing superposition/blend training). The model absorbs what the filter keeps having to correct; the filter's
  residual job shrinks toward evidence that was *not available at training time* (new corpora, new judges, fresh
  user filings). It never reaches zero — that residual is precisely what cannot be amortized.

## The three-way learn (user)

The model learns **separate per-channel heads plus a fused head**:

```text
mu_graph  = model's amortized graph judge      (what would the walk say?)
mu_LLM    = model's amortized LLM judge        (what would the judge say?)
mu_PoE    = model's amortized FUSION           (the learned correlated PoE)
```

Keeping `mu_graph` and `mu_LLM` distinct preserves the channels the explicit filter needs — evidence is not
pre-collapsed. The fused head is the amortized filter: **subsumption lives in one head while the channels stay
exposed for the hybrid.**

### Two target types for `mu_PoE` — different roles, not interchangeable

1. **Derived target (model's own readouts):** `poe(mu_graph, mu_LLM)` computed from the model's own heads. This is
   a **consistency regularizer**, NOT supervision — and it must use a **stop-gradient on the inputs**, or it
   creates the endogenous feedback loop flagged in the pseudo-judge design (readouts chase a target computed from
   themselves).
2. **Measured target (real data):** `poe(walk_hit_prob, llm_score)` from actual measurements. This is the
   **supervision anchor**. Guardrail carried over from `DESIGN_product_kalman_poe.md`: this is still a
   *constructed* fusion target (a fusion rule applied to real inputs), not a measured joint event — fine as
   distillation, labeled as such.

### Why a `mu_PoE` head at all (the tautology trap)

If `mu_PoE` were trained ONLY on the derived target it would be a **tautology** — a closed-form function of the
model's own outputs, learnable by construction, adding nothing. Its entire value is **where it deviates from the
naive product**: the naive product assumes independent experts, and ours are correlated (shared trunk, shared e5
inputs). So:

```text
naive product of own readouts  = shrinkage PRIOR (regularize toward it, stop-grad)
measured-data fusions          = ANCHOR that licenses deviation from the prior
the deviation itself           = the learned CORRELATED-PoE correction
```

This is the "regularize toward PoE/diagonal, learn the correction" prescription from the two-judge future-work
section, turned into a training recipe.

### What stays OUT of the mu heads

The error geometry. `Sigma(hop)`/`V` remains a **separate head**: mu heads estimate means, the V head supplies the
fusion weights/gain. Folding V into `mu_PoE` would re-confound mean estimation with error geometry — the type
error `DESIGN_product_kalman_poe.md` warns about ("PoE is a mean/prior mechanism; joint covariance is the error
geometry around it").

## The bound lattice: lower, middle, upper (user)

PoE is the AND-like *lower* proxy; we also derived the noisy-OR *upper* proxy. A convex mixture (MoE) sits between:

```text
mu_lower = prod_i mu_i ^ w_i              (PoE   — AND-like, <= min-ish)
mu_mid   = sum_i  w_i * mu_i              (MoE   — mixture, between the bounds)
mu_upper = 1 - prod_i (1 - mu_i) ^ w_i    (noisy-OR — non-rejection, >= max-ish)
```

Train all three as heads (same two target types each). The payoff is an internal **disagreement interval**:
`[mu_lower, mu_upper]` wide = sources disagree = route to the expensive judge or abstain; narrow = consensus,
trust the cheap path. Calibrated abstention falls out of the fuzzy algebra. (Per the earlier guardrail, the
interval is a disagreement diagnostic, not automatically a calibrated credible interval — calibrate on held-out.)

## Function-name embeddings (user)

Replace the indexed `JUDGES`/operator embedding table with conditioning vectors **derived from the e5 embedding of
the function's name** through a learned translation:

```text
cond(f) = W_translate · e5("descriptive name of function f")   [+ optional small learned residual delta_f]
```

- **Open vocabulary:** new functions (`PoE lower bound`, `noisy-OR upper bound`, `mixture`, `Kalman posterior`,
  any future judge) without resizing an embedding table — the `num_embeddings >= 9` headroom problem disappears.
- **Zero-shot composition:** an unseen name ("geometric mean of graph and LLM judges") lands near its trained
  relatives in e5 space.
- **Unification:** operators (SYM/HIER/ELEM/LINEAGE), judges (graph, LLM, blend, dir-blend), and fusion functions
  (lower/mid/upper) become ONE mechanism — text-conditioned readouts.

Two refinements:
1. **Descriptive phrases, not opaque tokens.** e5 can't place "dir-blend"; it can place "blended direction
   estimate from graph walk and LLM judges". "Lower bound" vs "upper bound" differ by one word — the learned
   translation can amplify that axis, but give it words to work with.
2. **Optional learned residual per frequent function** (name-translation + small delta) so heavily-used judges can
   specialize beyond what their name says, while rare/new functions ride on the name alone.

## Confounding risks (user asked; answered honestly)

| risk | verdict | mitigation |
|---|---|---|
| `mu_PoE` from own readouts = tautology/feedback | REAL | stop-grad on derived targets; measured-data anchor required |
| naive product target imports the independence error | REAL | that error is the point — it's the *prior*; the correlated correction comes from measured data |
| constructed fusion target treated as ground truth | REAL | label as distillation of a fusion rule (guardrail from PoE design) |
| folding V into the fused head | REAL | keep V/Sigma a separate head |
| three heads + bounds + names = too much at once | MANAGEABLE | the pieces are separable; build/evaluate incrementally (below) |

## Build order (each step separately evaluable)

1. **Run the existing 6-way comparison** (`DESIGN_product_kalman_poe.md` evaluation plan) on the already-scored
   real data — the harness exists, no new scoring needed. This is the missing WHAT-answer for the ~25 merged
   Product-Kalman infra PRs.
2. **Per-channel heads** (`mu_graph`, `mu_LLM`) trained on measured targets only — check they don't degrade the
   trunk (the judge-independence result says varying judges helps it).
3. **Fused + bound heads** (`mu_lower/mid/upper`) with the two-target recipe (stop-grad prior + measured anchor).
   Evaluate the disagreement interval for routing/abstention value (AURC, margin-gated selective risk).
4. **Function-name embeddings** — swap the embedding table for the e5-translation, ablate vs indexed embeddings.
5. **Wire `V(hop)`** as the gain for an explicit at-inference update over the per-channel heads; compare
   filter-at-inference vs `mu_PoE`-head-only on held-out fresh data (the two-timescale question, measured).

## Where the Kalman filter fits (user follow-up, 2026-07-08)

*(`mu_MOE` above was a typo for `mu_PoE`; the mixture row in the bound lattice is kept because MoE is genuinely the
middle of the lattice — and see the gate note below.)*

**The filter does not learn — it is the closed-form algebra that USES what the model learned.** The gain
`K = P H^T (H P H^T + R)^-1` and the posterior are fixed math; everything learnable lives in its inputs:

| Kalman object | supplied by | learned? |
|---|---|---|
| prior mean | mu heads (model readouts) | yes — amortized per-channel estimates |
| prior covariance `P` | `Sigma(hop, corpus)` head | yes — the confirmed result IS "P is predictable from graph position" |
| measurement `y` | graph walk / LLM judge at inference | no — evidence |
| measurement noise `R` (+ cross-channel corr) | residual calibration (`fit_residual_covariance`, calibration diagnostics) | yes — from held-out residuals |
| gain `K` | computed | never |

So the division of labor is: **the model learns the mean and correlation statistics; the Kalman update is the
closed-form use of them.** Architectural bet, stated crisply: *learn the uncertainty, derive the fusion weights
for free* — vs an attention/gating network that learns weights directly. Kalman form = sample-efficient +
interpretable where the Gaussian approximation holds; keep it as structure and learn the statistics feeding it.
(MoE note: the Kalman gain IS a mixture-of-experts gate, just computed from covariances rather than learned;
true learned MoE would enter later as regime routing, e.g. per-corpus experts.)

**Single update vs the actual filter.** A pair scored once uses only the measurement-update step (= Gaussian
conditioning, one Bayes step). The *filter* — the recursion — earns its name on a SEQUENCE, and the filing system
is one:

- **State:** stored `(mu, P)` per relation/membership in the filing DB.
- **Measurements over time:** user files an item, a judge scores a pair, a graph edge appears — each is one
  incremental update on the stored `(mu, P)`, NO retraining.
- **Static state (`Q=0`):** reduces to recursive least squares — `P` shrinks ~1/n as evidence accumulates.
- **Drift (`Q>0`):** ontology evolves / categories reorganized / judge calibration shifts — process noise gives
  exponential forgetting; **`Q` is the learnable drift-rate statistic** (classical adaptive filtering).

Deployment picture: model supplies amortized priors → filing DB holds per-relation `(mu, P)` posteriors updated
incrementally as evidence arrives → slow timescale distills accumulated posteriors back into the model. In filter
language, `mu_PoE` is the model *amortizing the update itself* — predicting the filter's posterior so the common
case needs no explicit algebra.

## What the model can and cannot learn about its own error (user, 2026-07-08)

*User: "means and correlations are something the model could also learn — but perhaps not the means and
correlations of the error."* Confirmed, and it is the deepest constraint in the design:

- **Signal means/correlations: fully learnable** (the mu heads, the joint structure).
- **The error MEAN: unlearnable from inside, in a strong sense.** Any predictable component of the model's own
  error gets absorbed into the mean estimate and *ceases to be error* — so the in-distribution error mean is ~0
  by construction, and what remains is exactly what the model cannot see. Out-of-distribution the error mean
  (bias) becomes nonzero precisely where the model has no way to know it; bias is visible only against EXTERNAL
  measurements (Kalman: bias states are unobservable without measurements).
- **The error COVARIANCE: learnable, but only from outside the training loop.** Second moments survive mean-
  fitting, so a `Sigma(hop)`-style head can live in the model — under two conditions the pipeline already
  respects: (1) fit on HELD-OUT residuals (training residuals are optimistically small; joint mean+variance NLL
  has the variance-eats-the-loss pathology — fit mean first, then residual covariance, as `fit_hetero`/
  `sigma_hop_confirmatory` do); (2) in-distribution only — error statistics are the first casualty of shift,
  which is why the fresh-corpus confirmation mattered most for the ERROR model.

**Crisp form: signal statistics are statistics of the world; error statistics are statistics OF THE MODEL, and
estimating statistics of the model requires standing outside it** — held-out residuals, innovations, fresh
corpora. Adaptive Kalman filtering does exactly this: `R`/`Q` are estimated from the innovation sequence
(residuals against real measurements), never from the filter's own predictions.

**This closes subsume-vs-hybrid with an epistemic argument, not a convenience one:** even under perfect
distillation, the model's self-estimated error is trained on PAST innovation statistics; when the world shifts,
only real measurements reveal the new error mean. The innovation loop is the only mechanism that can see the
model's bias — so the explicit filter (fast timescale) is necessary, and the distillation (slow timescale) can
absorb everything EXCEPT the role of standing outside.

## Error statistics as metastable random-walk states (user, 2026-07-08)

*User: the dynamic errors aren't learnable, but their STATISTICS can be modeled as a metastable state with drift
(a random walk).* Yes — this is the classical closure: **promote the error statistics into the state vector with
random-walk dynamics**, converting "unlearnable constants estimated offline" into *trackable states estimated
online from innovations*:

```text
b_{t+1}     = b_t     + w_b       (error MEAN/bias as random-walk state — classical "bias augmentation")
log s_{t+1} = log s_t + w_s       (error VARIANCE as random-walk state — stochastic volatility)
rho_{t+1}   = rho_t   + w_rho     (error correlation likewise)
```

The epistemic constraint survives: these states are observable only THROUGH the innovation stream (real
measurements). But the "standing outside" is now wired into the filter as a timescale hierarchy:

- **Level 0:** relation state `z` — fast, per-pair.
- **Level 1:** error statistics `(b, sigma, rho)` — slow random walk (the metastable state with drift).
- **Level 2:** drift rates `Q_1` of level 1 — quasi-fixed (how fast do my error statistics wander?).

`Sigma(hop, corpus)` slots in as the ATTRACTOR MAP: **the model learns the metastable attractors from pooled
history; the online filter random-walks around the current attractor; distillation periodically updates the
attractor map.** The two-timescale hybrid restated at the meta-level — the model holds the climate, the filter
tracks the weather.

**Jumps vs drift (the "metastable" refinement):** a pure random walk is diffusion and over-smooths JUMPS, and this
system has known jump sources — a judge version change (gpt-5.5 → gpt-6) is a STEP in `R_LLM`, not a drift; a
category reorganization is a step in the corpus Sigma. Metastability = plateaus + occasional transitions, which
wants either heavy-tailed process noise (Student-t `w`, jump-tolerant) or a switching state-space model (regime
variable + per-regime statistics). A switching Kalman filter IS a mixture of experts over regimes — the MoE hunch
in its natural home (third appearance: lattice middle, gate-from-covariances, regime routing). Practical middle
ground: random walk + innovation-consistency test (normalized-innovation-squared chi-square; sustained excess ⇒
temporarily inflate `Q` — classical adaptive fading), handling jumps without full switching machinery.
