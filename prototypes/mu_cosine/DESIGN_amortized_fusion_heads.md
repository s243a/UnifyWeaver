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
