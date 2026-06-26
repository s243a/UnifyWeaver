# Multiple μ sources & the estimation architecture — dual vs unified

Extends `DESIGN_inferred_operator_superposition.md`. There, the operator of an *inferred* relation is drawn
from a posterior `P(operator | μ, …)` measured by a single μ (the trained model's prediction). This doc adds
two things: (1) the distribution should condition on **more than one** μ estimate, and (2) **who owns the
distribution** — a separate estimator (dual) or the model itself (unified).

## 1. The distribution conditions on multiple μ sources

Every model is a *measurement* of μ for a pair — the frozen **e5** cosine, the **MuAttention** we are
training, later an **LLM** / **miniLM**. The posterior conditions on the set:

```
P(relation | μ_e5, μ_model, …)  ∝  P(relation) · ∏_k P(μ_k | relation)^{w_k}    (product-of-experts, NOT naive independence)
```

We start with **two** sources — e5 (static) and the training model (dynamic) — and the set is open (add an
LLM source later with the same interface). Each source's μ is mapped onto a comparable scale by a per-source
calibration **transform** before it enters `P(μ_k | relation)`.

> **API note (proposed, not yet built).** The transform lives on a *proposed* `CalibratedModel(name, scorer,
> transform)` wrapper. The **current** ensemble API (`bridge_ensemble.py`) is plain `(name, scorer_fn)` tuples
> in `BridgeEnsemble` — no transform. `CalibratedModel` is the future-work addition; do not treat it as
> existing code.

### ⚠ The sources are NOT conditionally independent — don't use a naive product
A naive `∏_k P(μ_k | relation)` assumes the μ sources are independent given the relation. **They are not:**
the MuAttention model **consumes e5 features and is warm-started toward e5**, so `μ_model` and `μ_e5` are
strongly correlated. Multiplying their likelihoods **double-counts the shared e5 evidence** and *over-tightens*
the posterior — which would **undercut** the very claim that adding sources reduces measurement noise. The
naive product is therefore an **optimistic bound**, not the estimator.

**Worked example (why it over-confidences).** Say for a pair the true relation is `see_also` but e5 is
mildly fooled, so both correlated sources report a membership-ish μ with per-source likelihood ratio
`P(μ|element_of)/P(μ|see_also) = 3`. Treated as independent, the product gives a ratio of `3 × 3 = 9` →
`P(element_of) ≈ 0.9`. But the two "votes" are mostly the *same* e5 signal seen twice; the honest combined
ratio is closer to `3` (≈0.75) — or, if `μ_model` adds little beyond e5, barely above `3` at all. The naive
product turned one piece of (wrong) evidence into a confident error.

**Corrections (any of):**
- **Down-weight correlated sources** — exponents `w_k < 1` in the product-of-experts above; in the extreme,
  treat `{e5, model}` as **one** expert until the model's *residual* over e5 is shown to carry signal.
- **Measure the dependence on the tagged set** — estimate `corr(μ_e5, μ_model | relation)`; set `w_k` (or a
  full covariance) from it. Only the part of `μ_model` *not predictable from* `μ_e5` is new evidence.
- **Model the residual** — condition on `(μ_e5, μ_model − E[μ_model | μ_e5])` so the second factor is the
  decorrelated increment.
- **Calibrated log-linear combiner / temperature** — learn the combination weights on held-out tagged data
  so the posterior is *calibrated* (its confidence matches its accuracy) rather than asserted.

The static-vs-dynamic framing below is partly *why* this matters: e5 is the shared backbone the model is
built on, so "two sources" is really "one anchor + the model's increment over it."

## 2. Static vs dynamic sources (and why it matters)

- **e5 μ — STATIC.** e5 is frozen, so `P(μ_e5 | relation)` is estimated **once** from the tagged data. It is
  an **anchor**: it gives real signal from **step 0** (before the model is any good), and because it never
  drifts it directly damps **noise source (d), model churn**.
- **model μ — DYNAMIC.** `P(μ_model | relation)` must be **re-estimated from the error** as the model
  improves; it starts uninformative and *sharpens* over training. Early on the posterior leans on e5; late,
  on the model.

Weight each source by how well its `P(μ_k | relation)` actually **separates** the relations (e.g. by the
mutual information between μ_k and relation on the tagged set) — a poorly-discriminating source contributes
little. The static/dynamic split also schedules itself: e5 dominates early, the model takes over as its μ
becomes discriminative.

`P(μ | relation)` is the **generative** direction (easy, robust — one histogram per relation per source);
inverting to `P(relation | μ)` is **Bayesian inversion**. That is the sensible way to build it: estimate the
easy generative term, invert for the discriminative posterior.

## 3. Who owns the distribution — dual vs unified

| | **Dual model** | **Unified model** |
|---|---|---|
| structure | a **distribution estimator D** (owns `P(μ\|relation)` for each source) **+** a **function model F** (μ predictor / operators) | one **F**; training is *constrained* so F's error distribution matches `P(μ\|relation)` |
| how they meet | **EM**: E-step re-estimates D from F's current predictions on the tagged set; M-step trains F with D fixed | one loss: latent relation marginalised, `P(relation\|μ)` taken from F's *own* μ |
| stability | **stable** — D is a fixed target during each M-step; D can be plain histograms, swappable, or a tiny calibrator | **fragile** — F can "explain away" by reshaping *both* its μ and the implied distribution ⇒ collapse to the easy operator |
| static e5 | drops in trivially as a fixed part of D | must be **pinned** as a non-movable posterior term to anchor against collapse |
| cost | two things to maintain; D lags F by one E-step | one model; co-adapts; harder to reason about |

### The fork collapses
A **unified model with a stop-gradient, slowly-updated (EMA / periodic) posterior estimate IS the dual
model**, implemented cheaply inside one network — the **target-network / self-distillation** pattern. The
distribution becomes a *target F fits but cannot game within a step*. Two ingredients keep it from the
degenerate solution:
- **stop-gradient** on the posterior estimate (F fits it; the loss can't reshape it to cheat), and
- the **static e5 anchor** — a model-independent term in the posterior F can't move at all.

### Recommendation
Go **dual-in-spirit**: keep distribution estimation **out of the gradient** (a statistical target, EMA'd),
conditioned on `{e5 (static), model (dynamic, stop-grad)}`, e5 anchoring against collapse. Concretely D is:
per source, per relation, an EMA'd histogram of μ on the tagged set; the static e5 histograms are computed
once. F trains inferred rows via the **soft posterior-weighted operator loss** (see
`DESIGN_inferred_operator_superposition.md`, *"The model — operator as a random superposition"*) using D.

A **fully coupled unified** model is the elegant end-state but needs the entropy + out-of-set-mass + anchor
regularisers (see `DESIGN_inferred_operator_superposition.md`, *"What the noise actually captures"*) *just to
not collapse* — the same machinery as the dual approach with more ways to fail. Treat it as a later
experiment, not the first build.

## 4. Build order (after the theory branch merges)
1. **D, static part:** compute `P(μ_e5 | relation)` once from tagged pairs (e5 cosine; build the proposed
   `CalibratedModel` transform on top of the existing `BridgeEnsemble` `(name, scorer_fn)` API).
2. **Measure e5↔model dependence** on the tagged set (per §1's correction) and set the source weights `w_k`
   accordingly — do this *before* combining, so the posterior is calibrated, not naively multiplied.
3. **D, dynamic part:** every N steps, EMA-update `P(μ_model | relation)` from F's predictions on tagged
   pairs (stop-gradient).
4. **F training:** for inferred rows, soft posterior-weighted operator loss using the *weighted* combined
   posterior; tagged rows untouched. A/B vs v1 (fixed-breadth) and no-switch.
5. **Later:** add an LLM μ source; explore the fully-unified variant with the anti-collapse regularisers.
