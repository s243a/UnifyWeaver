# Multiple μ sources & the estimation architecture — dual vs unified

Extends `DESIGN_inferred_operator_superposition.md`. There, the operator of an *inferred* relation is drawn
from a posterior `P(operator | μ, …)` measured by a single μ (the trained model's prediction). This doc adds
two things: (1) the distribution should condition on **more than one** μ estimate, and (2) **who owns the
distribution** — a separate estimator (dual) or the model itself (unified).

## 1. The distribution conditions on multiple μ sources

Every model is a *measurement* of μ for a pair — the frozen **e5** cosine, the **MuAttention** we are
training, later an **LLM** / **miniLM**. The posterior conditions on the set:

```
P(relation | μ_e5, μ_model, …)  ∝  P(relation) · ∏_k P(μ_k | relation)      (naive-Bayes; assume k ⟂ given relation)
```

This is exactly the **bridge ensemble** feeding the operator posterior, and the `CalibratedModel(name,
scorer, transform)` wrapper (see `REPORT_bridge_ensemble.md` discussion) is where each source's μ is mapped
onto a comparable scale before it enters `P(μ_k | relation)`. We start with **two** sources — e5 (static)
and the training model (dynamic) — and the set is open (add an LLM source later with the same interface).

Combining independent measurements **tightens** the posterior (reduces noise source (c), measurement
uncertainty) and lets disagreement among sources surface genuinely ambiguous pairs.

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
once. F trains inferred rows via the **soft posterior-weighted operator loss** (`DESIGN_inferred_operator_
superposition.md` §1) using D.

A **fully coupled unified** model is the elegant end-state but needs the entropy + out-of-set-mass + anchor
regularisers (op-superposition §3) *just to not collapse* — the same machinery as the dual approach with more
ways to fail. Treat it as a later experiment, not the first build.

## 4. Build order (after the theory branch merges)
1. **D, static part:** compute `P(μ_e5 | relation)` once from tagged pairs (e5 cosine via the
   `CalibratedModel` wrapper).
2. **D, dynamic part:** every N steps, EMA-update `P(μ_model | relation)` from F's predictions on tagged
   pairs (stop-gradient).
3. **F training:** for inferred rows, soft posterior-weighted operator loss using the combined posterior;
   tagged rows untouched. A/B vs v1 (fixed-breadth) and no-switch.
4. **Later:** add an LLM μ source; explore the fully-unified variant with the anti-collapse regularisers.
