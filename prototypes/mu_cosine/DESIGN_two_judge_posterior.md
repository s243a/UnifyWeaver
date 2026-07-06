# Two-judge operator posterior: P(op | d, LLM_op) — combiner ladder, soft constraints, pseudo-judges

*Theory note from the 2026-07-05 pivot discussion (after the Wikipedia transitive-superposition arc showed the naive
blend fails). Captures the reframing that the right combiner is a JOINT POSTERIOR over the operator conditioned on
two judges — not a superposition of them — plus the soft-constraint / pseudo-judge modeling path. Design; not built.*

## The problem the superposition hit
`dir-blend` combined graph and LLM by **collapsing them into one blended target** (`μ_fwd = w_g·walk + w_e·llm…`).
That's a *linear superposition*, and it failed for structural reasons: (a) it blends incommensurate SCALES (the
branch-diluted walk probability, floor ~0.18, vs the LLM's fuzzy membership); (b) it conflates DIRECTIONAL and
LATERAL relations into one number; (c) `μ_rev` had no principled value (`0` vs `1−p^h` was a symptom).

## The reframe: two SEPARATE judges, one joint posterior
`op(x,y)` is a **distribution over relations** {subcategory, subtopic, element_of, super_category, see_also, assoc,
none}. Two judges condition it:
- **`d(x,y)`** — the GRAPH judge (IS): walk hit-prob on a multi-parent DAG, hop count on a tree. Asymmetric ⇒ it
  already carries direction. Free at inference.
- **`LLM_op(x,y)`** — the LLM judge (OUGHT): the semantic relation reading. Expensive at inference.

Combine by **conditioning, not blending**: `P(op | d, LLM_op)`. Nothing gets averaged, so nothing needs to be
commensurate — the scale mismatch, the directional/lateral conflation, and the `μ_rev` puzzle all dissolve
(`super_category`, `assoc` are just other coordinates of `op`). This IS `mu_posterior.py` / JointPosterior (#3359)
with `d` and `LLM_op` added as features.

## The combiner ladder (a moment expansion of the log-likelihood)
For a log-linear posterior `log P(op|φ)=θ·φ−A(θ)`: the **gradient** `φ_obs−E[φ]` is LINEAR in features (1st order);
the **Hessian** `−Cov(φ)` is the feature COVARIANCE — the `φᵢφⱼ` outer/cartesian products (2nd order = correlation).

| rung | uses | drops | cost |
|---|---|---|---|
| linear superposition | main effects (1st order) | all (co)variance | cheap; also a TRAINING device (distills a judge into the trunk) |
| confidence-weighted | + diagonal of Cov (each `σᵢ²`) | off-diagonal | needs the variances |
| joint distribution | + off-diagonal `σᵢⱼ` (the products) | — | needs a head + calibration + both judges as features |

The joint SUBSUMES the others (recovers them if the data supports it) and is the only rung that survives
**correlated** judges (`e5↔graph ≈ +0.75`): the off-diagonal it keeps is exactly the term linear/PoE throw away.
Superposition (`⊕`, a sum) cannot represent an interaction (`⊗`, a product) — correlation is rank-2, it lives in `⊗`.

## Soft constraints = Mahalanobis fusion (the second-order structure IS a Lagrangian penalty)
Fusing judges = minimizing `(estimate − judges)ᵀ Σ⁻¹ (estimate − judges)` — a quadratic objective whose metric is
the inverse covariance. That's a soft-constraint/penalty form (`λ·g²`), with the correlation as the penalty metric.
The **off-diagonal of Σ⁻¹ is a coupling constraint between judges** ("not free to disagree independently"). Its job
is to **stop double-counting** correlated evidence — the over-confidence we hit when we hand-set inverse-variance
weights. The same applies to OPERATOR correlations: `subcat↔element` anti-correlation is a mutual-exclusion soft
constraint on the operator simplex.

**Constraints must be SOFT, and crossing them is the mechanism, not a failure** (user): the correlation is itself
estimated from a churning, randomly-sampled corpus, so it has a standard error. A hard constraint overfits the
sample; the soft one is regularized, its stiffness tracking estimation/transfer certainty. The local likelihood
should *cross* it where domains genuinely diverge — and the stochastic minibatch churn makes the fit orbit the
boundary, enforcing it in expectation, not instantaneously.

## The diffusion data (P(op | d), applies-mass vs hop h; 40–50 pairs/hop, gpt-5.5-low)
| corpus | | h1 | h2 | h3 | h4 | h5 |
|---|---|---|---|---|---|---|
| SimpleMind | directional (sub+subtop) | 0.99 | 0.72 | 0.75 | 0.62 | 0.51 |
| SimpleMind | symmetric (see+assoc) | 0.66 | 0.60 | 0.50 | 0.49 | 0.41 |
| Wikipedia | directional | 1.36 | 1.16 | 0.94 | 0.74 | 0.60 |
| Wikipedia | symmetric | 0.30 | 0.39 | 0.34 | 0.27 | 0.34 |
| both | super_category | ~0.03 | ~0.01 | ~0.02 | ~0.00 | ~0.01 |

Findings: directional decays at *similar* rates in both (not the "clean persists longer" I predicted). SimpleMind
has **~2× the symmetric/`assoc` mass** — mindmap nodes are *concepts* (hierarchical AND associative); Wikipedia
*categories* are purer taxonomy. `super_category≈0` confirms the reverse-directional is ~0 either way. The corpora
have **different `P(op|d)`**, so a hard-transferred constraint would be violated — validating the soft/crossable view.

**Large-n consequence (user):** directional and symmetric CO-OCCUR (both high at h2), so they are positively
correlated — a real off-diagonal. At large n this is estimable and **must be modeled**; ignoring it (independent
operators) reads co-occurring mass as competing mass. This RECONCILES the "SimpleMind should be cleaner than the
LLM's marginals say" intuition: high `assoc` does NOT imply low directionality once the correlation is in — the
graph edge stays directional and the `assoc` rides alongside. The marginals hide it; the off-diagonal shows it.
(User disputes the LLM's high mindmap `assoc` on separate grounds — to revisit.)

## Modeling proposal: PSEUDO-JUDGES for the second-order constraints (user)
Represent each second-order structure (a correlation / soft constraint) as a **pseudo-judge** — a synthetic input
added to the judge set — so the existing linear/superposition + `judge_emb` machinery captures it. Two flavors:
- **Interaction pseudo-judge:** reading = a product of real judges (e.g. `readout_directional × readout_symmetric`,
  or `d × LLM_op`). A linear model over [real judges + interaction pseudo-judges] is second-order — the polynomial-
  feature / kernel trick. **This un-breaks linear superposition**: the product is non-linear in the original judge
  space but LINEAR in the augmented space (lift the `⊗` term to a new `⊕` coordinate).
- **Constraint pseudo-judge (virtual observation):** a fictitious observation encoding a KNOWN constraint (e.g. the
  Wikipedia-measured `directional↔symmetric` off-diagonal) added as pseudo-data with precision = constraint
  strength — the transferable soft prior, moved from the big messy corpus to the small clean one.

Why this fits: it reuses `judge_emb` (each pseudo-judge gets its own calibration row), the readout vector, and
superposition training — no new architecture. **Regularizing the pseudo-judge weights gives everything at once:**
n-dependent complexity (interactions come online only when n licenses them), soft/crossable constraints (weights
shrink under the prior, overridden by strong local data), and transfer (carry the pseudo-judge weight as a prior).

### The smallest build target: 2 operators → 3 pseudo-judges
Simplest case — a directional operator `D` and a symmetric operator `S`. The second-order structure is a symmetric
2×2 matrix: 4 entries, `Σ_DS = Σ_SD`, so **3 unique terms → 3 pseudo-judges** (= the degree-2 monomials of `[D,S]`,
= dim of a symmetric 2×2):

| pseudo-judge | reading | matrix entry | role |
|---|---|---|---|
| self-`D` | `μ_D²` | `Σ_DD` (diagonal) | directional reliability/curvature → *confidence-weighted rung* |
| self-`S` | `μ_S²` | `Σ_SS` (diagonal) | symmetric reliability/curvature → *confidence-weighted rung* |
| cross | `μ_D·μ_S` | `Σ_DS` (off-diagonal) | directional↔symmetric **correlation** → *joint rung; the large-n term* |

The two self-terms are the diagonal (the confidence-weighted rung); the one cross-term is the off-diagonal — the
large-n directional↔symmetric correlation that carries the co-occurrence. Scaling: **k operators →
`k(k+1)/2` pseudo-judges** (`k` self + `k(k−1)/2` cross). Start at k=2 (3 pseudo-judges): the smallest model that
contains the one correlation term that matters; only switch on further cross-terms as `n` licenses them.

### Two realizations of the pseudo-judges — GLM form and ATTENTION form
Same 3 pseudo-judges, two ways to compute them:
- **GLM form (append features):** add scalar columns `[μ_D², μ_S², μ_D·μ_S]` to the logistic feature vector; three
  learned weights = the three unique `Σ⁻¹` entries. Cheap, convex, interpretable (read off which carries weight).
- **ATTENTION form (compute the Gram matrix):** if the operator readouts are VECTORS, `μ_D·μ_S` is a *dot product*
  — which **is** the (uncentered) correlation. The three pseudo-judges are then the three unique entries of the
  **Gram matrix** `G = [⟨μ_D,μ_D⟩, ⟨μ_D,μ_S⟩; ·, ⟨μ_S,μ_S⟩]` (= `‖μ_D‖², ‖μ_S‖², μ_D·μ_S`), and `G` *is* the
  second-moment/correlation matrix. So "3 unique covariance terms" and "3 unique Gram entries" are the same object.

The closure: the model is **μ-attention** — its native op is the dot product (query·key). So the second-order
correlation is computed the *same way the model computes everything*: the cross pseudo-judge is an **attention score
between operators** (learned bilinear `μ_Dᵀ W μ_S`), the self-terms are operator self-attention (`‖μ_·‖²`), and a
single small operator-attention head over the operator representation vectors *learns the whole second-order
structure*. Two readings of the vectors:
- **sample vectors over a region** (`μ_D=[μ_D(xᵢ,yᵢ)…]` across a neighborhood): `μ_D·μ_S` = the empirical regional
  correlation — ties to per-region confidence (`c_mem`); the correlation is a neighborhood property, computed as a
  dot product over that neighborhood.
- **feature vectors per pair** (`μ_D∈Rᵏ`): `μ_Dᵀ W μ_S` = a learned bilinear alignment of the two operator
  representations for that pair.

This also tames the scaling: `k(k+1)/2` cross-terms is one `k×k` (or low-rank) `W` as a bilinear form — attention
handles the quadratic count natively instead of enumerating features. **First build:** k=2 GLM form (3 scalar
pseudo-judges) to validate the correlation term cheaply; then the operator-attention head as the architectural
realization if it earns it.

## Deployment: teacher/student
The joint `P(op | d, LLM_op)` needs `LLM_op` as a LIVE feature — expensive. So: **teacher** = the joint posterior
(LLM in the loop, offline label-maker); **student** = the model distilled on the FREE features (e5 readouts + `d` +
interaction pseudo-judges), superposition-style, for LLM-free serving. Joint-posterior quality, linear-deploy cost.
This also bounds how much LLM scoring we need — only enough to fit the teacher.

## Open questions / next
- Is `d` the walk hit-prob everywhere (needs multi-parent — Pearltrees) or hops-on-trees (SimpleMind) / walk-on-DAGs?
- Full non-parametric teacher vs a 2nd-order GLM with cartesian-product features (cheaper, interpretable — read off
  which pseudo-judges carry weight). Lean GLM-with-interactions.
- Fit the off-diagonal at Wikipedia scale; carry it as a constraint pseudo-judge to SimpleMind/Pearltrees.
- Revisit whether the LLM over-assigns `assoc` to mindmap concepts (user's dispute) — a judge-calibration question.
