# Inferred-relation operators as a μ-conditioned superposition — theory & alternatives

When a relation is **tagged** (an exact section header — "Subtopics", "Subcategories", …) we know its operator.
When it is **inferred** (a typo'd / missing section → structural fallback; or a title-match bridge) the
operator is *uncertain*. Rather than commit an inferred relation to one operator, we treat the operator as a
**random variable** and train under its distribution. This documents that model, what its noise actually
captures, and the alternatives considered. Companion to `DESIGN_provenance_and_representation.md` (the
`confidence`/inferred tag) and `REPORT_graded_round.md` (the harvest→categorise split).

## 1. The model — operator as a random superposition

The uncertain latent is the **relation** (`element_of`, `subcategory`, `see_also`, …); the **operator** /
readout head (`ELEM`, `WIKI`, `SYM`) is a *deterministic downstream map* of it
(`element_of→ELEM`; `subcategory`/`super_category`/`subtopic→WIKI`; `see_also`/`assoc`/`bridge→SYM`). So we
reason about a posterior over **relations** and apply the map; the title says "operator" because that map is
what the *trainer* ultimately switches.

```
P(relation | μ, node_type, breadth)        operator = OP_OF(relation)
```

— a *superposition* over the candidate relations (and an out-of-set "other"), weighted by that posterior,
**plus noise**. The measurement is the model's **predicted** μ for the pair — NOT the inferred target. The
target was the uncertain guess; the model's μ is the *evidence* that reconsiders it (a self-training / EM
flavour). Warm-starting makes the measured μ trustworthy from step 0. (Because the map is many-relations→one-
operator, distinct relations that share an operator — element_of/subcategory both touch membership but map to
ELEM/WIKI — are exactly the cases μ alone can't resolve; see §2.)

Two realisations of "train under the distribution":
- **Hard sampling** — draw one operator per step from the posterior (a stochastic switch). Simple; higher
  gradient variance.
- **Soft posterior-weighted loss** (preferred) — the *expectation* over operators,
  `L = Σ_op P(op | μ, type, breadth) · mse_under(op)`. Lower variance; the posterior's spread *is* the noise
  (a flat posterior automatically trains a mixture of operators; a peaked one trains essentially one).

Hard sampling is the Monte-Carlo approximation of the soft loss.

## 2. The posterior, and why it factors

μ does **not** carry all the information, so the posterior factors along (roughly) orthogonal axes:

- **μ → membership vs associative.** Tagged μ clusters: membership (`element_of`, `subcategory`) ≈ 0.9;
  associative (`see_also` ≈ 0.4, `assoc` ≈ 0.3). So μ answers *"is this inferred see_also actually a
  membership relation?"* — exactly the reconsideration that motivated this.
- **node_type / breadth → element vs subcategory.** μ **cannot** separate `element_of` from `subcategory`
  (both target ≈ 0.9). That axis comes from the endpoint **node-type** (page → element, category →
  subcategory) and the container's **breadth** (a broad domain is likelier to hold subcategories than leaf
  elements). This is what the v1 fixed-breadth switch approximated.

So `P(op | μ, type, breadth) ∝ P(μ | op) · P(type | op) · prior(op | breadth)`, with `P(μ | op)` estimated
**from the label data** — the model's predicted-μ distribution over the *tagged* examples of each relation —
re-estimated every N steps as the model evolves. The tagged rows are the calibration set; the inferred rows
are the things being classified.

## 3. What the noise actually captures (it is not one thing)

The "operator noise" is a sum of distinct uncertainties — keeping them separate is what makes the model
honest rather than a fudge factor:

| source | what it is | how it enters |
|---|---|---|
| **(a) μ-residual** | how much of the operator choice μ does **not** determine — the element↔subcategory axis, and within-band overlap | the **entropy** of the in-set posterior `H(P(op | μ,…))` — a flat posterior ⇒ more spread |
| **(b) out-of-set mass** | the probability the true operator is **not in the candidate set** at all (an unenumerated relation) | a reserved "other" mass / floor in the posterior (open-world term) — never 0 |
| **(c) measurement / estimation uncertainty** | μ itself is a noisy estimate (finite capacity, the pair's e5, ancestor sampling) | propagate a measurement width δ: integrate the posterior over μ±δ (blurs P(op|μ)) |
| **(d) model churn** | μ drifts as the model trains; a posterior estimated at step t is stale at t+k | absorbed as non-stationarity — widen δ / re-estimate `P(μ|op)` more often / EMA the estimate |

(a) and (b) live in the **posterior over operators**; (c) and (d) live in the **measurement of μ** that feeds
it. Total injected noise = posterior spread (a,b) blurred by measurement spread (c,d). In the soft-loss form
all four show up as how *distributed* the operator-weighted gradient is.

## 4. Alternatives considered

| # | approach | verdict |
|---|---|---|
| A | **Blanket `element_of`** (harvester contentType default) | ✗ the original bug — conflates subcategory / super_category / see_also into element_of |
| B | **Blanket `see_also` fallback** | ✗ loses specificity; throws away the section signal that *is* there |
| C | **Fixed-breadth hard switch** (v1, shipped) | ~ element_of→subcategory with p ∝ breadth. Works (discrimination 89%→**97%**), but ignores μ and the see_also axis — a crude stand-in for the posterior |
| D | **μ-conditioned hard sampling** | ✓ samples op ~ `P(op | μ, type, breadth)`; stochastic, higher variance |
| E | **μ-conditioned soft posterior-weighted loss** (recommended) | ✓✓ expectation over operators; entropy = built-in noise; lower variance; subsumes C and D |
| F | **E + full noise decomposition** (out-of-set mass + measurement width + churn) | ✓✓ the honest model — separates the four noise sources above |
| — | **μ-band → see_also relabel** ("maybe", earlier) | a *special case* of E: a hard threshold on μ instead of a learned posterior. Keep as a cheap baseline |

**Recommendation: E, growing into F.** v1 (C) stays as the gated fallback / A-B control.

## 5. Estimation & calibration details

- **`P(μ | relation)`**: bin the model's predicted μ (e.g. 20 bins on [0,1]) over the tagged examples of each
  relation; Laplace-smooth; EMA across re-estimations to damp churn (d). Re-estimate every N steps.
- **Prior `P(op)`**: the tagged relation frequencies (or uniform if you want the measurement to dominate).
- **Out-of-set floor (b)**: reserve a small constant mass to an "other" operator so the posterior never
  asserts certainty it doesn't have; tune as a hyper-parameter.
- **Measurement width δ (c,d)**: a single scalar (or grow it with model churn) that blurs `P(op|μ)`.
- **Gating**: applies to **inferred** rows only (`confidence < 1`); **tagged** rows (conf 1.0) keep their
  operator untouched. The categorisation `method`/`confidence` is the gate (provenance, per
  `DESIGN_provenance_and_representation.md`).

## 6. Status

- **Shipped (v1, C):** `confidence` carried through fuse → graded pairs → trainer; `--infer-switch`
  hard-switches inferred `element_of`→`subcategory` with p ∝ breadth. Discrimination 89% → **97%**.
- **Next (E→F):** estimate label-data `P(μ | relation)`, switch the trainer to the soft posterior-weighted
  operator loss, add the out-of-set mass + measurement-width terms; A/B against v1 and against no-switch.
