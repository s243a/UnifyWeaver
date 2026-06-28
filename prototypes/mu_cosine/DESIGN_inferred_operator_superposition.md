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
- **Hard sampling** — draw one operator per step from the posterior (a stochastic switch). Simple; one
  forward; higher gradient variance.
- **Soft posterior-weighted loss** — the *expectation* over operators in LOSS space,
  `L = Σ_op P(op | μ) · mse_under(op)`. Lower variance, but **K forwards** per inferred row (one per operator).
- **Random operator EMBEDDING** (preferred) — the expectation in **INPUT space**: build one op token that is a
  random *superposition* of the candidate operator embeddings,
  `op_token = (w ~ Dirichlet(α · P(op|μ))) · op_emb  + out_of_set_noise·ε`, and do a **single forward**. The
  model literally sees a "superposition operator." `w @ op_emb` is a torch matmul so gradients still reach the
  operator embeddings (`w` is a detached random constant, like a dropout mask). The noise decomposition's two
  knobs live here: **α** = posterior-spread + measurement/churn variance (a,c,d); **out_of_set_noise** = the
  true operator is none of the candidates (b). `sample_operator_weights` / `random_operator_embedding` in
  `mu_posterior.py` (torch-free reference + tests); tagged rows keep their fixed `op_emb[op]`.

Hard sampling is the Monte-Carlo approximation of the soft loss. The **random embedding is NOT the same
expectation** — `model(Σ wᵢ·op_embᵢ)` ≠ `Σ wᵢ·model(op_embᵢ)` because the model is nonlinear in the operator
token (Jensen) — it is a cheaper **stochastic surrogate**: a single forward through an input-space mixture of
operator embeddings, which makes "operator = random superposition + noise, drawn from the fitted joint"
literal at the cost of being an approximation, not the loss-space expectation. (It equals the expectation only
if the readout were linear in the op token.) It is the build target *because* it is cheap (one forward) and
injects the uncertainty as input noise; the soft loss is the exact-but-K-forward alternative to A/B against.

## 1b. The probability structure: a finite categorical, realised as attention

The superposition weights are a **finite categorical on a simplex** (positive, sum to 1) over the finite
basis — `token = Σ wᵢ·valueᵢ` is a finite sum, **no integration**. (The alternative — a continuous *measure*
over superposition-vectors, with expectations as integrals — is declined; §1.)

This is exactly an **attention** read. Each basis entry is a **(key, value)** pair:
- **query** = the μ-feature vector; **keyᵢ** = the direction that sets how much entry *i* fires
  (`wᵢ = softmax(query·keyᵀ)ᵢ`); **valueᵢ** = what entry *i* contributes to the token.
- **key ≠ value** in general (untied). For a **frozen anchor**: `value` = `e5(tag phrase)` (frozen, the stable
  interpretable contribution) while the `key` is **learnable but calibrated** by the anchor-confidence KL
  (labels teach *when* it fires; *what* it contributes stays pinned). For a **learnable atom** (unknown
  relation): both `key` and `value` are learned — it competes with the anchors in the same softmax for the
  residual attention mass.
- The basis need **not** be orthogonal — softmax normalises regardless, and we only ever take convex
  combinations of values.
- The Dirichlet noise is **sampled** (one simplex point per step), not integrated; the soft-loss form is a
  finite sum over the basis. Both integration-free.

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

## 5b. Training-integration spec (the build)

Four rules govern how the random operator embedding is wired into training:

1. **Blend on UNLABELED rows only.** Tagged rows keep their fixed `op_emb[op]` and exact μ target; the random
   superposition + the joint-posterior assignment apply to **inferred** (`confidence < 1`) rows. Labelled
   data is ground truth — never blur it.

2. **Relation→operator marginalisation — embedding at OPERATOR level, target at RELATION level (PR #3359
   review).** The joint head outputs `P(relation | μ_vec)`, but the model only has 4 operator tokens, and the
   map is **many-to-one** (`element_of→ELEM`; `subcategory/super_category/subtopic→WIKI`;
   `see_also/assoc/bridge→SYM`). So split the two consumptions:
   - **Op-token embedding** uses the **operator marginal**
     `P(op|μ_vec) = Σ_relation P(op|relation)·P(relation|μ_vec)` (`P(op|relation)` is the deterministic map) —
     `op_token = (w ~ Dirichlet(α·P(op|μ_vec)))·op_emb`.
   - **The μ target stays at the RELATION level** — crucial because relations sharing an operator have
     **different** targets (under `SYM`: `bridge ≈ 0.9`, `see_also ≈ 0.4`, `assoc ≈ 0.3`). So
     `μ_target_dir = Σ_relation P(relation|μ_vec) · target_dir(relation)`, NOT a single per-operator value.
     Marginalising the target to the operator first would wrongly collapse SYM's three relations to one μ.

3. **Asymmetric operators carry TWO prior μ — both directions.** `WIKI`/`ELEM` are directional, so each
   relation under them has forward `target_fwd ≈ 0.9` and reverse `≈ 0.1`; `SYM` relations are symmetric (one
   value each). So an inferred pair generates **two** training examples — `(node, root)` and `(root, node)` —
   each with the *same* random-superposition op token but the **direction-specific relation-blended target**
   from rule 2 (forward uses each relation's forward target, reverse its reverse). This is why the readout
   vector keeps `wiki_fwd/rev` and `elem_fwd/rev` separate.

   **Inference-time discipline (PR #3359 review):** the Dirichlet *sampling* is training-time only — at
   eval/inference use the **deterministic mean** (`α→∞`, i.e. `op_token = P(op|μ_vec)·op_emb`, no noise), and
   keep the training-time sampling on its own isolated RNG (as the v1 switch already does).

4. **Curriculum — labelled first, then unlabelled.** The posterior is only as good as the model's μ readouts
   and the joint head fitted on them; both are poor early. So **train on tagged data only until the joint
   `P(relation|μ_vec)` is reasonably fit** (e.g. held-out accuracy/log-loss has plateaued, or a fixed warm-up
   of steps), *then* introduce the inferred rows with the blend. This breaks the chicken-and-egg (a bad
   posterior would mis-assign operators to unlabelled data while the model is still bad) — a warm-up gate,
   not a hard switch. The joint head is re-fit periodically on the tagged set (EMA / stop-grad) as the model
   sharpens.

## 6. Status

- **Shipped (v1, C):** `confidence` carried through fuse → graded pairs → trainer; `--infer-switch`
  hard-switches inferred `element_of`→`subcategory` with `p = base·min(1,breadth/scale)·(1−conf)`, drawing
  from an **isolated** RNG (so switch-off/on share the batch-sampling/masking trajectory).
- **A/B (clean, isolated RNG, same seed — only the operator differs):**

  | metric | switch OFF | switch ON |
  |---|---|---|
  | discrimination (argmax) | 89% (32/36) | **94% (34/36)** |
  | WIKI order-acc | 99.8% | **100.0%** |
  | SYM held-out | +0.830 | +0.834 |
  | ELEM corr | +0.698 | +0.662 (small trade-off) |

  **Correction:** an earlier run reported 89% → **97%**, but that used a *shared* RNG, so switch-on perturbed
  the whole training trajectory — the A/B was confounded (PR #3356 review, high-severity). With the RNG
  isolated the honest gain is **89% → 94%** (+2 examples on a 36-item probe): a real but **modest**
  improvement, ~half the originally-claimed magnitude, with a small ELEM trade-off. Treat as suggestive, not
  decisive, at this probe size.
- **Next (E→F):** estimate label-data `P(μ | relation)`, switch the trainer to the soft posterior-weighted
  operator loss, add the out-of-set mass + measurement-width terms; A/B against v1 and against no-switch.

## 7. The superposition is a REGULARIZER — blending tagged data, and the capacity bound

§1–§6 framed the random operator superposition narrowly as a way to handle **inferred**-label uncertainty,
and §5/§5b accordingly **gate it to inferred rows only** (tagged rows keep a hard `op_emb[op]`). That is a
*special case*. The mechanism is more general: the Dirichlet-sampled `op_weights` + noise is **stochastic
regularization on the operator axis** — the same family as dropout and label-smoothing. It spreads each
example's operator information across parameters rather than letting one hard operator token carve a brittle
path, so it improves robustness/generalisation, not just inferred-label calibration.

**Unification (every row, tagged or not).** A row's operator distribution is `label_prior × μ_posterior`, and
its **confidence sets the Dirichlet concentration α** (sharpness ⇒ how much noise):
- **Tagged** row → a *sharp* label prior, lightly softened by the μ-posterior. A label is **evidence, not
  certainty** — "it is a label" ≠ "100% confidence information" — so its prior is sharp but **not a delta**.
  High α ⇒ mild spread ⇒ *operator-label-smoothing*.
- **Inferred** row → *no* label prior, so the μ-posterior is the whole mean. Low α ⇒ wide spread.

So the current "tagged rows untouched" gating = the `confidence = 1.0 ⇒ α→∞ ⇒ no spread` corner. The general
knob is a **tagged confidence `< 1.0`** (a single `--blend-tagged-conf`, or a per-method value): run tagged
rows through the *same* sampler and they become a mild regularizer over the **whole** set.

**Why this matters here.** `REPORT_infer_blend_cx.md` found the blend at *parity* with the cheap v1 switch
because fuzzy tagging shrank the inferred set to **136 / 3370** (low-diversity, mostly element_of/subtopic +
bridges) — the regularizer was **starved**. Blending tagged data removes that bottleneck: the regularizer
sees all 3370 rows, with the label as an **additional prior** alongside the μ-priors, instead of only the
inferred remnant. This is the lever to pull "if we need more blended data" without harvesting more.

**The binding constraint is CAPACITY.** You can only regularise to the degree the model has spare capacity to
absorb the spread — "spread the information across the parameters" needs parameters to spread into. This is
exactly the **2→3-layer ELEM-interference** finding (`REPORT_element_operator.md`: 2 layers couldn't
co-serve discrimination + page-centrality; 3 could). So more blend-noise wants more capacity, and
**blend-strength must be evaluated *paired* with capacity** — a fixed-capacity sweep over tagged-blend
confidence conflates "regularisation helped" with "the model had room for it". Larger models tolerate
(and benefit from) more. (On *why* a parameter-count criterion like AIC can't answer "noise or layers" for a
net — and the singular-learning-theory analog that can, Watanabe's **WAIC / WBIC** with the learning
coefficient in place of `k` — see `NOTES_model_selection_capacity.md`.)

**Proposed test (paired, multi-seed — methodology per `REPORT_infer_blend_cx.md`).** A 2×2+ over
{capacity: layers 3 vs 4 / wider d_model} × {tagged-blend: off vs `--blend-tagged-conf` 0.8–0.9}, ≥3 seeds,
reading the **train-vs-held-out generalisation gap** (the regulariser's actual target) alongside
discrimination/SYM. Hypotheses: (i) tagged-blend narrows the generalisation gap; (ii) the gain is larger at
higher capacity; (iii) at fixed small capacity, over-blending *underfits* (degrades discrimination), the
signature of exceeding the capacity budget.

## 8. Generalisation: an ANCHORED BASIS + a learnable residual (open-set relations)

§1–§7 superpose a **fixed, all-learned** operator basis. The generalisation: make the relation/operator basis
**two-part — frozen label-tied anchors ∪ K learnable residual atoms** — keeping the same finite-categorical
machinery (the basis stays finite, so the weights stay a probability simplex and the token stays a finite
sum: *no integration*, per §1's measure-vs-categorical choice).

### The basis
- **Frozen anchors** — one per known principle tag (`element`, `subcategory`, `bridge`, `see_also`; the full
  relation set as needed). Embeddings are **fixed** (`requires_grad=False`), naturally seeded from the e5
  phrase embeddings (`e5("see also")`, …) — i.e. the existing `EMBED_EXEMPLARS`, consistent with the
  frozen-e5 philosophy. Frozen ⇒ stable meaning + stable calibration; they cannot drift.
- **K learnable atoms** — `--n-learnable-atoms` (default **5**, configurable). Free embeddings that absorb
  structure the fixed tags do not name. This makes the **out-of-set mass** (noise source (b), §3) *explicit
  and learnable* instead of a generic "other" floor.

### The token
`token = Σ wᵢ · basis[i]` over the **enlarged** basis (anchors ++ atoms), with `w = softmax(...)` on the
simplex. Two distinct bases are in play (the key/value distinction): the **projection** directions that
*produce* `w` (classifier weights / `P(relation|μ)`) need not equal the **superposition** basis that gets
*blended* (anchors ++ atoms).

### The binding constraint (the crux — keeps the known part honest)
Without a constraint the model would route mass to the learnable atoms and quietly ignore the labels (the
collapse risk of §7 / the learned-confidence discussion). Fix: **on labelled rows, the probability mass on
the frozen anchors must match the label and its confidence** — a cross-entropy/KL term pinning `P(anchors)`
to the confidence-calibrated target. The atoms then receive only the **residual** mass (the uncertain /
out-of-set portion); they cannot override a confident `element_of`. The anti-collapse anchor is exactly the
**provenance confidence** (the `--blend-tagged-conf` dial, here expressed as the anchor-loss weight).

### What the atoms buy
- **Representation** — a relation like *by order* (a faceting/organising link that is none of the four cleanly)
  can put mass on an atom instead of being forced onto subcategory/see_also.
- **Discovery** — inspect *what routes to the atoms*: cluster the inputs that load on them and you have
  candidate **new principle tags** to promote to frozen anchors. The atoms' non-interpretability is a feature
  for this — they are where the unnamed relations accumulate.

### Knobs & relation to the rest
- **K** (`--n-learnable-atoms`, default 5): too many → atoms overfit / siphon anchor mass; too few → limited
  residual capacity.
- **anchor-loss weight**: how hard the frozen anchors are pinned to confidence (= the `--blend-tagged-conf`
  dial as a constraint).
- §1–§7 are the **K = 0** special case (anchors only). This is an **open-set / prototype-plus-residual**
  model: fixed, calibrated, interpretable prototypes for what we know + a bounded learnable reserve for what
  we don't, with the supervised KL keeping the known part faithful.

### Build sketch (deferred — doc-first)
Split the operator/relation embedding into a frozen anchor block (`requires_grad=False`, e5-seeded) ++ K
learnable rows; `JointPosterior` outputs a distribution over `anchors ++ atoms`; add the anchor-confidence
KL on labelled rows. Then A/B vs K=0 on the generalisation gap + the discrimination/SYM headline, and probe
what the atoms specialise to.

## 8b. Choosing K, and growing it — incremental warm-started atoms

**Default K = 5.** Rationale is the *asymmetric cost* + diversity, NOT matching the current residual: an unused
atom merely gets no gradient (a few idle parameters, harmless), whereas too few atoms *bottleneck* distinct
emergent relations into one. So err high. A learned basis **on the order of the fixed anchors** is also a
clean symmetric design — comparable expressive room in the unknown space as the known. (An earlier
"cut to 2–3 to match the ~4% residual" was optimising the wrong risk.)

**Read utilisation as a FLOOR check, not a ceiling.** Instrument per-atom **mass-share** + atom **embedding
diversity**; the question is whether all K are saturated *and* the generalisation gap is still open (⇒ grow),
not whether to trim.

**Grow-and-prune (the adaptive K — a finite, warm-started Dirichlet-Process / CRP flavour):**
- **Spawn** a new atom when the existing atoms are all busy (high mass-share) AND residual/out-of-set mass is
  still high / the gap isn't closing — "open a new table when the occupied ones fill up." Self-terminates:
  when a freshly-spawned atom stays idle, the elbow is reached.
- **Warm-start** the survivors — keep previously-learned atom embeddings as initialisation; only the new atom
  starts fresh. (The "build up by fine-tuning" philosophy applied to the basis itself.)
- **Seed the new atom from the residual direction** — the mean / top principal direction of the inputs
  currently in the worst-fit / out-of-set mass — so growth is *targeted* and the atom specialises fast
  instead of wandering. A mild diversity term stops it duplicating an existing atom.
- **Prune** atoms that decay to near-zero mass. Grow + prune together converge K to the elbow automatically.
- **Re-equilibrate** with the early-stop loop after each growth; the **frozen anchors stay pinned to
  label-confidence throughout** — growth lives only in the learnable-residual subspace, so adding capacity
  never threatens the calibrated known part. (This is the safety property: grow freely, the supervised anchor
  keeps the labelled relations honest.)

**Sequencing:** build **fixed-K (=5)** first to validate the anchored-basis + learnable-residual mechanism
(it needs the utilisation metric anyway), THEN layer the grow/prune controller on top as v2 — the spawn/prune
triggers are defined in terms of that same utilisation signal.

## 8c. Query construction — fuse the categoriser output, the raw text, and the μ-evidence

The attention **query** (what we attend *from*) should fuse three signals, so the model learns *how much to
trust each*:
1. **Categoriser output** (provenance) — the fuzzy/lexical decision: `category` + `confidence` + `method`
   (embed the category as a small vector ++ the confidence scalar ++ the method). The **strong, calibrated**
   signal for KNOWN relations; it drives high-confidence anchor activation. The model **learns** the anchor
   calibration *from this signal* rather than having it imposed — and the **anchor-confidence KL target is
   derived from the same categoriser decision**, so feeding it as a feature + supervising with it is
   feature+label, not circular: the model learns to *reproduce a confident categoriser hit on the anchor
   block* ("high confidence for the fixed labels, **because** the categoriser said so").
2. **Raw-text e5 embedding** — `e5(section/node text)`. Direct semantic access: catches what the categoriser
   missed (→ atoms), lets the model **override/modulate** when the categoriser is wrong/absent, and handles
   the `None` / "by order" case (no categoriser signal ⇒ flows to raw-text + atoms).
3. **μ-feature vector** — the model's own predicted μ (e5, sym, wiki_fwd/rev, elem_fwd/rev): the **evidence
   that reconsiders** the label (§1).

Fusion: `q = q_proj([μ_vec ++ provenance ++ e5_raw])` → attend over the anchors ++ atoms. The learned weighting
is **confidence-modulated trust**: trust the categoriser when it fired confidently; fall back to raw-text +
atoms otherwise. (Implemented at the trainer-wiring increment — `AnchoredBasis` already takes a generic
`d_query`, so this is purely how the query is assembled.)

## 9. Haiku-expectation training + eval (the settled scheme)

After the anchored-basis A/B (architecture validated; the explicit per-operator distribution and the
text-in-query did NOT beat the simple version — `REPORT_anchored_basis_ab.md`), the operating scheme:

**Train on the EXPECTATION, not the full distribution.** The deliverable is the read-out μ, so train the
*blended* μ toward an expectation target, reusing the infer-blend (op_weights override + MSE). We do NOT
supervise `P(op)` per-operator (the elaborations that leaned on it didn't pay off). Trade accepted: we lose
the direct `P(op)_model` vs `P(op)_ref` eval and the explicit decomposition — but optimise the thing we use.

**Stochastic 70/30 target (per row, per step):**
- with prob = the row's label-confidence `c` (≈0.7): target the **label's** μ (calibrated, both directions);
- with prob `(1-c)`: target **Haiku's expectation** `E[μ] = Σ P_Haiku(op)·μ_op` (both directions).
This is the Monte-Carlo realisation of `c·label + (1-c)·Haiku` applied to the μ expectation. Keep
`op_weights` and the target on the **same** mixture (mismatching them distorts the per-op μ's).

**Asymmetric operators → both permutations:** train `E[μ(node|root)]` and `E[μ(root|node)]` separately
(REL_SPEC already stores fwd/rev).

**Eval against Haiku ALONE (not the training mixture).** Using the label-biased mixture as the eval reference
is circular (the model trained on the label). Independent reference = `corr(E[μ]_model, E[μ]_Haiku)` over the
**inferred** held-out rows — the population the anchored basis exists for, which SYM/disc never measured.

**`none` anchor (μ≈0):** a fixed "no relation" anchor for true negatives, DISTINCT from the learnable atoms
(novel *positives*). Closes the open-world set: known-positive anchors + `none` + atoms = "X / nothing /
something-unnamed". Lets Haiku place "unrelated" mass and pulls the expectation toward 0 for negatives.

**Haiku budget scope (one cached pass):** score each row ONCE → cache; reuse every epoch. The *must-Haiku*
set is the **inferred tail** (no label → full Haiku `E[μ]`). Clean labeled rows' `(1-c)` share can come free
from the model's own posterior; spend Haiku only on the tail (+ suspect labels). The same pass serves BOTH
the training target and the eval reference.

## 10. Overlap-aware partition — the softmax as learned effective disjoint boundaries (proposed)

### Why the finite categorical is valid even when relations overlap (the conceptual closure of §1b)
The relations are **not truly disjoint** — they correlate (the operator μ-readouts are ≈ +0.6–0.7 correlated;
element_of/subcategory co-vary). The naive worry: if relations overlap, "P sums to 1" breaks (independent
marginals would sum to > 1) and `E[μ]` is no longer a clean convex combination.

Resolution: **the softmax operates on the EVIDENCE space (the query / μ-vector), not on the relations** — and
there it imposes an **effective, *learned* partition.** "Disjoint" is a property of the partition the softmax
carves, not of the relations:
- It is a **soft tessellation** of evidence-space — each region gets a dominant anchor/atom ("in *this*
  region, μ is mostly this anchor").
- **Correlation lives in the *boundaries*, not the cells.** Where two relations genuinely overlap, the
  softmax *splits* the weight (a soft boundary) or an **atom** owns that region. Non-disjointness is absorbed
  as fuzzy boundaries / dedicated overlap cells — never a contradiction.
- The softmax **sharpness = how disjoint to treat them**, and it is *learnable* — correlated relations get
  softer boundaries, separable ones sharp boundaries; tuned per region.
- So `E[μ] = Σ w·μ` is a **smooth mixture mean over the learned partition**, bounded in [0,1] *regardless* of
  the true correlation. The categorical is valid because the softmax *manufactures* the effective
  disjointness in evidence-space.

### Proposed design: the disjoint-event partition
Generalise the operator distribution from "one relation per pair" to a **disjoint partition of the
relation-combination space** (the joint over relation *subsets*), realised directly by the anchored basis:
- **Singleton cells = the fixed anchors** — the known individual relations (e5-seeded).
- **Overlap / novel cells = the learnable atoms** — an atom firing for "element_of *and* subcategory" pairs
  *is* a disjoint overlap event with its own μ; an atom for an unnamed relation is a novel cell.
- **`none` cell = the negative anchor** (μ≈0) — closes the partition for unrelated pairs (absorbs mass,
  pulls `E[μ] → 0`; zero contribution, non-zero role).
- **softmax over [anchors ++ atoms ++ none] = the learned effective partition** (disjoint fractions, Σ = 1),
  and `E[μ] = Σ_cell P(cell)·μ_cell` is the convex combination over it.

**Mutual exclusivity is the special case** (mass only on singletons + none). The **multi-label
generalisation** activates from data: *promote an overlap atom* when a region of evidence-space persistently
**splits** weight between two anchors AND its μ differs from both — the §8b grow rule applied to overlaps.
That split-and-distinct signal is exactly how we'd detect we've outgrown the single-softmax model.

**Training / eval** ride §9 unchanged: target the **expectation** `E[μ]` over the partition (Haiku `E[μ]`
30% / label 70%, both directions for asymmetric cells); eval the model's `E[μ]` vs Haiku's over the inferred
tail. Nothing new is bolted on — the anchored basis *is* the partition, the atoms *are* the overlap/novel
cells, the `none` anchor closes it; we only add the *interpretation* + the overlap-promotion rule needed to
handle correlated relations principledly.

## 11. Training mechanics — sample superpositions, don't feed the mean (corrects §9 wording)

**Why we sample instead of feeding one averaged input.** The operator weights enter the model in two places:
the linear readout *and* the operator token that is **added to the input and passed through the non-linear
transformer** (attention + MLP). Because that path is non-linear, **the output of the *averaged* input is NOT
the average of the outputs** — feeding the mean superposition gives `μ(mean)`, which is *not* the expectation
`E[μ] = Σ P·μ`. (This is Jensen's inequality, and it's the §1 caveat: the blended-mean is a *biased shortcut*,
not the true expectation.) Each *sampled* superposition is its own non-linear forward; they don't collapse to
the mean. So we **sample**.

**Training — the §9 70/30, made precise:**
- **70% of steps:** train on the **label** (its operator + μ).
- **30% of steps:** train on a **random superposition informed (at least in part) by Haiku** — draw a
  superposition shaped by Haiku's distribution, feed it, backprop on its target.

**Coverage beats exact correlations — for *training*.** Training only needs good **coverage** of the
superposition space so the model learns the (non-linear) function across it. It does **not** need the exact
joint / correlations. Random Haiku-informed superpositions already give coverage; **knowing the correlations
just makes each sample more informative (better-placed), not required.** The §10 partition / exact
correlations matter most for an accurate *inference* `E[μ]`, far less for training coverage — so we don't
gate training on having them.

**Inference `E[μ]`** (where correlations *do* matter): the true expectation is the **Monte-Carlo average over
sampled superpositions**, not a single mean-fed pass (that pass is fast but Jensen-biased). For the
per-pair expectation, the samples are drawn from `P(op | pair)` (the §10 partition, correlations included).

**One-line summary:** *sample, don't average the input; for training, coverage (random Haiku-informed
superpositions) is enough; for inference, average the samples and let the correlations shape where they're
drawn.*

## 12. Resolutions to the PR #3373 review

Resolves the council's six comments + two spec-level blockers. **Authoritative where it clarifies §9–§11.**

**(1) `E[μ]` is CELL-level, never operator-level.** `E[μ] = Σ_cell P(cell)·μ_cell`, where a **cell is a
relation (or relation-combination, §10)** with its *own* μ — not an operator. Operators are only the shared
**readout head** (many cells → one operator's readout, §2); μ stays per-cell so co-operator cells with
different μ (SYM's `bridge`≈0.9 vs `see_also`≈0.4) never collapse. `P(cell)` may be marginalised to `P(op)`
for the embedding/readout, but **the expectation and its targets are computed at the cell level.**

**(2) The 30% Haiku branch — one contract (§11 supersedes §9's wording).** Per step, the Haiku branch: **draw
one cell** from `P_Haiku(cell)` (isolated RNG — see (4)), feed it **as sampled**, backprop on **that cell's
μ** (both directions for asymmetric cells). It does **NOT** target the scalar `E[μ]` — that is the
Jensen-biased mean (§11). `E[μ]` is an *inference* quantity (MC average — see (6)), not a per-step target.
§9's "target Haiku's `E[μ]`" is corrected to this **sampled-cell** form.

**(3) Haiku eval — held-out protocol + honest independence.** Split the inferred tail into **train-inferred**
and **held-out-inferred** *before* scoring; Haiku-score both once and **cache** (row-keyed). Train only on
train-inferred; eval only on held-out-inferred — **no per-row leakage**. *Correction:* this is independent at
the **row** level but NOT of Haiku's **style** (the model trains toward Haiku on other rows), so it measures
**agreement-with-Haiku on unseen rows, not ground truth**. §9's "no circularity" is **softened to "no per-row
leakage."** A genuinely independent check needs a **human-verified (or alternate-model) subset** — the
gold-standard upgrade.

**(4) Isolated RNG for the new draws.** All Haiku-branch cell draws use a **dedicated RNG**
(`random.Random(seed + fixed_offset)`), isolated from the batch/masking rng, so a Haiku-branch on/off A/B
shares the identical batch trajectory (the #3356 isolated-RNG lesson). No new draw touches the training rng.

**(5) Overlap partition — operational grouping key.** A row's **cell = the set of relations Haiku scores ≥ τ**
(`{rel : P_Haiku(rel) ≥ τ}`): one relation → a singleton cell; two+ → an overlap cell. **Atoms are promoted
per recurring relation-set** (the §8b grow rule, keyed on the recurring subset that persistently splits
weight and carries a distinct μ). The partition's grouping key is thus operational: Haiku's thresholded
multi-label relation-set per row.

**(6) MC inference — sample count + error bar.** `E[μ] = (1/N) Σ_i μ(sample_i)` over **N** sampled cells;
choose **N so `SE = σ_μ/√N ≤ target`** (default 0.02; with per-pair `σ_μ ~ 0.1` ⇒ N ≈ 25, so **N = 32
default**), and **report SE** per estimate; eval correlations carry their CI. The mean-fed single pass remains
a fast, Jensen-biased point estimate but is **not** the reported `E[μ]`.
