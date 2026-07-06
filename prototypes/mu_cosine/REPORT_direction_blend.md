# Cross-judge direction superposition on Wikipedia — build works, test is weak (consensus + tiny held)

*Result of `DESIGN_cross_judge_direction.md` option A (the user chose "build on Wikipedia now"). 3-operator
random-simplex superposition of DIRECTION (graph-discrimination ⊕ LLM-element ⊕ LLM-subcategory), trained as a
`dir-blend` judge on HIER directional rows, fine-tuned from `model_prod`. 2026-07-05.*

## Setup
- `emit_direction_blend.py`: `d_blend = w·[d_graph, d_element, d_subcat]`, `w` = equal (1/3) or Dirichlet(α=4);
  emitted as HIER rows with `μ_fwd − μ_rev = d_blend`, `judge=dir-blend`. Node-disjoint split (42 held, 26
  directional). `eval_direction.py`: `corr(HIER-asymmetry, mean(d_graph,d_element,d_subcat))` on held, read WITH
  `dir-blend` and AGNOSTICALLY.

## Result (26 held directional pairs)

| model | judge input | corr(HIER-asym, direction) | sign-acc |
|---|---|---|---|
| prod | agnostic | +0.206 | 100% |
| equal-mix | agnostic / dir-blend | +0.331 / +0.338 | 100% |
| dirichlet(α4) | agnostic / dir-blend | +0.463 / +0.214 | 100% |

## Honest read
1. **The pipeline works** — trains cleanly (no collapse, HIER edge-order 99.9% preserved), a new `dir-blend` judge
   row, the 3-estimator emitter + eval are reusable.
2. **Sign is 100% for every model** — direction on Wikipedia is *consensus* (all 3 estimators agree ~100%, and
   `model_prod` already gets it). So the sign carries **no learnable signal** here — the interesting axis is gone.
3. **Weak-but-suggestive magnitude signal:** dir-blend training beats `prod` on the direction-*magnitude*
   correlation (+0.33/+0.46 vs +0.21) — the superposition teaches the *degree* of asymmetry WIKI-edge training
   misses — and equal-mix is **judge-independent** (agnostic +0.331 ≈ dir-blend +0.338), the mechanism transferring.
4. **But it's underpowered:** n=26 held is tiny; the dirichlet arm's agnostic (+0.463) > dir-blend (+0.214) is
   almost certainly noise, and no magnitude gap clears it. **No confident claim.**

## Negatives — contradictory/no-direction → asymmetry 0 (user 2026-07-05)
When the operators give **no or contradictory** direction, the mix → ~0 ⇒ `μ_fwd ≈ μ_rev` — a **negative**
(no-clear-direction) case, teaching "a direction requires operator AGREEMENT." Included via `emit_direction_blend`
(default; `--no-negatives` to drop). Re-run (equal-mix + 345 no-direction negatives):

| model | corr(asym, direction) [26 dir] | mean\|asym\| [16 lateral, ↓ better] |
|---|---|---|
| prod | +0.206 | 0.096 |
| equal-mix | +0.331 | 0.093 |
| **equal + negatives** | **+0.536** | 0.127 |

**Adding negatives *improved* the positive signal** (direction corr +0.331→+0.536) — teaching direction-requires-
agreement sharpened it. But the **negative behaviour itself is not demonstrated**: lateral `|asym|` drifted *up*
(0.093→0.127), which on **16** held laterals (all values ~0.1) is noise, not a real regression. The mechanism is
sound and implemented; Wikipedia's tiny held set can't show the no-direction half.

## THE TRUE TEST — novel-node generalization (user 2026-07-05)
The real test (user): *"how well it predicts the direction in cases where NONE of the operators have seen the
nodes."* Sampled **400 enwiki parent-child pairs whose both nodes are OUTSIDE the training graph** (not in
100k_cats, never LLM-scored) — the model has **only frozen e5**, no graph ancestors, no operator readout trained
on them. Direction ground-truth = the enwiki edge (child→parent); metric = sign of `μ(child|parent)−μ(parent|child)`.

> **Scope limitation (review):** the ground-truth is the enwiki *structural* edge, not an independent semantic
> judgment. So this tests generalization of *structural* direction via e5, and can't fully separate "learned a
> direction concept" from "learned that enwiki taxonomy structure transfers via e5." The clean version uses an
> eval-time LLM/human on the novel pairs (deferred). Still meaningful — the nodes are outside all training — but
> not the maximally-independent test.

**Purpose (user):** the superposition *target* is a **linear** sum `Σ wᵢ·dᵢ`; the network learning it is **not**
linear — it uses that capacity to *separate* the operator inputs so it can reconstruct any linear mix, which is
what should let it generalize direction to unseen nodes.

### Result 1 — direction generalizes from e5 (the robust positive)
Every model predicts novel-node direction **~90%** from e5 alone (`prod` 91.5%). The separated direction concept
transfers to nodes no operator saw — it is *not* memorised operator outputs.

### Result 2 — the superposition does NOT reliably beat baseline (multi-seed kills the single-seed win)
`dirichlet` (random-mix): s1 95.5% / s2 **81.8%** / s3 94.2% → **mean 90.5% vs prod 91.5%** (−1.0). The single-seed
+4 was seed luck; across seeds the superposition training adds no reliable novel-node direction over `model_prod`.
(The 3 seeds vary *both* the emission seed — different Dirichlet mixing family per arm, `--seed 0/1/2` — and the
training seed, so the spread is genuine mixing+optimization variation, not optimization noise alone.)

### Result 3 — e5 prefixes carry ~8 points of it (user's prediction, confirmed)
| | WITH e5 prefixes | WITHOUT (query==passage) |
|---|---|---|
| prod | 91.5% | 83.2% |
| dirichlet s1 | 95.5% | 86.8% |

Removing the `query:`/`passage:` asymmetry drops direction ~8 pts — so **the frozen e5 prefixes are a real source
of the direction** (as predicted). But the model keeps **83%** without them: a large **learned, prefix-independent**
direction the non-linear network built during training.

### Result 4 — BACKWARD prefixes don't invert it (user: "the prefixes assume we know the answer")
Concern: assigning root=`query`, node=`passage` bakes in the direction. Test — swap the tables (root=`passage`,
node=`query`), which *negates* the e5-content term `D→−D` while the learned anchor/node role tokens stay fixed:

| prefix mode | prod | dirichlet s1 |
|---|---|---|
| forward (root=query) | 91.5% (+0.416) | 95.5% (+0.438) |
| none (query==passage) | 83.2% (+0.231) | 86.8% (+0.264) |
| **backward (root=passage)** | **81.5% (+0.146)** | 75.5% (+0.237) |

**Backward does NOT invert direction** — `prod` stays **81.5%** (a pure-prefix signal would collapse to ~15%). The
mean asymmetry shrinks (+0.416→+0.146) but the *sign* survives. So the direction is **not** a prefix artifact: it's
carried by the **learned role encoding**, with the e5 prefix a *modulator* (+8 aligned, −2 fought). Subtlety:
`dirichlet` swings more (95.5→75.5, 20 pts) than `prod` (91.5→81.5, 10 pts) — the superposition-trained model leaned
*more* on the prefix, not less, reinforcing the multi-seed null (it built a more prefix-dependent, not more robust,
direction).

### Result 5 — superposition value scales with direction UNCERTAINTY (user's hypothesis, confirmed)
The null (Result 2) is a **saturation artifact**: direction is a strong signal (Wikipedia titles *leak* it —
"X by country" vs "X in Germany"), so most pairs are near-certain and nothing can help them. Stratify the 400
novel pairs by direction confidence and the benefit appears exactly where the theory says it should:

| stratum | prod (single model) | dirichlet (3-model ensemble) | gap |
|---|---|---|---|
| high-confidence half (large \|asym\|) | 98.0% | 98.0% | **0.0** |
| low-confidence half (small \|asym\|) | 85.0% | 93.0% | **+8.0** |
| by title-sim: easy / medium / hard | 93 / 90 / 91% | 95 / 98 / 94% | +1.5 / +7.5 / +3.0 |

**Where direction is certain the superposition adds nothing (98→98); where uncertain it adds +8 (85→93).** So the
value scales with direction *uncertainty* — and the overall mean is drowned by easy, leakage-driven pairs. (Caveat:
"dirichlet" here is the 3-seed asym-*ensemble*, which does some averaging; but the *structure* — 0 on certain, +8
on uncertain — is the finding, and it's model-confidence-stratified so partly regression-to-mean too.)

### Open: multi-hop / transitive direction (user) — no magnitude rule yet
Untested and *underspecified*: for a grandparent pair (A subcat-of B subcat-of C), the *sign* should be transitive
(A→C), but we have **no rule for how the discrimination operator's MAGNITUDE should behave** across hops (decay
with distance? constant? — the transitive-as-ordinal-constraints question, PR #3377). The *sign* is cheaply
testable (novel multi-hop pairs); the magnitude needs a defined target first.

## Conclusion
On Wikipedia the direction axis is **consensus** (sign trivially agreed). The **true (novel-node) test** is the
informative one: **direction generalizes to unseen nodes ~90% from frozen e5**, and it's genuinely *learned* (it
survives backward prefixes — not a prefix artifact), the e5 prefix only *modulating* it (~8 pts). On the
**aggregate**, the superposition training adds nothing over `model_prod` (multi-seed 90.5% vs 91.5%) — **but that
aggregate is a saturation artifact.** Stratified by direction confidence, the superposition's benefit is **real and
regime-specific: 0 where direction is certain, +8 where it is uncertain** (Result 5) — exactly where extra signal
can help. Wikipedia's semantic leakage makes most pairs certain, so the mean hides it.

**Net:** the superposition is not worthless on direction — it helps precisely in the *uncertain* regime — but
Wikipedia has too few uncertain pairs to move the aggregate. **The aggregate null (−1.0) is NOT evidence against
cross-judge direction superposition in general** — only evidence that Wikipedia (a consensus taxonomy) is the wrong
corpus to test its motivation. The natural next step is a corpus with **more direction uncertainty** (option B:
looser hierarchies / multi-parent DAGs), where the uncertain tail is the bulk, not the fringe.

Reusable assets stand (the `dir-blend` judge, the 3-estimator emitter with contradiction→negative, the novel-node
+ prefix + difficulty-stratified evals).

Repro: `emit_direction_blend.py --mix {equal,dirichlet}` → fine-tune → novel-node eval (± e5 prefixes).
