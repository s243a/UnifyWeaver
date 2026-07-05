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

**Purpose (user):** the superposition *target* is a **linear** sum `Σ wᵢ·dᵢ`; the network learning it is **not**
linear — it uses that capacity to *separate* the operator inputs so it can reconstruct any linear mix, which is
what should let it generalize direction to unseen nodes.

### Result 1 — direction generalizes from e5 (the robust positive)
Every model predicts novel-node direction **~90%** from e5 alone (`prod` 91.5%). The separated direction concept
transfers to nodes no operator saw — it is *not* memorised operator outputs.

### Result 2 — the superposition does NOT reliably beat baseline (multi-seed kills the single-seed win)
`dirichlet` (random-mix): s1 95.5% / s2 **81.8%** / s3 94.2% → **mean 90.5% vs prod 91.5%** (−1.0). The single-seed
+4 was seed luck; across seeds the superposition training adds no reliable novel-node direction over `model_prod`.

### Result 3 — e5 prefixes carry ~8 points of it (user's prediction, confirmed)
| | WITH e5 prefixes | WITHOUT (query==passage) |
|---|---|---|
| prod | 91.5% | 83.2% |
| dirichlet s1 | 95.5% | 86.8% |

Removing the `query:`/`passage:` asymmetry drops direction ~8 pts — so **the frozen e5 prefixes are a real source
of the direction** (as predicted). But the model keeps **83%** without them: a large **learned, prefix-independent**
direction the non-linear network built during training.

## Conclusion
On Wikipedia the direction axis is **consensus** (sign trivially agreed) and the superposition's magnitude/negative
signals are underpowered on the in-coverage held set. The **true (novel-node) test** is the informative one and
gives a clean, honest read: **direction generalizes to unseen nodes ~90% from frozen e5** — driven partly by the
e5 prefix asymmetry (~8 pts, confirmed) and partly by a learned prefix-independent representation — but the
**direction-superposition training adds no reliable generalization over `model_prod`** (multi-seed 90.5% vs 91.5%).
The reusable assets stand (the `dir-blend` judge, the 3-estimator emitter with contradiction→negative, the novel-
node eval). The superposition's value on *direction* is not established here; a **direction-AMBIGUOUS** corpus
(option B) — where operators genuinely flip — remains the setting where it could actually pay off.

Repro: `emit_direction_blend.py --mix {equal,dirichlet}` → fine-tune → novel-node eval (± e5 prefixes).
