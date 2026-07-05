# Multi-hop direction: the 0.9^h transitive rule + upward/downward semantic leakage

*Follow-up to `REPORT_direction_blend.md` (user, 2026-07-05). Two questions: (a) does the discrimination operator's
magnitude decay like `p^h` (p≈0.9 base, h hops) in the transitive case? (b) is upward semantic leakage weaker than
downward? 300 ancestor-descendant chains sampled at exact hop distances h=1..5 from 100k_cats; μ_HIER(desc|anc)
from the trained model, plus frozen-e5 directional dot-products (model-independent leakage).*

## (a) Transitive magnitude — the model is DIRECT-membership, not 0.9^h

| h | μ_fwd(desc\|anc) | μ_rev(anc\|desc) | sign-acc | 0.9^h |
|---|---|---|---|---|
| 1 | 0.880 | 0.022 | 99.7% | 0.900 |
| 2 | 0.363 | 0.003 | 97.0% | 0.810 |
| 3 | 0.069 | 0.000 | 88.0% | 0.729 |
| 4 | 0.010 | 0.003 | 73.3% | 0.656 |
| 5 | 0.003 | 0.001 | 51.3% | 0.590 |

The trained μ_HIER **collapses far faster than 0.9^h** (0.88 → 0.36 → 0.07 → ~0 by h=3): it learned membership ≈
**direct edge**, because the WIKI training edges are direct parent-child. `0.9^h` is a sensible *target* but the
model doesn't produce it for free — it would need **explicit transitive training** (emit μ=p^h targets at h hops).
Note the **sign** (direction) degrades *gracefully* — correct to ~h=4, chance (51%) by h=5 — even as the magnitude
collapses; so the model keeps *directional* information a couple hops past where it keeps *membership* magnitude.

**Design implication (answers the open multi-hop question):** if we want the discrimination operator to be
transitive, `p^h` is the rule to train toward, with **p a per-source leakage base** (Wikipedia ≈ 0.88 at h=1). A
less-leaky corpus would have lower p ⇒ faster decay ⇒ more uncertain multi-hop direction ⇒ (per
`REPORT_direction_blend.md` Result 5) more room for the superposition to help.

## (b) Upward vs downward semantic leakage (frozen e5, model-independent)

Leakage is a property of the frozen e5 (what titles reveal), not the trained model. e5's `query:`/`passage:`
prefixes make it directional: UP = `passage:desc · query:anc` (narrow-as-content vs broad-as-query), DOWN reversed.

| h | e5 UP (p:desc·q:anc) | e5 DOWN (p:anc·q:desc) | up>down |
|---|---|---|---|
| 1 | 0.852 | 0.851 | 48.7% |
| 2 | 0.819 | 0.815 | 57.0% |
| 3 | 0.794 | 0.786 | 65.7% |
| 4 | 0.778 | 0.768 | 70.7% |
| 5 | 0.768 | 0.756 | 77.0% |

**Upward leakage is slightly HIGHER than downward** — the *opposite* of the hypothesis (which was flagged as
uncertain) — and the gap *grows* with h (up>down on 49% → 77% of pairs). Reading: a specific descendant as a
`passage` matched to a broad ancestor `query` scores marginally better than the reverse; broad-→narrow (downward)
is (very slightly) harder. Both stay high (~0.8) because Wikipedia titles are semantically clustered overall — the
leakage is *large* in both directions, with a small, systematic upward tilt.

## (c) Training on transitive hops → the operator becomes CONTINUOUS (user's fix, works)
Emitted transitive HIER targets (`emit_transitive_hops.py`): μ_fwd(desc|anc)=p^h, μ_rev(anc|desc)=1−p^h,
p=0.9, h=1..5, node-disjoint held. Fine-tuned from `model_prod` (+ the h=1 element/subcat superposition).

| h | target 0.9^h | prod μ_fwd | transitive μ_fwd | transitive sign-acc |
|---|---|---|---|---|
| 1 | 0.900 | 0.878 | 0.810 | 100.0% |
| 2 | 0.810 | 0.314 | 0.635 | 99.3% |
| 3 | 0.729 | 0.074 | 0.465 | 100.0% |
| 4 | 0.656 | 0.007 | 0.328 | 97.3% |
| 5 | 0.590 | 0.003 | 0.258 | 93.7% |

`prod` collapses to ~0 by h=3 (direct-only); the **transitive model decays smoothly** (0.81→0.26) and keeps
**direction correct 94% at h=5 vs prod's 51% (chance)**. It undershoots the exact p^h target (direct WIKI edges,
μ≈1 at h=1, pull the curve down), but the operator is now **continuous across hops** — the goal. A cleaner match to
p^h would come from down-weighting the direct edges or a larger transitive round.

## Open: LLM judges can score transitive pairs DIRECTLY (user 2026-07-05)
The graph side needs a *mathematical* transitive rule (p^h) because the graph has only direct edges. The **LLM
side does not** — it can judge a multi-hop pair *directly* ("is `1941_songs` ultimately under `Sound`?"). So the
transitive superposition should use **direct LLM scores** on the multi-hop pairs (no composition), and — bonus —
those scores **validate p^h**: does the LLM's transitive membership actually decay like 0.9^h? (Needs a scoring
run on the multi-hop chains; see §(d) below.)

## (d) Effective-h from semantic distance (user 2026-07-05) — calibration
Make the operator continuous *within* a hop: measure avg semantic distance per h, calibrate it, and use a
continuous **effective-h** (blending graph hops with an external-judge distance) in μ=p^(effective_h).

| h | mean e5 cos (query·query) | dist=1−cos | 0.9^h | adjacent-hop Cohen's d |
|---|---|---|---|---|
| 1 | **0.902** | 0.098 | 0.900 | h1–h2: 1.01 |
| 2 | 0.855 | 0.145 | 0.810 | h2–h3: 0.77 |
| 3 | 0.820 | 0.180 | 0.729 | h3–h4: 0.66 |
| 4 | 0.792 | 0.208 | 0.656 | h4–h5: 0.41 |
| 5 | 0.777 | 0.223 | 0.590 | |

Findings: (i) **mean h=1 cos = 0.902 ≈ p=0.9** — the base rate *is* the average h=1 semantic similarity, a clean
interpretation. (ii) distance is **smooth & monotonic** in h ⇒ calibratable to effective-h. (iii) **separable at
low h** (d≈1.0) but **poor at high h** (d=0.41, h4–h5) — leakage compresses distant nodes (e5 cos decays *slower*
than 0.9^h, 0.78 at h=5 vs 0.59). ⇒ effective-h refines within ~3 hops on Wikipedia; a **lower-leakage corpus or a
better embedding** extends the resolution. Construction: `effective_h = blend(graph_hops, embedding_distance)`,
μ=p^effective_h — connects to the reciprocal-target struct embedding (`structural_embedding.py`, `1/d = 3/(1+‖Δ‖)`).

### The two-part architecture (user 2026-07-05) — each side gets the transitive treatment it needs
The transitive superposition has **two distinct parts**, and the two ideas map to them 1:1:
- **GRAPH part** (discrimination operator): continuous via **effective-h** — integer hops refined by an
  **embedding** distance, `μ = p^(effective_h)`. Structure is only direct-edged, so it needs the calibrated rule.
- **NON-GRAPH part** (element / subcategory operators): **direct LLM** scores on the transitive pairs — the LLM
  judges any pair directly, so *no* mathematical composition is needed (and those scores also validate `p^h`).

So the LLM is the judge for the *non-graph* operators; the *embedding* is the distance for the *graph* effective-h.
*Deferred: (graph) build effective-h target + train; (non-graph) LLM-score the multi-hop chains, then superpose.*

## Takeaways
- The discrimination operator as trained is **direct-membership** (μ collapses by h=3); transitive `p^h` behaviour
  is a **design target requiring explicit training**, with p a measurable per-source leakage base (≈0.88 here).
- **Direction (sign) survives ~2 hops past membership magnitude** — a cheap transitive signal is there even without
  a magnitude rule.
- Semantic leakage is **large and slightly upward-tilted** on Wikipedia — consistent with why direction is "too
  easy" here; a lower-leakage source is where transitive + superposition effects would actually show.

Repro: sample chains from 100k_cats at h=1..5 → build e5 (with ancestor closure) → μ_HIER(desc|anc) vs h + frozen-e5 up/down dots.
