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
> **Superseded:** §(e) gives a graph-native alternative that needs no embedding, and the DECISION at §(f) adopts it.
> This section is retained for comparison; skip to §(e) for the chosen approach.
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

## (e) Graph-native effective-h via random walk — no second model (user 2026-07-05)
Can the *graph alone* give a continuous effective-h (avoiding the embedding/second model for §d)? Yes — the
**up-walk hitting probability** `P(uniform-random up-walk from desc visits anc)` (exact DP: `h(anc)=1`,
`h(x)=mean over parents`, roots→0):

| h | mean hit-prob | within-h CV | 0.9^h | mean e5 cos |
|---|---|---|---|---|
| 1 | 0.529 | 0.48× | 0.900 | 0.900 |
| 2 | 0.327 | 0.83× | 0.810 | 0.851 |
| 3 | 0.246 | 1.02× | 0.729 | 0.814 |
| 4 | 0.201 | 1.12× | 0.656 | 0.775 |
| 5 | 0.181 | 1.21× | 0.590 | 0.775 |

- **Continuous** — 418 distinct values over 1500 pairs (vs 5 integer hops); graph-only.
- **Mean-reverting** — flattens toward a base-rate floor (~0.18) instead of decaying to 0 like `p^h`. Fixes the
  `p^h → 0` asymptote (a distant ancestor is a near-random pair) *for free* — the walk dilutes across branches, so
  "the probability of drifting back rises the further you drift" (user) is intrinsic.
- **Within-hop variance grows with h** (0.48×→1.21×) — refines exactly where integer-h is coarsest.
- **Direction encoded for free** — `P(up-walk anc→desc)=0` structurally (up-walks never descend), so `μ_rev=0`
  automatically; the fwd/rev asymmetry needs no `1−p^h` construction, and it converges toward the common root.
  **Two distinct "reverses" (user):** (a) *invert the arguments* → `μ(anc|desc)=hit_prob(anc,desc)=0` — the correct
  reverse *membership* (an ancestor is genuinely not under its descendant); this is the right `μ_rev`, and it shows
  why `1−p^h` was wrong (it conflated direction-*uncertainty* with reverse-*membership*, inflating μ_rev to 0.41).
  (b) *reverse the walk direction* → a **down**-walk `P(anc→desc)` is small-nonzero but is still a *forward*
  containment viewed top-down (fan-out-diluted: a broad ancestor "reaches" any one member weakly) — a useful
  *specificity-weighted* second direction signal, **not** the reverse.
- **Path-agnostic** — `hit_prob` sums over *all* routes, so a pair sampled at shortest-path h via BFS may read a
  hit-prob for a *shorter effective distance* if alternate short paths exist (part of why h=1 mean is 0.53 not 0.9:
  branching, not distance). This is intended (effective, not shortest-path, membership).

**Design principle — IS vs OUGHT (user 2026-07-05).** `μ_rev=0` answers *is* the ancestor structurally a member of
its descendant (no — a descriptive graph fact). The `1−p^h` complement was reaching for *ought* it be uncertain at
distance (an epistemic claim). Different questions — and the architecture assigns each to its right source: the
**graph** component supplies the crisp structural **IS** (`μ_rev=0` — trains faster and keeps a clean basis for
source-separation), and the epistemic **OUGHT** (confidence softening at distance) enters through the **judge we
superpose with** (the LLM/semantic side). So the *blended* `μ_rev` is not 0 — it is the graph's structural 0 mixed
with the partner's uncertainty. We deliberately keep the graph target factual and let the superposition carry the
ought; baking ought into the graph would muddy the very basis the superposition needs to separate.
- Caveat: **+0.27 corr with e5 cos** — *complementary* to semantic distance, not a replacement; at h=1 it's 0.53
  (not 0.9) because branching dilutes membership when a node has many parents (arguably *correct* for effective
  membership — each of many parents is a weaker container).

**Implication:** a graph-native transitive target `μ_fwd(desc|anc)=hit_prob`, `μ_rev=0` needs **no embedding and no
LLM** for the graph side of the superposition — it is continuous, mean-reverting, root-converging, and
direction-correct by construction. Relative to §(d)'s effective-h, this *replaces* the second model rather than
calibrating against it.

### (f) Walk-target TRAINING — direction FLOORS above 0.5 (user), and walk beats p^h at depth
Trained a model on `--target walk` (`μ_fwd=hit_prob, μ_rev=0`, + the h=1 element/subcat superposition), evaluated
`μ_fwd` and **direction sign-accuracy** vs h to h=8 on held nodes (n=200/hop):

| h | prod μ / dir% | p^h μ / dir% | walk μ / dir% |
|---|---|---|---|
| 1 | 0.880 / 100% | 0.807 / 100% | 0.807 / 99% |
| 3 | 0.065 / 88% | 0.446 / 98% | 0.166 / 98% |
| 5 | 0.000 / 52% | 0.202 / 91% | 0.069 / 94% |
| 6 | 0.001 / 40% | 0.144 / 86% | 0.050 / 90% |
| 7 | 0.000 / 30% | 0.079 / 79% | 0.031 / 87% |
| 8 | 0.000 / **25%** | 0.071 / 66% | 0.032 / **79%** |

**Direction floors ABOVE 0.5 (user's prediction, confirmed):** root convergence makes the more-root-ward node
consistently the ancestor, so even where the *magnitude* floors near the base rate the *direction* survives — walk
holds **79% at h=8**, p^h 66%, both far above chance. `prod` (no transitive training) instead falls **below** chance
(25% at h=8) — its μ collapses to 0 and noise dominates, so it's actively wrong deep down. **Walk beats p^h at every
deep hop** (79 vs 66% at h=8): the graph-native mean-reverting/root-converging target teaches a *more robust*
deep-hop direction than the exponential `p^h`, and needs no second model to define it. **Single-seed — the
walk-vs-p^h *margin* should be validated multi-seed before it is cited as a quantitative claim** (the direction
*floor* trend and the `μ_rev=0` structure are robust; the exact margin is not yet).

**DECISION (user 2026-07-05): the WALK target is the graph side of the transitive superposition.** The deciding
reason is *architectural, not the accuracy margin*: the walk needs **only the graph** — the same definition on
every corpus — so it **avoids the "which model/embedding measures effective distance?" question** that §(d)'s
effective-h forces (pick an embedding, verify it separates hops on that corpus — e5 barely does past ~3 hops here,
§d — then re-tune). §(d) effective-h and the `p^h` rule (§c) are demoted to **deferred alternatives** kept for
comparison. (The walk-beats-p^h deep-hop margin is single-seed and a bonus; the decision stands on self-sufficiency
+ the confirmed direction floor.)

## Takeaways
- The discrimination operator as trained is **direct-membership** (μ collapses by h=3); transitive `p^h` behaviour
  is a **design target requiring explicit training**, with p a measurable per-source leakage base (≈0.88 here).
- **Direction (sign) survives ~2 hops past membership magnitude** — a cheap transitive signal is there even without
  a magnitude rule.
- Semantic leakage is **large and slightly upward-tilted** on Wikipedia — consistent with why direction is "too
  easy" here; a lower-leakage source is where transitive + superposition effects would actually show.

Repro: sample chains from 100k_cats at h=1..5 → build e5 (with ancestor closure) → μ_HIER(desc|anc) vs h + frozen-e5 up/down dots.
