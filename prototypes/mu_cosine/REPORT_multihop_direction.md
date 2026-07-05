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

## Takeaways
- The discrimination operator as trained is **direct-membership** (μ collapses by h=3); transitive `p^h` behaviour
  is a **design target requiring explicit training**, with p a measurable per-source leakage base (≈0.88 here).
- **Direction (sign) survives ~2 hops past membership magnitude** — a cheap transitive signal is there even without
  a magnitude rule.
- Semantic leakage is **large and slightly upward-tilted** on Wikipedia — consistent with why direction is "too
  easy" here; a lower-leakage source is where transitive + superposition effects would actually show.

Repro: sample chains from 100k_cats at h=1..5 → build e5 (with ancestor closure) → μ_HIER(desc|anc) vs h + frozen-e5 up/down dots.
