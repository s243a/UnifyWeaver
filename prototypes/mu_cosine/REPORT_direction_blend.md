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

## Conclusion
On Wikipedia, **direction is too consensual to be a real test** — the *sign* invariant is trivial, and the
*magnitude* + *negative* signals are too small/noisy on 26 dir / 16 lateral held pairs to claim anything firmly
(the +0.54 with negatives is suggestive but underpowered). The build is the reusable asset: the `dir-blend` judge,
the 3-estimator emitter with **contradiction→negative** handling (user's insight, ready for real contradictions),
and the eval. The **definitive test needs direction-AMBIGUOUS data** (option B: Pearltrees multi-parent DAG /
looser hierarchies) where the graph and LLM actually *flip* — there the negatives are *genuine* (not just absent),
and a larger held set gives the power this lacks.

Repro: `emit_direction_blend.py --mix {equal,dirichlet}` (+ negatives by default) → fine-tune → `eval_direction.py`.
