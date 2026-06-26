# Label-data μ→relation estimator — static e5 source + label-anomaly review

First increment of the inferred-operator posterior (`DESIGN_inferred_operator_superposition.md`,
`DESIGN_mu_sources_and_estimation.md`): `mu_posterior.py` estimates `P(μ | relation)` from the **tagged**
pairs per μ source and Bayes-inverts to `P(relation | μ)`. This runs the **static e5 source now** (no
training loop) and ships the **label-anomaly review** rule.

## What it does
- `MuPosterior.fit_source(source, (relation, μ)…)` — smoothed per-relation histogram of μ for a source.
- `.posterior({source: μ})` — `P(relation | μ's)` as a **weighted product-of-experts** (`weights[source]`),
  not a naive product (the sources are correlated — see the design doc).
- `.band/.is_anomalous` — a relation's expected μ band ([q,1−q] quantiles); a TAGGED label whose measured μ
  is outside it is flagged for review.
- Pure-Python + numpy (torch-free core; `test_mu_posterior.py` 6/6). The e5 source loads an e5 cache.

## Finding 1 — e5 weakly separates relations (spread 0.040)
Per-relation e5-μ means on the tagged graded round:

| relation | mean μ_e5 | band[0.05] | n |
|---|---|---|---|
| bridge | 0.932 | [0.906, 0.954] | 88 |
| see_also | 0.853 | [0.787, 0.921] | 90 |
| super_category | 0.843 | [0.789, 0.899] | 154 |
| subcategory | 0.842 | [0.786, 0.904] | 396 |
| subtopic | 0.824 | [0.770, 0.883] | 80 |
| element_of | 0.818 | [0.763, 0.888] | 1098 |
| bridge_neg | 0.795 | [0.752, 0.840] | 182 |

e5 cleanly ranks the extremes — `bridge` (identity) top, `bridge_neg` (random) bottom — but the
membership/associative relations **overlap** (0.82–0.85). This **empirically confirms** the design claim:
e5 separates *membership vs associative / identity vs random* but **cannot** separate `element_of` from
`subcategory`. So e5 is a real but **weak** source ⇒ small weight; the *model* μ (dynamic, next increment)
must carry the finer axis. `P(relation | μ_e5)` is accordingly prior-dominated, shifting toward `bridge` only
at high μ.

## Finding 2 — label-anomaly review fires correctly (the side-note rule)
**214 / 2088** tagged labels have a μ_e5 outside their relation's band ⇒ flagged for LLM/human review. They
are exactly the suspect cross-dataset links: `norbit`, `ladybird-of-szeged`, `fast-and-frugal-trees ∈
cybernetics`, `BELBIC` — the same set the bridge ensemble surfaced. (e5 is recall-heavy here: it also flags
legit-but-opaque acronyms, which the *model* source — knowing the graph — would rescue; that's why the rule
is **review**, not auto-drop.)

## Finding 3 — dynamic model source separates far better, but is strongly correlated with e5
Adding the **model** μ source (`--model`, symmetric SYM μ, masked provenance) on the same tagged set:

| relation | model μ mean | (e5 μ mean) |
|---|---|---|
| bridge | 0.885 | 0.932 |
| see_also | 0.672 | 0.853 |
| subcategory | 0.648 | 0.842 |
| super_category | 0.644 | 0.843 |
| element_of | 0.541 | 0.818 |
| subtopic | 0.536 | 0.824 |
| bridge_neg | 0.413 | 0.795 |

- **Model μ separates the relations 3.4× better** — spread **0.136 vs e5's 0.040** — across 0.41→0.89, and
  even pulls `subcategory` (0.648) above `element_of` (0.541), the axis e5 couldn't touch (it has node-type).
- **e5 ↔ model μ correlation = +0.751** (strong) — because the model consumes e5. A **naive product would
  over-count** this shared evidence (the #3357 review concern, now measured).
- **Principled product-of-experts weights** (model 1.0; e5 = `(1−r²)·separability-share`): **e5 → 0.099.**
  e5 is a weak, mostly-redundant anchor (~10%); the **model carries the discriminative signal**. This is the
  non-independence correction applied with a number, not asserted.

## Next increments
1. **Soft posterior-weighted operator loss** for inferred rows (replacing v1's fixed-breadth switch), using
   `P(relation | μ_model, μ_e5)` with these weights; A/B against v1 / no-switch with the clean isolated-RNG
   harness. (The model source must be refreshed in-loop — EMA, stop-grad — as the model trains.)
2. Route the 214 anomalies through a budget-gated LLM/human pass and re-confidence.
4. Route the 214 anomalies through an LLM/human pass (budget-gated) and re-confidence.
