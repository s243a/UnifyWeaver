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

## Finding 4 — the full μ-readout vector + outlier rejection + the correlation structure
The posterior should condition on the **full vector** of μ readouts, not one number: raw e5 μ, the symmetric
SYM μ, and — for the ASYMMETRIC operators — **both directions** (`wiki_fwd/wiki_rev`, `elem_fwd/elem_rev`).
`mu_posterior.py --model --reject-outliers` fits all six and reports separability + the correlation matrix.

**Outlier rejection (all relation types, e5 out-of-band):** 214/2088 rejected, by relation
`element_of 110, subcategory 40, bridge_neg 20, super_category 16, see_also 10, bridge 10, subtopic 8` —
**element_of is by far the noisiest label** (110, ≈10%), confirming it most needs outlier rejection.

**Per-source separability:** `elem_fwd/rev 0.146 > sym 0.137 > wiki_fwd/rev 0.059 > e5 0.041`.

**Correlation matrix (the structure that matters):**
| | e5 | sym | wiki_fwd | wiki_rev | elem_fwd | elem_rev |
|---|---|---|---|---|---|---|
| e5 | +1.00 | +0.66 | **+0.15** | +0.12 | +0.56 | +0.51 |
| wiki_fwd | | | +1.00 | **−0.72** | +0.65 | −0.37 |
| elem_fwd | +0.56 | +0.71 | +0.65 | | +1.00 | **+0.11** |

Three reads:
- **e5 is the most *independent* source** of the directional readouts (e5↔wiki_fwd +0.15) — the weak but
  label-independent anchor; the operator readouts are mutually +0.6–0.7 (**circular** — trained together).
- **`wiki_fwd ↔ wiki_rev = −0.72`** — the two directions of subcategory's operator are strongly
  **anti-correlated**: conditioning on both *is* the asymmetry (your insight, realised without a hand-built
  axis).
- **`elem_fwd ↔ elem_rev = +0.11`** (near-symmetric) vs WIKI's −0.72 ⇒ **element_of is much *less*
  directional than subcategory** — an independent confirmation of the verified-Wikipedia check (subcategory
  asym 0.89 vs element 0.37), and *against* the "element strictest" intuition. Robust across two methods.

**Strictness verdict (verified Wikipedia + this correlation structure):** subcategory is the more
directional/asymmetric relation (proper-subset containment), element_of is more symmetric (instance ≈ its
topic). But both are **trained** to be directional, so directional asymmetry is largely a *trained artifact*;
it should NOT be a stand-alone "strictness axis" — instead feed `wiki_fwd/rev`, `elem_fwd/rev` as ordinary
readouts and let the weighted/decorrelated combiner use only their non-redundant part.

## Next increments
1. **Soft posterior-weighted operator loss** for inferred rows, conditioning on the **full readout vector**
   with weights that discount the mutual redundancy (e5 the independent anchor, model readouts down-weighted
   by their +0.6–0.7 correlation); A/B vs v1 / no-switch with the clean isolated-RNG harness. (Model readouts
   refreshed in-loop — EMA, stop-grad.)
2. Route the (now rejected) out-of-band labels through a budget-gated LLM/human pass and re-confidence.
