# Infer-blend A/B on the fuzzy-tagged, Complex-augmented round — a 3-seed honest re-test

The follow-up `REPORT_infer_blend.md` promised: re-run the posterior infer-blend vs the v1 fixed-breadth
switch on a **larger, cleaner, more diverse** inferred set — the regime where the principled blend was
predicted to finally separate from the cheap heuristic. This is that re-test, on the four-map systems-theory
round **with the new Complex systems theory map** and **fuzzy** section categorisation.

## The round
`build_graded_round.py` over cybernetics(2h) + dynamical(3h) + LTI(3h) + **complex(3h)**, `--section-method
fuzzy`:
- **1199 nodes, 1785 fused edges → 3370 graded targets** (was 2900 / 2488 before Complex).
- **INFERRED: only 136 / 3370** — vs **500** on the old `exact_phrase` round. The fuzzy layer (typo +
  tag/qualifier + paren + parent-signal) tagged almost everything, so the inferred set *shrank* rather than
  grew. Its composition: 88 structural-fallback (conf 0.40 — element_of 38, subtopic 28, assoc 22; pearls in
  no recognised section) + 48 bridge title-match (conf 0.80).

## A/B (warm-start `model_nodetype.pt`, 700 steps, bs 128, `--layers 3 --use-nodetype`, 3 seeds)

| metric | seed | no-switch | v1 switch | infer-blend |
|---|---|---|---|---|
| **SYM held-out corr** | 1 | +0.638 | +0.656 | +0.678 |
| | 2 | +0.641 | +0.647 | +0.657 |
| | 3 | +0.746 | +0.700 | +0.683 |
| | **mean** | **+0.675** | **+0.668** | **+0.673** |
| **multi-domain discrimination** | 1 | 88% | 92% | 96% |
| | 2 | 92% | 100% | 92% |
| | 3 | 96% | 100% | 100% |
| | **mean** | **92.0%** | **97.3%** | **96.0%** |

(WIKI held-out order-accuracy is 99.8–99.9% in every cell — saturated, not discriminating.)

## Honest verdict (the skepticism check earned its keep)
**Seed 1 alone looked like a clean monotonic win** for the blend (disc 88→92→96%, SYM +0.638→+0.656→+0.678).
Across **3 seeds it does not hold**:

1. **SYM corr is noise-dominated.** All three arms land ~0.67 (0.675 / 0.668 / 0.673) — indistinguishable.
   The no-switch arm *alone* swings 0.638→0.746 across seeds (≈±0.06), larger than any between-arm gap; the
   held-out SYM set is only 40 positives.
2. **Discrimination: both switch methods beat no-switch** (97.3% / 96.0% vs 92.0%, ~+4–5pp) — a modest,
   consistent edge (the switch/blend arm is ≥ no-switch in every seed). But the gaps are 1–2 probes on 25.
3. **The posterior blend does NOT beat the cheap v1 heuristic.** They are tied within noise (v1 is in fact
   marginally ahead on mean discrimination, 97.3 vs 96.0; tied on SYM). The seed-1 "blend > v1" was noise.

So this **replicates and strengthens** the prior parity finding: the principled posterior superposition
**matches** the fixed-breadth switch; it does not out-score it — and, contrary to the hypothesis, the
*cleaner* round did **not** change that.

## Why the cleaner round didn't help (the mechanism)
The hypothesis was "more/cleaner data → more diverse inferred → the blend's generality pays off." The opposite
happened to the *inferred* set: fuzzy tagging is **so effective** that it converts the genuinely-ambiguous
section headers into tagged rows, leaving an inferred set that is **small (136) and low-diversity** —
dominated by structural-fallback `element_of`/`subtopic` and `bridge`, exactly the element→subcategory case
v1 already covers. There is little left for the posterior's see_also↔membership↔subcategory generality to
exploit. The lexical labeling success (a real win) **removes the very ambiguity the blend was built to
handle**.

## Takeaways
- Ship the switch (either v1 or blend) — both give a small, consistent discrimination edge over no-switch.
- The blend's value remains **generality/principle**, not a number; it is at parity with v1 here.
- The blend would need a **genuinely ambiguous** inferred set to separate — which fuzzy tagging now prevents
  on this corpus. The place to find it is a corpus where section headers are paraphrase/semantic (the
  embedding layer's target, `REPORT_section_embedding.md`) rather than typo'd — i.e. *other users'*
  Pearltrees, not these well-labeled systems-theory maps.
- Methodology: **single-seed A/B on a 25-probe discrimination metric is not trustworthy** here; the seed
  variance (±1–2 probes, ±0.06 SYM) exceeds the effect. Report multi-seed.
