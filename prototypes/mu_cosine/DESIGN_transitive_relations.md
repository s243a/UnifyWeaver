# Transitive relations as ordinal constraints (proposal — for review)

**Status:** design proposal, not built. Captures open questions for review (several conceptual points are
genuinely unsettled — flagged in §Open questions).

## Motivation
The μ model is permutation-invariant attention over the **two endpoints' e5** — it has *no structural access
to the A→B→C path*. So it cannot learn **transitive decay** (distant-but-reachable ⇒ moderate μ) unless we
teach it. Composing the graph's **tagged (conf=1.0)** edges yields transitive pairs that are **clean (no LLM,
no judge-noise), free, and deterministically measurable** — the opposite of the inferred-tail augmentation,
whose ~80% judge-noise left it inconclusive (`REPORT_haiku_tail_pilot.md`). This is graph-truth structural
signal, and the cleanest scaling lever we've identified.

## What is transitive
Hierarchical relations compose; lateral ones do not:
- `subcategory ∘ subcategory ⇒ subcategory` (and `subtopic`; `super_category` is the reverse direction);
- `element_of ∘ subcategory ⇒ element_of` (an element of a subset is an element of the superset);
- `bridge ∘ R ⇒ R` (identity passes a relation through — see open Q5).
- `see_also` / `assoc`: **NOT transitive** (lateral) — excluded.

Compose **only tagged edges** — composing the noisy conf<1.0 tail would re-import the noise we just measured.

## The structure: a PAIRWISE ORDINAL CONSTRAINT, not a point target
For a chain A→B→C we do **not** regress μ(A→C) to a guessed value — we know the **order** (decay) but not the
**value**. We impose an inequality between the model's *own* predictions for the transitive pair and its
constituent direct pair(s):

> `μ(A→C) ≤ μ(A→B)`  and  `μ(A→C) ≤ μ(B→C)`   — a chain is no stronger than any of its links.

It is a constraint **between two model outputs** (pairwise prediction), enforced via a ranking loss — *not* a
regression to a number.

### Is the bound the product? (No — and why this is the key choice)
Two candidate compositions (fuzzy **t-norms**):
- **min** (Gödel): `μ(A→C) = min(links)` — "as strong as the weakest link"; no compounding.
- **product** (probabilistic): `μ(A→C) = Π links` — compounds, attenuates each hop (0.9·0.9 = 0.81).

The **product was the original *point-target* idea**, and it over-commits to a value we can't verify. The
**inequality `μ(A→C) ≤ min(links)`** is the **common consequence of *both*** (for values in [0,1],
`product ≤ min ≤ each link`), so it is **robust to which t-norm is actually right** — it enforces only what
min and product agree on (decay), without picking one. (Product can return as an optional *soft floor* — below.)

## The loss — ranking cross-entropy
Per transitive pair + its bounding direct pair, push the direct above the transitive:

> `L_trans = −log σ( s·(μ_direct − μ_transitive − m) )`   (BCE: "the shorter hop wins")

One-sided — it penalises only when the transitive μ exceeds `(direct − m)`. Composes with the trainer's
existing margin/ranking machinery (`--margin-weight`); gated by a new `--transitive-weight`.

## The "too low" risk + the band guard
A pure `≤` can be satisfied **trivially by collapsing μ(A→C) → 0**. Three guards:
1. **Direct (tagged) pairs are still regressed** to their REL_SPEC μ — anchors the absolute scale.
2. **Small margin `m`** → gentle per-hop decay, not a cliff.
3. **Optional floor (band):** `μ(A→C) ≥ baseline_unrelated` → "weaker than direct, but still clearly a
   relation." The **product μ could serve as this soft floor**, turning the constraint into a *band*
   `[floor, direct − m]` rather than a one-sided push.

**Bounded hops (2–3)** cap cumulative decay so distant-but-real pairs don't vanish.

## Mental model: fuzzy set vs distribution (open conceptual question)
μ admits two readings, which suggest different compositions:
- **Fuzzy-set membership** (Zadeh): μ(A⊆C) is graded containment; AND/intersection → **min** t-norm. Maps
  cleanly to subset/element as graded set-ops.
- **Probabilistic / expectation** (the §10–§11 reading of the operator-superposition doc): μ = E[membership]
  over a distribution; chaining independent steps → **product**, and "pruning the distribution" ≈
  *conditioning* along the chain.

The reviewer's worry — that the distribution view "may not map cleanly to subset/sub-element" — is real: set
containment is crisp-logical, μ is graded, and the bridge is the t-norm choice (min vs product), which is
*not* uniquely fixed by the semantics. **This is exactly why we adopt the inequality**: it is the shared
consequence of both readings, so we need not resolve "is μ a fuzzy degree or an expectation?" to use it.
(Resolving it is open — it decides whether `product` is the right *floor*.)

## Multiple graph paths (open question, raised in review)
If A→C is reachable via B *and* via D, the paths **reinforce** — more independent routes ⇒ *higher*
membership, a fuzzy **OR / t-conorm**:
- **max** (Zadeh): `μ(A→C) ≥ max_path(strength)`;
- **noisy-OR** (probabilistic): `μ(A→C) = 1 − Π_path(1 − strength_path)`.

This **complicates the upper bound**: legitimate multi-path reinforcement can push μ(A→C) *above* a single
weak path's `min`, violating a naively-applied per-path `≤`. Practical handling:
1. **Start single-path:** build the constraint from the **dominant (shortest / strongest) path** per (A,C)
   pair only — avoids contradictory bounds; bound = `min(links of the strongest path)`.
2. **Defer** full max / noisy-OR aggregation — and note path-multiplicity is itself a signal the
   endpoint-only model cannot see, so it may deserve to be a feature later.

## Eval (finally clean — no judge-noise ceiling)
- **Constraint satisfaction:** fraction of held-out transitive pairs with `μ_transitive ≤ μ_direct − m`.
- **Decay curve:** mean μ vs hop-distance along chains (should decrease monotonically).

Both are **deterministic graph-truth** — no point-guessing, no judge disagreement. Unlike the inferred-tail
metric (±0.15 noise at 84 rows, judges agree only +0.28), this metric can fully resolve whether the
constraint is learned.

## The constraint is statistical, not logical (loss semantics — clarifies the whole design)
**The inequality `μ(A→C) ≤ μ(A→B)` is a high-confidence *statistical* statement, not a hard logical law** —
and the ranking cross-entropy is the right loss *precisely because* it encodes this. `σ(s·(μ_direct −
μ_transitive − m))` **is the modeled probability that the inequality holds**; CE trains that probability, it
does **not** forbid violations.

- **"Violation = error" is the wrong frame.** A violation is a **low-likelihood event, not a mistake** — e.g.
  multi-path reinforcement (§Multiple paths) legitimately lifting μ(A→C) above a single weak link. Soft CE
  absorbs these gracefully (bounded, proportional loss); a **hard** constraint (projection / ∞-hinge) would be
  wrong — it treats structural likelihood as inviolable law.
- **The scale `s` is the confidence** of the statistical inequality ("fairly high" ⇒ a moderately sharp `s`,
  kept soft enough that genuine exceptions aren't crushed).
- **Entropy is likelihood, not error.** The uncertainty in a transitive relation is a *statement of how
  likely*, not noise to eliminate — so we **model the probability** (soft CE), we don't drive entropy to zero.

This favors the **probabilistic reading** of μ (open Q4): μ = **E[membership] over a distribution**; transitive
decay = the **expectation dropping along the chain** (each hop adds uncertainty ⇒ lower expected membership,
*with high probability*); the constraint's confidence is a property of that distribution. The **product
t-norm is the independent-chaining special case**; the inequality is the assumption-light version. And it
**softens the "too low" worry** — we make decay *likely*, not *forced*, so μ_transitive sits modestly below
its link rather than collapsing.

**Distinct from the inferred-tail noise (ties the arc together):** there, judge-disagreement was partly
*measurement error* (two judges, one truth, ~80% noise) — to **down-weight**. Here the entropy is *inherent
structural likelihood* of a graded relation — to **represent**. Same word "uncertainty"; opposite treatment.
That is the difference between an **error model** and a **likelihood model**.

## Open questions (for review)
1. **Bound:** `≤ min(links)` alone (robust), or add `product` as a soft **floor** (band)? What floor / baseline?
2. **Hyperparameters:** margin `m`, scale `s`, and `--transitive-weight` relative to the direct regression.
3. **Multiple paths:** dominant-path-only to start vs max / noisy-OR aggregation — and should path-multiplicity
   become an explicit feature?
4. **Conceptual:** is μ better modelled as a **fuzzy degree** (→ min) or an **expectation/probability**
   (→ product)? Does the answer matter *given the inequality*, or only for the optional floor?
5. **`bridge`:** should identity compose with the hierarchy, and should it decay at all (identity arguably
   shouldn't attenuate)?
6. **Risk the reviewer flagged:** will the chained value come out **too low**? Are the band + small margin +
   bounded hops sufficient, or is an explicit per-hop decay floor needed?

## If approved — build sketch
1. `build_graded_round --transitive`: compose tagged hierarchical edges ≤N hops → emit transitive **triples**
   (the transitive pair + its bounding direct pair), dominant-path-only.
2. Trainer: `--transitive-weight` ranking-CE term (`−log σ(s·(μ_direct − μ_trans − m))`, optional floor).
3. Eval: constraint-satisfaction % + μ-vs-hop decay curve on a held-out transitive slice.
