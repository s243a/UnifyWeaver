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

**Multi-hop `μ_direct` (review).** For chains >2 hops, the bound generalises: `μ_direct` is **`min` over the
chain's constituent direct (1-hop, tagged) links** — equivalently a *recursive* monotone chain
`μ(k-hop) ≤ μ((k−1)-hop sub-path) ≤ … ≤ min(links)`. The term can be applied at either granularity
(transitive-vs-link or transitive-vs-sub-path); `μ_direct` below denotes whichever bounding quantity is used.

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

## Multi-path: semiring closure (max) vs path enumeration (noisy-OR)
The log-space additivity that makes generation a Dijkstra also resolves multi-path: transitive μ with
multiple routes is `⊕` over paths of ( `⊗` over the path's links ) — a genuine **semiring only for some `⊕`** (below):
- **`⊗` (chain / AND)** = `Π μ(links)` = **`+` in log-μ space** (why Dijkstra works).
- **`⊕` (combine paths / OR)** = the multi-path combiner — *its* choice IS the multi-path formulation:

| `⊕` | behaviour | additive in |
|---|---|---|
| **max** (max-product semiring) | best single path; **no reinforcement** | `log μ` → Dijkstra (already specified) |
| **noisy-OR** | paths **reinforce** (μ rises with multiplicity) | **`log(1−μ)`** (the complement / "survival") |

**Caveat (review): only `max` gives a true semiring closure.** `max`-product *is* a semiring — product
distributes over `max` (`c·max(a,b)=max(ca,cb)`) — so Dijkstra / Floyd–Warshall compute its closure exactly.
**`noisy-OR` with product chaining is NOT a semiring**: distributivity fails (`a·(b⊕c) ≠ (a·b)⊕(a·c)` unless
`a=1`), so there is **no closure algorithm** — it requires **explicit path enumeration**. The `log(1−μ)`
additivity still holds for combining a *fixed, enumerated* path set (`log(1−μ)=Σ_p log(1−s_p)`), but that is
an *aggregation*, not an algebraic-path closure. So the "one clean semiring" framing is rigorous only for
`max`-product; reinforcement (noisy-OR) costs enumeration.

**So multi-path = "pick `⊕`":** start `⊕ = max` (the Dijkstra we already have — single best path, simplest);
upgrade to `⊕ = noisy-OR` for reinforcement (**no closure — enumerate the top-k paths per pair and aggregate
via `log(1−μ)`**; more expensive, but bounded by the curriculum's high-product head). This also resolves the
earlier contradiction: under reinforcement the bound `μ(A→C) ≤ min(links)` correctly **relaxes** (extra paths
legitimately raise μ).

## Eval (clean graph-truth — but guard collapse and leakage)
- **Constraint satisfaction:** fraction of held-out transitive pairs with `μ_transitive ≤ μ_direct − m`.
- **Decay curve:** mean μ vs hop-distance (should decrease monotonically).
- **Anti-collapse (review):** the two above are **gamed by `μ→0` everywhere**, so pair them with a **level**
  metric — `μ_direct` stays at its REL_SPEC value, and transitive μ sits in the **band** `[floor, direct−m]`
  (band occupancy), not at 0. Ordering + level together.
- **Leakage-aware split (review):** a random *pair* split **leaks** — a held-out transitive pair shares its
  constituent links and endpoint nodes with training. Split by **node / subgraph** so a held-out pair's links
  and endpoints are unseen; else the model memorises rather than generalises the decay.

Still deterministic graph-truth (no judge-noise ceiling); with these guards the metric resolves
*generalisation* of the constraint, not just in-sample satisfaction.

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

## The loss must be over the predicted DISTRIBUTION, not the point error (heteroscedastic)
Error-vs-likelihood is not just framing — it changes the **loss form**. A point error (MSE / naive gradient)
assumes the target is exact and descends on `(μ − target)²`. A *statistical* relation has **variance in the
predicted error**, so the loss must be a **proper scoring rule over the model's predicted distribution** — the
gradient flows through the **spread**, not only the mean ("a gradient on the distribution that predicts the
error").

The naive ranking CE `−log σ(s·(E_dir − E_trans − m))` bakes a **global** confidence `s` — it is
*homoscedastic*. The principled form makes confidence **per-pair**, from the model's own predicted spread:

> `L = −log Φ( (E_dir − E_trans − m) / √(Var_dir + Var_trans) )`

`s` is replaced by `1/√(Var_dir + Var_trans)`: a violation under **high** predicted variance costs little
(within the spread — a genuine "unlikely-but-possible"); under **low** variance it costs a lot (a confident
ordering broken). The gradient updates **both mean and variance**.

**The variance comes from the superposition — but it is NOT "free" (review).** `Var[μ] = Σ P(cell)·(μ_cell −
E[μ])²` *looks* closed-form, but the per-cell `μ_cell` are themselves **nonlinear readouts** (§11: μ is
non-linear in the operator weights through the transformer), so obtaining them needs **R hard-cell forwards
(or an MC estimate)** — the §11 sampling cost, not zero. So the variance is *available from the existing
superposition machinery* (no new **head**), but it is **real compute**, not a free byproduct of one forward.

**Independence caveat (review):** `√(Var_dir + Var_trans)` assumes `μ_dir ⟂ μ_trans`, but they are **not**
independent — same model, shared endpoint (A is in both A→B and A→C), shared e5. Properly `Var(μ_dir −
μ_trans) = Var_dir + Var_trans − 2·Cov(dir,trans)`, and `Cov` is likely **positive** (shared factors), so the
independence sum **over**estimates the denominator → an **under-confident** loss. Estimate `Cov` from joint MC
samples, or treat the sum as a conservative bound.

**Proper scoring ⇒ no gaming.** The variance sits in the denominator, so inflating it to dodge a violation
*also* dulls the reward on confident-correct orderings; NLL penalises the net. The model is forced to learn
**calibrated** uncertainty rather than escape the constraint. The naive fixed-`s` CE remains the cheap
homoscedastic first cut — but it *cannot* represent that some transitive pairs are confidently ordered and
others genuinely ambiguous, which is the whole statistical point.

*Caveat:* Gaussian-`Φ` is an approximation for bounded μ ∈ [0,1]; a **logit-space** difference or a **Beta**
parameterisation of μ is the more correct distributional form, at some complexity. Open for review.

### Variance source: product-propagation along the chain (IMPLEMENTED)
The section above assumed the per-pair variance comes from the **superposition** (`Var[μ] = Σ P(cell)·(μ_cell −
E[μ])²`) — which costs R hard-cell forwards / MC and is a *model-side* quantity. **What we built instead is
cheaper and more apt: propagate the variance through the chain, exactly as μ propagates through the product.**
This is **standard error-propagation** (cf. any error-analysis text): for a product `μ_trans = Π μ_link`,
*relative* variances add — equivalently it is **additive in log-variance**, the dual of the log-μ chaining that
makes generation a Dijkstra:

> per-link relative variance `(1−μ)/μ` (Bernoulli) → accumulated **`V = Σ_links (1−μ_i)/μ_i`** →
> per-pair scale **`s_pair = s / √(1 + V)`**.

Properties: it is **structural** (known at generation time — emitted as the triple's `var` column, no variance
head, no extra forwards), and it **grows with chain length** (longer/weaker chains → larger `V` → softer
constraint), which the global-`s` form cannot express. Built as `--transitive-hetero` (off → global `s`). The
superposition-variance variant remains a deferred alternative if a per-pair (non-chain) uncertainty is wanted.

## A multi-factor loss — judge (absolute) + pairwise (relative), as complementary likelihoods
The loss need not be one term. Different supervisors know different things; each enters as a **likelihood
factor**, weighted by its reliability:
- **Direct-step model judge** → anchors the **absolute** μ of a direct relation (calibration). Distributional,
  and **down-weighted by reliability** — the inferred-tail finding (judge μ ~80% noise on hard rows) *is* this
  weight; the soft-outlier-rejection `--tail-weight` is a factor weight.
- **Pairwise transitive constraint** → shapes the **relative** structure (`μ_transitive ≤ μ_direct`), the
  heteroscedastic ordinal term above. Graph-truth, clean, higher weight.

**Complementary and jointly well-posed:** the judge anchors `μ_direct` (so the inequality can't be gamed by
inflating the direct side), and the pairwise term then places `μ_transitive` below that anchored value —
**absolute × relative**, each closing a degree of freedom the other leaves open.

**Principled form:** total loss = a **joint negative log-likelihood** — a sum of per-factor log-likelihoods,
each weighted by its reliability (a *product-of-experts / factor graph*). The distributional treatment applies
to *every* factor: the judge's E[μ] carries its disagreement-variance, the pairwise carries the
superposition's variance — so each is a likelihood under its own predicted distribution, and the **weights are
reliabilities, not arbitrary knobs**.

This is the **multi-source-of-truth** structure (deferred earlier) realised as a loss: REL_SPEC regression
(tagged, clean) + model-judge (noisy, down-weighted) + transitive ordinal (graph-truth) = **factors in one
joint likelihood**, each contributing where it is strong.

## Measured caveat — judge *confidence* ≠ judge *reliability* (use inter-judge agreement)
Tempting hypothesis: "where μ is high the LLM judge isn't noise." Checked on the 84-row Haiku+Sonnet holdout,
stratified by Haiku μ — it **fails for judge-rated μ**: disagreement *rises* with Haiku-μ (mean |Δ| 0.12 mid →
0.27 high; Sonnet stays ~0.3 even where Haiku says >0.75). I.e. **Haiku is *overconfident* on ambiguous tail
pairs** (small-n caveat: high bins are 25 / 2 rows).

The hypothesis survives only with the right population: the transitive chains compose **tagged (conf=1.0,
human-curated) edges**, whose high μ is *not* a noisy judge rating — so there the judge is plausibly reliable
(untested: tagged edges were never judge-scored). The tail merely refutes the *judge-rated-μ* version.

**Design consequence:** the judge factor's reliability weight must come from **inter-judge agreement**
(ensemble variance), **not single-judge confidence** — a judge's own high μ is not a reliability signal. So
calibrating the judge-factor weight needs ≥2 judges *on a sample* (Sonnet already serves; doesn't change the
defer-external-judges call). This composes with the heteroscedastic loss: the judge factor's variance is the
*ensemble* variance, the pairwise factor's is the superposition variance.

## Open questions (for review)
*Update: several are now resolved by the implementation (see Implementation status) — Q7 (loss form: both the
fixed-`s` CE and the heteroscedastic product-propagation are built; dual-ascent λ added), and the multi-path
Q3 (`max`/Dijkstra built; noisy-OR deferred). The rest stand.*
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
7. **Loss form:** naive global-`s` CE (homoscedastic, cheap) vs the distributional/heteroscedastic
   `Φ((ΔE−m)/√ΣVar)` using the superposition's own variance — and Gaussian-Φ vs logit/Beta for bounded μ.
8. **Multi-factor loss:** the factor weights (reliabilities); avoiding double-counting when a direct
   edge is both REL_SPEC-regressed *and* judge-supervised; which judge anchors which factor; are weights
   fixed, annealed, or learned (as inverse-variance)?
9. **Judge-factor weight:** estimate it from inter-judge agreement (ensemble variance) on a calibration
   sample, not single-judge confidence (measured: judges *disagree more* at high single-judge μ on the tail).

## Generation order — rank by the PRODUCT of link μ (greedy / Dijkstra), curriculum
Generate transitive pairs **ranked by the product of their link μ** (= the estimated transitive μ under the
product t-norm), **descending**. Product, not `min`, because it is **length-aware**: a 5-hop all-0.9 chain has
`min=0.9` but `product=0.59` — correctly deprioritised. (Product *ranks* generation; `≤ min` remains the
*constraint bound* in the loss, robust to the t-norm choice — no conflict.)

**The search is exact, not heuristic — it's Dijkstra.** Maximising `Π μ(links)` = maximising `Σ log μ` =
minimising `Σ −log μ` (each `−log μ > 0`); so the **highest-product path = shortest path under edge cost
`−log μ`**, with optimal substructure (highest-product path to C goes through that to B). Run greedy/Dijkstra
from each root, emit (root, node, product) transitive pairs, sort by product, take the top as the curriculum.

Rationale, all aligned with the loss design:
- **Highest confidence / lowest variance:** strong links ⇒ the inequality holds most certainly ⇒ the
  most-informative, sharpest-gradient constraints under the distributional loss. Cleanest first signal.
- **Dodges the "too low" worry:** strong chains keep the transitive μ well above any floor (0.9·0.9 = 0.81 —
  clearly still a relation); weak chains (0.4·0.4 = 0.16) are where decay collapses and the floor gets murky.
- **Curriculum + cheap falsification:** establish the decay *pattern* on the clearest chains, then expand to
  weaker ones; and if it fails to help even on the strongest chains, the idea is refuted early.
- **Bounds the blow-up:** a high-μ prefix of all transitive pairs, grown on demand — pairs with the budget.

## Implementation status (BUILT vs proposed)
Stages 1–3 + two stage-2 upgrades are built, verified, and documented (how-to:
[`README_transitive.md`](README_transitive.md); results: [`REPORT_transitive_verification.md`](REPORT_transitive_verification.md)).

**BUILT:**
1. **Generation** — `transitive_closure.py`: compose tagged hierarchical edges ≤N hops, max-product
   dominant-path (Dijkstra on `−log μ`), emit triples (transitive pair + bounding edge + product/min/hops/var).
2. **Ranking-CE loss** — `--transitive` / `--transitive-weight` (`−log σ(s·(μ_bound − μ_trans − m))`).
3. **Held-out eval** — `--eval-transitive`: constraint-satisfaction + anti-collapse level, leakage-aware node-split.
4. **Dual-ascent λ** — `--transitive-target-sat`: the Lagrangian multiplier auto-tuned to a target satisfaction.
5. **Heteroscedastic** — `--transitive-hetero`: per-pair `s/√(1+V)`, `V` the product-propagated chain variance.

**Verified:** generalises (94%/93% on a leakage-aware holdout, 2-seed), no-collapse (μ_bound rose to ~0.85),
survives convergence (baseline plateaus below at 0.595), dual-ascent λ self-tunes, hetero runs (95%, no collapse).

**Proposed (not built):** superposition-variance hetero (vs the built product-propagation), noisy-OR multi-path
(needs enumeration), LLM-anchored multi-factor `μ_bound`, product soft-floor band. See Open questions.

## Rejected alternatives (and why)
Each design choice has a discarded counterpart; reviewers should feel free to challenge the *rejections*.

| alternative | why rejected | chosen instead |
|---|---|---|
| **Point target** `μ(A→C)=Π μ` (regress to the product) | over-commits to a value we can't verify; bakes in one t-norm | the **inequality** `μ(A→C) ≤ min(links)` — robust to min-vs-product |
| **Hard constraint** (projection / ∞-hinge) | treats a *statistical* likelihood as inviolable law; can't absorb legitimate violations | **soft ranking CE** — `σ(s·Δ)` = P(holds), trained not forbidden |
| **`min` for generation ranking** | length-blind (5-hop all-0.9: min=0.9 but product=0.59) | **product** (Dijkstra on `−log μ`) — length-aware; `min` kept only as the *bound* |
| **Homoscedastic point-error loss** (fixed global `s` / MSE residual) | can't express that some pairs are confidently ordered, others ambiguous | **heteroscedastic** `−log Φ((ΔE−m)/√ΣVar)`, variance from the superposition (via R hard-cell forwards / MC — *not* free; naive fixed-`s` kept as a cheap first cut) |
| **Scale single-judge LLM labels** (more tail augmentation) | measured ~80% judge-noise; no transfer to an independent judge | **graph-truth ordinal** data (clean, free, measurable) — this proposal |
| **Compose inferred (conf<1.0) edges** | re-imports the judge noise just measured | **tagged (conf=1.0) edges only** |
| **Judge self-confidence as the reliability weight** | measured: judges disagree *more* at high single-judge μ (overconfidence) | **inter-judge agreement** (ensemble variance) |
| **Hope transitivity emerges** (no explicit data) | the model sees only endpoints — no path access — so it *cannot* compose from structure | **explicit** transitive training pairs |
| **Fixed decay schedule** (`μ − δ` per hop) | imposes a guessed rate | **learned** via the ordinal constraint |
| **Unbounded transitive closure** | combinatorial blow-up | **bounded + product-curriculum** (high-product head, grown on demand) |
| **Hard dataset cleaning** (drop "bad" rows) | must *decide* which rows are bad; expensive | **soft down-weighting** (reliability / inverse-variance); defers cleaning |
| **Add external judges (agy/codex) now** | premature cost+complexity; value is *independence*, only needed once the judge factor is binding | **deferred**; within-Anthropic cascade for calibration |
| **`⊕ = noisy-OR` from the start** (multi-path) | reinforcement adds complexity before the core is validated | **start `⊕ = max`** (Dijkstra); noisy-OR as the upgrade |

## References (theory anchors)
The design composes several established frameworks; this maps each choice to its canonical source. (Cited
from memory — verify exact year/venue/page before publication.)

**Fuzzy membership, t-norms, transitivity** (the μ model + chaining)
- L.A. Zadeh, "Fuzzy sets," *Information and Control* 8(3), 1965 — graded membership μ∈[0,1].
- L.A. Zadeh, "Similarity relations and fuzzy orderings," *Information Sciences* 3(2), 1971 — **sup-T
  transitivity** of fuzzy relations; the formal basis for transitive fuzzy μ.
- E.P. Klement, R. Mesiar, E. Pap, *Triangular Norms*, Kluwer, 2000 — t-norms (**min**=Gödel,
  **product**=probabilistic/Goguen) and dual t-conorms (the AND/OR connectives; the min-vs-product choice).

**Ordinal / pairwise supervision** (the inequality + ranking CE)
- R.A. Bradley, M.E. Terry, "Rank analysis of incomplete block designs," *Biometrika* 39, 1952 — pairwise
  comparison as `σ(score difference)`.
- C. Burges et al., "Learning to rank using gradient descent" (**RankNet**), *ICML* 2005 — the pairwise
  logistic/cross-entropy ranking loss used here.

**Distributional / heteroscedastic prediction** (variance-aware loss; variance from the superposition)
- D. Nix, A. Weigend, "Estimating the mean and variance of the target probability distribution," *ICNN* 1994.
- A. Kendall, Y. Gal, "What uncertainties do we need in Bayesian deep learning for computer vision?," *NeurIPS*
  2017 — aleatoric uncertainty as learned per-example variance ("gradient on the distribution").
- L. Vilnis, A. McCallum, "Word representations via Gaussian embedding," *ICLR* 2015 — entities as
  **distributions (mean+variance)** with asymmetric/containment relations.
- J.R. Taylor, *An Introduction to Error Analysis*, 2nd ed., 1997 (also Bevington & Robinson) — **product
  error-propagation** (relative variances add / add in quadrature) — the basis for the *implemented*
  chain-propagated transitive variance `V = Σ (1−μ)/μ`.
- T. Gneiting, A. Raftery, "Strictly proper scoring rules, prediction, and estimation," *JASA* 102, 2007 — why
  NLL can't be gamed by inflating variance.

**Order / containment in embedding space** (transitive structure)
- I. Vendrov, R. Kiros, S. Fidler, R. Urtasun, "Order-embeddings of images and language," *ICLR* 2016 —
  partial-order (entailment/containment, transitive) in embedding space.
- L. Vilnis, X. Li, S. Muresan, A. McCallum, "Probabilistic embedding of knowledge graphs with box lattice
  measures," *ACL* 2018 — graded containment as probability.

**Multi-factor loss / combining noisy supervisors**
- G.E. Hinton, "Training products of experts by minimizing contrastive divergence," *Neural Computation* 14,
  2002 — joint model as a product of experts (sum of log-factors).
- inverse-variance weighting — classical (e.g. W.G. Cochran, "The combination of estimates from different
  experiments," *Biometrics* 10, 1954) — weight ∝ 1/variance for combining noisy estimates.
- B. Frénay, M. Verleysen, "Classification in the presence of label noise: a survey," *IEEE TNNLS* 25, 2014.
- L. Zheng et al., "Judging LLM-as-a-judge with MT-Bench and Chatbot Arena," *NeurIPS* 2023 — LLM-judge biases
  and agreement (why judge confidence ≠ reliability → use inter-judge agreement).

**Multi-path / algebraic path problem** (the log-semiring)
- J. Pearl, *Probabilistic Reasoning in Intelligent Systems*, Morgan Kaufmann, 1988 — the **noisy-OR** gate
  (multi-path reinforcement).
- E.W. Dijkstra, "A note on two problems in connexion with graphs," *Numerische Mathematik* 1, 1959 — shortest
  path; here on `−log μ` edges = highest-product path (max-product semiring).
- M. Mohri, "Semiring frameworks and algorithms for shortest-distance problems," *J. Automata, Languages and
  Combinatorics* 7, 2002 — the **algebraic path problem**: `⊕`/`⊗` semiring closure generalising shortest path.
- M. Gondran, M. Minoux, *Graphs, Dioids and Semirings*, Springer, 2008 — dioids/semirings for path algebras.

**Curriculum**
- Y. Bengio, J. Louradour, R. Collobert, J. Weston, "Curriculum learning," *ICML* 2009 — high-confidence
  examples first (our highest-product-chains-first ordering).
