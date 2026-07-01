# PATH operator — multi-path ancestor membership over the DAG (design / future-work)

Design note (nothing built yet). Follows the finding that single-path LINEAGE is redundant to HIER, but a
**multi-path** ancestor operator is *not* — because a DAG node has many ancestor chains, and the harvested
single-path collapse (`parent_map` precedence) threw all but one away.

## 1. Motivation

`LINEAGE` (as built) scored `μ(node | ONE materialized ancestor chain)` — the jsonL picks a single parent per node
(RefPearl "inside" over AliasPearl "cross-link"), collapsing the **DAG** (multi-parent: Pearltrees aliases, Wikipedia
categories) to one arbitrary path. A **PATH** operator instead scores the node against a **superposition over all
valid ancestor chains** — the full set of hierarchical contexts it belongs to ("this AI paper is under both
`CS ▸ ML` and `Research ▸ ToRead`"). Keep LINEAGE (the canonical chain); add PATH (the multi-path superposition).

## 2. Unification: one operator, a METHOD parameter (a higher-order operator)

Ancestor-path membership is a *single relation*; LINEAGE and PATH differ only in **how the paths are aggregated**.
So factor it: **operator = ancestor-path membership (what); method = aggregation/sampling strategy (how).**
`operator × method` is a **higher-order operator** — the operator takes a *strategy* as a parameter:

| name today | = |
|---|---|
| `LINEAGE` | `ancestor-path(method = canonical)` — the single primary chain |
| `PATH` | `ancestor-path(method = random-walk)` — uniform-parent superposition |
| (future) | `ancestor-path(method = edge-weighted / PPR)` — richer variants |

This unifies lineage/path into one operator and generalizes the pattern (any operator with multiple computation
strategies can be method-parameterized).

## 3. "Random superposition" = sampling (cheap; reuses the variant cache)

Realize the superposition **stochastically**: each training step, **sample ONE ancestor path** from the method's
distribution, embed it, apply wildcard masking (id-dropout + prefix-dropout), and use it as the passage. The model
learns the *expectation over paths* across steps — no need to embed every path and weight-average explicitly. The
wildcards are an orthogonal masking augmentation on top of the path sampling.

## 4. The three path-sampling methods (document all three)

**(1) Uniform random walk — default, cheapest.** Going up the DAG, pick a parent **uniformly** at each node;
a path's probability is `∏ 1/|parents(level)|`. Parameter-free — literally "sample a parent at each level." Weights
paths by *structural* (branching) likelihood: a node with two parents splits its mass evenly. Start here.

**(2) Edge-weighted — canonical strength.** Weight edges by **type/confidence** — RefPearl (inside/owned) ≫
AliasPearl (cross-link), or graph-edge confidence — so a path's probability = normalized product of its edge
weights. Down-weights incidental cross-link lineages; encodes "primary home vs cross-reference." The right refinement
once uniform over-weights incidental multi-parenting.

**(3) PPR / flux — principled continuous.** Personalized PageRank from the node over the **reverse (ancestor)**
graph; each path's mass = the PPR flow along it, giving exp-decay / power-law weighting by depth × branching for
free. Reuses the scan-strategy flux machinery (the Green's-function / KCL framing). The continuous version of (1)+(2).

**Weighting semantics.** (1) = structural likelihood, (2) = lineage strength, (3) = both / continuous. Default (1);
move to (2) (effectively *structural × confidence*) because a node's *true* contexts are its canonical ancestors,
and pure branching over-weights incidental multi-parenting; (3) if/when the full flux is worth computing.

## 5. Method as a model input — higher-order, but adopt lazily

Conceptually `method` is a **factored token** like `corpus_emb`/`judge_emb`: the model conditions on
`(operator, method)`, and PATH becomes **queryable by method at inference** ("give me the PPR-weighted membership").
But **with only one method live, a method token is constant = no signal.** So:
- **Now:** method is a **data-generation choice** (how we sample the training paths) — *not* a model input.
- **Promote to a model input (`method_emb`)** only when **(a) ≥2 methods are trained** *and* **(b) inference needs to
  select/distinguish them.** Then the higher-order operator is realized.

(Same conclusion as the source-type discussion: design for it; adopt the machinery when the instance count justifies
it, not before.)

## 6. Why PATH (multi-path) is *not* redundant to HIER, though LINEAGE was

`HIER` = the single category-hierarchy **edge**. Single-path `LINEAGE` = one chain of `HIER` edges ⊂ (`HIER` +
transitive closure) — hence redundant. But the **multi-path superposition** is the DAG's *ancestor set / closure
with weights* — it expresses multi-context membership that a single chain (or a bare edge) does not emphasize. That
is the part our experiments did **not** test, and the reason to build PATH rather than conclude "paths don't help."

## 7. Start

`ancestor-path` operator, `method = random-walk`, sampled per step with wildcard masking, method kept as a
data-generation knob (no token yet). Add the `method` token when a second method (edge-weighted or PPR) is trained
and inference wants to select. Eval with the graceful-degradation metrics already built (path-overlap / matched-depth
on the paired hard subset) — the question is whether multi-path recovers ancestor branches better than HIER alone.
