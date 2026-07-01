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

## 3b. Passage representation — per-node tokens, NOT a fixed whole-path embedding

Once paths are *sampled* (random walk × stop-β × multi-path × wildcards), the space is **combinatorial** — you cannot
precompute a fixed e5 embedding for every sampled path (the variant-cache trick that worked for LINEAGE's ~6
variants/folder breaks entirely). **Resolution: stop embedding the whole path as one e5 string.** Represent the path
as a **set of per-node embeddings** — embed each ancestor node's *title* **once** (cached, linear in #nodes) — and
let the model **attend / aggregate over the sampled subset**. Then all the sampling (which nodes, how deep, which of
the DAG's paths, wildcard-masked) is just **selecting which cached node-tokens enter the passage**:
- No re-embedding, ever — the combinatorial path space is handled by *token selection over a linear node cache*
  (this also kills the re-embed-per-step compute concern that dogged the whole-path-string approach).
- The μ-attention block already supports a **multi-token passage**, so this is a representation change, not an
  architecture change.
- The materialized-id line / wildcards become per-node attributes rather than substrings of one blob.

So the path operator factors on **three** axes: **operator** (ancestor-path membership) × **method** (sampling incl.
stop-β) × **representation** (per-node tokens, attention-aggregated). The whole-path-string embedding was scaffolding
that doesn't scale; per-node composition is the structure that does — and it's what the decoder's per-node /
optimizer view wanted anyway (§0 of the decoder sketch: the optimizer reads a per-node partial-path state).

## 4. The three path-sampling methods (document all three)

**(1) Uniform random walk (+ stop-β) — default, cheapest.** Going up the DAG, pick a parent **uniformly** at each
node, and **terminate with probability β at each step**. Path prob ≈ `∏ 1/|parents(level)|` × the geometric
stopping term. Parameter-free but for β — literally "sample a parent at each level, sometimes stop early." Two
bonuses: **(a)** the stop gives geometric-depth (variable-length) paths ⇒ a *principled depth sampler* (graded-depth
ancestor membership — a better-motivated prefix-dropout); **(b)** a uniform up-walk with per-step stop **IS discrete
PPR** — β is the PPR restart — so **methods (1) and (3) collapse into one knob** (β→0 walks to the root; larger β
stays shallow). Start here.

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

**Correction / the real mechanism — a learned MLP encoder, not more tokens.** A `method_emb` *token* (and the fixed
`op_emb` **codebook** generally) fits a *small discrete* set — fine for `{SYM, HIER, ELEM}`. But once operators are
**higher-order and parameterized** (`ancestor-path × method × stop-β × edge-weights`) the space is **continuous and
compositional**, and no lookup table — nor a handful of factored tokens — can span it (you can't *index* a real-valued
β, nor enumerate compositions). The mechanism that can: **compute the operator encoding with a small MLP from an
operator SPECIFICATION** — `op_encoding = MLP([relation_emb, method_emb, params…])` — instead of looking it up. The
MLP is the *higher-order function realized in the architecture*: the "operator constructor" that takes the parameters
(method, β, edge-weights) and emits the `d_model` operator token.
- **Parameterized:** β / edge-weights enter as real-valued inputs (a codebook holds points, not a continuum).
- **Compositional + interpolable:** unseen (method, β) configs map smoothly; the base relations `{SYM, HIER, ELEM}`
  become special *points* the MLP can still produce — the current codebook is the degenerate "lookup for one-hot
  specs" case.
- **Hypernetwork-flavored:** a small net generating the conditioning vector from a spec — the honest way to feed a
  *function-with-parameters* to the model. Spec source = learned per-component embeddings (relation/method) + scalar
  params; optionally grounded in e5-of-description (but that reintroduces prompt-rewriting; the MLP mediates it into a
  *learned* encoding).
- **Scope:** overkill for 3 discrete relations; it is the **enabling piece** for parameterized/higher-order operators
  — it's what turns "operator × method × representation" from a *factoring* into something the model can actually
  consume. Adopt it *when building* PATH-with-methods, and the codebook stays valid for the base relations meanwhile
  (the MLP can be introduced as `op_encoding = MLP(spec)` with the current ops as its first, one-hot-spec outputs).

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
