# Applications of the μ model — and a geometric (bivector) theory note

**Status: FUTURE-WORK / roadmap.** Most of this is *proposed*, not built: the retrieval algorithm, the
hop-stratified eval, and the geometric (bivector / root-centred) theory are directions, not shipped code. It
builds on a few **existing** pieces — `emit_dense` → `dense_mu_attn_*.tsv` (density maps), the bridge detectors
(`bridge_ensemble.py`, PR #3322), and the `DESIGN_bidirectional_walk.md` traversal — and on the **built +
verified** transitive μ (`DESIGN_transitive_relations.md`) it relies on for distance-awareness. The "Build
plan" at the end is the concrete next increment.

The μ-attention model produces **`μ(node | root)`** — a *directional, fuzzy* relatedness/membership degree of
`node` to `root` over a concept graph, learned over **frozen e5** embeddings with a permutation-invariant
operator-attention readout. This doc collects the **applications** that consume μ, the **new retrieval
algorithm**, how it relates to **prior (distance-metric) approaches**, and a short **theory** section — the
last of which flags a genuinely uncommon framing (embedding **bivectors** / root-centred embeddings) worth
recording.

## μ as the primitive
- **Directional:** `μ(a|b) ≠ μ(b|a)` (a member is mostly-not its container). Most vanilla embedding similarities
  (cosine/dot) are *symmetric* and cannot express this.
- **Fuzzy/graded:** μ ∈ [0,1], a membership degree, not a hard edge.
- **Operator-structured:** μ is read out under operators (SYM = lateral `see_also`/`assoc`; WIKI/ELEM =
  directional `subcategory`/`element_of`/…). An **equal-weight superposition** of operators (e.g.
  `⅓·subcategory + ⅓·element_of + ⅓·see_also`) is the **unconditional** relatedness — "how related, regardless
  of which relation" — the `E[μ]` over a uniform operator prior (operator-superposition design §1b/§12).
- **Distance-aware:** trained with the transitive ordinal constraint so μ **decays sensibly with hop-distance**
  (`DESIGN_transitive_relations.md`) — which is what makes it usable for *multi-hop* retrieval.

## Built foundation (what the roadmap enhances)
The applications and roadmap below build on these **shipped, documented** pieces — described here so the
proposed *enhancements* have context:
- **μ-attention model** — directional fuzzy membership over **frozen e5**, permutation-invariant
  operator-attention readout (SYM = lateral; WIKI/ELEM = directional) + the §8 **anchored basis** (frozen
  label-tied anchors ∪ learnable atoms). The core estimator.
- **Operator superposition** — μ read under an operator distribution; **equal-weight = the unconditional
  `E[μ]`** (`DESIGN_inferred_operator_superposition.md`).
- **Transitive ordinal constraint** — μ **decays with hop-distance**: ranking-CE + dual-ascent λ +
  heteroscedastic product-propagated variance. Built **+ verified** (generalises, no-collapse,
  convergence-stable) — `DESIGN_transitive_relations.md`, `README_transitive.md`, `REPORT_transitive_verification.md`.
- **Density maps** — `emit_dense` → `dense_mu_attn_*.tsv`: μ(·|root) over all nodes (the density-explorer feed).
- **Bridge detection** — `bridge_ensemble.py` (here) + PR #3322's declarative bridge detector (Prolog foreign
  predicate).
- **Bidirectional walk** — depth-balanced traversal sampler (`DESIGN_bidirectional_walk.md`,
  `validate_bidir_walk.py`).
- **Inferred-tail augmentation** — `score_inferred_tail.py` + `cell_sampler.py` (LLM `E[μ]` for the inferred
  tail; measured ~80% judge-noise → soft-rejected, off by default).

## Applications
| application | how μ is used | existing pieces |
|---|---|---|
| **Graph RAG / graph search** | rank graph nodes by relatedness to a query node (incl. multi-hop) | the new algorithm (below); `emit_dense` |
| **Bridge-node identification** | nodes whose μ links *across* domains (the `bridge` structure / cross-domain μ) | `bridge_ensemble.py`; PR #3322 declarative bridge detector (Prolog foreign pred) |
| **Semantic visualisation (density explorer)** | μ as a relatedness/density field around a root | `emit_dense` → `dense_mu_attn_*.tsv`; root-centred chart (theory below) |

## The retrieval algorithm (new) — greedy bidirectional gather + μ-superposition sort
Structure **∩** semantics: the graph gives a cheap, structure-aware *candidate set*; μ gives the *semantic
ranking*.
1. **Greedy bidirectional gather** from the root — best-first expansion following edges in **both** directions
   (child *and* parent, using the directional asymmetry; depth-balanced so it stays lateral, not drifting to
   hubs/leaves — reuses the `DESIGN_bidirectional_walk.md` traversal insight, but greedy and for retrieval, not
   random sampling). Gathering, not scoring-all-10k → efficient.
2. **Sort by the operator-superposition μ** — score each gathered candidate by the equal-weight unconditional
   `E[μ](candidate | root)`; return top-k.
3. **Diagnostic — μ-vs-hop scatter (top-k, e.g. k=25):** x = hop-distance, y = μ. Expect a *decaying-but-spread
   cloud* (semantics, not a step function). The **off-diagonal points are the payoff**: high-μ-far (semantically
   close, graph-distant — what μ adds over shortest-path) and low-μ-near (graph-adjacent but weak — what μ
   *corrects*). The scatter is a visual proof of whether semantics-on-structure beats structure-alone.

### Relation to prior approaches
Prior graph retrieval uses **distance metrics** — most relevantly **weighted shortest path** (and the WAM
core's effective-distance). Those are *structural only*: graph-near ≠ semantically-related. The new algorithm
keeps the structural candidate-gathering but **replaces structural distance with the learned μ** for ranking —
or hybridises (μ-weighted edges in the greedy priority). The eval (below) measures exactly this gap.

## Theory note — μ is a geometric (scalar + bivector) object
This is the uncommon framing worth recording. Standard embeddings give each entity a **vector** `v` and a
**symmetric** similarity (cosine/dot) — which structurally *cannot* represent a directional relation. Various
fixes exist (order-embeddings, box/Gaussian embeddings, bilinear `vᵀMv`), but they bolt asymmetry on.

**The geometric-algebra view makes the asymmetry intrinsic.** For two vectors `a, b`, the **geometric
product** decomposes by grade:

> `ab = a·b + a∧b`  —  a **scalar** (symmetric inner product) + a **bivector** (antisymmetric *oriented* area).

Our operators split *exactly* along this seam:
- **symmetric operators** (`see_also`, `assoc`, `bridge`) ↔ the **scalar** `a·b` — un-oriented relatedness;
- **directional operators** (`subcategory`, `element_of`, `super_category`) ↔ the **bivector** `a∧b` —
  `a∧b = −b∧a` encodes the parent/child **orientation** (the `μ(a|b) ≠ μ(b|a)` asymmetry is the sign flip).

So μ's symmetric/directional operator structure *is* the scalar/bivector grade structure of a multivector — the
asymmetry is geometric **by construction**, not bolted on.

**Embedding bivectors (novel-ish):** rather than embed each node as a vector and patch the similarity, embed
the **relationship** as a multivector — the directional content lives in the **bivector grade**. (GA / Clifford
layers exist in recent ML, but tying the *fuzzy-membership operator split* to the scalar/bivector grades is
uncommon.) This is a research thread, not a dependency of the applications.

**Root-centred embeddings / tangent chart (for visualisation):** fix the **root as origin**; node embeddings
become root-relative, and the relationship to the root is the oriented `node ∧ root`. A **local chart** for the
density explorer: 2D/3D coordinates where displacement ≈ μ-relatedness and the **axes are the principal
statistical variations of the root's neighbourhood** (local PCA / a "tangent plane" at the root). Less common
than global vector embeddings, and naturally per-root (each root its own chart). Also a viz/research thread.

**For ranking we need none of this** — μ *is* the distance measure; embeddings/bivectors are for visualisation
and a possible re-derivation, not for retrieval.

## Roadmap (enhancements to built things + new things)
**Enhancements to built things:**
- **Transitive μ** — the deferred stage-2 items (noisy-OR multi-path, LLM-anchored multi-factor `μ_bound`,
  product soft-floor; `DESIGN_transitive_relations.md` open questions). Most bite only in the **weak/long-chain
  regime**, so pursue them *when retrieval shows that regime matters* (the hetero A/B was neutral precisely
  because the strong-chain curriculum doesn't exercise them).
- **Density explorer** — **root-centred tangent charts** (theory below): a per-root local view, vs the current
  global dense map.
- **Bridge detection** — fuse with the transitive-decay μ and the retrieval gather (a bridge is a node with
  high cross-domain μ reachable in the bidirectional walk).
- **Operator superposition** — learned (non-uniform) operator weights per query instead of the equal-weight
  `⅓+⅓+⅓` default.

**New things:**
- **Graph RAG / retrieval algorithm** — greedy bidirectional gather + μ-superposition sort (above).
- **Hop-stratified retrieval eval** + the μ-vs-hop top-k scatter.
- **Embedding bivectors / geometric (GA) re-derivation** — the theory note (research thread).

## Build plan (the retrieval core — first concrete increment, no LLM, runs against existing models)
1. **Hop-stratified retrieval eval** — ground-truth = graph reachability from a root (relevant = reachable at
   ≤H hops; stratify metrics by hop-distance). Metric: recall@k / AUC (related vs unrelated), per hop.
2. **Baseline = weighted shortest path** on that eval.
3. **New algorithm** = greedy bidirectional gather + μ-superposition sort, same eval.
4. **μ-vs-hop scatter (top-k)** as the diagnostic.
Compare: does semantics-on-structure beat structure-alone, *especially at 2–3 hops* (where the transitive μ
should pay off)?

## References
- D. Hestenes, *New Foundations for Classical Mechanics* / *Space-Time Algebra* — geometric (Clifford) algebra;
  the scalar+bivector geometric product.
- J. Brandstetter et al., "Clifford Neural Layers for PDE Modeling," 2022; D. Ruhe et al., "Geometric Clifford
  Algebra Networks," ICML 2023 — GA/Clifford structure in deep learning.
- I. Vendrov et al., "Order-Embeddings of Images and Language," ICLR 2016; L. Vilnis, A. McCallum, "Word
  representations via Gaussian embedding," ICLR 2015 — asymmetric/containment embeddings (the bolted-on fixes).
- P. Lewis et al., "Retrieval-Augmented Generation…," NeurIPS 2020; Edge et al., "From Local to Global: A Graph
  RAG Approach," 2024 — RAG / graph-RAG.
- `DESIGN_transitive_relations.md` (transitive decay), `DESIGN_bidirectional_walk.md` (depth-balanced
  traversal), `DESIGN_inferred_operator_superposition.md` (the operator superposition / unconditional E[μ]).
