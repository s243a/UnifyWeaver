# Applications of the μ model — and a geometric (bivector) theory note

**Status: mixed — roadmap + a built core.** The **retrieval algorithm itself is now built + verified**
(`eval_retrieval`, three-way DENSE/WSP/GREEDY comparison — see "Built + verified" below). Still *proposed*: the
**formal hop-stratified recall@k / AUC eval** against a held-out-edge ground truth, and the geometric (bivector /
root-centred) theory. It builds on a few **existing** pieces — `emit_dense` → `dense_mu_attn_*.tsv` (density maps), the bridge detectors
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

### Built + verified (`eval_retrieval` in `train_mu_attention.py`, `--eval-retrieval`)
Stages 1–3 are implemented (`--eval-retrieval "Root1,Root2" --retrieval-k 25 [--retrieval-out scatter.tsv]`),
reporting **three rankings side-by-side** — DENSE-μ (score all nodes), WSP (graph-hop, μ tiebreak; the
structural baseline), GREEDY (μ-ranked among graph-reachable; the new algorithm) — plus per-hop mean μ and the
top-k hop distribution. Verified on `model_nodetype.pt` over three roots spanning the generalisation gradient:
- **Physics (in-distribution)** — all three rankings agree; dense∩greedy overlap **21/25**.
- **Music (clean OOD, not trained)** — all three agree, **25/25** — frozen e5 generalises cleanly outside the
  physics training region.
- **Cooking (confusable OOD)** — the discriminating case. DENSE-μ is **polluted** by `Movies_directed`,
  `Movie_studios`, `Basketball_movie` (hops 4–6, μ 0.4–0.6) because e5 conflates cooking-TV → film. GREEDY
  **removes all of them** (they're not in the gather's graph region) and recovers the actual cooking
  subcategories ranked by μ; dense∩greedy overlap collapses to **4/25**. It *also* beats WSP, which keeps
  `Home(0.04,h1)`/`Nutrition(0.18,h2)` purely for graph-adjacency — greedy's μ-ranking demotes them.

The two corrections, both visible in Cooking: **vs DENSE**, structure removes high-μ-far false positives (the
movie leak); **vs WSP**, semantics demotes low-μ-near false positives (`Home`). **Bonus —
dense∩greedy overlap is a free reliability self-diagnostic**: high (Physics/Music) ⇒ structure and semantics
agree, trust the result; low (Cooking 4/25) ⇒ e5 is leaking across a domain boundary, trust the
structure-constrained greedy over raw μ. No ground truth required.

### Formal eval — bookmark-filing ground truth (`eval_filing.py`)
The non-circular head-to-head (Build-plan §1–2), using **real Pearltrees filing decisions** as labels (a
bookmark's actual `treeId` = which folder it belongs in — a human decision, not graph distance). 335 candidate
folders (≥3 bookmarks), 500 sampled query bookmarks; rank folders by μ(bookmark|folder); recall@k / MRR. Three
rankers: **e5-cos** (raw e5 cosine, no model), **mu-super** (equal-weight operator superposition), **mu-elem**
(the `element_of` operator — the membership relation filing *is*).

| ranker | recall@1 | recall@5 | recall@10 | MRR | med.rank |
|---|---|---|---|---|---|
| **e5-cos** | **0.202** | **0.410** | **0.480** | **0.299** | **14** |
| mu-super | 0.088 | 0.198 | 0.248 | 0.151 | 52 |
| mu-elem  | 0.096 | 0.208 | 0.256 | 0.160 | 48 |

**Raw e5 cosine ~doubles the μ model on every metric** (random@335 ≈ MRR 0.019, so both beat chance — but the
learned readout *underperforms its own frozen substrate* here). `mu-elem` edges out `mu-super` — the membership
operator is the right one, and directionality helps a hair even untrained. **Why μ loses — it's zero-shot OOD,
not an architecture ceiling:** the checkpoint was trained on *simplewiki category→category membership*
(Physics→Energy), never on *bookmark→folder filing*; bookmarks are noisy web-article titles, folders are
Pearltrees collections; and the gather runs with **no lineage** (empty DAG — the "absent lineage = off-manifold
noise" regime). The learned transform, tuned for a different regime, *distorts* e5 for this task. Since the
readout sits **on top of** frozen e5 and can represent near-identity, filing fine-tuning should recover *and*
beat `e5-cos` via directionality — `e5-cos` MRR **0.299** is now the bar.

**Stratified by distance to the trained region** (`--core-anchors`, max folder-similarity to
Physics/Math/Chem/CS/Eng — testing "μ only helps inside its region"): the e5-cos − mu-elem MRR gap is
**flat at ~0.5× ratio across all three bins** (FAR 0.309/0.161, MID 0.327/0.185, NEAR-STEM 0.263/0.136). μ does
**not** close the gap near the core — the loss is *uniform*, not region-specific. Two confounds keep this from
refuting the region hypothesis: (1) weak stratifier (e5-small's high cosine floor — even "FAR" folders sit at
0.74 to *Physics*); (2) **the lineage confound** — the model trained *with* ancestor lineage but runs here on
**cold, lineage-free** bookmarks (empty DAG), a uniform handicap that could itself flatten the ratio. **NB —
the transformer is nonlinear, so multi-region is *not* a capacity question.** A single μ model can represent
many regions separated by boundary manifolds (the nonlinear analog of routed per-cluster Procrustes — the
*linear* model needs discrete routing precisely because it can't); so the flat ~2× loss is cold-start + lineage,
*not* a single-global-transform ceiling. The mixture is the linear workaround, not the fix here. So "μ
loses ~2× everywhere" conflates OOD-task + cold-start + (maybe) wrong-region; stratification rules out
wrong-region-as-sole-cause but not the rest.

**Decisive next test — μ vs e5-cos on home turf:** hold out *simplewiki* category→parent memberships
(in-distribution, *with* lineage) and run the same recall@k. If μ beats e5-cos there, the filing loss is fully
OOD + cold-start (your "with training data it wins" holds → filing fine-tune, then per-region/mixture μ like
the routed per-cluster Procrustes). If μ still loses on home turf, that's a finding about the readout itself and
reshapes the roadmap. Only *after* that: the filing fine-tune learning-curve (warm-start, folder-disjoint split,
data fractions vs the flat e5-cos bar).

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

**Rotations on the sphere → bivectors as the generator.** e5 embeddings are **unit-normed**, so they live on a
sphere and the relationship root→node is a **geodesic rotation**: rotor `R = exp(−½ θ B)`, applied
`x_node = R x_root R̃`, where the **plane** `B = (x_root ∧ x_node)`-normalized and the **angle**
`θ = arccos(x_root·x_node)`. The bivector encodes *both* — its plane is the rotation plane, its magnitude
`|x_root ∧ x_node| = sin θ` is the angle. **Off the sphere = rotation + scalar dilation:** `x' = s·R x R̃`, a
*scaled rotor* (similarity transform) splitting magnitude (`s`) from orientation (`R`). This is why
the bookmarking agent's exact-Procrustes used **logm/expm** (the `so(n)` Lie algebra *is* the bivector space) —
exact, but matrix-exp/log expensive. **Why bivector, not cross product:** in 3D they are Hodge-duals
(`a∧b = I(a×b)`), but the cross product exists *only* in 3D/7D; for 384-dim e5 the bivector is the only correct
generalisation of "oriented plane of rotation."

**Embedding bivectors from the tangent space of each input (the concrete construction).** Rather than feed raw
`x_node, x_root` and hope attention discovers directionality, feed the **local-tangent bivector**
`B = x_root ∧ x_node` (or its compact rotor/log form) as an explicit feature. At a sphere point `x`, the tangent
space `T_x` is everything ⟂ `x`; `B` lies in the root's tangent plane and *is* the geodesic direction toward the
node. Two payoffs, both geometric-by-construction rather than learned: (1) **directionality is the sign of the
bivector** (`x_r∧x_n = −x_n∧x_r` ⇒ `μ(a|b)≠μ(b|a)`); (2) because the tangent frame is **local**, `B` is
**automatically region-adaptive** — the same construction yields different oriented relations in different parts
of the sphere, the *continuous* analog of routed per-cluster transforms (the tangent space **is** the local
chart, so no discrete clusters). The model then learns μ *on top of* the right primitive (`a∧b`, the
antisymmetric part the scalar cosine throws away — exactly the part `e5-cos` ignores yet still beats a fumbling
learned readout with). This is the model variant the GA note points at, now with a definite construction.

**Root-centred embeddings / tangent chart (for visualisation):** fix the **root as origin**; node embeddings
become root-relative, and the relationship to the root is the oriented `node ∧ root` (the same tangent bivector,
used for *display* not features). A **local chart** for the density explorer: 2D/3D coordinates where
displacement ≈ μ-relatedness and the **axes are the principal statistical variations of the root's
neighbourhood** (local PCA / a "tangent plane" at the root). Naturally per-root (each root its own chart).

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
- **Graph RAG / retrieval algorithm** — greedy bidirectional gather + μ-superposition sort. **Built + verified**
  (`eval_retrieval`); the discriminating Cooking case shows it fixes the dense-μ domain-leak. The μ-vs-hop
  top-k scatter is built too.
- **Hop-stratified retrieval eval** (formal recall@k / AUC vs held-out-edge ground truth) — *still proposed*;
  the head-to-head number against the WSP baseline.
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
