# Repeated-judge source regions — topology-only no-spend audit

## Bottom line

No preregistered source-region count passes the joint topology gate.  The audit used the frozen canonical
graphs and `K={64,96,128}`, but no scores, historical labels, candidate triples, embeddings, or judge calls.

The negative result has two different causes:

- on exploratory, the exact three-hop cores retain at most 2.95% of nodes and the optimistic four-endpoint
  capacity bound fails for the larger registered campaign sizes; and
- on fresh, the four-endpoint capacity bound passes, but core retention is at most 32.66%, below the frozen
  50% gate.

Increasing `K` does not rescue the design.  It reduces exploratory core retention and its `U_4(800)` bound.
There is therefore no selected source-region partition, and every history, candidate, Nomic, live-campaign,
covariance, QR, and CUDA gate remains closed.

## What a source region means

`source_region` was tested as an exclusive engineering unit for the 10% concentration cap, fold containment,
and dependence sensitivity.  It is deliberately distinct from `weak_component_id`, the true connected-
component diagnostic.  One weak component may contain many source regions; a source region never crosses a
weak component.

The topology-only construction is frozen before any endpoint exclusion:

1. canonicalize the parent/child maps to one undirected graph with nonempty string node IDs;
2. allocate one region to each true weak component, then repeatedly give the next region to the eligible
   component maximizing `node_count/current_region_count`, with ties resolved by canonical component order;
3. root a deterministic BFS spanning tree at the node farthest from the canonical seed;
4. jointly choose a tree edge and integer part allocation minimizing
   `|subtree_size * parts - node_count * subtree_parts|`; and
5. recurse on the descendant subtree and its complement.

Each tree cut leaves a spanning tree on both sides, so the output contains exactly `K` nonempty,
induced-connected, exclusive regions.  Stable region and weak-component IDs hash sorted canonical node
content rather than paths or traversal order.  The streamed floor/ceiling search uses linear working memory;
balanced recursion is approximately `O((V+E) log K)` and the articulation-heavy worst case is
`O(K(V+E))` for the frozen `K<=128` grid.

The first implementation used a distance ordering followed by a threshold split.  A six-node star requested
at two regions exposed that this could fragment one side and return four regions.  The real exploratory
`K=64` diagnostic returned thousands rather than 64.  That implementation was discarded before the reported
audit.  The spanning-tree regression now pins exact count and connectivity on hub graphs.

## Exact three-hop core

For source region `R` and canonical undirected radius `r=3`, the core is

```text
R_core = {v in R : B_r(v) is a subset of R}.
```

Equivalently, let `B` contain both endpoints of every cross-region edge.  The excluded halo is

```text
H_r = {v : distance(v, B) < r}.
```

Thus a boundary endpoint and nodes one or two hops from it are excluded, while a same-side node exactly three
hops from it is retained when its complete three-hop ball stays inside the region.  Cross-region core nodes
are at least `2r+1=7` hops apart, so their radius-three graph-feature supports are disjoint.

This is a feature-support statement, **not an independence statement**.  Nomic similarity, global judge
effects, prompt blocks, paths through the same weak component, and residual correlation may all cross source-
region boundaries.  Those terms must be estimated or bounded rather than erased by terminology.

## Frozen gates

For every `K`, corpus, and registered `G in {160,320,512,800}`, all four distinct candidate endpoints would
have to lie in one source-region core.  A disconnected distant comparator is consequently invalid.  Under a
10% per-region component cap,

```text
U_4(G) = sum_r min(floor(|R_core,r| / 4), max(1, floor(.10 G))).
```

The necessary topology gate requires:

- `U_4(G) >= G` for every registered `G`;
- at least 50% of graph nodes retained in source-region cores;
- at least 20 regions with four or more core nodes;
- exact count, full exclusive coverage, induced connectivity, and weak-component containment; and
- the exact radius-three core/support guarantee.

The coarsest jointly passing `K` would be selected.  There is no fallback outside the grid and no threshold is
changed after inspecting the audit.  `U_4` is only an optimistic upper bound: a pass would not establish the
later 32-cell, history-disjoint, Nomic-agreement, or exact-packing constraints.

## Frozen inputs

| corpus | retained graph | bytes | SHA-256 |
|---|---|---:|---|
| exploratory | SimpleWiki `100k_cats/category_parent.tsv` | 10,126,922 | `4881beedfd876e3abb9f1783cbc3fb8a7350e108e3f531cc4de28ef9956dc8ec` |
| exploratory identity | `100k_cats/metadata.json` | 308 | `15a632799adebd8b4f736c8420add8969bb5e5e4f20aa21ecaae9e2d95a61577` |
| fresh | enwiki scoped LMDB `data.mdb`, retained under `Behavior` | 1,319,403,520 | `3bcfe59a3f85870f377fad1ea77547f7c3566370f6172e27748f4f7ceba5d690` |

The canonical exploratory graph has 84,136 nodes, 196,876 undirected non-self edges, and 34 weak components.
The fresh graph has 75,901 nodes, 99,971 edges, and one weak component.  Its loader also binds the exploratory
graph because SimpleWiki-title overlap is excluded from the fresh slice.  LMDB `lock.mdb` is runtime state and
is excluded from scientific identity.  The content-addressed graph files are an immutable/read-only
operational precondition during the audit; assignment/core records expose any later reproduction mismatch.

## Results

| corpus | `K` | cut-edge fraction | core nodes | core fraction | effective regions | `U_4(160)` | `U_4(320)` | `U_4(512)` | `U_4(800)` | capacity ESS at 800 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exploratory | 64 | 41.87% | 2,485 | 2.95% | 26 | 173 | 247 | 304 | 391 | 7.01 |
| exploratory | 96 | 44.19% | 2,087 | 2.48% | 28 | 169 | 218 | 256 | 314 | 6.62 |
| exploratory | 128 | 45.49% | 2,037 | 2.42% | 32 | 164 | 199 | 237 | 295 | 6.19 |
| fresh | 64 | 9.77% | 24,789 | 32.66% | 64 | 992 | 1,906 | 2,889 | 4,056 | 56.49 |
| fresh | 96 | 10.11% | 23,745 | 31.28% | 95 | 1,464 | 2,769 | 3,885 | 4,805 | 78.88 |
| fresh | 128 | 10.62% | 23,273 | 30.66% | 125 | 1,852 | 3,190 | 4,218 | 5,003 | 91.06 |

`effective regions` counts cores with at least four nodes.  Capacity ESS is
`(sum contribution)^2 / sum(contribution^2)` for the `G=800` per-region `U_4` contributions; it diagnoses
how strongly the nominal upper bound is concentrated in a few regions.

Exploratory fails both retention and capacity, so lowering only the 50% retention threshold would not produce
a joint pass.  Its large weak component has hub/small-world expansion: recursive connected cuts expose many
boundary edges, and a three-hop halo consumes nearly all of the large component's pieces.  The intact small
weak components then dominate the remaining capacity, which explains the very low capacity ESS.

Fresh has much lower cut-edge fractions and ample optimistic capacity, yet three-hop expansion still removes
more than two thirds of its nodes.  More regions create additional cap slots but slightly reduce retained
core mass.  This separation between capacity and retention is why both gates are reported rather than
choosing whichever tells the preferred story.

## Alternatives and disposition

| alternative | disposition | reason |
|---|---|---|
| relabel weak components | already rejected | literal weak components fail the prior 10% capacity audit |
| top-level rooted branches | reject | fresh has only eight direct branches and multi-branch assignment needs an explicit exclusive rule |
| overlapping regions | reject | one endpoint would have ambiguous cap, fold, and resampling ownership |
| recompute regions after history exclusions | reject | exclusions would manufacture new cap slots and leak selection into the dependency unit |
| partition using outcomes, residuals, judges, or Nomic | reject for this gate | makes the supposedly outcome-blind source unit depend on the signal being tested |
| hash/random balanced regions | reject | balances counts by destroying graph locality and gives no local-support interpretation |
| remove or shorten the halo after this result | reject for this audit | changes a frozen support guarantee after observing its feasibility result |
| Louvain/METIS/spectral/min-cut communities | defer to a new exploratory family | potentially lower boundary, but exact count, portable determinism, dependency/version identity, and prospective gates must be frozen first |
| treat source regions as independent | reject | the graph, Nomic, prompt, and judge processes provide cross-region dependence paths |
| model cross-region dependence explicitly | recommended next fork | aligns inference with the eventual joint covariance/QR conditioner instead of demanding a false hard partition |

Graph-community alternatives may be explored in a separate diagnostic PR, but they cannot be substituted into
this frozen result.  A new partition family needs a prospective algorithm/version contract and must rerun the
same full grid.  A more promising path is to keep an exclusive concentration/fold label while replacing hard
three-hop independence with a powered cross-region dependence model: report graph-kernel exposure, estimate or
upper-bound effective source information, and calibrate prompt-block/source-region/graph-aware inference under
the complete selection procedure.

## Authorization and next work

All authorization fields are false.  In particular, this PR does not migrate the current v2 selector's
`source_component` schema.  A future compatible schema must require both `source_region` and
`weak_component_id`, use only source region for caps/folds, propagate both IDs, require all four endpoints in
one permitted core, and reject a legacy alias.  Implementing that migration against a failed partition would
create an unusable contract and is therefore deferred.

The next no-spend PR should preregister and power the dependence-aware fork before reading history or Nomic:

1. distinguish the exclusive concentration label from a claim of independence;
2. define a topology-only cross-region exposure matrix from the already frozen graph geometry;
3. compare hard core exclusion with conservative graph-aware multiplier/parametric-bootstrap inference and
   report effective source information under sensitivity envelopes;
4. require familywise error and power across both corpora and all registered `G`; and
5. unlock the attempted-input historical inventory only if that bridge passes.

Only then should the pipeline proceed to structural enumeration, a revision-pinned independent Nomic cache,
exact packability, prompt/position pilots, repeated judge collection, covariance promotion, and the joint
square-root/QR/CUDA implementation.  `JointPosterior` remains the learned decision comparator; the QR
conditioner remains the numerical implementation of a statistically promoted covariance model.

Verification passes 25 focused source-region/runner tests, 117 repeated-judge regressions, and 15 graph-
geometry tests.  Python compilation and `git diff --check` are clean.  The multiprocessing repeated-judge and
graph-geometry suites are run in separate pytest processes so their BLAS-runtime provenance is not affected by
test-order library loading.

## Reproduction

```bash
python prototypes/mu_cosine/run_repeated_judge_source_regions.py \
  --artifact-repo /path/to/UnifyWeaver-with-ignored-graph-artifacts \
  --out /tmp/repeated_judge_source_regions.json
```

The default command writes the complete blocked audit and exits with status 2.  `--audit-only` returns zero
after writing identical JSON for explicit reporting workflows and unlocks nothing.  The tracked artifact is
`repro/repeated_judge_source_regions/summary.json`: 64,449 bytes, SHA-256
`66e3fd0deb4f63a3c20e52d04ef1210490bb17f3dc3f33794156dc4e67d3a46f`.  The runner stores content records,
not machine paths or elapsed time.
