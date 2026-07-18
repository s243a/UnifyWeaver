# Root-Anchored Metrics — Philosophy

## What this is

A design for expressing **per-node graph metrics that are anchored at a root**
— minimum distance to root, maximum distance to root, and additive
"effective distance" / flux — as **declarative Prolog difference equations with
a boundary condition at the root**, which the compiler recognises and lowers to
an efficient **node-based dynamic program** (and, when the root is fixed,
materialises once at ingest into the fact store).

The deliverable in one sentence: *write the metric as a recurrence pinned at the
root in Prolog; let the target compute it in time linear in the graph (not
exponential in its paths), and store it so queries become lookups.*

This sits directly above two threads that already landed:

- **Demand-set-at-ingest** (`project_demand_set_at_ingest`, the scoped-subtree
  builder + `WAM_DEMAND` flag): do the expensive whole-graph work once at
  ingest, store it, make queries cheap. Root-anchored metrics are the natural
  next quantity to precompute and store alongside the scoped subtree.
- **Kernel shape recognition** (`KERNEL_SHAPE_RECOGNITION.md`) and the WAM
  lowering pipeline: the compiler already detects recursive kernel shapes and
  emits specialised code. Root-anchored metrics are a new shape to recognise.

## Why now

Benching the `category_ancestor` effective-distance query on the **coherent**
enwiki content graph (3-dump Correct-mode ingest; see
`project_enwiki_correct_ingest`) exposed a wall the older fixtures hid:

- On the genuine content DAG (**~2.98 edges/node**, deep), the current
  effective-distance kernel costs **~24 ms/seed** — a full 2.6M-seed query would
  take **~17 h** single-threaded.
- The kernel is **path-based**: per seed it walks to the root and records a hop
  value *for every distinct path*. On a near-tree (simplewiki, 1.03 edges/node)
  the path count ≈ the node count, so it was microseconds. On a dense DAG the
  number of distinct seed→root paths grows **multiplicatively** with branching
  — exponential — so a single seed spawns millions of paths.
- The earlier "fast" enwiki numbers were measured on the **broken flat-star
  graph** (mixed id-space ingest, 1.03 edges/node, almost nothing to traverse).
  They were a degenerate artifact, not real performance — a textbook case for
  `feedback_perf_skepticism`.

The data is now correct; the bottleneck has moved to the **algorithm**. The fix
is not to optimise path enumeration but to *stop enumerating paths* — compute a
per-node value once and reuse it.

## The core principle: the Prolog is the *what*, the target is the *how*

The same idea that motivates the whole WAM-target project applies here. The
Prolog should be a **specification of the quantity**, not an algorithm. It must
capture exactly the details that (a) determine correctness and (b) let the
compiler recognise *which* algorithm to apply — and nothing about the algorithm
itself.

A root-anchored metric, written declaratively, *is* a fixed-point with a
boundary. For minimum distance:

```prolog
dist_to_root(Root, Root, 0).
dist_to_root(Root, Node, D) :-
    parent(Node, P), dist_to_root(Root, P, D0), D is D0 + 1.

min_dist_to_root(Root, Node, D) :-
    aggregate_all(min(D0), dist_to_root(Root, Node, D0), D).
```

The base clause `dist_to_root(Root, Root, 0)` is the **fixed value at the root**.
Run naively this Prolog *is* the path enumeration that explodes. The point is
that the compiler recognises the pattern — `min` aggregation over an additive
`+1` transitive closure with a root boundary — and lowers it to **BFS from the
root with a visited set / stored distance**, never literal backtracking. The
declarative text stays simple; the efficiency lives in the target.

## Theory: one recurrence skeleton, three semirings

All three metrics share a recurrence shape — a node's value is an aggregate over
its parents' values, combined with a per-edge step, pinned at the root:

```
value(Root) = identity_boundary
value(Node) = AGGREGATE over p in parents(Node) of  COMBINE(value(p), edge_step)
```

They differ only in the `(AGGREGATE, COMBINE)` pair — i.e. in the **semiring**:

| Metric                | AGGREGATE | COMBINE      | boundary | algorithm class            |
|-----------------------|-----------|--------------|----------|----------------------------|
| min distance to root  | `min`     | `+1`         | `0`      | single-source shortest path|
| max distance to root  | `max`     | `+1`         | `0`      | longest path / longest walk|
| effective distance    | `+` (sum) | `× decay`    | `1`      | linear fixed point (KCL)   |

- **Min** is the tropical `(min, +)` semiring → Dijkstra/BFS. Each node's value
  is its shortest hop-distance to root.
- **Max** is `(max, +)` → longest path. Well-defined only on a DAG; on a cyclic
  graph it must be bounded (see *Cycles* below).
- **Effective distance** is the ordinary `(+, ×)` ring with a decay weight per
  edge → a linear difference equation `V = decay·Mᵀ·V` pinned at the root
  (`V(root)=1`). This is the discrete KCL / Green's-function form already
  sketched in `project_scan_strategy_cost_functions` (the additive
  flux / PPR variants). It reproduces the path-sum `Σ_paths (len+1)^(-n)` **without
  enumerating paths**, by counting paths bucketed by length:
  `count[Node][L] = Σ_parents count[parent][L-1]`, then
  `S(Node) = Σ_{L≤max_depth} count[Node][L]·(L+1)^(-n)`. `L` is bounded by
  `max_depth`, so each node carries a tiny fixed-width vector and the whole
  computation is one pass, `O(edges × max_depth)`.

For the related symmetric circuit construction in which independent semantic
embeddings modulate edge resistance and every node can leak to a common bath,
see [LEAKY_GRAPH_DIFFUSION.md](LEAKY_GRAPH_DIFFUSION.md).

Seeing all three as the same skeleton over different semirings is what lets a
single recognition rule + a single lowering template cover them.

## Cycles

The Wikipedia category graph has cycles, so cycle behaviour is part of
correctness, not an afterthought:

- **Min** is cycle-safe for free: BFS never re-visits, and a cycle can only ever
  offer a *longer* path, so it never improves the minimum.
- **Max** diverges on a cycle (you can loop forever to grow the length). It is
  only meaningful as a **depth-bounded longest walk** (cap at `max_depth`, the
  bound the kernel already uses) or after **SCC condensation** (collapse each
  strongly-connected component to one node). Depth-bounding is the simpler
  default and matches existing kernel behaviour.
- **Effective distance** with `decay < 1` converges even with cycles (it is a
  contraction), but the depth-bounded truncation makes it finite and exact to
  `max_depth` regardless; that is the recommended default.

## Alternatives considered

### A. Keep path enumeration, optimise constants
Status quo. Memoise sub-walks, prune aggressively, parallelise. **Rejected as the
primary fix**: the work is exponential in graph density; constant-factor wins
don't change the asymptotics. (Parallelism still helps once the algorithm is
linear.)

### B. In-kernel node-DP (memoised, per run)
Recognise the shape and emit a memoised node-DP that runs at query/load time.
Linear in edges, no schema change. **Recommended for ad-hoc / runtime-chosen
roots** — the analogue of the query-time demand BFS.

### C. Ingest-materialised per-node value (stored in the DB)
Compute the metric once at ingest (when the root is known) and store
`node → value` in a sub-db; queries become O(1) lookups plus the prune.
**Recommended as the default for a known fixed root** — the exact analogue of the
pre-scoped subtree DB, and it composes with it (store the metric *in* the scoped
DB).

The B-vs-C choice mirrors the demand-set "option 3" conclusion: support both,
default to ingest-materialisation (C) when the root is fixed, fall back to
in-kernel (B) at small scale or for a runtime root.

### D. How explicitly should the Prolog declare the pattern?
- **Inferred** — write pure `aggregate_all(min(...), closure, ...)` and have the
  compiler detect the shape. Cleanest Prolog, but brittle: inference must catch
  every spelling.
- **Declared** — keep the relational clauses and add a small directive naming the
  recurrence class. UnifyWeaver already leans on directives (`visited_set/2`, the
  algorithm manifest, demand directives), so this is consistent and robust.

**Recommendation: declared.** A directive that names the five details below is
the right "spec surface"; inference can be a later convenience layer on top.

## What the Prolog must capture (and what it must not)

Must capture — the five details that fix correctness *and* select the algorithm:

1. **Edge relation + direction** — `parent/2`, i.e. which way is "up."
2. **Boundary** — the root node and its fixed value (`0` for distance, `1` for flux).
3. **Combine** — the per-edge step (`succ`/`+1`, or `× decay`).
4. **Aggregate** — `min` / `max` / `sum`; this one choice picks the semiring.
5. **Depth bound** — `max_depth` (and, implicitly, the cycle policy it induces).

Must **not** capture — these belong to the target and existing machinery:
traversal order, the visited set, memoisation, the iteration/convergence scheme,
SCC handling, the demand set, LMDB layout, parallelism.

That five-tuple is exactly what distinguishes min from max from flux while
leaving the algorithm entirely to the backend. The specification doc turns it
into a concrete directive; the implementation plan sequences min-distance first.

## See also

- `ROOT_ANCHORED_METRICS_SPECIFICATION.md` — the directive + semantics.
- `ROOT_ANCHORED_METRICS_IMPLEMENTATION_PLAN.md` — phased delivery.
- `KERNEL_SHAPE_RECOGNITION.md`, `WAM_DEMAND_FILTER_PHILOSOPHY.md`,
  `SCAN_STRATEGY_PHILOSOPHY.md`, `COST_FUNCTION_PHILOSOPHY.md`,
  `TREE_LIKENESS_INDEX_THEORY.md` (graph density is what triggers the blowup).
- `project_enwiki_correct_ingest`, `project_demand_set_at_ingest` (memory).
