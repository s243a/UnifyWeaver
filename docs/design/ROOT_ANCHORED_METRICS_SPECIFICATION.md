# Root-Anchored Metrics — Specification

Companion to `ROOT_ANCHORED_METRICS_PHILOSOPHY.md`. This doc fixes the **spec
surface**: the directive a user writes, the semantics of each metric, the
semiring abstraction, cycle/depth rules, the recognition contract, and the
materialised output layout. It does not prescribe the algorithm (that is the
implementation plan's job) beyond what correctness requires.

## 1. The directive

A root-anchored metric is declared with one directive next to the relational
clauses that define the edge relation:

```prolog
:- root_metric(Name/Arity,
       [ edge(EdgePred/2, Direction),   % Direction = up | down
         boundary(RootTerm, V0),        % fixed value at the root
         combine(Combine),              % succ | plus(K) | scale(Decay)
         aggregate(Aggregate),          % min | max | sum
         max_depth(D),                  % non-negative integer
         cycles(Policy),                % bounded | scc | ignore (default bounded)
         materialize(Where) ]).         % ingest | kernel (default kernel)
```

`Name/Arity` is the user-facing predicate the metric is exposed as, conventional
arity 3: `Name(Root, Node, Value)`. The relational clauses may still be present
(for the Prolog reference semantics / non-WAM targets); the directive tells the
compiler the recurrence class so it can lower efficiently instead of executing
them literally.

### Field semantics

- **`edge(EdgePred/2, Direction)`** — the adjacency relation. `Direction = up`
  means the metric is computed by following `EdgePred` from a node toward the
  root (child → parent); `down` follows the reverse. Internally the engine walks
  *from the root outward* over the opposite direction (see §4), so it needs both
  the relation and which way "toward root" points.
- **`boundary(RootTerm, V0)`** — the root node (a term, possibly a variable bound
  at query time) and the value assigned to it. `V0 = 0` for distances, `1.0` for
  flux.
- **`combine(Combine)`** — the per-edge step applied to a parent's value:
  - `succ` ≡ `plus(1)` — add one hop.
  - `plus(K)` — add a constant edge weight `K`.
  - `scale(Decay)` — multiply by `Decay` (0 < Decay ≤ 1) for flux.
- **`aggregate(Aggregate)`** — how a node combines contributions from multiple
  parents: `min`, `max`, or `sum`. This selects the semiring (§3).
- **`max_depth(D)`** — contributions beyond `D` hops from the root are not
  counted. Bounds cost and makes cyclic graphs well-defined.
- **`cycles(Policy)`** — `bounded` (default; rely on `max_depth`), `scc`
  (condense strongly-connected components first), or `ignore` (assume acyclic;
  undefined behaviour on a cycle — only for known DAGs).
- **`materialize(Where)`** — `kernel` (default; recompute per run as a node-DP)
  or `ingest` (precompute once, store `node → value`; see §6).

## 2. The three canonical instances

### 2.1 Minimum distance to root

```prolog
:- root_metric(min_dist_to_root/3,
       [ edge(parent/2, up), boundary(Root, 0),
         combine(succ), aggregate(min),
         max_depth(10), cycles(bounded), materialize(ingest) ]).
```

`min_dist_to_root(Root, Node, D)` — `D` is the fewest `parent` hops from `Node`
to `Root`, or unbound/`+inf` (absent from the materialised table) if `Node`
cannot reach `Root` within `max_depth`. Cycle-safe with no special handling.

### 2.2 Maximum distance to root

```prolog
:- root_metric(max_dist_to_root/3,
       [ edge(parent/2, up), boundary(Root, 0),
         combine(succ), aggregate(max),
         max_depth(10), cycles(bounded), materialize(ingest) ]).
```

`max_dist_to_root(Root, Node, D)` — the longest `parent` *walk* from `Node` to
`Root` of length `≤ max_depth`. With `cycles(bounded)` this is the longest walk
within the budget; with `cycles(scc)` it is the longest path in the condensed
DAG; with `cycles(ignore)` it is the DAG longest path (undefined if a cycle
exists).

### 2.3 Effective distance (flux)

```prolog
:- root_metric(effective_distance/3,
       [ edge(parent/2, up), boundary(Root, 1.0),
         combine(scale(0.2)), aggregate(sum),
         max_depth(10), cycles(bounded), materialize(kernel) ]).
```

`effective_distance(Root, Node, V)` — the linear difference equation
`V(Node) = Decay · Σ_{p∈parents} V(p)`, `V(Root) = 1`, truncated at `max_depth`.
This reproduces the path-sum `Σ_paths f(len)` without path enumeration; see §5
for the equivalence to the existing `effective_distance_sum` kernel.

## 3. The semiring abstraction

Each `(aggregate, combine)` pair is a semiring `(⊕, ⊗, 0̄, 1̄)`. The node value is

```
value(Node) = ⊕_{p ∈ parents(Node)}  ( value(p) ⊗ edge_weight )
value(Root) = 1̄   (the boundary V0 maps to the semiring identity / source)
```

| Metric    | ⊕ (aggregate) | ⊗ (combine)     | annihilator 0̄ | identity 1̄ |
|-----------|---------------|-----------------|---------------|-------------|
| min dist  | `min`         | `+`             | `+inf`        | `0`         |
| max dist  | `max`         | `+`             | `-inf`        | `0`         |
| flux      | `+`           | `× decay`       | `0`           | `1`         |

A single lowering template parameterised by the semiring covers all three. New
metrics (e.g. count-of-paths with `(sum, ×1)`, or PPR with a teleport term) are
added by naming a new semiring, not by writing a new traversal.

## 4. Evaluation semantics (what any correct lowering must produce)

Independent of algorithm, a conforming implementation must compute, for the
declared root `R` and every node `N` within `max_depth`:

- the **least fixed point** of the recurrence in §3 for `min`/`max`
  (shortest/longest under the depth bound), and
- the **depth-`D` truncation** of the linear fixed point for `sum` flux:
  `value(N) = Σ_{k=0}^{D} (contributions along walks of exactly k steps)`.

Canonical evaluation order (what the engine actually does): **start at the root**
and relax outward over the *reverse* of `edge`'s `Direction` (for `up`, walk the
reverse `child` adjacency `down` from the root). This is why both the relation
and its direction are required. Nodes unreachable from the root within
`max_depth` have value `0̄` and are omitted from the materialised table.

Determinism: results must be independent of node visitation order (the semiring
operations are associative/commutative). This is the same purity contract the
existing kernels rely on (`purity_certificate`).

## 5. Equivalence to the existing effective-distance kernel

The current kernel computes, per seed, `Σ over discovered hops of (hop+1)^(-n)`.
With `aggregate(sum)` + `combine(scale(1))` and a post-map, the bucketed-by-length
DP yields the identical sum:

```
count[N][L] = Σ_{p∈parents(N)} count[p][L-1],   count[R][0] = 1
effective_distance_sum(N) = Σ_{L=0}^{max_depth} count[N][L] · (L+1)^(-n)
```

This is the formal statement that the node-DP is **not an approximation** — it is
the same quantity, reorganised from "sum over paths" to "sum over path-lengths
weighted by path-count," which collapses the exponential to `O(edges·max_depth)`.

## 6. Materialised output layout (`materialize(ingest)`)

When materialised at ingest, the metric is stored in the Phase-1 LMDB next to the
graph (mirrors the scoped-subtree marker convention in
`build_scoped_subtree_lmdb.py`):

- A sub-db `metric_<name>` mapping `int32_le node → value`:
  - distances: `int32_le` value;
  - flux: `f64` little-endian (8-byte) value.
- A `meta` entry recording the metric's directive tuple + root + max_depth +
  build provenance, so a consumer can verify the stored table matches the query
  it intends (same root, same depth, same aggregate).

Query semantics against a materialised table:

- `Name(Root, Node, V)` is a single keyed lookup; absent ⇒ unreachable within
  depth.
- **Pruning** falls out: any traversal that needs "can a path through `N` still
  improve the best?" consults `metric_min_dist_to_root[N]`; if
  `stored_min + remaining_budget` cannot beat the incumbent, the branch is cut.
  This is the branch-prune the philosophy doc motivates.

A materialised table is only valid for the root it was built against; a different
root needs either its own table or the `kernel` mode.

## 7. Recognition contract

The compiler recognises a `root_metric/2` directive and, for each declared
metric, must:

1. Validate the tuple (known `combine`/`aggregate`/`cycles`; `max_depth ≥ 0`;
   `edge` predicate exists; boundary arity matches).
2. Emit (or look up) the semiring for `(aggregate, combine)`.
3. Choose the lowering by `materialize`: a node-DP kernel (`kernel`) or an
   ingest-time table builder + lookup stub (`ingest`).
4. Reuse existing infrastructure where it already fits — the shortest-path
   kernels (`weighted_shortest_path3`, `astar_shortest_path4`), the demand BFS
   over `category_child`, the visited-set machinery, and the LMDB fact source.

Inference of the pattern from bare `aggregate_all/3` over a transitive closure
(without the directive) is **out of scope for v1** and listed as a later
convenience in the plan.

## 8. Non-goals (v1)

- Multiple simultaneous roots / multi-source metrics (one root per table).
- Negative edge weights for `min` (would break the shortest-path lowering).
- Exact (non-truncated) flux on cyclic graphs via linear solve — the
  depth-bounded truncation is the supported semantics.
- Automatic inference without the directive (§7).

## See also

`ROOT_ANCHORED_METRICS_PHILOSOPHY.md`, `ROOT_ANCHORED_METRICS_IMPLEMENTATION_PLAN.md`,
`ALGORITHM_MANIFEST_SPECIFICATION.md`, `KERNEL_SHAPE_RECOGNITION.md`,
`WAM_DEMAND_FILTER_SPECIFICATION.md`.
