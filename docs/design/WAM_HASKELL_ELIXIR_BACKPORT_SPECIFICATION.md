# WAM Haskell Backport Specification From Elixir Large-Scale Benchmarks

## Scope

This document specifies planner-visible features for porting the useful
large-scale WAM-Elixir benchmark lessons back to WAM-Haskell. The semantic
filtering clauses are included as a second-stage cross-target extension, not as
part of the initial Haskell-only backport.

The initial target is the effective-distance family:

```prolog
category_ancestor(Cat, Parent, 1, Visited) :-
    category_parent(Cat, Parent),
    \+ member(Parent, Visited).

category_ancestor(Cat, Ancestor, Hops, Visited) :-
    category_parent(Cat, Mid),
    \+ member(Mid, Visited),
    category_ancestor(Mid, Ancestor, H1, [Mid|Visited]),
    Hops is H1 + 1.
```

The same concepts should generalize to other graph kernels when their direction
and bound arguments are known.

## Planner Declarations

The planner should support explicit declarations rather than infer semantic
filtering silently.

### Kernel Selection

```prolog
:- use_kernel(category_ancestor/4, ffi).
:- use_kernel(category_ancestor/4, demand_filter(root_bound)).
```

`ffi` requests the native Haskell kernel where the target supports it.
`demand_filter(root_bound)` allows structural filtering when the root argument
is bound.

### Graph Shape

```prolog
:- predicate_property(category_ancestor/4,
    graph_kernel(edge_predicate(category_parent/2),
                 direction(child_to_parent),
                 start_arg(1),
                 root_arg(2),
                 visited_arg(4))).
```

This declaration lets the planner compute a reverse adjacency map and demand
set without depending on predicate names.

## Stage 2: Cross-Target Semantic Planning

Semantic filtering is intentionally a second stage. The declarations below are
target-neutral planner inputs that Haskell, Elixir, Rust, Scala, or a
distributed backend could consume. The first Haskell backport does not depend on
them.

### Embedding Availability

```prolog
:- has_embeddings(category/1, wikipedia_category_minilm).
:- embedding_index(wikipedia_category_minilm,
    artifact('data/embeddings/wiki_categories.minilm.index')).
```

`has_embeddings/2` states that terms in a domain have precomputed vectors.
Targets may use this for semantic prefilters only when a query explicitly opts
in.

### Semantic Prefilter

```prolog
:- query_hint(effective_distance/3,
    semantic_prefilter(top_k(5000000),
                       endpoint_args([1, 2]),
                       embedding_source(wikipedia_category_minilm),
                       cache_prepopulate(true))).
```

The top-K value is a planning budget. The planner may score candidates by
similarity to one or more endpoints, select local top-K per shard, merge the
candidate streams, and prepopulate the LMDB cache for the resulting candidate
set.

Unless paired with a completeness declaration, semantic prefiltering is a
heuristic planning hint. It may reduce recall if used to prune the actual query
domain. It is safe for cache prepopulation because cache warming does not remove
solutions by itself.

## Required Semantics

### Demand Filter

For a child-to-parent edge relation and a bound root:

1. Build or access child -> parents adjacency.
2. Build reverse adjacency parent -> children.
3. Run BFS from the root over reverse adjacency.
4. The visited set is the demand set: nodes that can reach the root.
5. Before calling the recursive kernel for a seed, return zero/no solution if
   the seed is outside the demand set.
6. Inside the kernel, ignore outgoing edges whose destination parent is outside
   the demand set.

This preserves semantics for root-bound reachability queries.

### Path-Preserving vs Folded Kernel

The planner must distinguish these modes:

```prolog
:- kernel_mode(category_ancestor/4, path_preserving).
:- kernel_mode(category_ancestor/4, folded_aggregate(sum)).
```

`path_preserving` enumerates the same rows the original predicate would
enumerate.

`folded_aggregate(sum)` may fold hop weights directly only when the enclosing
query observes the aggregate result and does not observe paths, hop rows, or
witnesses.

Correctness tests must include:

- Diamond graphs where duplicate paths are semantically meaningful.
- Cycles guarded by the visited argument.
- Multiple roots and roots with tiny demand sets.
- Seeds outside the demand set.
- A comparison between folded aggregate output and path-preserving output
  aggregated externally.

### Cache Contract

LMDB cache modes remain bounded. Collision overwrite is allowed only when the
stored key is checked on lookup.

Cache prepopulation modes:

```prolog
:- cache_policy(category_parent/2, lmdb(two_level)).
:- cache_prepopulate(category_parent/2, structural_demand(root_bound)).
:- cache_prepopulate(category_parent/2,
    semantic_top_k(5000000, wikipedia_category_minilm)).
```

Structural prepopulation may populate all demand-set adjacency lists.
Semantic prepopulation may populate candidate adjacency lists selected by the
embedding plan. Semantic prepopulation must not by itself prune the query unless
an additional completeness contract exists.

## Metrics

Targets should emit comparable metrics:

- `seed_count`
- `demand_set_size`
- `demand_total_nodes`
- `demand_filtered_nodes`
- `cache_preload_entries`
- `cache_hits`
- `cache_misses`
- `query_ms`
- `setup_ms`
- `aggregation_ms`

These metrics are required to distinguish storage latency from duplicated graph
exploration.
