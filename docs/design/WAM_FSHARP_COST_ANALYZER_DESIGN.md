# F# Edge Store Cost Analyzer: Design

**Status**: Design. Ready for implementation.
**Date**: 2026-05-26
**Depends on**: CSR reader (done), dual-CSR builder (done),
`WcLookupSources` auto-wiring (done)

## 1. Problem

The F# WAM target currently requires manual selection of the edge
store backend via codegen options:

```prolog
write_wam_fsharp_project(Preds, [
    lmdb_path('/path/to/lmdb'),
    lmdb_materialisation(cached),   % eager | lazy | cached
    csr_path('/path/to/csr'),       % optional reverse CSR
    csr_parent_path('/path/to/fwd') % optional forward CSR
], Dir).
```

The user must know when CSR beats LMDB, when eager beats cached,
and when to build CSR at all. The cost analyzer should make this
choice automatically based on workload metadata.

## 2. Decision variables

| Variable | Controls | Values |
|---|---|---|
| `edge_store` | backing store for category_parent | `lmdb_cached`, `lmdb_eager`, `csr`, `dual_csr` |
| `materialisation` | LMDB access pattern | `eager`, `lazy`, `cached` |
| `reverse_store` | backing store for category_child | `none`, `lmdb`, `csr` |
| `index_backend` | CSR index lookup method | `sorted_array`, `lmdb_offset` |

## 3. Inputs to the cost model

| Input | Source | Type | Example |
|---|---|---|---|
| `expected_query_count` | caller / manifest | int | 1000 |
| `expected_lookups_per_query` | workload analysis | int | 50 |
| `edge_count` | corpus manifest / LMDB meta | int | 6000 |
| `parent_count` | corpus manifest | int | 750 |
| `graph_mutability` | caller declaration | `static` / `dynamic` | `static` |
| `needs_reverse` | kernel analysis | bool | `true` (bidirectional) |
| `available_memory_mb` | system / option | int | 8000 |
| `id_density` | computed from max_id/count | float | 0.8 |

## 4. Cost model

### 4.1 Per-lookup cost (microseconds, approximate)

| Store | Cold | Warm (cached) | Notes |
|---|---|---|---|
| LMDB cursor | 5-10 | 2-5 | B-tree traversal |
| LMDB cached (L1 hit) | - | 0.01 | Array index |
| LMDB cached (L2 hit) | - | 0.1 | ConcurrentDict |
| CSR binary search | 1-3 | 1-3 | O(log n) on index |
| CSR + cache (L1 hit) | - | 0.01 | Same as LMDB cached |
| Eager Map.tryFind | - | 0.5 | O(log n) tree |
| Eager Dict.TryGetValue | - | 0.1 | O(1) hashtable |

### 4.2 Materialisation cost (milliseconds)

| Operation | Cost formula | Example (6k edges) |
|---|---|---|
| LMDB eager load | `edge_count * 0.005` | 30 ms |
| CSR build from LMDB | `edge_count * 0.01` | 60 ms |
| CSR open (sorted_array) | `parent_count * 0.016` | 12 ms (load .idx) |
| CSR open (lmdb_offset) | `parent_count * 0.001` | 0.75 ms (open env) |

### 4.3 Break-even calculation

```
total_lookups = expected_query_count * expected_lookups_per_query
csr_break_even = csr_build_cost / (lmdb_per_lookup - csr_per_lookup)
```

For 6k edges, 50 lookups/query:
- CSR build: 60 ms
- LMDB cached warm lookup: 0.1 us
- CSR cached warm lookup: 0.01 us (negligible difference when cached)
- CSR cold lookup: 2 us vs LMDB cursor: 7 us (5 us savings)
- Break-even: 60ms / 5us = 12,000 lookups = 240 queries

At >240 queries, CSR wins. Below that, LMDB cached is simpler.

### 4.4 When each store wins

| Workload | Recommended | Why |
|---|---|---|
| <100 queries, dynamic graph | `lmdb_cached` | No preprocessing, handles mutations |
| >100 queries, static, small graph | `lmdb_eager` | Fast Dict after one-time load |
| >500 queries, static, large graph | `csr` | Amortizes build cost, compact |
| Bidirectional search needed | `dual_csr` | Both directions available |
| Memory-constrained | `lmdb_cached` | Bounded L2, no eager load |

## 5. Implementation

### 5.1 Predicate: `resolve_edge_store/2`

Location: `src/unifyweaver/core/cost_model.pl`

```prolog
%% resolve_edge_store(+Options, -ResolvedOptions)
%  Resolve edge_store(auto) into a concrete store selection.
%  Adds/replaces: lmdb_materialisation, csr_path, csr_parent_path.
resolve_edge_store(Options, ResolvedOptions) :-
    option(edge_store(auto), Options, auto),
    !,
    option(expected_query_count(Q), Options, 1),
    option(expected_lookups_per_query(L), Options, 50),
    option(edge_count(E), Options, 0),
    option(graph_mutability(Mut), Options, static),
    option(needs_reverse(NeedsRev), Options, false),
    TotalLookups is Q * L,
    CsrBuildMs is E * 0.01,
    PerLookupSavingsUs is 5,
    BreakEvenLookups is CsrBuildMs * 1000 / PerLookupSavingsUs,
    (   Mut = dynamic
    ->  Store = lmdb_cached
    ;   TotalLookups < BreakEvenLookups
    ->  (E < 50000 -> Store = lmdb_eager ; Store = lmdb_cached)
    ;   NeedsRev = true
    ->  Store = dual_csr
    ;   Store = csr
    ),
    apply_store_selection(Store, Options, ResolvedOptions).
resolve_edge_store(Options, Options).  % not auto, pass through
```

### 5.2 Integration point

In `write_wam_fsharp_project/3`, call `resolve_edge_store/2` early
in the option resolution chain (before `generate_program_fs`):

```prolog
write_wam_fsharp_project(Predicates, Options0, ProjectDir) :-
    resolve_edge_store(Options0, Options1),
    ... existing code with Options1 ...
```

### 5.3 Applying the selection

```prolog
apply_store_selection(lmdb_cached, Opts0, Opts) :-
    merge_options([lmdb_materialisation(cached)], Opts0, Opts).
apply_store_selection(lmdb_eager, Opts0, Opts) :-
    merge_options([lmdb_materialisation(eager)], Opts0, Opts).
apply_store_selection(csr, Opts0, Opts) :-
    % Build CSR from LMDB at project generation time
    option(lmdb_path(LmdbPath), Opts0),
    build_csr_artifact(LmdbPath, CsrPath, category_parent),
    merge_options([csr_parent_path(CsrPath)], Opts0, Opts).
apply_store_selection(dual_csr, Opts0, Opts) :-
    option(lmdb_path(LmdbPath), Opts0),
    build_csr_artifact(LmdbPath, ParentCsrPath, category_parent),
    build_csr_artifact(LmdbPath, ChildCsrPath, category_child),
    merge_options([csr_parent_path(ParentCsrPath),
                   csr_path(ChildCsrPath)], Opts0, Opts).
```

### 5.4 Index backend selection

```prolog
resolve_index_backend(Options, Backend) :-
    option(parent_count(N), Options, 0),
    option(max_id(MaxId), Options, 0),
    (   MaxId > 0, N > 0
    ->  Density is N / MaxId,
        (Density < 0.1 -> Backend = lmdb_offset ; Backend = sorted_array)
    ;   Backend = sorted_array
    ).
```

Sparse IDs (density < 10%) benefit from LMDB offset index.
Dense IDs work fine with binary search.

## 6. User-facing API

```prolog
%% Auto mode (cost analyzer chooses):
write_wam_fsharp_project(Preds, [
    lmdb_path(Path),
    edge_store(auto),
    expected_query_count(1000),
    edge_count(6000)
], Dir).

%% Explicit mode (user overrides):
write_wam_fsharp_project(Preds, [
    lmdb_path(Path),
    edge_store(dual_csr),
    lmdb_materialisation(cached)
], Dir).
```

When `edge_store(auto)` is not present, the existing behavior
is preserved (manual options pass through unchanged).

## 7. Testing

```prolog
test_resolve_edge_store :-
    % Small static graph, few queries -> lmdb_eager
    resolve_edge_store([edge_store(auto), edge_count(1000),
                        expected_query_count(10)], R1),
    member(lmdb_materialisation(eager), R1),

    % Large static graph, many queries -> csr
    resolve_edge_store([edge_store(auto), edge_count(100000),
                        expected_query_count(10000),
                        lmdb_path('/tmp/test')], R2),
    member(csr_parent_path(_), R2),

    % Dynamic graph -> always lmdb_cached
    resolve_edge_store([edge_store(auto), edge_count(100000),
                        graph_mutability(dynamic)], R3),
    member(lmdb_materialisation(cached), R3).
```

## 8. References

- CSR philosophy: `docs/design/WAM_FSHARP_CSR_PHILOSOPHY.md`
- Cost model: `src/unifyweaver/core/cost_model.pl`
- LMDB lazy philosophy: `docs/design/WAM_LMDB_LAZY_PHILOSOPHY.md`
- Existing cost predicates: `resolve_auto_cache_strategy` in
  `wam_haskell_target.pl` (similar pattern for Haskell)
- CSR builder: `examples/benchmark/build_csr_artifact.py`
- CSR benchmark: `docs/reports/reverse_csr_scale_sweep.md`
