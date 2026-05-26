# F# CSR Kernel Integration: Benchmark + Bidirectional Search

**Status**: Design. Ready for implementation.
**Date**: 2026-05-26
**Depends on**: CSR reader (done), dual-CSR (done), cost analyzer (done)

## 1. The big picture

The CSR work so far has been data plumbing: reading packed binary
files, wiring them into WcLookupSources, and auto-selecting the
store via the cost analyzer. None of this has changed the *algorithm*.

The payoff is algorithmic: **bidirectional effective-distance search**.
The current `category_ancestor/4` kernel only walks upward (child ->
parent). With CSR providing parent -> children lookup, we can search
from both ends and meet in the middle. This reduces the search space
exponentially for deep graphs.

This doc has two phases:
1. Benchmark the cost analyzer on the real effective-distance workload
2. Implement and benchmark bidirectional search

## 2. Phase A: Benchmark the cost analyzer

### 2.1 Goal

Run the effective-distance workload with different `edge_store` modes
and verify that:
1. The cost analyzer picks the right mode for each scale
2. The CSR-backed path produces identical results to LMDB
3. The performance characteristics match the cost model predictions

### 2.2 Benchmark matrix

| Scale | Edges | Seeds | edge_store=auto resolves to | Expected winner |
|---|---|---|---|---|
| dev | 198 | 21 | lmdb_eager (few queries, small) | lmdb_eager |
| small | 2k | 100 | lmdb_cached or lmdb_eager | lmdb_cached |
| medium | 6k | 300 | depends on query_count | threshold test |
| large | 20k | 1000 | csr (many queries, static) | csr |

### 2.3 Implementation

Create `tests/core/test_wam_fsharp_cost_analyzer_bench.pl`:

1. Generate LMDB fixtures at each scale using
   `generate_synthetic_phase1_lmdb.py`
2. Build CSR artifacts using `build_csr_artifact.py`
3. For each scale, generate F# projects with:
   - `edge_store(lmdb_cached)` (baseline)
   - `edge_store(lmdb_eager)`
   - `edge_store(csr)` (forward CSR only)
   - `edge_store(auto)` (cost analyzer picks)
4. Run each project, compare:
   - Correctness: all modes produce same BFS ancestor counts
   - Timing: which mode is fastest at each scale
   - Cost analyzer accuracy: did auto pick the fastest mode?

### 2.4 Key insight for the benchmark

The current F# E2E benchmark (`test_wam_fsharp_lmdb_e2e_bench.pl`)
uses a BFS kernel that calls `lookupParents` per node. This kernel
works with ANY `ILookupSource` — LMDB, CSR, eager Map, cached.
So we can benchmark different stores without changing the kernel.

The `resolveFactLookup "category_parent" ctx` call in the kernel
dispatch automatically uses whatever is in `WcLookupSources`. So
wiring a CSR into `WcLookupSources["category_parent"]` makes the
kernel use CSR transparently.

## 3. Phase B: Bidirectional search kernel

### 3.1 Problem with upward-only search

The current `category_ancestor/4` does DFS upward:

```
ancestor(Cat, Root, Hops, Visited) :-
    parent(Cat, Mid),       % upward step
    \+ member(Mid, Visited),
    (Mid = Root -> Hops = 1
    ; ancestor(Mid, Root, H, [Mid|Visited]), Hops is H + 1
    ).
```

For a graph where root is 10 hops away, this explores ALL upward
paths from Cat, including branches that never reach Root. The
demand-set filter (reverse BFS from Root) prunes unreachable nodes,
but the remaining search tree can still be large.

### 3.2 Bidirectional search

With CSR providing parent -> children lookup, we can search from
both ends:

```
bidirectional_ancestor(Cat, Root, MaxDepth) :-
    % Phase 1: BFS downward from Root to find reachable descendants
    bfs_down(Root, MaxDepth, RootDescendants),
    % Phase 2: BFS upward from Cat, stopping when we hit RootDescendants
    bfs_up_to_set(Cat, RootDescendants, MaxDepth, Hops).
```

This is the **meet-in-the-middle** optimization: instead of searching
depth D from one end (exploring O(b^D) nodes where b is branching
factor), search depth D/2 from each end (exploring O(2 * b^(D/2))
nodes). For b=3, D=10: upward-only explores 59049 nodes;
bidirectional explores 486 nodes — a 121x reduction.

### 3.3 Implementation approach

The bidirectional kernel is a native F# function (not WAM-compiled).
It plugs into the existing kernel detection + FFI dispatch:

```fsharp
/// Bidirectional effective-distance kernel.
/// Phase 1: BFS down from root via category_child CSR.
/// Phase 2: BFS up from seed via category_parent, stopping at
///          the root-reachable set.
let bidirectionalAncestor
    (lookupParents: int -> int list)
    (lookupChildren: int -> int list)
    (cat: int) (root: int) (maxDepth: int) : int list =
    // Phase 1: BFS down from root (using CSR category_child)
    let rootSet = bfsDown lookupChildren root (maxDepth / 2)
    // Phase 2: BFS up from cat, each hit on rootSet is a path
    bfsUpToSet lookupParents cat rootSet (maxDepth - maxDepth / 2)
```

### 3.4 Where it lives

- Template: `templates/targets/fsharp_wam/kernel_bidirectional_ancestor.fs.mustache`
- Kernel detection: extend `detect_kernels/2` in `wam_fsharp_target.pl`
  to recognize when both `category_parent` and `category_child` are
  available and the kernel is `category_ancestor/4`
- Option: `kernel_mode(bidirectional)` or auto-detect when `csr_path`
  is set

### 3.5 Correctness constraint

Bidirectional search must produce the SAME effective-distance values
as upward-only search. The Hops values may differ in ordering but
the sum `Sigma (Hops+1)^(-n)` must be identical. Test this by
running both kernels on the same fixture and comparing results.

## 4. Sequencing

```
Phase A: Benchmark cost analyzer (1 session)
    |
    v
Phase B: Bidirectional kernel (1-2 sessions)
    |
    v
Benchmark: bidirectional vs upward-only speedup
```

Phase A is independent and validates existing infrastructure.
Phase B is the algorithmic payoff and depends on Phase A for
the benchmark harness.

## 5. Success criteria

- Phase A: cost analyzer picks the fastest mode at 3/4 scales
- Phase B: bidirectional search produces identical effective-distance
  values with measurable speedup (>2x) on deep graphs

## 6. References

- CSR reader: `templates/targets/fsharp_wam/csr_reader.fs.mustache`
- CSR philosophy: `docs/design/WAM_FSHARP_CSR_PHILOSOPHY.md`
- Cost analyzer: `docs/design/WAM_FSHARP_COST_ANALYZER_DESIGN.md`
- Existing BFS kernel: `templates/targets/fsharp_wam/kernel_category_ancestor.fs.mustache`
- E2E benchmark: `tests/core/test_wam_fsharp_lmdb_e2e_bench.pl`
- Reverse index artifacts: `docs/design/WAM_REVERSE_INDEX_ARTIFACTS.md`
