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

### 3.2 Bidirectional search with path-cost threshold

With CSR providing parent -> children lookup, we can search in both
directions. But child fan-out is much larger than parent fan-out, so
we need to constrain the search space.

**Key design**: two separate concerns:

1. **Inclusion threshold** (path-cost budget): which paths to explore.
   Each step has a direction-dependent cost. Paths are pruned when
   cumulative cost exceeds the budget. This controls the search space.

2. **Distance metric** (weighting function): how to score included
   paths in the effective-distance sum. Currently power-law
   `(hops+1)^(-n)`. Could later be flux-based or other schemes
   (see `COST_FUNCTION_PHILOSOPHY.md`). The metric is independent
   of the inclusion threshold.

These two are specified separately. You CAN use the same cost
function for both, but you don't have to.

#### Path-cost function (for pruning)

```
step_cost(parent_hop) = 1.0    -- cheap, natural direction
step_cost(child_hop)  = k      -- expensive, k > 1 (e.g., 3.0)
path_cost(path)       = sum of step_cost for each hop in path
include(path)         = path_cost(path) <= budget
```

With k=3 and budget=10: a path can take 10 parent hops, or 3 child
hops, or 1 child + 7 parent hops, etc. This naturally limits child
exploration without a hard depth cap.

#### Distance metric (for d_eff, unchanged initially)

```
d_eff = (Σ (hops+1)^(-n))^(-1/n)   -- current power-law, raw hop count
```

Initially we keep the existing power-law metric using raw hop count
(not weighted by direction). The inclusion threshold determines WHICH
paths contribute to the sum; the metric determines HOW they contribute.
Later we can explore direction-weighted metrics or flux-based schemes.

#### Kernel signature

```fsharp
let bidirectionalAncestor
    (lookupParents: int -> int list)
    (lookupChildren: int -> int list)
    (cat: int) (root: int)
    (parentStepCost: float) (childStepCost: float) (costBudget: float)
    : int list =
    // BFS with per-step costs, pruning when cumulative > budget.
    // Returns list of hop counts for paths that reach root.
    ...
```

#### Why bidirectional finds new paths

Upward-only search finds paths like: seed -> A -> B -> root (always
going up). Bidirectional finds additional "non-carrot-shaped" paths
like: seed -> A -> B <- C <- root (going up from seed AND down from
root). These are paths through shared ancestors that the upward-only
kernel cannot discover.

The effective-distance with bidirectional search will generally be
**lower** (more paths = more terms = lower d_eff). This is a feature:
it captures more of the graph's connectivity.

### 3.3 Implementation approach

The bidirectional kernel is a native F# function (not WAM-compiled).
It plugs into the existing kernel detection + FFI dispatch:

```fsharp
/// Bidirectional effective-distance kernel with path-cost pruning.
///
/// Explores paths in both directions (parent hops and child hops).
/// Each step has a direction-dependent cost. Paths are pruned when
/// cumulative cost exceeds the budget. Returns hop counts for all
/// paths that reach root within the budget.
///
/// The distance metric (how hop counts become d_eff) is applied
/// outside this function -- it only returns raw hop counts.
let bidirectionalAncestor
    (lookupParents: int -> int list)
    (lookupChildren: int -> int list)
    (cat: int) (root: int)
    (parentCost: float) (childCost: float) (budget: float)
    : int list =
    // DFS/BFS with cumulative cost tracking.
    // Each frontier entry: (node, cumulative_cost, hop_count, visited_set)
    // Parent steps add parentCost; child steps add childCost.
    // Prune when cumulative_cost > budget.
    ...
```

### 3.4 Where it lives

- Template: `templates/targets/fsharp_wam/kernel_bidirectional_ancestor.fs.mustache`
- Kernel detection: extend `detect_kernels/2` in `wam_fsharp_target.pl`
  to recognize when both `category_parent` and `category_child` are
  available and the kernel is `category_ancestor/4`
- Option: `kernel_mode(bidirectional)` or auto-detect when `csr_path`
  is set

### 3.5 Correctness and comparison

Bidirectional search finds a SUPERSET of paths (upward-only paths
plus non-carrot-shaped paths via child hops). So:

- `d_eff_bidirectional <= d_eff_upward_only` (more paths = lower distance)
- When `childStepCost` is very high (effectively infinite), bidirectional
  degenerates to upward-only and results should match exactly
- Test: run both kernels, verify `d_eff_bidir <= d_eff_upward + epsilon`,
  and verify exact match when child hops are disabled

The interesting comparison is: how much does d_eff change when we
allow child hops? This measures how much connectivity the upward-only
kernel misses.

### 3.6 Future: alternative weighting schemes

The current distance metric uses power-law `(hops+1)^(-n)`. Future
work can explore:

- Direction-weighted metric: child hops contribute differently to
  the distance calculation (not just the inclusion threshold)
- Flux-based weighting: see `docs/design/COST_FUNCTION_PHILOSOPHY.md`
- The separation of inclusion threshold from distance metric makes
  it easy to swap metric functions without changing the search logic

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

- Phase A: cost analyzer picks the fastest mode at 3/4 scales;
  all modes produce identical BFS results
- Phase B: bidirectional kernel with childCost=infinity matches
  upward-only exactly; with finite childCost, d_eff is lower
  (finds more paths); path-cost pruning keeps the search space
  manageable (child fan-out doesn't explode)

## 6. References

- CSR reader: `templates/targets/fsharp_wam/csr_reader.fs.mustache`
- CSR philosophy: `docs/design/WAM_FSHARP_CSR_PHILOSOPHY.md`
- Cost analyzer: `docs/design/WAM_FSHARP_COST_ANALYZER_DESIGN.md`
- Existing BFS kernel: `templates/targets/fsharp_wam/kernel_category_ancestor.fs.mustache`
- E2E benchmark: `tests/core/test_wam_fsharp_lmdb_e2e_bench.pl`
- Reverse index artifacts: `docs/design/WAM_REVERSE_INDEX_ARTIFACTS.md`
