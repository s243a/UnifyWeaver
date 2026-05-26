# Prompt: CSR Benchmark + Bidirectional Search Kernel

## Context

The F# WAM target now has:
- CSR reader (`CsrLookupSource`) with sorted_array and lmdb_offset backends
- Dual-CSR builder (`build_csr_artifact.py`) for both directions
- Cost analyzer (`resolve_edge_store/2`) that auto-selects LMDB vs CSR
- Auto-wiring into `WcLookupSources` via `csr_path` and `csr_parent_path`
- Program.fs mustache template with `{{match materialisation}}` dispatch

**What's missing**: nobody has run the cost analyzer on a real workload
to verify it picks the right mode, and nobody has used the reverse CSR
(parent -> children) for anything algorithmic yet.

The CSR reverse lookup isn't just a data store optimization — it's the
enabler for **bidirectional effective-distance search** (meet in the
middle), which can reduce search space exponentially on deep graphs.

## Task: Two phases

### Phase A: Benchmark the cost analyzer (do this first)

Create `tests/core/test_wam_fsharp_cost_analyzer_bench.pl` that:

1. Generates synthetic LMDB fixtures at multiple scales (200, 2k,
   6k, 20k edges) using `generate_synthetic_phase1_lmdb.py`
2. Builds CSR artifacts for each scale using `build_csr_artifact.py`
3. For each scale, generates F# projects with different `edge_store`
   modes: `lmdb_cached`, `lmdb_eager`, `csr`, and `auto`
4. Runs each project with a BFS kernel (reuse the BFS from
   `test_wam_fsharp_lmdb_e2e_bench.pl` — it works with any
   `ILookupSource`)
5. Compares: correctness (same BFS counts), timing, and whether
   `auto` picked the fastest mode

The BFS kernel already uses `resolveFactLookup "category_parent" ctx`
which transparently dispatches to whatever is in `WcLookupSources`.
So different stores are tested without kernel changes.

**Key files to reference:**
- `tests/core/test_wam_fsharp_lmdb_e2e_bench.pl` — existing BFS benchmark pattern
- `tests/core/test_wam_fsharp_csr_bench.pl` — CSR vs LMDB comparison pattern
- `examples/benchmark/generate_synthetic_phase1_lmdb.py` — fixture generator
- `examples/benchmark/build_csr_artifact.py` — CSR builder

### Phase B: Bidirectional search kernel (after Phase A works)

Create `templates/targets/fsharp_wam/kernel_bidirectional_ancestor.fs.mustache`:

```fsharp
let bidirectionalAncestor
    (lookupParents: int -> int list)
    (lookupChildren: int -> int list)
    (cat: int) (root: int) (maxDepth: int) : int list =
    // Phase 1: BFS down from root (using category_child CSR)
    let rootSet = bfsDown lookupChildren root (maxDepth / 2)
    // Phase 2: BFS up from cat, stopping when we hit rootSet
    bfsUpToSet lookupParents cat rootSet (maxDepth - maxDepth / 2)
```

This is the **meet-in-the-middle** optimization: instead of exploring
O(b^D) nodes upward, explore O(2 * b^(D/2)) from each end. For
branching factor 3 and depth 10: 59049 vs 486 nodes.

**Critical**: the bidirectional kernel must produce the SAME
effective-distance values as the upward-only kernel. Test by running
both on the same fixture and comparing `sum ((hops+1)^(-n))`.

The kernel plugs into the existing FFI dispatch via
`executeForeign` / kernel detection.

See `docs/design/WAM_FSHARP_CSR_KERNEL_INTEGRATION.md` for the
full design including correctness constraints and implementation
approach.

## What NOT to do

- Don't work on the Haskell mutable registers (that's a separate
  track, already 52/55 arms converted in this session)
- Don't modify the cost analyzer logic (already implemented and
  tested in PR #2493)
- Don't modify the template system (stable, tested)

## Success criteria

- Phase A: benchmark report showing cost analyzer picks fastest
  mode at 3+ scales; all modes produce identical BFS results
- Phase B: bidirectional kernel produces identical effective-distance
  with measurable speedup (>2x) on a deep synthetic graph
