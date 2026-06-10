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
    (cat: int) (root: int)
    (parentCost: float) (childCost: float) (budget: float)
    : int list =
    // DFS/BFS with cumulative cost tracking.
    // Parent steps add parentCost, child steps add childCost.
    // Prune when cumulative cost > budget.
    // Returns hop counts for paths that reach root.
    ...
```

**Key design**: two separate concerns:

1. **Inclusion threshold** (path-cost budget): `parentCost=1.0`,
   `childCost=3.0`, `budget=10.0` means at most 10 parent hops
   or 3 child hops or any mix. This constrains child fan-out
   without a hard depth limit.

2. **Distance metric** (unchanged): the existing `(hops+1)^(-n)`
   power-law uses raw hop count. The inclusion threshold determines
   WHICH paths contribute; the metric determines HOW. These are
   specified separately so we can later swap in flux-based or
   direction-weighted metrics independently.

**Correctness**: bidirectional finds a SUPERSET of paths (upward
paths plus non-carrot-shaped paths via child hops). So:
- `d_eff_bidir <= d_eff_upward` (more paths = lower distance)
- When `childCost` is very high (infinity), it degenerates to
  upward-only and results should match exactly
- Test both: exact match with childCost=infinity, and verify
  d_eff decreases with finite childCost

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
- Phase B: bidirectional kernel with childCost=infinity matches
  upward-only exactly; with finite childCost, d_eff is lower
  (finds more paths through non-carrot-shaped routes); path-cost
  pruning keeps search space manageable
