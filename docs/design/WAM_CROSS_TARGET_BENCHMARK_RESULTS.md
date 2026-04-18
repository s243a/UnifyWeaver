# WAM Cross-Target Benchmark Results

Benchmark results comparing WAM execution across multiple compilation targets.
All measurements at **scale 300** (6008 category_parent facts, 770 article_category facts).

## Test Environment

- **Date**: 2026-04-18
- **Platform**: Linux 6.1.158 (amd64)
- **SWI-Prolog**: 9.2.9
- **Go**: 1.24.2
- **Rust**: 1.95.0 (release build, `--release`)
- **Python**: 3.12.8 (CPython)
- **Haskell/GHC**: 9.6 (from prior measurements)
- **Root category**: `Physics`
- **Repetitions**: 3 per query, median reported

## Effective Distance (WAM-compiled predicate)

The benchmark predicate is `category_ancestor$effective_distance_sum_bound/3`,
which computes effective distance via `category_ancestor/4` with path pruning
and `begin_aggregate`/`end_aggregate` for sum accumulation.

| Target | Median (ms) | Status | Notes |
|--------|-------------|--------|-------|
| SWI-Prolog (seeded) | **428** | Valid | Full WAM interpretation + unification |
| SWI-Prolog (accumulated) | **517** | Valid | Seeded accumulation variant |
| Haskell + FFI (1-core) | **107** | Valid | From WAM_PERF_OPTIMIZATION_LOG |
| Haskell + FFI (4-core) | **75** | Valid | Parallel, 3.3x speedup |
| Go WAM bytecode | N/A | **Runtime incomplete** | Y-register clobbering (see below) |
| Rust WAM lowered | N/A | **Runtime incomplete** | Y-register clobbering + missing dispatch |
| Python WAM | N/A | **Runtime incomplete** | Aggregate instructions not implemented |

## Shortest Path (Transitive Closure)

| Target | Median (ms) | Status | Notes |
|--------|-------------|--------|-------|
| SWI-Prolog (branch_pruning=auto) | **3273** | Valid | BFS with branch pruning |
| Haskell + FFI (4-core) | **70** | Valid | From WAM_PERF_OPTIMIZATION_LOG |

## Weighted Shortest Path (Dijkstra, variant=min)

| Target | Median (ms) | Status | Notes |
|--------|-------------|--------|-------|
| SWI-Prolog | **124** | Valid | Dijkstra with semantic weights |
| Haskell + FFI (4-core) | **90** | Valid | From WAM_PERF_OPTIMIZATION_LOG |

## Speedup Analysis (Scale 300, Effective Distance)

| Comparison | Speedup |
|-----------|---------|
| Haskell FFI (4-core) vs SWI-Prolog (seeded) | ~5.7x |
| Haskell FFI (1-core) vs SWI-Prolog (seeded) | ~4.0x |

## WAM Runtime Failure Analysis

### Root Cause: Y-Register Clobbering (Go, Rust, Python)

All three transpiled WAM runtimes share the same fundamental defect:
**Y-registers (permanent variables) are stored in a flat global register array
instead of per-environment stack frames.**

In a correct WAM implementation, Y-registers are allocated on the control stack
within environment frames. Each `allocate`/`deallocate` pair creates a new frame
with its own Y1, Y2, etc. This means a callee's Y-registers are independent of
the caller's.

The transpiler emits Y-registers as global offsets: `Y1 → reg[200]`, `Y2 → reg[201]`,
etc. When `category_ancestor/4` (callee) stores the visited-list in `Y2` (reg 201),
it overwrites `power_sum_bound/3` (caller)'s `Y2` (also reg 201), which the caller
later uses as the target variable for `is/2` (the power computation).

**Observed failure in Go bytecode path:**
```
PC=9:  GetVariable {201 3}  — stores A4=[Energy,[]] into reg 201 (callee's Y2)
...
PC=81: PutValue {201 0}     — reads reg 201 expecting unbound var for is/2 result
PC=85: BuiltinCall is/2     — FAILS: arg1 is List [Energy,[]], not Unbound
```

This is a transpiler code-generation defect, not a runtime bug. The fix requires
either:
1. Making Y-registers per-environment-frame (significant refactor of all runtimes), or
2. Having the transpiler emit unique Y-register offsets per predicate (e.g., callee
   uses Y100+, caller uses Y200+), or
3. Using the WAM stack properly: `allocate` saves Y-regs, `deallocate` restores.

### Go WAM Bytecode: Additional Details

- **Fact table lookup**: Successfully implemented via `callIndexedAtomFact2` with
  first-argument indexing (`IndexedAtomFactMap`) and choice-point backtracking.
- **`is/2` arithmetic**: Works correctly for standalone expressions.
- **`begin_aggregate`/`end_aggregate`**: Implemented in runtime, correctly clones
  sub-VM and iterates solutions. However, the cloned sub-VM inherits the clobbered
  register state.
- **Blocking issue**: Y-register clobbering causes the aggregate body to fail after
  the first `category_ancestor/4` call. The aggregate collects 0 solutions.

### Rust WAM Lowered: Additional Details

- **Missing predicate dispatch**: Lowered functions call `vm.labels.get("category_parent/2")`
  and `vm.labels.get("category_ancestor/4")`, but fact tables aren't in the label map
  and lowered predicates aren't registered as labels. Needs a dispatch mechanism that
  tries: labels → lowered functions → indexed fact tables.
- **Y-register clobbering**: Same issue as Go — the lowered code emits
  `vm.put_reg("Y2", ...)` which maps to global reg 201.
- **Fact table registration**: `register_indexed_atom_fact2_pairs` exists and works.

### Python WAM: Additional Details

- **Aggregate instructions**: `begin_aggregate`/`end_aggregate` are not implemented
  in the `run_wam()` interpreter loop (marked `# SKIP` in generated code).
- **No `state.run()` method**: The runtime uses `run_wam(code, labels, entry, state)`
  as a free function, but generated predicates call `state.run()`.
- **Y-register clobbering**: Same fundamental issue.

## Errata: Previous "Native DFS" Results Were Invalid

The previous version of this document reported:
- Rust WAM: 3ms effective distance (143x faster than SWI-Prolog)
- Python WAM: 2.3ms effective distance (186x faster than SWI-Prolog)

**These results were invalid.** The benchmark drivers used hand-written native
DFS (`collect_native_transitive_distance_results`) that bypassed the WAM entirely.
They measured pure in-memory hash-map graph traversal, not WAM predicate execution.
The results have been removed.

## Larger Scale Reference Numbers (SWI-Prolog)

| Scale | Effective Distance Seeded (ms) | Shortest Path (ms) | Weighted SP (ms) |
|-------|-------------------------------|--------------------|--------------------|
| 300 | 428 | 3273 | 124 |
| 1k | 337 | 2053 | 124 |
| 5k | 1304 | 10676 | 311 |
| 10k | 3013 | 21200 | 596 |

## Previously Recorded Haskell Numbers (from WAM_PERF_OPTIMIZATION_LOG.md)

| Configuration | Scale 300 (ms) | Scale 5k (ms) | Scale 10k (ms) |
|--------------|---------------|--------------|----------------|
| Haskell + FFI 1-core | 107 (total), 32 (query) | - | - |
| Haskell + FFI 4-core | 75 (total) | 213 | 604 |
| Shortest path BFS (Haskell 4-core) | 70 | - | 420 |
| Weighted SP Dijkstra (Haskell 4-core) | 90 | - | 441 |

## Bug Fixes Applied

1. **Go emitter: SharedWamCode/SharedWamLabels undefined** (wam_go_target.pl)
   - `compile_predicates_for_project` emitted `sharedWamCode`/`sharedWamLabels` (unexported)
   - Added exported aliases: `var SharedWamCode = sharedWamCode` / `var SharedWamLabels = sharedWamLabels`

2. **Rust lowered emitter: backslash escape in string literals** (wam_rust_lowered_emitter.pl)
   - `builtin_call \+/1` was emitted as `"\+/1"` (invalid escape in Rust)
   - Added `escape_rust_string/2` import and call in `emit_one(builtin_call(...), ...)`

3. **Go lowered emitter: same backslash escape issue** (wam_go_lowered_emitter.pl)
   - Added `escape_go_string/2` import and call in `emit_one(builtin_call(...), ...)`

4. **Python target: module-qualified predicates** (wam_python_target.pl)
   - `compile_one_predicate/3` didn't handle `Module:Pred/Arity` format
   - Added clause to strip module prefix before delegation

5. **Go WAM runtime: callIndexedAtomFact2 for foreign fact lookup** (runtime.go)
   - Added inline fact table dispatch in Call/Execute handlers
   - First-argument indexed HashMap (`IndexedAtomFactMap`) for O(1) lookup
   - Choice point backtracking for multi-match results

## Recommended Next Steps

1. **Fix Y-register allocation in the transpiler** — Either emit per-predicate
   Y-register ranges (cheapest fix) or implement proper environment frame Y-register
   storage in all three runtimes.

2. **Add predicate dispatch in Rust lowered path** — Implement a `call_predicate(name)`
   method that tries: label-based bytecode → lowered function table → indexed fact
   table. Currently lowered functions can only call bytecode labels.

3. **Implement aggregate instructions in Python runtime** — The `begin_aggregate`/
   `end_aggregate` pair needs to clone the VM state, iterate all solutions via
   backtracking, and accumulate results.
