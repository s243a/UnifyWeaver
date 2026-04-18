# WAM Cross-Target Benchmark Results

Benchmark results comparing WAM execution across multiple compilation targets.
All measurements at **scale 300** (6004 category_parent facts, 757 article_category facts).

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

## Effective Distance (DFS with cycle detection)

| Target | Rep 1 (ms) | Rep 2 (ms) | Rep 3 (ms) | Median (ms) | Notes |
|--------|-----------|-----------|-----------|-------------|-------|
| SWI-Prolog (seeded) | 428 | 429 | 428 | **428** | Full WAM interpretation + unification |
| SWI-Prolog (accumulated) | 405 | 517 | 530 | **517** | Seeded accumulation variant |
| Haskell + FFI (1-core) | - | - | - | **107** | From WAM_PERF_OPTIMIZATION_LOG |
| Haskell + FFI (4-core) | - | - | - | **75** | Parallel, 3.3x speedup |
| Rust WAM (native DFS) | 3 | 3 | 4 | **3** | Native hash-indexed DFS |
| Python WAM (native DFS) | 2.6 | 2.3 | 2.3 | **2.3** | Native dict-indexed DFS |
| Go WAM | - | - | - | **N/A** | Build OK, fact loading not wired up |

## Shortest Path (Transitive Closure)

| Target | Rep 1 (ms) | Rep 2 (ms) | Rep 3 (ms) | Median (ms) | Notes |
|--------|-----------|-----------|-----------|-------------|-------|
| SWI-Prolog (branch_pruning=auto) | 3273 | 3232 | 3332 | **3273** | BFS with branch pruning |
| Haskell + FFI (4-core) | - | - | - | **70** | From WAM_PERF_OPTIMIZATION_LOG |
| Rust WAM (native DFS) | <1 | <1 | <1 | **<1** | 42 reachable nodes |
| Python WAM (native DFS) | <1 | <1 | <1 | **<1** | 42 reachable nodes |

## Weighted Shortest Path (Dijkstra, variant=min)

| Target | Rep 1 (ms) | Rep 2 (ms) | Rep 3 (ms) | Median (ms) | Notes |
|--------|-----------|-----------|-----------|-------------|-------|
| SWI-Prolog | 124 | 124 | 125 | **124** | Dijkstra with semantic weights |
| Haskell + FFI (4-core) | - | - | - | **90** | From WAM_PERF_OPTIMIZATION_LOG |

## Speedup Analysis (Scale 300, Effective Distance)

| Comparison | Speedup |
|-----------|---------|
| Rust native vs SWI-Prolog (seeded) | ~143x |
| Python native vs SWI-Prolog (seeded) | ~186x |
| Haskell FFI (4-core) vs SWI-Prolog | ~5.7x |
| Rust native vs Haskell FFI (4-core) | ~25x |

## Architecture Notes

### Why Native DFS Is So Fast

The Rust and Python benchmarks use **native hash-indexed DFS** — category_parent
facts are pre-indexed into a `HashMap<String, Vec<String>>`, and the DFS walks
the graph directly. This bypasses:
- WAM instruction decode/dispatch
- Unification and trail management
- Choice point save/restore
- Register-based argument passing

The SWI-Prolog and Haskell numbers include the full WAM overhead, making them
more representative of general Prolog workload performance but slower for this
specific graph traversal pattern.

### Go Status

The Go WAM target builds successfully (after fixing `SharedWamCode`/`SharedWamLabels`
export aliases), but the generated `main.go` template doesn't wire up fact data
(category_parent/2) into the runtime. The lowered predicates reference `category_parent/2`
through WAM label lookup, but the facts are not compiled into WAM instructions.
This requires either:
- Compiling all 6004 facts as WAM try_me_else chains, or
- Implementing a foreign fact lookup similar to Rust's `indexed_atom_fact2`

### Larger Scale Reference Numbers (SWI-Prolog)

| Scale | Effective Distance Seeded (ms) | Shortest Path (ms) | Weighted SP (ms) |
|-------|-------------------------------|--------------------|--------------------|
| 300 | 428 | 3273 | 124 |
| 1k | 337 | 2053 | 124 |
| 5k | 1304 | 10676 | 311 |
| 10k | 3013 | 21200 | 596 |

### Previously Recorded Haskell Numbers (from WAM_PERF_OPTIMIZATION_LOG.md)

| Configuration | Scale 300 (ms) | Scale 5k (ms) | Scale 10k (ms) |
|--------------|---------------|--------------|----------------|
| Haskell + FFI 1-core | 107 (total), 32 (query) | - | - |
| Haskell + FFI 4-core | 75 (total) | 213 | 604 |
| Rust native DFS (hand-fused, 1-core) | 126 | - | - |
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
