# Hybrid WAM targets comparison: Haskell, Rust, LLVM, C++, F#

Focused comparison of the five hybrid WAM backends that are furthest
along on generalization + optimization. Companion to
[`WAM_TARGET_ROADMAP.md`](WAM_TARGET_ROADMAP.md) (full fleet),
[`WAM_PERF_CROSS_TARGET.md`](WAM_PERF_CROSS_TARGET.md) (perf idioms),
and [`WAM_CROSS_TARGET_CONFORMANCE.md`](WAM_CROSS_TARGET_CONFORMANCE.md)
(classic-program harness).

**Verdict in one line:** pick by workload shape — **Rust** for
single-core tight graph kernels, **Haskell** for parallel fusion +
LMDB scale, **F#** for .NET + richest LMDB materialisation modes,
**LLVM** for portable native/WASM codegen, **C++** for ISO-error /
systems-correctness reference.

## What “hybrid WAM” means here

Prolog → WAM bytecode → target-language VM (instruction array +
`step`/`run`), with an optional **lowered emitter** that turns some
predicates into direct host functions that bypass the dispatch loop.
Graph hot paths may further lower to **foreign kernels** (FFI /
native DFS / BFS / Dijkstra / A*).

All five targets ship:

| Layer | Present? |
|---|---|
| WAM instruction lowering (`wam_*_target.pl`) | yes |
| Per-predicate lowered emitter (`wam_*_lowered_emitter.pl`) | yes |
| Project generation (buildable artifact) | yes |
| Choice points / trail / unification | yes |

They diverge on kernels, materialisation, parallelism, ISO errors,
and host-runtime idioms.

## At-a-glance matrix

Code sizes are approximate line counts of the Prolog codegen modules
(runtime templates and host sources are additional).

| Dimension | **Haskell** | **Rust** | **LLVM** | **C++** | **F#** |
|---|---|---|---|---|---|
| Architectural question | cheap materialisation at scale; GHC fusion | hand-tuned native FFI kernels | portable native IR / WASM | systems substrate + ISO errors | Haskell-shaped WAM on .NET |
| Codegen size (`*_target` + `*_lowered`) | ~8.2k | ~7.9k | ~22.8k | ~11.8k | ~7.0k |
| State / registers | immutable `WamState` + `IntMap` regs | `&mut self`, `Vec` regs | SSA + stack alloca; arena heap | mutable cells / maps | immutable record + in-place `WsRegs` (post `#2428`) |
| Lowered emitter | dual; clause-1 of multi-clause | deterministic / clause-1 | M1–M4 (single + multi-clause hybrid, pattern match, cross-pred closures) | deterministic / clause-1 / ITE | dual; mirrors Haskell |
| Shared graph kernels (7 kinds) | yes | yes | yes (7 foreign kinds incl. countdown/list_suffix) | foreign dispatch surface; not the full shared-kernel set | yes (+ bidirectional ancestor) |
| Extra kernels | `parMap` fork path | effective-distance matrix; **bidirectional_ancestor**; CSR child index | auto `foreign_lowering(true)` | `call_foreign` trampolines | bidirectional + CSR + cost analyzer |
| LMDB / facts | FactSource; eager + cached (lazy degenerate) | LookupSource; eager shipped; lazy/cached planned | arena only — **no LMDB** | arity-2 LMDB FactSource (v1) | **eager + lazy + cached** (+ two-level L1/L2) |
| Atom interning | compile-time `atom_intern_id` | u32 IDs in FFI hot path (~8× on scale-300) | IR constants / native | dynamic / runtime | mostly string/Map; LMDB path uses int IDs |
| Parallelism | `parMap rdeepseq` (gated) | primarily single-core kernels | host-dependent | host threads if wired | TPL `Parallel.map` / negation race |
| Classic conformance harness | yes (opt-in cabal) | yes | **not registered** | yes (conformant on onboarding) | **not registered** |
| ISO errors (`is_iso` / catch-throw) | mostly missing | mostly missing | not a reference adopter | **reference adopter** | substrate + variants shipped |
| Runtime parser | compiled (opt-in; default off) | compiled (opt-in; default off) | not in capability table | **native default** + compiled | compiled (opt-in; default off) |
| Deploy shape | Cabal / GHC binary | Cargo crate | `.ll` → native or WASM | g++ / cmake-style project | `.fsproj` / `dotnet run` |
| Approx. dedicated tests | ~11 | ~44 | ~40 (mostly `tests/core/`) | ~6 focused + generator suite | ~25+ |

Kernel kinds shared across mature targets (from the roadmap detector):
`transitive_closure2`, `category_ancestor`, `transitive_distance3`,
`transitive_parent_distance4`, `transitive_step_parent_distance5`,
`weighted_shortest_path3`, `astar_shortest_path4`.

## Effective-distance perf snapshot (scale 300)

From [`design/WAM_CROSS_TARGET_BENCHMARK_RESULTS.md`](design/WAM_CROSS_TARGET_BENCHMARK_RESULTS.md)
(same Prolog shape; numbers are indicative, not a live CI board):

| Target | query_ms | total_ms | Notes |
|---|---:|---:|---|
| Rust WAM + FFI + u32 interning | **17** | **32** | Beats pruned native DFS on query; single core |
| F# WAM (functions mode) | **11** | 159 | Query competitive; .NET startup dominates total |
| F# LMDB cached (fact access only) | **2** | — | Lookup path, not full WAM query |
| Haskell WAM + FFI (4-core) | 32 | 75 | Parallel `parMap` path |
| Haskell WAM + FFI (1-core) | 107 | 193 | Same kernels, sequential |
| Rust WAM interpreter (no FFI) | 137 | 151 | Baseline hybrid interpreter |
| SWI-Prolog (optimized) | 336 | 409 | Source-language reference |

**LLVM** has foreign-kernel and dispatch microbenches
(`test_wam_llvm_benchmark.pl`, realdata smoke) but is not yet on the
same effective-distance matrix as Rust/Haskell/F#. **C++** has
generator and lowered e2e coverage; graph-kernel matrix presence is
weaker than Rust/Haskell/F#.

## Per-target profiles

### Haskell — scale + fusion

- **Wins when:** large fact sets (LMDB), pure recursive numeric /
  list fusion, multi-core fanout where sparks amortise.
- **Strengths:** mature FactSource story; compile-time atom
  interning; dual lowering; parallel fork/negation paths; broad
  design doc surface (perf, mode analysis, intra-query, Elixir
  backport).
- **Gaps:** ISO-error adoption; conformance builds are heavy
  (cabal); lazy LMDB mode is degenerate (cap 0); parser E2E is
  slow when enabled.
- **Path forward (roadmap):** Elixir-style cost gates; keep LMDB on
  the safe key/value API (raw-pointer path abandoned).

### Rust — single-core kernel king

- **Wins when:** one process, hot DFS/BFS/A* loops, u32-interned
  edges, optional CSR reverse child index.
- **Strengths:** densest kernel surface (shared 7 + matrix +
  bidirectional); largest hybrid-WAM test footprint among the five;
  conformance green; LookupSource over LMDB; atom interning is a
  measured ~8× lever.
- **Gaps:** lowered emitter is deterministic-only; LMDB lazy/cached
  still planned (R7/R8); no full FactSource facade like Haskell;
  ISO errors not adopted; strings elsewhere outside FFI IDs.
- **Path forward:** simplewiki-scale bidirectional vs F#;
  distribution-cache phases; FactSource generalisation.

### LLVM — portable native / WASM

- **Wins when:** you need a **native binary or WASM** from one IR
  pipeline, without tying to Rust/GHC/.NET toolchains.
- **Strengths:** largest codegen module; richest lowered emitter
  milestones (M1–M4 closures); arena allocator with growable
  trail/stack/CP/heap; seven auto-detectable foreign kernels;
  arena-reset cleanup (~18% per-query on dispatch microbench).
- **Gaps:** **no LMDB / FactSource**; not in classic conformance
  harness; not in runtime-parser capability table; real-workload
  matrix thinner than Rust/Haskell/F#; hybrid clause-1 trail
  rollback still called out as follow-up.
- **Path forward:** effective-distance-class benchmarks; LMDB
  fact-source; conformance adapter.

### C++ — correctness / ISO reference

- **Wins when:** ISO `catch`/`throw` and `is_iso`/`is_lax` semantics
  matter; you want a mutable systems runtime that already passed
  classic conformance without onboarding fixes.
- **Strengths:** first ISO-error reference consumer (with Elixir);
  native + compiled runtime parser; arity-2 LMDB FactSource design;
  lowered emitter (deterministic / clause-1 / ITE); audit-clean
  quoted-atom handling.
- **Gaps:** not the graph-kernel performance leader; parser-bundled
  projects compile very slowly (~11 min for ~42k-line
  `generated_program.cpp`); fewer dedicated kernel e2e tests than
  Rust/LLVM/F#.
- **Path forward:** keep ISO contract as shared reference; treat
  compile-time cost of compiled parser as a first-class constraint;
  grow shared-kernel parity if C++ is used for graph benches.

### F# — .NET hybrid with richest LMDB modes

- **Wins when:** .NET deployment; you need **eager/lazy/cached**
  LMDB with two-level cache; query-time competitiveness with Rust
  on graph workloads.
- **Strengths:** mirrors Haskell lowering; TPL parallelism;
  bidirectional ancestor + CSR reader; cost analyzer; ISO substrate
  shipped; runtime parser smoke 42/42; scale-300 query_ms matches
  Rust FFI class (startup dominates total_ms).
- **Gaps:** not in classic conformance harness; roadmap row for
  materialisation is stale (LMDB modes **are** shipped — see
  `WAM_FSHARP_PARITY_AUDIT.md`); dynamic DB is partial vs
  C++/Python/Haskell.
- **Path forward:** register conformance adapter; follow Rust on
  scan/segregation LMDB contract; keep cost-gate calibration aligned
  with Elixir/Haskell lessons.

## Workload → target cheat sheet

| Workload shape | Prefer | Why |
|---|---|---|
| Single-process tight graph recursion | **Rust** | Native + u32 interning + kernel density |
| Multi-core independent subqueries | **Haskell** (or F# TPL) | `parMap` / Parallel.map with fusion or .NET tasks |
| >~100k facts, memory-mapped | **F#** or **Haskell** | F# has full eager/lazy/cached; Haskell has production LMDB |
| Portable native or browser WASM | **LLVM** | One IR → `llc`/clang or WASM |
| ISO error / catch-throw fidelity | **C++** (then F#) | Reference adopter; F# has substrate |
| .NET shop / CLR embedding | **F#** | First-class `.fsproj` + LightningDB |
| Fastest classic-conformance CI among natives | **C++** or **Rust** | Both harness-green; C++ onboarded clean |

## Maturity ranking (orthogonal axes)

Do not collapse these into one score:

1. **Kernel / graph perf maturity:** Rust ≈ F# ≈ Haskell > LLVM > C++
2. **Materialisation maturity:** F# ≥ Haskell > Rust > C++ > LLVM
3. **Semantic / ISO maturity:** C++ > F# > (Haskell, Rust, LLVM)
4. **Portable codegen maturity:** LLVM > Rust > C++ > Haskell > F#
5. **Conformance harness maturity:** Rust = C++ = Haskell > (F#, LLVM unregistered)
6. **Test-surface breadth:** Rust ≥ LLVM > F# > Haskell > C++

Overall “most mature hybrid WAM” depends on the axis. For UnifyWeaver’s
graph-algorithm mission, **Rust / Haskell / F#** are the performance
tier; **C++** is the correctness-contract tier; **LLVM** is the
deployment-portability tier.

## Related docs

- [`WAM_TARGET_ROADMAP.md`](WAM_TARGET_ROADMAP.md) — full target fleet
- [`design/WAM_CROSS_TARGET_BENCHMARK_RESULTS.md`](design/WAM_CROSS_TARGET_BENCHMARK_RESULTS.md) — scale-300 numbers
- [`design/WAM_FSHARP_PARITY_AUDIT.md`](design/WAM_FSHARP_PARITY_AUDIT.md) — F# vs Haskell/Rust baselines
- [`WAM_PERF_CROSS_TARGET.md`](WAM_PERF_CROSS_TARGET.md) — why perf fixes rarely backport
- [`WAM_BACKEND_CONVENTIONS.md`](WAM_BACKEND_CONVENTIONS.md) — cons cells, `//`, placeholders
- [`WAM_RUNTIME_PARSER_STATUS.md`](WAM_RUNTIME_PARSER_STATUS.md) — parser modes
- [`design/WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md`](design/WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md) — ISO adoption

## Document status

Descriptive snapshot of repo state (codegen sizes, harness
registration, and feature tables as of the comparison branch). Not a
binding roadmap — work items live in PRs and `docs/proposals/`.
