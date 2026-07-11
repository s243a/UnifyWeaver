# Hybrid WAM targets comparison

Compare every UnifyWeaver **hybrid WAM** backend (Prolog → WAM →
target-language VM, optionally plus lowered emitters / foreign
kernels). Companion to [`WAM_TARGET_ROADMAP.md`](WAM_TARGET_ROADMAP.md).

**Not covered here:** non-WAM direct compilers (`haskell_target.pl`,
`rust_target.pl`, `llvm_target.pl`, `fsharp_target.pl`, `go_target.pl`,
…). Those have separate docs (`HASKELL_TARGET.md`, `RUST_TARGET.md`,
…).

## Status documents (read these first)

| Target | Status / primary doc | Notes |
|---|---|---|
| **Haskell** | [`WAM_HASKELL_STATUS.md`](WAM_HASKELL_STATUS.md) | **New** living status; session summary + `design/WAM_HASKELL_*` |
| **Rust** | [`WAM_RUST_STATUS.md`](WAM_RUST_STATUS.md) | **New**; `design/WAM_RUST_*` + reports |
| **LLVM** | [`WAM_LLVM_STATUS.md`](WAM_LLVM_STATUS.md) | **New**; transpilation trilogy |
| **C++** | [`WAM_CPP_STATUS.md`](WAM_CPP_STATUS.md) | **New**; ISO + LMDB design docs |
| **F#** | [`WAM_FSHARP_STATUS.md`](WAM_FSHARP_STATUS.md) | **New** status extract; usage in [`WAM_FSHARP_TARGET.md`](WAM_FSHARP_TARGET.md) + [`design/WAM_FSHARP_PARITY_AUDIT.md`](design/WAM_FSHARP_PARITY_AUDIT.md) |
| **Elixir** | [`design/WAM_ELIXIR_STATUS.md`](design/WAM_ELIXIR_STATUS.md) | Architectural reference baseline |
| **Scala** | [`WAM_SCALA_TARGET.md`](WAM_SCALA_TARGET.md) | Usage guide (no separate STATUS yet) |
| **R** | [`WAM_R_TARGET.md`](WAM_R_TARGET.md) | Usage + [`handoff/wam_r_session_handoff.md`](handoff/wam_r_session_handoff.md) |
| **Go** | [`design/WAM_GO_PARITY_AUDIT.md`](design/WAM_GO_PARITY_AUDIT.md) | Parity audit vs Rust/Haskell |
| **Python** | [`design/WAM_PYTHON_PARITY_AUDIT.md`](design/WAM_PYTHON_PARITY_AUDIT.md) | Parity + partial ISO |
| **Lua** | [`design/WAM_LUA_PARITY_AUDIT.md`](design/WAM_LUA_PARITY_AUDIT.md) | 2026 builtin parity pass |
| **WAT** | [`targets/wam-wat.md`](targets/wam-wat.md) | Overview + architecture |
| **C** | [`WAM_C_TARGET_NEXT_STEPS.md`](../WAM_C_TARGET_NEXT_STEPS.md) | Living checklist (functions as status) |
| **Clojure** | proposals only (`WAM_CLOJURE_*`) | No STATUS yet |
| **JVM / ILAsm / Kotlin** | plans / family overview | Early scaffolds |

Cross-cutting: [`WAM_CROSS_TARGET_CONFORMANCE.md`](WAM_CROSS_TARGET_CONFORMANCE.md),
[`WAM_BACKEND_CONVENTIONS.md`](WAM_BACKEND_CONVENTIONS.md),
[`WAM_RUNTIME_PARSER_STATUS.md`](WAM_RUNTIME_PARSER_STATUS.md),
[`design/WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md`](design/WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md),
[`WAM_PERF_CROSS_TARGET.md`](WAM_PERF_CROSS_TARGET.md).

## Verdict (five “mature” backends)

Pick by **axis**, not a single score:

| Prefer | When |
|---|---|
| **Rust** | Single-core tight graph kernels; u32-interned FFI |
| **Haskell** | Multi-core fanout + LMDB scale + GHC fusion |
| **F#** | .NET deploy; richest LMDB eager/lazy/cached (+ L1/L2) |
| **LLVM** | Portable native binary or WASM from one IR pipeline |
| **C++** | ISO `catch`/`throw` / `is_iso` fidelity |

Elixir remains the **architectural reference** (dual lowering default,
ISO reference, conformance default CI, all 7 kernels) but still lacks
mmap LMDB. Scala anchors **generalization** (classics suite).

## Fleet inventory

Seventeen `wam_*_target.pl` modules. Line counts are approximate
(codegen Prolog only; runtime templates extra). Conformance =
registered in `tests/test_wam_cross_target_conformance.pl`.

| Target | target.pl | lowered | Tests≈ | Conformance | Kernels | LMDB / facts | ISO | Maturity band |
|---|---:|---:|---:|---|---|---|---|---|
| **Elixir** | ~6.8k | ~2.3k | 7 | **default CI** | all 7 | FactSource (no mmap) | **reference** | Reference / primary |
| **Haskell** | ~7.0k | ~1.3k | 14 | opt-in ✓ | 7 + bi + parMap | FactSource eager/cached | substrate | Primary (scale) |
| **Rust** | ~7.1k | ~0.8k | 44 | opt-in ✓ | 7 + matrix + bi + CSR | LookupSource partial | — | Primary (kernels) |
| **F#** | ~5.3k | ~1.6k | 27 | **no** | 7 + bi + CSR | **eager/lazy/cached** | partial | Primary (.NET) |
| **LLVM** | ~20.6k | ~2.1k | 52 | **no** | 7 foreign kinds | arena only | — | Primary (portable) |
| **C++** | ~11.0k | ~0.8k | 6 | opt-in ✓ | foreign surface | arity-2 LMDB v1 | **reference** | Primary (ISO) |
| **Scala** | ~1.4k | ~0.8k | 10 | **default CI** | all 7 opt-in | 4 backends + arity-N LMDB | — | Primary (breadth) |
| **C** | ~6.6k | — | 23 | opt-in ✓ | all 7 | TSV + LMDB | — | Strong systems |
| **Go** | ~3.7k | ~0.8k | 12 | opt-in ✓ | category_ancestor+ | TSV/LMDB atom facts | — | Strong runtime |
| **R** | ~2.0k | ~1.0k | 49 | **no** | all 7 | optional LMDB | tryCatch only | Strong (campaign) |
| **Python** | ~2.8k | ~1.3k | 7 | opt-in ✓ | — | — | partial | Mid (parity) |
| **WAT** | ~6.7k | ~0.5k | 6 | opt-in ✓ | — | — | — | Mid (WASM) |
| **Clojure** | ~0.9k | ~1.5k | 6 | **no** | — | LMDB JNI + caches | — | Mid (LMDB niche) |
| **Lua** | ~0.8k | ~0.6k | 5 | **no** | — | — | — | Mid (builtins) |
| **ILAsm** | ~2.0k | — | 2 | **no** | — | — | — | Early |
| **JVM** | ~0.7k | — | 1 | **no** | — | — | — | Early |
| **Kotlin** | ~0.5k | — | 1 | **no** | — | — | — | Early |

Shared kernel kinds (detector): `transitive_closure2`,
`category_ancestor`, `transitive_distance3`,
`transitive_parent_distance4`, `transitive_step_parent_distance5`,
`weighted_shortest_path3`, `astar_shortest_path4`.

## Tier A — primary hybrid backends

### Haskell / Rust / LLVM / C++ / F#

Deep profiles live in the STATUS docs linked above. Snapshot:

| | Haskell | Rust | LLVM | C++ | F# |
|---|---|---|---|---|---|
| Role | scale + fusion | single-core kernels | native/WASM IR | ISO reference | .NET + LMDB modes |
| Lowered | dual, clause-1 | deterministic only | M1–M4 hybrid | det / clause-1 / ITE | dual (Haskell-like) |
| State | immutable + IntMap | `&mut` Vec | SSA + arena | mutable maps/cells | record + in-place regs |
| Scale-300 query | 32 ms (4c) / 107 ms (1c) | **17 ms** | microbench-class | not on matrix | **11 ms** (startup↑ total) |
| Status doc | ✅ new | ✅ new | ✅ new | ✅ new | ✅ new |

### Elixir — reference baseline

See [`design/WAM_ELIXIR_STATUS.md`](design/WAM_ELIXIR_STATUS.md).
Dual lowering **default**; full indexed dispatch; ISO three-form;
aggregates with witness-group bagof/setof; all 7 kernels; Y-regs
fix ~30–55× on chain bench. Deferred: mmap LMDB, atom interning
(deprioritized), Items API for lowered path.

### Scala — generalization anchor

See [`WAM_SCALA_TARGET.md`](WAM_SCALA_TARGET.md). Classics suite
(n-queens, Ackermann, fib, …); `emit_mode(functions)`; all 7 kernels
via `kernel_dispatch(true)`; four fact backends including arity-N
LMDB; aggressive compile-time atom interning. Default conformance CI
backend alongside Elixir.

## Tier B — strong but narrower

### C (`wam_c_target`)

Living checklist: [`WAM_C_TARGET_NEXT_STEPS.md`](../WAM_C_TARGET_NEXT_STEPS.md).
All 7 kernels, LMDB FactSource, aggregates/bagof/setof meta-goals,
lowered **helpers** prototype (no separate `wam_c_lowered_emitter.pl`).
Roadmap Table 2 historically understated size (~907 lines cited;
codegen is ~6.6k). Conformance green. Useful FFI/glue substrate.

### Go (`wam_go_target`)

[`design/WAM_GO_PARITY_AUDIT.md`](design/WAM_GO_PARITY_AUDIT.md):
broad builtin/IO/aggregate surface; `category_ancestor` FFI (roadmap
cites ~52× at scale-300); TSV/LMDB atom-fact paths; conformance green.
Default Go product path is still non-WAM `go_target.pl` — WAM via
`prefer_wam(true)`.

### R (`wam_r_target`)

[`WAM_R_TARGET.md`](WAM_R_TARGET.md) + session handoff: ~30-PR parity
campaign; **7/7 kernels**; rich builtins; native parser default;
optional LMDB (load-everything). Large generator test count; **not**
in conformance harness yet.

## Tier C — mid maturity

### Python

Parity audit + partial ISO (catch/throw, `is_iso`, compares, succ).
Conformance registered. No graph kernels / LMDB. Packaged
`WamRuntime.py` is the parity surface.

### WAT (WebAssembly text)

[`targets/wam-wat.md`](targets/wam-wat.md): 73-instruction fused
bytecode; hybrid clause-1 lowered + `$run_loop` fallback; conformance
green after cons/`is`/indexing fixes. Browser/sandboxed deploy.
Interpreter-bound (dispatch still hot).

### Clojure

LMDB JNI + cache policies; lowered emitter **deterministic-prefix
only** (no `switch_on_constant` lowering yet); no kernels; no
conformance registration. Docs are proposals, not STATUS.

### Lua

[`design/WAM_LUA_PARITY_AUDIT.md`](design/WAM_LUA_PARITY_AUDIT.md):
focused 2026 builtin/control/aggregate parity; lowered T4–T6; narrow
IO. No kernels/LMDB/conformance.

## Tier D — early scaffolds

| Target | Shape | Missing |
|---|---|---|
| **ILAsm** | .NET CIL from WAM (`switch` dispatch) | lowered emitter, kernels, conformance |
| **JVM** | Jamaica/Krakatau bytecode dual emit | lowered emitter; third JVM route after Scala/Clojure |
| **Kotlin** | hybrid partition + WAM fallback | lowered emitter; tiny test surface |

## Workload → target cheat sheet

| Workload | Prefer |
|---|---|
| Single-process tight graph recursion | **Rust** |
| Multi-core independent subqueries | **Haskell** / **F#** / **Elixir** (fanout) |
| >~100k facts, mmap | **F#** or **Haskell** (also Scala LMDB) |
| Portable native or browser WASM | **LLVM** or **WAT** |
| ISO error fidelity | **C++** then **Elixir** / **F#** / **Python** |
| Classic-program CI / breadth | **Scala** + **Elixir** |
| .NET shop | **F#** (ILAsm only if raw CIL needed) |
| BEAM / OTP | **Elixir** |
| Scripting embed | **Python** / **Lua** / **R** |
| Small C ABI / FFI glue | **C** (then **C++**) |

## Maturity on orthogonal axes

Do **not** collapse into one ranking:

1. **Kernel / graph perf:** Rust ≈ F# ≈ Haskell ≈ Elixir ≈ Scala/R/C > Go > LLVM > C++ > others  
2. **Materialisation:** F# ≥ Haskell ≈ Scala > Rust > C/C++/Clojure > Elixir (no mmap) > LLVM  
3. **ISO / exceptions:** C++ ≈ Elixir > F# ≈ Python > Haskell (substrate) > rest  
4. **Portable codegen:** LLVM > WAT > Rust/C/C++ > Haskell > F#/.NET  
5. **Conformance harness:** Scala + Elixir (default) > haskell/rust/c/cpp/go/python/wat > unregistered (F#, LLVM, R, Clojure, Lua, …)  
6. **Architectural completeness:** Elixir > Haskell/F#/Rust > Scala > Go/R/C > …

## Effective-distance snapshot (scale 300)

From [`design/WAM_CROSS_TARGET_BENCHMARK_RESULTS.md`](design/WAM_CROSS_TARGET_BENCHMARK_RESULTS.md):

| Target | query_ms | total_ms | Notes |
|---|---:|---:|---|
| Rust + FFI + u32 intern | **17** | **32** | Single core |
| F# functions mode | **11** | 159 | Startup dominates total |
| Haskell + FFI (4-core) | 32 | 75 | Parallel |
| Haskell + FFI (1-core) | 107 | 193 | |
| Rust interpreter | 137 | 151 | No FFI |
| SWI-Prolog optimized | 336 | 409 | Reference |

LLVM/C++/Elixir/Scala/Go appear in other benches or roadmap notes;
not all are on this exact matrix.

## Documentation gaps closed by this branch

| Gap | Action |
|---|---|
| No Haskell/Rust/LLVM/C++ STATUS | Added `WAM_*_STATUS.md` |
| F# STATUS scattered across TARGET + PARITY | Added `WAM_FSHARP_STATUS.md` + link from TARGET |
| Comparison covered only five targets | This doc now inventories all 17 |
| Roadmap F# “LMDB none” / Elixir “kernels none” | Corrected in `WAM_TARGET_ROADMAP.md` |
| Roadmap omitted C++ primary row | Added |

Still optional: dedicated `WAM_SCALA_STATUS.md`, `WAM_CLOJURE_STATUS.md`,
`WAM_C_STATUS.md` (C next-steps already serves), conformance adapters
for F# / LLVM / R.

## Document status

Descriptive snapshot for the hybrid comparison branch. Work items
live in PRs and `docs/proposals/`. Prefer updating per-target STATUS
docs when milestones land, then refresh the fleet table here.
