# Hybrid WAM targets comparison

Compare every UnifyWeaver **hybrid WAM** backend (Prolog → WAM →
target-language VM, optionally plus lowered emitters / foreign
kernels). Companion to [`WAM_TARGET_ROADMAP.md`](WAM_TARGET_ROADMAP.md).

**Confidence:** High-risk rows (F# kernel templates, Rust LMDB matrix
path, Haskell LMDB emit knobs, LLVM-7 vs shared-7, C 7+bi, Go all-7
FFI, Elixir LMDB + emit-mode default, conformance registration) were
re-checked against SOURCE with Composer explore agents. Perf numbers
are cited from existing bench docs — not re-run in this PR.

**Not covered here:** non-WAM direct compilers (`haskell_target.pl`,
`rust_target.pl`, `llvm_target.pl`, `fsharp_target.pl`, `go_target.pl`,
…). Those have separate docs (`HASKELL_TARGET.md`, `RUST_TARGET.md`,
…).

## Status documents (read these first)

| Target | Status / primary doc | Notes |
|---|---|---|
| **Haskell** | [`WAM_HASKELL_STATUS.md`](WAM_HASKELL_STATUS.md) | Living status; session summary + `design/WAM_HASKELL_*` |
| **Rust** | [`WAM_RUST_STATUS.md`](WAM_RUST_STATUS.md) | Living status; `design/WAM_RUST_*` + reports |
| **LLVM** | [`WAM_LLVM_STATUS.md`](WAM_LLVM_STATUS.md) | Living status; transpilation trilogy |
| **C++** | [`WAM_CPP_STATUS.md`](WAM_CPP_STATUS.md) | Living status; ISO + LMDB design docs |
| **F#** | [`WAM_FSHARP_STATUS.md`](WAM_FSHARP_STATUS.md) | Living status; usage in [`WAM_FSHARP_TARGET.md`](WAM_FSHARP_TARGET.md) + [`design/WAM_FSHARP_PARITY_AUDIT.md`](design/WAM_FSHARP_PARITY_AUDIT.md) |
| **Elixir** | [`design/WAM_ELIXIR_STATUS.md`](design/WAM_ELIXIR_STATUS.md) | Architectural reference baseline |
| **Scala** | [`WAM_SCALA_STATUS.md`](WAM_SCALA_STATUS.md) | Living status; usage in [`WAM_SCALA_TARGET.md`](WAM_SCALA_TARGET.md) |
| **R** | [`WAM_R_STATUS.md`](WAM_R_STATUS.md) | Living status; usage + [`handoff/wam_r_session_handoff.md`](handoff/wam_r_session_handoff.md) |
| **Go** | [`WAM_GO_STATUS.md`](WAM_GO_STATUS.md) | Living status; [`design/WAM_GO_PARITY_AUDIT.md`](design/WAM_GO_PARITY_AUDIT.md) |
| **Python** | [`WAM_PYTHON_STATUS.md`](WAM_PYTHON_STATUS.md) | Living status; [`design/WAM_PYTHON_PARITY_AUDIT.md`](design/WAM_PYTHON_PARITY_AUDIT.md) |
| **Lua** | [`WAM_LUA_STATUS.md`](WAM_LUA_STATUS.md) | Living status; [`design/WAM_LUA_PARITY_AUDIT.md`](design/WAM_LUA_PARITY_AUDIT.md) |
| **WAT** | [`WAM_WAT_STATUS.md`](WAM_WAT_STATUS.md) | Living status; [`targets/wam-wat.md`](targets/wam-wat.md) |
| **C** | [`WAM_C_STATUS.md`](WAM_C_STATUS.md) | Living status; checklist [`WAM_C_TARGET_NEXT_STEPS.md`](../WAM_C_TARGET_NEXT_STEPS.md) |
| **Clojure** | [`WAM_CLOJURE_STATUS.md`](WAM_CLOJURE_STATUS.md) | Living status; proposals `WAM_CLOJURE_*` |
| **ILAsm** | [`WAM_ILASM_STATUS.md`](WAM_ILASM_STATUS.md) | Living status (early scaffold) |
| **JVM** | [`WAM_JVM_STATUS.md`](WAM_JVM_STATUS.md) | Living status (early scaffold) |
| **Kotlin** | [`WAM_KOTLIN_STATUS.md`](WAM_KOTLIN_STATUS.md) | Living status (early scaffold) |

Cross-cutting: [`WAM_CROSS_TARGET_CONFORMANCE.md`](WAM_CROSS_TARGET_CONFORMANCE.md),
[`WAM_BACKEND_CONVENTIONS.md`](WAM_BACKEND_CONVENTIONS.md),
[`WAM_RUNTIME_PARSER_STATUS.md`](WAM_RUNTIME_PARSER_STATUS.md),
[`design/WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md`](design/WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md),
[`WAM_PERF_CROSS_TARGET.md`](WAM_PERF_CROSS_TARGET.md).

## Verdict (five “mature” backends)

Pick by **axis**, not a single score:

| Prefer | When |
|---|---|
| **Rust** | Single-core tight graph kernels; u32-interned FFI; boundary cache |
| **Haskell** | Multi-core fanout + LMDB scale + GHC fusion |
| **F#** | .NET deploy; richest LMDB eager/lazy/cached/auto (+ L1/L2) |
| **LLVM** | Portable native binary or WASM from one IR pipeline |
| **C++** | ISO `catch`/`throw` / `is_iso` fidelity |

Elixir remains the **architectural reference** (ISO reference,
conformance default CI, all 7 kernels, FactSource including LMDB
adaptor) but lacks F#-style lazy/cached two-level policies; codegen
default is still `interpreter` while tests pass `emit_mode(lowered)`.
Scala anchors **generalization** (classics suite). **C** is
underestimated historically — it ships 7+bi kernels + reverse CSR.

## Fleet inventory

Seventeen `wam_*_target.pl` modules. Line counts ≈ codegen Prolog
only. **Tests≈ = test *files*** (not plunit cases — e.g. C++ has ~6
files but ~400 cases in the generator; ILAsm has 2 files / ~45 cases).
Conformance = registered in `tests/test_wam_cross_target_conformance.pl`.

| Target | target.pl | lowered | Tests≈ | Conformance | Kernels | LMDB / facts | ISO | Maturity band |
|---|---:|---:|---:|---|---|---|---|---|
| **Elixir** | ~6.8k | ~2.3k | 7† | **default CI** | all 7 | FactSource incl. LMDB | **reference** | Reference / primary |
| **Haskell** | ~7.0k | ~1.3k | 14 | opt-in ✓ | 7 + bi (templates) | `use_lmdb` + cache_mode tiers | substrate | Primary (scale) |
| **Rust** | ~7.1k | ~0.8k | 44 | opt-in ✓ | 7 + matrix + bi + **boundary** | LookupSource; lazy/cached in matrix path | weak | Primary (kernels) |
| **F#** | ~5.3k | ~1.6k | 27 | **no**‡ | detector all; **templates: 2** (CA + bi) | **eager/lazy/cached/auto** | partial | Primary (.NET) |
| **LLVM** | ~19.2k | ~2.0k | 52 | **no** | **LLVM-7** (≠ shared-7) | arena only | — | Primary (portable) |
| **C++** | ~10.7k | ~0.7k | 6† | opt-in ✓ | foreign surface only | arity-2 LMDB v1 | **reference** | Primary (ISO) |
| **Scala** | ~1.4k | ~0.8k | 11† | **default CI** | all 7 opt-in | 4 backends + arity-N LMDB | — | Primary (breadth) |
| **C** | ~6.6k | — | 9 | opt-in ✓ | **7 + bi** + reverse CSR | TSV + LMDB | — | Strong systems |
| **Go** | ~3.7k | ~0.8k | 13 | opt-in ✓§ | **all 7** FFI | TSV/LMDB atom facts | — | Strong runtime |
| **R** | ~2.0k | ~1.0k | 7† | **no** | all 7 | optional LMDB | tryCatch only | Strong (campaign) |
| **Python** | ~2.8k | ~1.3k | 8† | opt-in ✓ | interpreter graph ops only | — | partial | Mid (parity) |
| **WAT** | ~6.7k | ~0.5k | 6 | opt-in ✓ | — | — | — | Mid (WASM) |
| **Clojure** | ~0.9k | ~1.5k | 5 | **no** | foreign CA handlers | LMDB JNI + caches | — | Mid (LMDB niche) |
| **Lua** | ~0.8k | ~0.6k | 5 | **no** | — | — | — | Mid (builtins) |
| **ILAsm** | ~1.9k | — | 2† | **no** | — | — | — | Early |
| **JVM** | ~0.7k | — | 1 | **no** | — | — | — | Early |
| **Kotlin** | ~0.5k | — | 1† | **no** | — | — | — | Early |

† File count understates cases (Elixir classics 50; C++ generator ~400;
R generator ~94; ILAsm ~45; Kotlin includes Gradle e2e).  
‡ F# has a **dedicated** main-workflow + LMDB oracle job, but is absent
from classic conformance matrix.  
§ Go has all 7 via `go_foreign_lowering` / FFI dispatch (`go_supported_shared_kernel/1`
lists 5; weighted/A* are separate arms). Conformance requires
`prefer_wam(true)` — default Go path is non-WAM `go_target`.

**Shared-7** detector kinds: `transitive_closure2`, `category_ancestor`,
`transitive_distance3`, `transitive_parent_distance4`,
`transitive_step_parent_distance5`, `weighted_shortest_path3`,
`astar_shortest_path4`.

**LLVM-7** differs: drops parent/step-distance + bidirectional; adds
`countdown_sum2`, `list_suffix2`.

## Corrections from parallel source review (2026-07-11)

Four explore subagents (Haskell/Rust, LLVM/C++/C, F#/Elixir/Scala,
remaining nine) re-checked SOURCE against this doc. High-signal fixes
already applied above / in STATUS pages:

1. **Rust lazy/cached LMDB is implemented** (templates + matrix benches);
   STATUS “R7/R8 planned” was stale.
2. **Haskell `lmdb_materialisation` does not drive emit** — real knobs
   are `use_lmdb` + `lmdb_cache_mode`.
3. **F# “7 kernels” overstated** — only CA + bidirectional templates
   exist; bi off by default; `lmdb_materialisation(auto)` shipped.
4. **LLVM-7 ≠ shared-7** — do not equate kernel counts across rows.
5. **C has 7+bi + reverse CSR**; roadmap “907 lines / less-developed”
   was wrong.
6. **Go has all 7 FFI kernels**, not “category_ancestor+”.
7. **Elixir has LMDB FactSource**; gap is rich lazy/cached policies, not
   absence of mmap. Lowered is test/production path, not unresolved
   codegen default.
8. **Clojure** strips `switch_on_constant` prefixes for T4 — it is not
   “no switch handling,” just no emitted switch table.
9. **Test≈ column** is file counts; several suites are much denser.

## Tier A — primary hybrid backends

### Haskell / Rust / LLVM / C++ / F#

Deep profiles: STATUS docs. Snapshot after source review:

| | Haskell | Rust | LLVM | C++ | F# |
|---|---|---|---|---|---|
| Role | scale + fusion | single-core kernels | native/WASM IR | ISO reference | .NET + LMDB modes |
| Lowered | dual, clause-1 + Phase I | det / T4–T6 / ITE | M1–M4 hybrid | det / clause-1 / ITE / T4–T6 | dual (default interpreter) |
| Kernels | shared-7 + bi (mustache) | shared-7 + bi + boundary + matrix | LLVM-7 | foreign trampoline only | templates: CA + bi only |
| LMDB | use_lmdb + cache_mode | LookupSource; matrix eager/lazy/cached | none | arity-2 v1 | eager/lazy/cached/**auto** |
| Scale-300 query | 32 ms (4c) / 107 ms (1c) | **17 ms** | microbench-class | not on matrix | **11 ms** (startup↑ total) |

### Elixir — reference baseline

See [`design/WAM_ELIXIR_STATUS.md`](design/WAM_ELIXIR_STATUS.md).
Tests use `emit_mode(lowered)`; unresolved codegen default remains
`interpreter`. Full indexed dispatch; ISO three-form; aggregates with
witness-group bagof/setof; all 7 kernels; FactSource including
**LMDB** / LmdbIntIds; Y-regs fix ~30–55× on chain bench. Deferred:
rich LMDB policies, Items API for lowered path, IEEE-754 lax divide.

### Scala — generalization anchor

See [`WAM_SCALA_TARGET.md`](WAM_SCALA_TARGET.md). Classics suite
(n-queens, Ackermann, fib, …); `emit_mode(functions)`; all 7 kernels
via `kernel_dispatch(true)`; four fact backends including arity-N
LMDB; aggressive compile-time atom interning. Default conformance CI
backend alongside Elixir.

## Tier B — strong but narrower

### C (`wam_c_target`)

Living checklist: [`WAM_C_TARGET_NEXT_STEPS.md`](../WAM_C_TARGET_NEXT_STEPS.md).
**All 7 shared kernels + `bidirectional_ancestor`**, reverse-CSR
child-index paths, LMDB FactSource, aggregates/bagof/setof meta-goals,
lowered **helpers** prototype (no separate `wam_c_lowered_emitter.pl`).
Conformance green. Useful FFI/glue substrate; historically undercounted
as “907 lines.”

### Go (`wam_go_target`)

[`design/WAM_GO_PARITY_AUDIT.md`](design/WAM_GO_PARITY_AUDIT.md):
**all 7 FFI kernels** via `go_foreign_lowering` / shared detector;
broad builtin/IO/aggregate surface; TSV/LMDB atom-fact paths;
conformance green **with `prefer_wam(true)`**. Default Go product path
is still non-WAM `go_target.pl`.

### R (`wam_r_target`)

[`WAM_R_TARGET.md`](WAM_R_TARGET.md) + session handoff: ~30-PR parity
campaign; **7/7 kernels**; rich builtins; **native parser default**;
optional LMDB (load-everything). Generator suite ~94 plunit cases;
classic conformance **opt-in** (`CONF-R`): all classic programs green
after R-SWITCH-INDEX-CONFORMANCE (fallthrough/A2 → existing handlers).

## Tier C — mid maturity

### Python

Parity audit + partial ISO (catch/throw, `is_iso`, compares, succ).
Conformance registered. No **FFI** graph-kernel set; still has
interpreter-level indexed-fact / `base_category_ancestor*` ops.
Packaged `WamRuntime.py` (~3.6k) is the parity surface.

### WAT (WebAssembly text)

[`targets/wam-wat.md`](targets/wam-wat.md): ~6.4k-line fused bytecode
VM; T4–T6 hybrid + `$run_loop` fallback; conformance green. Browser /
sandboxed deploy. Interpreter-bound (dispatch still hot).

### Clojure

LMDB JNI + cache policies; lowered emitter larger than target.pl;
T4 with **`switch_on_constant` prefix stripping** (no emitted switch
table). Foreign handlers for category_parent/ancestor — not shared-7
FFI. No conformance registration.

### Lua

[`design/WAM_LUA_PARITY_AUDIT.md`](design/WAM_LUA_PARITY_AUDIT.md):
focused 2026 builtin/control/aggregate parity; lowered T4–T6; narrow
IO. No kernels/LMDB/conformance.

## Tier D — early scaffolds

| Target | Shape | Missing |
|---|---|---|
| **ILAsm** | .NET CIL from WAM (`switch` dispatch) | lowered emitter, kernels, conformance; **~45** plunit cases in 2 files |
| **JVM** | Jamaica/Krakatau bytecode dual emit | lowered emitter; third JVM route after Scala/Clojure |
| **Kotlin** | hybrid partition + WAM fallback | lowered emitter; 9 plunit incl. **Gradle e2e** when available |

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

1. **Kernel / graph perf:** Rust ≈ Haskell ≈ Elixir ≈ Scala/R/C/Go > F# (templates thin) > LLVM > C++ > others  
2. **Materialisation:** F# ≥ Haskell ≈ Scala > Rust (matrix path) > C/C++/Clojure/Elixir > LLVM  
3. **ISO / exceptions:** C++ ≈ Elixir > F# ≈ Python > Haskell (substrate) > rest  
4. **Portable codegen:** LLVM > WAT > Rust/C/C++ > Haskell > F#/.NET  
5. **Conformance harness:** Scala + Elixir (default) > haskell/rust/c/cpp/go/python/wat > unregistered (F#, LLVM, R, Clojure, Lua, …)  
6. **Architectural completeness:** Elixir > Haskell/Rust > F#/Scala > Go/R/C > …

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

## Future-work scope (whole fleet)

Aggregated from each target's STATUS "Path forward". Grouped by the
lever that moves the most targets at once; per-target detail lives in
the linked STATUS docs. **Atomic, hand-off-ready task cards** (files to
touch, reference to copy from, steps, acceptance) for every cell below
are in [`WAM_FLEET_GAP_TASKS.md`](WAM_FLEET_GAP_TASKS.md).

| Lever | Targets it unblocks | Notes |
|---|---|---|
| **Conformance adapters** | F#, LLVM, R, Clojure, Lua | F#/LLVM/R are otherwise mature; registering them closes the biggest correctness-visibility gap. |
| **Two-level lazy/cached LMDB policies** | C, Go, Scala, R (Elixir too) | FactSource exists; the F#/Haskell policy tier is the missing piece for >~100k facts. |
| **ISO three-form error contract** | C, Go, Scala, R (finish Python, F#) | C++/Elixir are the references to copy. |
| **Finish foreign kernel templates** | F# (2 of 7 templated) | Detector fires for all kinds; only CA + bidirectional are emitted. |
| **Lowered emitters** | ILAsm, JVM, Kotlin | Early scaffolds — decide invest-vs-retire given Scala/Clojure/F# cover .NET+JVM maturely. |
| **Effective-distance bench rows** | LLVM, C++, C, Go, R | Perf-matrix coverage thinner than Rust/Haskell/F#. |
| **Runtime-parser capability entries** | C, Go, Scala, Clojure, Lua | Only R/C++ are native-default; Python/F#/Haskell/Rust have compiled opt-in. |

Maturity bands for prioritisation: **Primary** (Elixir, Haskell, Rust,
F#, LLVM, C++, Scala) → polish/parity; **Strong** (C, Go, R) →
conformance + LMDB tiers; **Mid** (Python, WAT, Clojure, Lua) →
targeted gaps; **Early** (ILAsm, JVM, Kotlin) → invest-or-retire.

## Documentation gaps closed by this branch

| Gap | Action |
|---|---|
| No Haskell/Rust/LLVM/C++ STATUS | Added `WAM_*_STATUS.md` |
| F# STATUS scattered across TARGET + PARITY | Added `WAM_FSHARP_STATUS.md` + link from TARGET |
| Comparison covered only five targets | Fleet inventory of all 17 |
| **No STATUS for the other 12 hybrid targets** | Added `WAM_{C,GO,SCALA,R,PYTHON,WAT,CLOJURE,LUA,ILASM,JVM,KOTLIN}_STATUS.md` — every hybrid WAM target now has a living STATUS doc |
| Stale / wrong claims vs source | Parallel subagent review corrections (§ above) |
| Roadmap F# LMDB / Elixir kernels / C “907 lines” | Corrected in `WAM_TARGET_ROADMAP.md` |

Every `wam_*_target.pl` module now has a dedicated `WAM_*_STATUS.md`.
Remaining scope-out work (tracked per-target under each STATUS "Path
forward"): conformance adapters for **F# / LLVM / R / Clojure / Lua**;
finish **F# kernel templates**; two-level lazy/cached LMDB policies for
**C / Go / Scala / R**; lowered emitters for **ILAsm / JVM / Kotlin**;
ISO three-form adoption for **C / Go / Scala / R** and completion for
**Python / F#**.

## Document status

Descriptive snapshot after parallel source exploration (Grok 4.5 +
Composer 2.5 explore agents). Prefer updating per-target STATUS docs
when milestones land, then refresh the fleet table here.
