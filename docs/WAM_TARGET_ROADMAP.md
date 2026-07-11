# WAM target roadmap: where we are, where we're going

## Mission

For a focused comparison of hybrid WAM backends (all 17 modules, with
deep coverage of Haskell / Rust / LLVM / C++ / F#), see
[`WAM_HYBRID_TARGETS_COMPARISON.md`](WAM_HYBRID_TARGETS_COMPARISON.md).
Per-target living status for the mature set:
[`WAM_HASKELL_STATUS.md`](WAM_HASKELL_STATUS.md),
[`WAM_RUST_STATUS.md`](WAM_RUST_STATUS.md),
[`WAM_LLVM_STATUS.md`](WAM_LLVM_STATUS.md),
[`WAM_CPP_STATUS.md`](WAM_CPP_STATUS.md),
[`WAM_FSHARP_STATUS.md`](WAM_FSHARP_STATUS.md)
(Elixir: [`design/WAM_ELIXIR_STATUS.md`](design/WAM_ELIXIR_STATUS.md)).

Two intertwined goals for the graph-algorithm pipeline:

- **Generalize** — how much Prolog can the transpiler accept, and how
  faithfully does each backend execute it? Validated by the breadth
  of the test set per target — at the high end, Scala's classic
  programs (n-queens, Ackermann, Fibonacci) set the upper bound on
  problem-shape coverage.
- **Optimize** — how fast does the resulting code run? Validated by
  cross-target graph-algorithm benchmarks (`benchmark_effective_distance_*.py`,
  `benchmark_category_influence_cross_target.py`,
  `benchmark_dependency_depth_cross_target.py`, etc.).

Both goals are pursued simultaneously: a backend that runs Ackermann
fast but can't transpile a typical recursive predicate isn't
interesting; a backend that can transpile everything but runs orders
of magnitude slower than alternatives also isn't.

## Origin: C# as the optimization-inspiration

C# was the **first target where we landed significant performance
optimizations**, drawing from a SQL/LINQ perspective. The connective
tissue is the algebraic identity:

- Prolog `,` (and) ≈ SQL `JOIN`
- Prolog `;` (or) ≈ SQL `UNION`
- Branch trimming (skipping clauses that can't contribute to the
  answer) is the Prolog analogue of SQL query planning.

The strategy was: re-express those LINQ/SQL-style optimisations as
**branch-pruning rewrites in the source Prolog itself**, and rely on
the transpiler chain (Prolog → WAM bytecode → target language) to
distribute the wins to every backend. Compiler-level optimisations
generalise; query-engine-level optimisations don't.

## What we discovered

Three layers of perf headroom turned up, each with a different
character:

1. **WAM-instruction lowering alone is not enough.** A backend that
   only translates WAM bytecode into a target-language `Array[Instruction]`
   and runs it through a step-loop interpreter sits in roughly the same
   perf class as a tuned bytecode VM. Generalisation is great; speedup
   over "just run the source Prolog" is modest.

2. **Per-predicate native fast-path emitters** (the `wam_*_lowered_emitter.pl`
   pattern) close part of the gap by emitting some predicates as direct
   target-language functions that bypass the dispatch loop entirely.
   Useful, but bounded.

3. **Kernel-based lowering** — dispatching specific hot graph
   operations to hand-tuned native FFI kernels — produces the
   **dramatic wins**. Existing data points in the repo:
   - `feat(wam-go): category_ancestor FFI kernel — 52x speedup at scale-300`
   - `bench(wam-rust): add effective-distance matrix ffi targets`
   - `feat(wam-haskell): parallel parMap on dupsort path`

   These aren't general-purpose optimisations; they identify a
   specific algorithm shape (transitive closure, effective-distance,
   ancestor lookup) and replace its WAM execution with a kernel
   tuned for the target's runtime.

4. **Materialization cost** is a separate, dominant bottleneck. For
   workloads above ~100k facts, the cost of re-deriving relations on
   every query overwhelms the per-query optimisations. Two production
   responses:
   - **C#**: cost measurement (`benchmark_csharp_query_source_mode_sweep.py`)
     plus memory-mapped file backing for materialised relations.
   - **Haskell**: LMDB (memory-mapped under the hood). The raw-pointer
     interface caused crashes; we moved to the safe key/value API.
     Less zero-copy, more reliable.

## Where Elixir is (the reference baseline)

Most architecturally complete of the WAM targets:

- Dual-mode lowering: WAM-instruction lowering + `wam_elixir_lowered_emitter.pl`
  per-predicate native emitter.
- Comprehensive builtins (PRs #1777 #1780 #1782 #1784 — comparisons,
  unification, list ops, output, term meta-programming).
- Inline-list-build fix (PR #1781).
- Two cost gates for Tier-2 parallelism: `forkMinCost` (static,
  PR #1783) and `runtime_cost_probe` (sticky-decision, PR #1785),
  calibrated against the benchmark grid (PR #1786 — both gates'
  defaults derived from the speedup=1.0 crossover).
- FactSource facade for externalisation: ETS, SQLite, TSV, fact_table.

Gaps relative to the lessons from other targets:

- **LMDB FactSource exists** (`FactSource.Lmdb` / `LmdbIntIds` in
  generated runtime) but lacks F#/Haskell-style
  eager/lazy/cached two-level `ILookupSource` policies; validate
  under real `:elmdb` installs.
- **Kernels: all 7 shared kinds shipped** (PRs #1799–#1826) — the
  older “no kernel-based lowering” gap list below this section was
  stale and is retained only as historical context in older commits.
- **Atom interning** is opt-in (`intern_atoms(true)`) and
  deprioritized for perf (string compares <1% post Y-regs fix).

## Table 1 — primary targets (Elixir-relevant lessons)

The set Elixir can most directly learn from. Each target is anchored
on a specific architectural question.

| Target | Architectural question | Generalization | Lowering | Materialization | Kernels | Path forward |
|---|---|---|---|---|---|---|
| **Elixir** | reference baseline | Phase 3/4 + comprehensive builtins | dual: WAM-instr + per-predicate emitter (tests use `emit_mode(lowered)`; codegen default still `interpreter`) | FactSource facade (ETS/SQLite/TSV/**LMDB**/LmdbIntIds); lacks F#-style lazy/cached two-level policies | all 7 shared kernel kinds (PRs #1799–#1826) | deepen LMDB policies; emitter-driven Tier-2 fanout |
| **Haskell** | how to make materialisation cheap at scale | broad WAM, parMap parallel | dual: WAM-instr + emitter | LMDB via `use_lmdb` + `lmdb_cache_mode` tiers (safe KV API); `lmdb_materialisation` option currently unused by emit | all 7 + bidirectional (mustache templates) + parMap | wire materialisation option into emit; Elixir-style cost gates |
| **C#** | SQL/LINQ-as-substrate; the optimisation-inspiration target | broadest aggregate/join/negation coverage | LINQ pipeline (not WAM-shaped); split into `csharp_target` + `csharp_query_target` + `csharp_native_target` | source-mode sweeps measure cost; memory-mapped file in flight | n/a (LINQ-inspired lineage) | continue source-mode benchmark sweep; tighten cost model |
| **Scala** | how aggressive can compile-time atom interning + classic-program coverage be | classic programs (n-queens, Ackermann, Fibonacci) — sets the generalisation upper bound | dual: WAM-instr + per-predicate emitter (`wam_scala_lowered_emitter.pl`, `emit_mode(functions)`) — clause-1 fast path with interpreter fallback | 4 backends (inline / file CSV / grouped TSV / **arity-N LMDB**, validated end-to-end) | all 7 kinds opt-in `kernel_dispatch(true)`; mode bench ~4×@depth100, ~9×@depth300 (shallow queries can regress) | cross-target bench vs Elixir/Haskell; ISO adoption |
| **Clojure** | LMDB JNI integration as a first-class data tier | deterministic-prefix lowering; sequential-only tests | dual: WAM-instr + emitter (T4; strips `switch_on_constant` prefix — no emitted switch table) | LMDB only (JNI loader) + cache policies (memoize / shared / two_level) | none (foreign handlers for category_parent/ancestor) | emit lowered switch tables; parallelism gates |
| **Rust** | hand-tuned FFI kernel route | det/T4–T6/ITE lowered emitter | dual: WAM-instr + emitter | LookupSource; eager/lazy/cached in matrix path + templates (promote into default project writer); CSR | all 7 + matrix + bidirectional + **boundary** extra | simplewiki bi-dir vs F#; default `lmdb_materialisation` option; FactSource facade |
| **F#** | Haskell-shaped target on the .NET runtime | mirrors Haskell coverage | dual: WAM-instr + emitter (default interpreter; functions ~1.0–1.07×) | LMDB `eager`/`lazy`/`cached`/`auto` (+ two-level L1/L2); CSR | detector fires for all kinds; **templates only for category_ancestor + bidirectional** (others stub); bi opt-in | finish kernel templates; classic conformance adapter; cost gates |
| **LLVM** | portable native-codegen via LLVM IR (native binary or WASM) | broad WAM coverage; lowered M1–M4 | dual: WAM-instr + lowered emitter | arena only — no LMDB | **LLVM-specific 7** (has countdown/list_suffix; lacks parent/step-distance + bidirectional) | conformance adapter; LMDB; trail-rollback |
| **C++** | systems substrate + ISO-error reference | broad WAM; classic conformance green on onboarding | dual: WAM-instr + lowered emitter (det / clause-1 / ITE / T4–T6) | arity-2 LMDB FactSource (v1) | `call_foreign` only — no shared-kernel detector | keep ISO reference; optional shared-kernel parity |
| **Typr** | typed-functional R wrapper with explicit raw-R fallback | recursion-pattern matcher (per-path-visited, tail, tree, mutual); raw-R fallback for producer goals | native pattern lowering (different model — pattern recognition, not WAM-bytecode lowering) | none — input modes (stdin/file/VFS/function) | n/a | continue typed lifting per `docs/handoff/typr-*.md`; doesn't generalise easily to graph-algorithm benchmarks |

## Table 2 — other hybrid WAM targets

| Target | Source size | Status | Notes |
|---|---|---|---|
| **WAT** (WebAssembly text) | `wam_wat_target.pl` (~6.4k) + `wam_wat_lowered_emitter.pl` | Substantial WAM-instruction lowering + T4–T6 hybrid; conformance green | Browser/sandboxed deploy; interpreter-bound |
| **C** | `wam_c_target.pl` (~6.1k) + `wam_c_runtime/wam_runtime.h` (~1k) | **Strong systems** — all 7 shared kernels + `bidirectional_ancestor`, LMDB, reverse CSR, aggregates, lowered helpers | Living checklist: `WAM_C_TARGET_NEXT_STEPS.md`. Conformance green. Not “907 lines / less-developed.” |
| **Go** | `wam_go_target.pl` (~3.5k) + lowered (~0.7k) | **All 7 FFI kernels** + TSV/LMDB atom facts; conformance via `prefer_wam(true)` | Default product path is still non-WAM `go_target.pl` |
| **JVM** | `wam_jvm_target.pl` (~650) | Jamaica/Krakatau dual emit; no lowered emitter | Third JVM route after Scala/Clojure |
| **Kotlin** | `wam_kotlin_target.pl` (~418) | Hybrid partition scaffold; Gradle e2e when available | No lowered emitter / kernels |
| **ILAsm** | `wam_ilasm_target.pl` (~1.9k) | CIL hybrid + ~45 plunit tests | No lowered emitter |

## Cross-cutting observations

1. **The Prolog-as-optimisation-carrier strategy works for the
   semantic layer, not the storage layer.** Branch pruning expressed
   as Prolog rewrites does generalise — every WAM target picks it up
   for free. But materialisation is below the WAM abstraction, so
   those wins don't transit. Each target has to solve materialisation
   in its own runtime.

2. **Two distinct research questions are running in parallel:**
   materialisation cost (C# query sweeps; Haskell LMDB; Clojure LMDB
   policies) and kernel-based lowering (Go category_ancestor; Rust
   effective-distance). They're independent: a target can solve one
   without the other, but the biggest scaling wins come from solving
   both.

3. **C# is structurally separate from the WAM targets.** Same
   optimised-Prolog input, different transpilation backbone. The
   parallel pipeline turns out to be a feature, not a bug — it lets
   C# explore the LINQ side of the design space while WAM targets
   explore the bytecode side, and we can compare results.

4. **Scala is the breadth-anchor.** The classics suite (n-queens,
   Ackermann, Fibonacci) is the most valuable generalisation test bed in
   the repo. Scala now has a per-predicate native fast-path emitter
   (`wam_scala_lowered_emitter.pl`, opt-in via `emit_mode(functions)`),
   closing the dual-mode-emitter gap and letting it benchmark against
   Elixir/Haskell on the same basis. The next levers are hot-path graph
   kernels and an LMDB sidecar.

   PR #1827 ports the **discipline** (subprocess invocation +
   true/false verification) to Elixir as a starter classic-programs
   suite (`tests/test_wam_elixir_classic_programs.pl`, fibonacci +
   ackermann). The reusable `with_elixir_project/4` +
   `verify_elixir_args/4` harness + shared `run_classic.exs` driver
   open the door to porting the rest of the Scala classics
   (list_reverse, nrev, expression_evaluator, n-queens) plus the
   builtin smoke tests (between, sort/msort, format) once
   `parse_arg/1` in the driver grows compound-term support.
   Runtime-correctness validation through the WAM-Elixir compiler
   was previously emit-and-grep only; this is the first end-to-end
   discipline.

5. **Interning approaches diverge.** Three strategies in flight:
   compile-time → LMDB-key (Haskell), aggressive compile-time IDs
   with pre-assigned well-known atoms (Scala), and strings everywhere
   (Elixir, Clojure, Rust, Typr, F#). Worth measuring whether the
   first two produce meaningful wins on heap-allocation-heavy
   workloads — the kernel-based-lowering data hints they should.

## Paradigm alignment: which target wins on which workload shape

Eight Elixir-kernel PRs (#1809–#1815 plus the parallel-fanout work)
on the cross-target `effective_distance` benchmark surfaced a
clear paradigm-alignment story. The underlying observation: **target
selection should follow workload shape, not target-ranked-by-aggregate-perf**.

| Workload shape | Best target | Reason |
|---|---|---|
| Single-process tight numeric recursion (no parallelism) | **Rust** | Native compile, register-allocated floats, LLVM auto-vectorisation. ~1× baseline. |
| Pure recursive numeric aggregation (compile-time fusion friendly) | **Haskell** | GHC list-comprehension fusion + strictness analysis collapse producer+consumer into one tight loop. Within 1-2× of Rust. |
| Fanout across many independent subqueries (cores available) | **Elixir** | Lightweight process spawn, no shared mutable state to coordinate, scheduler-aware load balance. Reaches Rust parity at ~1k scale on 4 cores; gap stays ~1.6× at 10k. **Wins outright at very high core counts** (untested in this measurement). |
| Distributed / fault-tolerant deployments | **Elixir** | OTP supervision, message-passing process model. Other targets need explicit infrastructure for the same. |
| Relational search with backtracking | **Prolog** | First-class unification, choice points, indexing. The source language for a reason. |
| Storage above ~100k facts | **any target with LMDB** | Memory-mapped storage amortises load cost. Haskell + C# + Scala have it shipping. Elixir has the FactSource adaptor (PR #1792) but needs runtime validation with `:elmdb`. |

Key surprises from the Elixir kernel work:
- The biggest single perf lever was not a kernel-internals
  optimization. It was the **caller-side data representation** —
  integer IDs in a tuple-as-array gave ~2× over atom-keyed `Map`
  because `elem/2` is O(1) without hashing (PR #1815). Lesson: profile
  the FactSource layer before optimising the kernel.
- The second-biggest lever was **outer-loop parallelism with chunking**
  (~3.4× on 4 vCPUs). Naive `Task.async_stream` with one task per
  item is a 1.5-4× regression because per-task spawn overhead exceeds
  per-item work. Pre-batching into ~`schedulers × 8` chunks amortises
  it. Lesson: always chunk for fine-grained parallelism on BEAM.
- **BEAM atom comparison and small-int comparison are equivalent**
  per-call (both immediate-word compare). The 10-15% delta between
  atom-Map and int-Map at scale is the `Map.get` hash path
  (atom-table precomputed hashes vs `:erlang.phash2` per call), not
  the comparison.
- BEAM has no Go-style compiled-switch jump table for free
  constant-time dispatch. Choice-point-stack walks for aggregate
  frames are O(D) by construction; the right BEAM idiom is "walk
  once at entry, thread through, reassemble at exit" rather than
  "walk every push" (PR #1814).
- Producer/consumer fusion on BEAM is best expressed as
  **callback-fold** — pass the consumer's fold INTO the producer
  (PR #1812). `Stream.unfold` is runtime laziness via closure
  trampolining and is *slower* than a materialised list for tight
  numeric inner loops. This is the BEAM analogue of GHC
  deforestation / Rust iterator monomorphization.

## Implications for Elixir's next work

In rough order of expected payoff:

1. **LMDB integration** for FactSource. Memory-mapped fact storage is
   the established response to materialisation cost above ~100k
   facts; Haskell and C# both have it shipping. This is the highest-
   value next step. Pattern to follow: Haskell's safe key/value API
   (the raw-pointer interface caused crashes — design doc note).

   Note: an earlier version of this doc claimed Haskell uses LMDB's
   storage IDs directly as the interning key. That's not what the
   Haskell target actually does — it maintains a compile-time
   `atom_intern_id` table (see `wam_haskell_target.pl` ~line 75) and
   stores facts in LMDB with binary keys. Scala uses a runtime
   `InternTable` per FactSource lookup. **Neither pushes interning
   into LMDB itself.** The LMDB-native design — three sub-databases
   (facts, key→id, id→key) within one env, with insert-time ID
   assignment — is captured in
   `docs/proposals/wam_elixir_lmdb_int_id_factsource.md` and a code
   stub of `WamRuntime.FactSource.LmdbIntIds` is shipped alongside
   it for emit-and-grep validation. Real runtime testing requires
   `:elmdb`, which the sandbox cannot install.

2. **Hot-path graph kernels.** Pick one or two graph operations
   (transitive closure, effective-distance) and emit them as Elixir
   modules that bypass WAM dispatch entirely, mirroring the
   Go/Rust kernel approach. Coverage status as of this writing:

   | Kernel kind | Rust | Haskell | Elixir | Scala |
   |---|:-:|:-:|:-:|:-:|
   | `transitive_closure2` | ✓ | ✓ | ✓ (PR #1799) | ✓ |
   | `category_ancestor` | ✓ | ✓ | ✓ (PR #1803, optimised through #1817) | ✓ |
   | `transitive_distance3` | ✓ | ✓ | ✓ (PR #1822) | ✓ |
   | `transitive_parent_distance4` | ✓ | ✓ | ✓ (PR #1823) | ✓ |
   | `transitive_step_parent_distance5` | ✓ | ✓ | ✓ (PR #1824) | ✓ |
   | `weighted_shortest_path3` | ✓ | ✓ | ✓ (PR #1825) | ✓ |
   | `astar_shortest_path4` | ✓ | ✓ | ✓ (PR #1826) | ✓ |

   **Coverage complete.** All 7 kernel kinds the shared detector
   (`recursive_kernel_detection.pl`) recognises now have native
   Elixir kernel + dispatch-wrapper implementations, matching the
   coverage Rust and Haskell already had.

3. **Tier-2 outer-loop parallelism, but emitter-driven.** The
   parallel-fanout numbers in `benchmarks/wam_effective_distance_cross_target.md`
   came from a hand-written caller-side bench. The same pattern
   should be auto-emitted by the WAM-Elixir target when it
   recognises an outer `forall` / `findall` / multi-pair query
   shape. Existing Tier-2 super-wrapper scaffolding (PR #1799 et seq.)
   is the starting point; the chunking heuristic (~`schedulers × 8`)
   needs to make it into the codegen.

4. **Transform-aware emitter pass for fold dispatch.** Recognise
   `aggregate_all(sum(W), (KernelGoal, W is f(Hops)), R)` shapes
   in the WAM IR, compile the per-solution arithmetic into an
   Elixir closure, route through `fold_hops/6`. Current dispatch
   handles only the bare case where the value register IS the
   kernel hop (PR #1814). Generalisation requires
   Prolog-arithmetic-to-Elixir compilation. Deferred because the
   parallel-fanout result moves the wall-clock more than this would
   on real workloads.

5. **Atom interning experiment.** The Haskell + Scala approaches
   are different; either could be ported. Measure on the existing
   Tier-2 benchmark grid before committing. Lower priority now that
   the int-tuple FactSource path (PR #1815) gives equivalent perf
   without atom-table pressure.

6. **NIF-backed kernels (Rustler / Zigler).** Largest single perf
   lever theoretically — port `collect_native_category_ancestor_hops`
   to native code, dispatch through the same FactSource contract.
   Closes the BEAM-vs-native gap entirely. Untested in this session
   because the sandbox cannot reach Hex.pm; viable in environments
   that can install Rustler.

7. **Cross-target parity automation** (Phase 4 proposal §9 Q5):
   compile the same Prolog through Haskell + Elixir, assert
   set-equal results in CI. Catches the kind of shared-WAM-compiler
   regressions that landed throughout this session.

## Validation strategy

Two test surfaces, mapped to the two mission goals:

| Goal | Validation |
|---|---|
| Generalize | Per-target test suites (Phase 3/4 for Elixir; classic programs for Scala; recursion patterns for Typr; aggregates/joins for C#). Coverage equates to "how much Prolog this target accepts." |
| Optimize | Cross-target benchmark suite (`examples/benchmark/benchmark_effective_distance_*.py`, etc.). Same Prolog, every target, measure wall-clock. The kernel-based-lowering wins show up here. |

The two surfaces interact: a generalisation gap (e.g., a missing
builtin) shows up as a benchmark crash, not a slow result. The
session that produced PRs #1776 #1777 #1780 #1781 #1782 #1784
discovered most of those gaps via failing benchmarks.

## Document scope

This document is **descriptive of current state and direction**, not
a binding plan. The "path forward" cells in the tables are starting
points for discussion, not committed work items. Work items live in
PR descriptions and the design proposals in `docs/proposals/`.

Updated as part of the session that produced the cost-gate
calibration (PR #1786). Earlier session-of-record context:

- PR #1767 — Phase 4d nested-fork suppression test + branch_sentinel fix
- PR #1770 — multi-clause findall compiler fix
- PR #1774 — Tier-2 findall parallel-vs-sequential benchmark
- PR #1776 — integer constant quoting fix
- PR #1777 — missing comparison builtins
- PR #1778 — arith benchmark + builtin coverage audit
- PR #1780 — extended builtin set + default-arm hardening
- PR #1781 — inline-list-build heap link fix
- PR #1782 — meta-builtins (functor, arg, =.., copy_term)
- PR #1783 — static cost-aware Tier-2 gate (`forkMinCost`)
- PR #1784 — meta-builtins build/compose modes + copy_term fresh-var rename
- PR #1785 — runtime cost probe (`runtime_cost_probe`)
- PR #1786 — cost-gate calibration from benchmark grid
