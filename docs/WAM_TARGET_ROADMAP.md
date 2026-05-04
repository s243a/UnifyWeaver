# WAM target roadmap: where we are, where we're going

## Mission

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

- **No memory-mapped fact storage.** The bottleneck above ~100k facts
  Haskell/C# both target.
- **No kernel-based lowering.** No graph-algorithm FFI kernels yet.
- **No interning.** Atoms are stored as Elixir strings everywhere.

## Table 1 — primary targets (Elixir-relevant lessons)

The set Elixir can most directly learn from. Each target is anchored
on a specific architectural question.

| Target | Architectural question | Generalization | Lowering | Materialization | Kernels | Path forward |
|---|---|---|---|---|---|---|
| **Elixir** | reference baseline | Phase 3/4 + comprehensive builtins | dual: WAM-instr + per-predicate emitter | FactSource facade (ETS/SQLite/TSV); no memory-mapped | none | LMDB integration (high-value for >100k); hot-path graph kernels |
| **Haskell** | how to make materialisation cheap at scale | broad WAM, parMap parallel | dual: WAM-instr + emitter | LMDB key/value (memory-mapped under the hood); raw-pointer interface abandoned due to crashes | parMap rdeepseq | calibrate fork-min-cost like Elixir; investigate cost-aware probing |
| **C#** | SQL/LINQ-as-substrate; the optimisation-inspiration target | broadest aggregate/join/negation coverage | LINQ pipeline (not WAM-shaped); split into `csharp_target` + `csharp_query_target` + `csharp_native_target` | source-mode sweeps measure cost; memory-mapped file in flight | n/a (LINQ-inspired lineage) | continue source-mode benchmark sweep; tighten cost model |
| **Scala** | how aggressive can compile-time atom interning + classic-program coverage be | classic programs (n-queens, Ackermann, Fibonacci) — sets the generalisation upper bound | WAM-instruction lowering only; no `wam_scala_lowered_emitter.pl`; step-loop interpreter | 3 backends (inline / file CSV / grouped TSV); auto-inline ≤128 rows | none | add per-predicate native fast-path emitter (single biggest gap); port classics to other targets to validate generalisation |
| **Clojure** | LMDB JNI integration as a first-class data tier | deterministic-prefix lowering; sequential-only tests | dual: WAM-instr + emitter (deterministic prefix only — no `switch_on_constant` lowering yet) | LMDB only (production-grade JNI loader, delay-wrapped) + cache policies (memoize / shared / two_level) | none | extend lowered emitter to non-deterministic prefixes; add parallelism gates |
| **Rust** | hand-tuned FFI kernel route | deterministic only in lowered emitter | dual: WAM-instr + emitter (deterministic only) | absent (no FactSource facade); FFI kernels are ad-hoc | effective-distance matrix FFI kernel (concrete demonstration of kernel-based lowering wins) | port the kernel pattern to other targets; add layout policies / FactSource generalisation |
| **F#** | Haskell-shaped target on the .NET runtime | mirrors Haskell coverage | dual: WAM-instr + emitter | none documented | TPL Parallel.map mentioned, no gating | follow Haskell's LMDB lessons + Elixir's cost-gate calibration |
| **Typr** | typed-functional R wrapper with explicit raw-R fallback | recursion-pattern matcher (per-path-visited, tail, tree, mutual); raw-R fallback for producer goals | native pattern lowering (different model — pattern recognition, not WAM-bytecode lowering) | none — input modes (stdin/file/VFS/function) | n/a | continue typed lifting per `docs/handoff/typr-*.md`; doesn't generalise easily to graph-algorithm benchmarks |

## Table 2 — less-developed hybrid WAM targets

Listed for completeness. These have substantial Prolog source but
haven't yet hit the kernel-or-LMDB inflection point.

| Target | Source size | Status | Notes |
|---|---|---|---|
| **WAT** (WebAssembly text) | `wam_wat_target.pl` (6325 lines) | Substantial WAM-instruction lowering pipeline | No `wam_wat_lowered_emitter.pl`; single-mode interpreter (Scala-shaped). Browser-deployment angle is the differentiator |
| **LLVM** | `wam_llvm_target.pl` (4499 lines) | Substantial WAM-instruction lowering | No emitter file; compiles to LLVM IR. Worth investigating for cross-target kernel reuse (LLVM IR as portable kernel format?) |
| **C** | `wam_c_target.pl` (907 lines) + `wam_c_runtime/wam_runtime.h` | Has a C runtime header | Smaller surface; useful as a portable substrate for FFI kernels (Rust/Go FFI kernels could share C glue) |
| **JVM** | `wam_jvm_target.pl` (711 lines) | Smaller than the Scala/Clojure entries | Generic JVM bytecode emit; both Scala and Clojure target the JVM via different routes — this is the third |

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

4. **Scala is the breadth-anchor and the dual-mode-emitter outlier.**
   The classics suite (n-queens, Ackermann, Fibonacci) is the most
   valuable generalisation test bed in the repo. Scala lacks the
   per-predicate native fast-path emitter; adding one would let it
   benchmark against Elixir/Haskell on the same basis.

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

   | Kernel kind | Rust | Haskell | Elixir |
   |---|:-:|:-:|:-:|
   | `transitive_closure2` | ✓ | ✓ | ✓ (PR #1799) |
   | `category_ancestor` | ✓ | ✓ | ✓ (PR #1803, optimised through #1817) |
   | `transitive_distance3` | ✓ | ✓ | ✓ (PR #1822) |
   | `weighted_shortest_path3` | ✓ | ✓ | — |
   | `transitive_parent_distance4` | ✓ | ✓ | — |
   | `astar_shortest_path4` | ✓ | ✓ | — |
   | `transitive_step_parent_distance5` | ✓ | ✓ | — |

   Remaining 4 kernels in rough order of implementation difficulty:
   `transitive_parent_distance4` (DFS, no cycle check, 3-tuple output —
   simplest); `transitive_step_parent_distance5` (DFS + first-step
   tracking — small extension of #1822); `weighted_shortest_path3`
   (Dijkstra over weighted edges — needs priority queue); 
   `astar_shortest_path4` (A* with heuristic — most complex, goal-
   directed instead of full enumeration).

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
