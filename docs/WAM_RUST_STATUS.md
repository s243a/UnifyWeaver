# WAM Rust Target — Status

Living summary of the hybrid WAM-Rust backend. Distinct from the
**non-WAM** direct Rust stream/project compiler documented in
[`RUST_TARGET.md`](RUST_TARGET.md) (`rust_target.pl`).

Companion docs:

- Design set under `docs/design/WAM_RUST_*` (transpilation, LMDB crate
  decision, boundary distribution, bridge detector, state retrospective).
- Reports under `docs/reports/wam_rust_*` (T7 parallel, cached scaling,
  bidirectional kernel benches).
- Handoffs: [`handoff/wam_rust_simplewiki_blocker.md`](handoff/wam_rust_simplewiki_blocker.md),
  [`handoff/rust_fsharp_parity_campaign.md`](handoff/rust_fsharp_parity_campaign.md).
- [`WAM_HYBRID_TARGETS_COMPARISON.md`](WAM_HYBRID_TARGETS_COMPARISON.md).

## Role

**Single-core kernel king.** Native compile, register-allocated hot
loops, u32 atom interning in FFI kernels, and the densest graph-kernel
surface in the fleet.

## Codegen surface

| Module | Approx. lines |
|---|---:|
| `src/unifyweaver/targets/wam_rust_target.pl` | ~7.1k |
| `src/unifyweaver/targets/wam_rust_lowered_emitter.pl` | ~0.8k |
| Dedicated tests | ~44 files |

## What's shipped

**Dual lowering.** Full WAM instruction VM + lowered emitter for
**deterministic** predicates / clause-1 of multi-clause (ITE via
`wam_ite_structurer.pl` where applicable).

**Kernels.** All seven shared kinds (inlined in `wam_rust_target.pl`,
unlike Haskell’s mustache templates), plus:

- Effective-distance **matrix** FFI path.
- **bidirectional_ancestor** (F# parity port: calibration + A*-pruned
  direction-cost search; `kernel_mode(bidirectional)`).
- **`category_ancestor_boundary`** (Rust-only extra kind) + large
  `boundary_cache.rs.mustache` distribution-cache substrate.
- Child direction from CSR artifact (`csr_child_index(true)`), LMDB
  `category_child`, or derived reverse table.

**Materialisation.** `LookupSource` trait over LMDB. Eager / lazy /
cached branches live in `materialisation_setup.rs.mustache` and are
exercised by the matrix benchmark generator
(`examples/benchmark/generate_wam_rust_matrix_benchmark.pl`); June
2026 reports document cached vs lazy speedups. Default
`write_wam_rust_project` does **not** yet take a first-class
`lmdb_materialisation(...)` option the way F# does. Reverse-CSR
artifact reader ships. No full Haskell-style FactSource facade yet.

**Atom interning.** u32 IDs in FFI hot path — measured **~7.9×**
query speedup at scale 300 (134 ms → 17 ms) vs string-keyed maps
([`design/WAM_CROSS_TARGET_BENCHMARK_RESULTS.md`](design/WAM_CROSS_TARGET_BENCHMARK_RESULTS.md)).

**Parallelism.** T7 `parallel_aggregates(true)` / `parallel(true)`
with rayon (`par_aggregate.rs.mustache`); measured ~3.39× on fib
BASE=20 (`docs/reports/wam_rust_t7_speedup_benchmark.md`).

**Classic conformance.** Registered; opt-in; green (cons-cell /
placeholder / integer `is/2` conventions applied).

**Runtime parser.** Opt-in compiled mode; default off; generated-runtime
tests cover round-trips.

**Lowered emitter.** Deterministic / T4 multi-clause / T5 switch
cascade / T6 match / ITE (`wam_rust_lowered_emitter.pl`) — broader
than “clause-1 only,” but still det-centric vs Elixir’s full lowered
default path in tests.

## Performance notes

- Scale-300 Rust WAM + FFI + interning: **17 ms query / 32 ms total**
  (beats pruned native DFS on query).
- Cached vs lazy 10k: ~88 vs ~279 query_ms (~3.17×) —
  `docs/reports/wam_rust_cached_scaling_sweep_2026-06-14.md`.
- Simplewiki median ~370 ms query —
  `docs/handoff/wam_rust_simplewiki_blocker.md`.
- Best target for single-process tight numeric / graph recursion with
  no parallelism requirement
  ([`WAM_TARGET_ROADMAP.md`](WAM_TARGET_ROADMAP.md) paradigm table).

## Known issues / gaps

- **Head-unification fixes landed 2026-08-06** (CONF-FIX-RUST-NESTED,
  CONF-FIX-RUST-EMPTYLIST) — four defects, all silent wrong answers,
  found by the new `nested`/`emptylist`/`repeatvar` conformance
  programs: `get_structure` deciding read-vs-write mode before
  dereferencing (so a bound argument got a fresh structure built over
  it, and a head matched a list element it should have rejected); its
  read branch re-appending the arity to an already-qualified functor key
  (`"s/1/1"`), which broke inner-functor clause discrimination;
  `unify_constant` using raw equality plus an unbound short-circuit
  instead of `unify()`; and `unify_value` accepting an unbound heap
  argument without binding it. Rust now passes all ten conformance
  programs with no xfail. **These were live in a Primary-band target and
  the six classic programs caught none of them** — worth remembering when
  sizing "is this target conformant?" for the remaining backends.
- Wire `lmdb_materialisation(...)` into default project writer (today
  mainly matrix-bench path).
- LMDB scan/segregation (R9/R10) still open.
- ISO three-form stack not adopted (catch/throw appears in builtin
  parity only).
- Simplewiki-scale bidirectional vs F# still an open measurement.
- Lowered TODO stub remains for some instruction shapes.
- **`Value` has no structural sharing**, and `save_regs()` deep-clones every
  live register into every choice point. A register holding a large term is
  copied per choice point, so symbolic workloads cost
  O(choice points x term size) in both time and memory. This is why the
  uw-resolve exercise measures ~163x slower than SWI and OOMs at 5000
  packages (see below). Prerequisite for any symbolic-workload performance
  claim.
- `call/1` is first-solution (the `call_goal_once` contract). A conjunction
  under `call/1` whose left conjunct is nondeterministic and whose right
  conjunct fails is refused loudly rather than answered incompletely.
- `bagof`/`setof` implement the witness-free semantics only; a non-empty
  ISO free-witness list refuses to compile.
- **Natively-lowered predicates have no WAM label** (`collect_wam_entries`
  skips `native` entries), so an interpreted caller cannot reach them: the
  `Call`/`Execute` arm falls through. Before the §7 class fix that was a
  silent failure; now such a name reaches builtin dispatch, which will run a
  *builtin* of the same name if one exists. Pre-existing and orthogonal to
  this work — the uw-resolve build compiles every predicate through the WAM
  fallback, so it does not arise there — but it is the next thing to check
  for a mixed native/interpreted project.

## Whole-program exercise (A2, 2026-09): known / suspected deficiencies

The peerhailer CLI-parser exercise (see
[`WAM_FLEET_GAPS.md`](WAM_FLEET_GAPS.md)) found three JS-WAM runtime bug
classes that are fleet-wide suspects. Rust's audit:

| # | Deficiency | Status | Evidence / reason |
|---|---|---|---|
| A1 | `sub_string/5` builtin missing | **partly closed** | `sub_atom/5` runs (probe `b04` in `tests/test_wam_rust_builtin_probes.pl`); it was unreachable in tail position until A3 was fixed. `sub_string/5` itself is still absent |
| A2 | Y-register clobber across `Call` of a no-`Allocate` callee | **was PRESENT in a second form; fixed 2026-09** | the aliasing form is indeed absent (string-named registers). But the *frameless-Y-write* form the original audit dismissed **does** occur: `wam_target.pl` reserves a permanent `Y` for an if-then-else barrier after deciding the clause needs no environment, so it emits `get_level Y1` in `Allocate`-less clauses (e.g. `satisfies/2`'s `gte` clause). Routing Y to the topmost env frame then wrote the **caller's** `Y1` — `pick/7` returned a choice-point depth instead of a version term. Fixed by keeping ITE barrier levels on the choice point (`ChoicePoint::levels`) instead of in a register; found by `examples/pkg_resolver/rust/` (15/2400 differential divergences) |
| A3 | `Execute` of a builtin doesn't return to the continuation | **FIXED 2026-09 (class fix)** | the `Execute` arm now falls back to the **whole** builtin table after label lookup fails and then takes `Proceed`'s return path (`pc = cp`), the Rust analogue of Go's `BuiltinExecute`; the `Call` arm got the same treatment (it had the same hand-listed gate). The old `is_iso_meta_builtin` / `atomic/1` special cases are gone — `is_iso_meta_builtin/1` is retained only as documentation. Probes `b04` (execute form) and `b05` (call form) in `tests/test_wam_rust_builtin_probes.pl`; an unresolvable goal can be reported with `UW_WAM_WARN_UNKNOWN=1` |
| A4 | String fidelity | **rung 0** | `Value` has no string variant (`value.rs.mustache:9-23`); D37's double-quoted literals intern as atoms |

Worth re-reading alongside the 2026-08-06 head-unification entry above:
that episode already demonstrated this doc's central caution — four
silent-wrong-answer defects lived in a Primary-band target that was
green on the six classic conformance programs. The whole-program
benchmark (`examples/cli_args/`, Class C in the fleet doc) is the
stronger gate.

Pattern lane: `rust_target.pl` compiles facts from `clause(Head, true)`
(`:7364-7372`) — the G-A3-8 execute-at-compile-time hazard is absent.
The G-A3 machinery (cross-predicate calls by callee output count,
multi-output tuple returns, semidet failure sentinel, compound terms as
runtime data) has no analogue; presume those gaps present until the
benchmark is attempted.

## Whole-program exercise (uw-resolve, 2026-09): eight silent-defect classes closed

`examples/pkg_resolver/resolver.pl` — the frozen P0.5 package resolver,
already gated on wam_javascript — was compiled through wam_rust
(`examples/pkg_resolver/rust/`, which carries the full write-up). It is a
harder program than `examples/cli_args/`: genuine backtracking, cuts at API
edges, if-then-else everywhere, `findall`/`bagof`/`setof`, and deep list
recursion over a term-shaped catalog.

**Gates now green:** 39/39 contract scenarios (selections *and* explanation
terms) and **2400 seeded differential cases with 0 divergences** across all
ten queries; 35/35 cut-semantics probes in four configurations.

**Eight defect classes it forced out — every one a silent wrong answer or a
silent failure, none a crash:**

1. **Empty list rejected by list builtins** (CONVENTIONS §1). `sort/2`,
   `keysort/2`, `sum_list/2`, `include/3`, `foldl/4`, … matched only
   `Value::List`; `put_constant []` delivers the **atom** `[]`. `sort([], X)`
   failed and took the clause with it. Fixed by routing all 22 list-argument
   sites through `deref_list_arg` (over the pre-existing `value_as_list`).
2. **A zero-solution aggregate halted the machine reporting success.** The
   finalisation PC was recorded by `EndAggregate` — which never runs when the
   inner goal has no solutions — so it read a stale `0`, and PC 0 is HALT.
   `findall(X, Goal, L)` with no solutions "succeeded" and dropped every goal
   after it. `BeginAggregate` now scans forward for its matching
   `EndAggregate` and carries the continuation on the aggregate frame.
3. **`==/2` / `\==/2` did not alias `[]` with the empty list**, so
   `L == []` was false whenever `L` came back from a builtin. Replaced raw
   `Value: PartialEq` with `terms_identical/2`.
4. **A3 (`Execute`/`Call` of a runtime builtin)** — see the table above.
5. **ITE barrier registers clobbered the caller's permanent variables** —
   see A2 above. This is the one that produced wrong *answers* rather than
   failures, and it is an upstream `wam_target.pl` shape that every backend
   routing Y to a frame (Haskell, Python, R, F# per §8) should re-audit.
6. **`!` inside a `findall`/`bagof`/`setof`/`aggregate_all` inner goal
   destroyed the aggregate frame** (§9 barrier-raising contexts).
   `BeginAggregate` now raises `cut_barrier` above its own choice point.
7. **`pending_cut_barrier` leaked out of `Allocate`-less clauses**: a fact
   predicate's `TryMeElse` parked a barrier no `Allocate` ever consumed, and
   the next unrelated `Allocate` took it — so a `!` in a single-clause callee
   cut past a preceding fact's choice point. The pending barrier is now
   paired with the PC that must consume it.
8. **`call/N` did not exist, and `bagof/3`/`setof/3` were never inlined.**
   `execute call/1` / `execute bagof/3` found no label and no builtin, so
   every clause ending in one was dead. `call/1`…`call/8` are now meta
   builtins routed through `call_goal_once` (an opaque cut scope per §9, plus
   `,/2` `;/2` `->/2` `\+/1` `!` in the meta-call walker); `wam_rust` now
   compiles with `inline_bagof_setof(true)` and implements the
   witness-free semantics on the aggregate frame.

**§9 (cut is a barrier) is now claimed**, with the required probe corpus:
`tests/test_wam_rust_cut_semantics.pl` — the 35 SWI-oracled probes ported
from the JS suite, run in `interpreter`, `pure` (fact tables and kernels
off), `functions` (lowered emitter offered everything), and a lowered-tier
mode that calls the generated `lowered_pNN_t_1` functions directly and is
oracled against `once/1`. `tests/test_wam_rust_builtin_probes.pl` holds the
minimal probe for each non-cut defect above.

**Refuse-to-compile-loudly** where semantics are not implemented:
`bagof`/`setof` with a non-empty ISO witness list (grouping) throws
`unsupported_wam_instruction(bagof_setof_witness_grouping(...))` rather than
emitting an ungrouped answer.

### Performance: the interpreter is memory-bound, and it is severe

On this workload wam_rust is **~163× slower than SWI-Prolog** (2400
differential cases: 306.9 s vs 1.88 s), and it **cannot run the 5000-package
scale catalog at all** — 8.5 GB RSS and still running after four minutes,
then OOM-killed, against SWI's 9.4 ms. Resolve time grows quadratically with
catalog size (3.5 s at 54 packages → 79 s at 363), while *load* (term
construction) is ~5× faster than SWI.

Cause: `save_regs()` clones every live register into every choice point, and
`Value` has no structural sharing — `Value::List(Vec<Value>)` and
`Value::Str(String, Vec<Value>)` deep-copy on clone. A register holding a
whole catalog term is therefore deep-copied per choice point, so cost and
memory scale as (choice points × term size). **Giving `Value` structural
sharing (`Rc`/`Arc` around the argument vectors) is the single highest-value
change available to this backend** and is a prerequisite for any
"single-core kernel king" claim on symbolic (as opposed to FFI-kernel)
workloads. Numbers and the full sweep:
[`../examples/pkg_resolver/rust/README.md`](../examples/pkg_resolver/rust/README.md).

## Path forward

1. Simplewiki-scale bidirectional benchmark vs F#.
2. Promote LMDB lazy/cached into default project options + FactSource
   generalisation.
3. Distribution-cache / boundary phases + builtins parity sweep.
4. Optional ISO three-form adoption once C++/Elixir/F# patterns are
   extracted.

## Document status

Snapshot for the hybrid comparison branch. Prefer updating this file
when kernel, LMDB, or conformance milestones land rather than only
adding one-off report files. 2026-09-01: added the whole-program (A2)
deficiency audit; see [`WAM_FLEET_GAPS.md`](WAM_FLEET_GAPS.md).
2026-09-03: uw-resolve exercise — A3 closed with a class fix, A2 reopened
and closed in its frameless-Y form, six further silent-defect classes
fixed, §9 claimed with a 35-probe corpus, and the deep-copy `Value`
performance blocker recorded.
