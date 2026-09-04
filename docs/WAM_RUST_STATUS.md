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

## Quadratic list handling: `unify` deep-derefs before dispatch (2026-09-04)

Found while implementing guard G1 of
[`proposals/RESOLVER_PRUNING_DESIGN.md`](proposals/RESOLVER_PRUNING_DESIGN.md)
(a per-call catalog index built in pure Prolog). The identical program is
**linear** on SWI and on wamjs and **quadratic** here, so this is a runtime
shape, not a spec property.

**Symptom.** `examples/pkg_resolver` B3 ladder, resolve ms, same box:

| packages | 500 | 1,000 | 2,000 | 5,000 |
| --- | ---: | ---: | ---: | ---: |
| baseline (no index) | 639 | 1,296 | 2,608 | 6,427 |
| with the G1 index | 1,850 | 6,557 | 26,668 | **abort: 4 GiB exhausted** |

Baseline doubles with size (linear); the indexed build goes ×3.5 then ×4.1
per doubling (quadratic) and cannot finish 5k inside 4 GiB.

**Isolation.** Six ten-line probes, each building an N-element list and then
doing exactly one more thing to it (ms, release build):

| probe | 1,000 | 2,000 | 4,000 | 8,000 | shape |
| --- | ---: | ---: | ---: | ---: | --- |
| list construction only | 26.5 | 51.7 | 126.7 | 224.9 | **linear** |
| + `keysort/2` | 57.9 | 187.5 | 824.8 | 3,500.6 | **quadratic** (×4.2) |
| + `sort/2` on tagged triples | 89.0 | 236.9 | 764.1 | 2,979.2 | **quadratic** (×3.3) |
| suffix-through-output-arg only, no sorting | 123.4 | 429.5 | 1,755.2 | 7,092.5 | **quadratic** (×4.0) |
| `sort/2` + `group_keyed/2` + `same_key/4` | 349.9 | 1,243.1 | 4,706.6 | — | quadratic |
| the whole index build (adds `build_tree/4`) | 727.4 | 2,679.0 | 10,655.9 | — | quadratic |

Constructing long lists is fine. **Two independent shapes are each
quadratic on their own**: the sort builtins, and any predicate that hands a
suffix of a long list back through an output argument (`Rest1 = [X|Rest]`).
Above ~8,000 list cells every probe also aborts with a Rust stack overflow.

**Cause.** callgrind on the suffix probe (1,853 M Ir): `_int_malloc` 14.3 %,
`_int_free` 12.0 %, `WamState::deref_heap` (recursive) 11.3 %, `malloc`
7.8 %, `free` 5.1 %, `malloc_consolidate` 4.7 %, `String::clone` 4.7 %,
`drop_in_place<Value>` 4.3 % — about 60 % of the program is allocating and
freeing copies of terms.

`WamState::unify` (`templates/targets/rust_wam/state.rs.mustache`) opens with

```rust
let dv1 = self.deref_var(&self.deref_heap(v1));
let dv2 = self.deref_var(&self.deref_heap(v2));
match (&dv1, &dv2) { ... }
```

`deref_heap` walks the **entire** term recursively, allocating a fresh
`Vec<Value>` per node and re-parsing `"functor/arity"` with `rfind('/')` at
every node. It is called eagerly on both arguments *before* the match learns
what they are — including on the very common `(Value::Unbound(n1), _)` arm,
which then just does `bind_var(n1, dv2.clone())`. So unifying a fresh output
variable against a suffix of a long list costs O(length of the suffix), and
doing that once per element is Θ(N²). The `if !changed { return val.clone() }`
short-circuit inside `deref_heap` avoids *rebuilding* the spine but has
already *walked* it, so it does not bound the cost.

**Consequence for the fleet.** The catalog index is gated behind
`index_threshold/1` (64 rows) in `examples/pkg_resolver/resolver.pl`, but the
Rust regression starts at 500 packages — far above any threshold that would
still be worth having — so **the threshold does not rescue this lane**. The
Rust lane is therefore left on the pre-P3 committed build and must not take
the index until this is fixed. wamjs is unaffected (9.2× faster at 5k) and
SWI is unaffected (it never sees the tree because it is faster without it,
but it is correct either way).

**Fix sketch (not attempted here — it is the hottest path in the runtime).**
Dispatch on the *undereferenced* shapes first and deref only the argument the
chosen arm actually needs; bind variables to the shallow value and let the
existing lazy deref resolve it later. A cheaper first step is to stop
re-parsing `functor/arity` per node (the callgrind profile shows 6.5 % in
`memrchr`/`CharSearcher` alone). Pin any fix with the six probes above before
and after.

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
- ~~`Value` has no structural sharing~~ — **fixed 2026-09-03 (D52).**
  `Value::Str` and `Value::List` now carry a shared `Args` spine
  (`Arc<Vec<Value>>` plus an offset), so `Value::clone` is O(1) and a list
  tail is the same allocation one element along. Choice-point creation costs
  O(live registers) instead of O(term size), and walking an N-element list is
  O(N) instead of N retained copies of the remainder. Numbers below.
- **The lowered tier is not reachable from the interpreter.** Lowered
  predicates keep their WAM label and no name-keyed call-site hook routes
  `Execute`/`Call` to a lowered function, so a project built with
  `emit_mode(functions)` runs entirely interpreted and the lowered functions
  are dead code. Measured, and scoped, below.
- **`arg/3` over a body-constructed compound returns the construction-time
  placeholder** instead of the argument: `T = k(5,2), arg(1,T,V)` leaves `V`
  unbound where SWI gives 5. Head unification of the same term
  (`T = k(V,_)`) is correct, which is why the uw-resolve differential — the
  resolver never calls `arg/3` — does not see it. Reproduces identically
  before and after the structural-sharing change, so it is an independent,
  pre-existing silent-wrong-answer defect; found while writing
  `tests/test_wam_rust_term_sharing.pl`, not yet fixed.
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

### Performance: structural sharing landed (D52)

`Value` now has structural sharing, and the memory blocker is gone.

| | before | after |
|---|---:|---:|
| B1 corpus (39 scenarios) | 102 ms | **34 ms** |
| B2 differential (2400 cases) | 306.9 s | **16.5 s** |
| B3 resolve, 363 packages | 79.2 s | **0.27 s** |
| B3 resolve, 7514 packages | OOM at 8.5 GB | **4.73 s, 680 MB peak RSS** |

SWI on the same box: B2 2.12 s, B3 5k load 31.1 ms / resolve 16.9 ms. The gap
to SWI on B2 fell from ~163x to ~7.8x; the 5k resolve is ~280x and now
finishes at all. Both B3 curves are linear in catalog size where they used to
be quadratic. Full ladder:
[`../examples/pkg_resolver/rust/README.md`](../examples/pkg_resolver/rust/README.md).

**The representation** (`templates/targets/rust_wam/value.rs.mustache`):

```rust
pub struct Args { spine: Arc<Vec<Value>>, off: usize }   // Deref -> [Value]
pub enum Value { .., Str(String, Args), List(Args), .. }
```

`Args::tail()` is O(1) — the same allocation viewed one element further along
— and `Deref<Target = [Value]>` keeps every read site (`len`, `iter`,
indexing, slicing, `split_first`, `to_vec`) source-compatible, so the change
touched construction sites and three mutation sites rather than the ~320
pattern matches across the target.

**Why sharing needs no trail cooperation.** Terms in this runtime are
immutable: a variable binding lives in `WamState::bindings` keyed by the
variable's *name*, never inside the term cell, and `unwind_trail_to` restores
bindings (and register slots), never cells. Nothing can mutate a spine that
another snapshot observes, so aliasing is unobservable and the trail did not
change. The one place that used to mutate a spine in place — prepending, in
`set_heap_or_list` and in `deref_heap`'s cons reconstruction — copies instead
(`Args::cons`), because the slot before a shared window may belong to another
view of the same allocation. `Arc` rather than `Rc` because `WamState` must
stay `Send` for the T7 parallel-aggregate substrate (the static assertion in
`state.rs` pins it).

**Two costs, not one.** Making `Value::clone` cheap did not on its own rescue
the 5k case: it got *faster*, so it allocated further before dying — 13.6 GB
peak RSS and still OOM-killed, versus 8.5 GB before. Memory was still
quadratic (cap-1000 peaked at 4.8 GB). The larger offender was `get_list`,
which built `Value::list(tail.to_vec())` at every
list step, so peeling an N-element list allocated N copies of the remainder
and every choice point retained one. Sharing the tail is what made the curve
linear. A third, smaller fix: `deref_heap` used to rebuild every compound and
list it walked, on every call; it now returns the same spine when no sub-term
dereferenced to a different cell.

`tests/test_wam_rust_term_sharing.pl` (9 probes, SWI-oracled) pins the
observable consequences: peeling under backtracking, a tail handed out as a
term in its own right, trail restore of a binding made inside a shared spine,
list builtins over a shared tail, nested spines, a register holding a shared
term across choice points, and consing two lists onto one tail.

### Where the remaining time goes, and what would remove it

A callgrind profile of the 363-package resolve, after sharing, attributes the
run entirely to interpreter machinery. There is no arithmetic or constraint
hot spot to capture — `execute_builtin` covers ALL builtins and is 5.6%:

| | inclusive Ir |
|---|---:|
| `WamState::step` (instruction dispatch) | 65% |
| `WamState::backtrack` | 21% |
| `restore_regs` | 12% |
| `deref_var` + `deref_heap` | 12% |
| `get_reg` | 11% |
| `trail_binding` | 8% |
| `unify` | 7% |
| `put_reg` | 6% |
| `execute_builtin` (every builtin) | 5.6% |
| malloc/free (flat) | 31% |
| binding / Y-register hashing (flat) | 6.5% |

That profile says the lever is the **lowered tier** — running predicates as
direct Rust functions instead of interpreting their bytecode — not
hand-written leaf builtins. (One cheap allocation win is still on the table
inside `deref_heap`: it builds the `derefed` vector eagerly and throws it away
when nothing changed, so an already-resolved N-element list still costs one
N-element allocation per call. Deferring that vector until the first changed
child removes it; worth ~5% on the profile above, and independent of the
lowered-tier work.)

**Lowered-tier scoping (the wamjs D41/D42 analogue).** Rebuilt with
`emit_mode(functions)`, 74 of the resolver's 79 predicates lower —
27 `deterministic`, 20 `multi_clause_n`, 15 `multi_clause_1`, 12
`ite_lowered` — and the crate compiles. It also changes nothing measurable:
answers are byte-identical and the 363-package resolve takes 267 ms against
244 ms interpreted, peak RSS 43.4 MB either way. Three things stand between
that and the wamjs result:

1. **No Execute-of-user protocol.** `collect_wam_entries/6` emits a WAM label
   for `lowered` entries exactly as for `wam` entries, and the only
   name-keyed call-site hook the generated `lib.rs` exposes is
   `fact_table_call(vm, pred, cont_pc) -> Option<bool>` (T9), which is empty
   for this program. `Execute`/`Call` therefore always take the interpreter
   path. That T9 hook is the shape to copy: a
   `lowered_call(vm, pred, cont_pc) -> Option<bool>` consulted from both arms
   before label lookup.
2. **T4's contract is first-solution.** `emit_multi_clause_n_rust` tries each
   clause inline and returns `bool` at the first success
   (`lo_clause_snapshot` / `lo_restore_clause` in between), leaving no choice
   point for the caller to retry. The resolver is genuinely nondeterministic
   — `candidates_high_first/4` enumerates versions through `member/2`,
   `pick/7` and `blocked_from/4` backtrack — so wiring dispatch without
   fixing this would silently drop solutions. A resumable protocol is needed:
   the lowered function pushes a choice point carrying its clause index and
   is re-entered on backtracking. The runtime already has that shape twice —
   `fact_table_attempt` for T9 and `finish_foreign_results` + `BuiltinState`
   for foreign results. Until it exists, the honest behaviour is to refuse to
   lower a predicate whose caller may retry it.
3. **Five predicates do not lower at all**, and the missing shapes are
   specific:
   * `acc_conflicts/4` — a solution-producing disjunction `(A ; B)`;
     `rust_structured_clause1` folds only ITE / `\+` / `once`.
   * `pick/7`, `safe_upgrade/4`, `dep_breaks/5`, `blocked_from/4` — a
     *multi-clause* predicate with an ITE or a negation inside a clause.
     `rust_all_clauses_lowerable` rejects any clause containing
     `try_me_else`, and the `ite_lowered` path is gated on the predicate
     being single-clause.

   The two front-end extensions are therefore: apply the ITE structurer
   per clause rather than only to clause 1 of a single-clause predicate, and
   treat `(A ; B)` as an alternatives block (which needs item 2 anyway).

Order matters: item 1 alone is unsound without item 2. The sound intermediate
available today is to dispatch only predicates classified `deterministic` or
`clause_chain` — for this program 27+ of 79 — and leave the rest interpreted:
a mixed tier with no semantic risk, measurable before the resumable protocol
exists.

## Path forward

0. **Make the lowered tier reachable** — the three-step scoping above
   (`lowered_call` call-site hook, a resumable multi-clause protocol, ITE
   per clause + `(A ; B)`). This is now the largest single lever on symbolic
   workloads; the sound intermediate (dispatch only `deterministic` /
   `clause_chain` predicates) can land and be measured first.
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
performance blocker recorded. 2026-09-04: the uw-resolve pruning round
isolated a **second** quadratic — `unify` deep-derefs both arguments before
dispatch — with a six-probe ladder and a callgrind profile; see the section
above. 2026-09-03 (D52): the earlier blocker **closed** —
`Value` gained structural sharing, B2 fell 306.9 s → 16.5 s and B3 went from
OOM-at-8.5 GB to 4.73 s / 680 MB peak RSS; a 9-probe sharing suite added; the
lowered tier measured to be unreachable from the interpreter, and scoped.
