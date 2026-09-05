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

## Quadratic list handling: `unify` deep-derefs before dispatch — FIXED (2026-09-05)

**Fixed 2026-09-05 by the shape-first `unify` restructure (D59 step 1c follow-up).**
`WamState::unify` now dispatches on the raw cell shape, following only the
variable-binding chain by reference (`deref_chain`, allocation-free), and
materialises at most ONE level of a term per side (`deref_shallow`) before any
element-wise work. Deep traversal happens only when both sides are genuinely
compound, and the spine walk is a loop (O(1) Rust stack). The bind-unbound arm
no longer pays a walk of the other side at all. `deref_heap` got two companion
fixes: cons chains are now flattened ITERATIVELY (`deref_cons_chain`) instead of
rebuilt one `Args::cons` at a time — that was a *second* Θ(N²), and the source
of the "stack overflow above ~8,000 cells" — and its argument vector is now
allocated lazily, only once a sub-term actually moves (the ~5% `deref_heap`
follow-up the profile flagged).

**Is it mechanical?** Mostly. The reorder itself is the mechanical part. The
one borrow-checker constraint: phase 1 (the allocation-free arms) borrows
`&self.bindings` through `deref_chain`, and an arm that must mutate — bind a
variable, or recurse into `unify(&mut self, …)` — cannot hold that borrow. The
structure that satisfies the checker is a two-phase body: a first `match` on the
two borrowed raw cells that only *reads* (returns for the atomic/identical-var
cases), then, once the borrows are dropped, phase 2 takes owned one-level values
(`deref_shallow` returns `Value`, O(1) for compounds/lists because the spine is
shared) and does the binding/recursion. That split is forced, not stylistic, but
it is a local rewrite of one function plus three small helpers — no change to
the `Value` representation, the trail, or any call site.

**Result — the six D59 probes, pristine vs fixed** (release, this box; the
suffix and build_tree probes are the ones the write-up isolated as quadratic):

| probe (ms) | 1,000 | 2,000 | 4,000 | 8,000 | shape now |
| --- | ---: | ---: | ---: | ---: | --- |
| construction only (pristine) | 33.8 | 118.6 | 446.7 | 1,794 | was already ~linear-ish, now flat |
| construction only (fixed) | 12.9 | 24.3 | 49.6 | 98.2 | **linear** (×2/doubling) |
| suffix-through-output-arg (pristine) | 173.6 | 697.7 | 2,620 | 10,474 | quadratic (×4) |
| suffix-through-output-arg (fixed) | 22.5 | 42.9 | 96.6 | 193.1 | **linear** (×2) |
| build_tree/keysort/group (pristine `pd`) | 397.5 | 1,451 | 6,884 | 30,943 | quadratic (×4) |
| build_tree/keysort/group (fixed `pd`) | 58.6 | 150.1 | 262.7 | 568.1 | **linear** (×2) |

`keysort/2` and `sort/2` over tagged triples also go linear (they were quadratic
only because they *walked* long lists through the same deref path). No probe
aborts any more — 32,000-cell `pd` finishes in 4.1 s where 8,000 used to be the
stack-overflow ceiling.

**The G1 index on the Rust lane, unblocked.** The scale ladder (scratch crate,
P3 rows stripped, `index_threshold(1)` to force the index on), resolve ms:

| packages | 744 | 1,511 | 3,022 | 7,522 |
| --- | ---: | ---: | ---: | ---: |
| pristine + G1 index | 1,358 | 4,914 | 18,490 | **abort (4 GiB)** |
| fixed + G1 index | 223 | 411 | 822 | 1,924 |
| fixed, index off | 480 | 876 | 1,717 | 4,402 |
| pristine, index off | 543 | 919 | 1,839 | 4,684 |

The indexed build is now **linear** (×2 per doubling) and, with the quadratic
gone, the index is a straight ~2.3× win at 5k rather than an OOM. The plain
index-off ladder does **not** regress (fixed ≈ pristine, both linear). So the
D59 index is sound on the Rust lane once the runtime is fixed — the committed
example stays PRE-P3 (rebuilding it against the current resolver is a separate
port round), but the blocker recorded below is cleared.

Pinned by `tests/test_wam_rust_unify_shape.pl` (9 SWI-oracled behaviour probes:
unbound-vs-suffix, var-var, nested compound-compound, cons/empty-list aliasing,
N-cell element-wise, partial-list tail fill, wide compound, deep-leaf failure)
and the re-run 9-probe sharing suite.

<details><summary>Original finding (2026-09-04), kept for the record</summary>

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

</details>

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
- ~~**The lowered tier is not reachable from the interpreter.**~~ **Partly
  closed 2026-09-05.** A `lowered_call` hook now routes `Execute`/`Call` to a
  lowered function for the dispatch-safe class (cp-clean fixpoint). Predicates
  that call a multi-clause predicate, and the multi-clause predicates
  themselves, are still interpreted pending the resumable protocol (item 2b).
  See the lowered-tier section below.
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

### The sound intermediate landed (2026-09-05) — `lowered_call` hook

Item 1 is now wired: a crate-level `lowered_call(vm, pred, cont_pc) -> Option<bool>`
(shaped exactly like T9's `fact_table_call`) is consulted from both the `Call`
and `Execute` arms *before* label lookup, and the emitter builds its match table
at compile time from the predicates it classifies dispatch-safe. On by default
under `emit_mode(functions)`; `lowered_dispatch(false)` turns it off.

**What is dispatched — and why the class is narrower than "deterministic or
clause_chain" first looks.** A lowered function reaches a user callee by running
it *interpreted* (`emit_one(call)` sets `pc` and calls `run()`), NOT by calling
the callee's own lowered function — so an interpreted MULTI-clause callee leaves
a live choice point, and a later failure inside the same lowered function
backtracks straight into it, escaping the function's scope. The runtime guard
(`WamState::lowered_dispatch`: run with `cp=0` and the cut barrier at entry
depth, then roll back and return `None` if the call left a choice point) catches
a non-deterministic *result*, but cannot undo mid-run corruption. So eligibility
is a greatest fixpoint: a predicate is dispatched only when every predicate it
(transitively) calls is **cp-clean when interpreted** — single-clause
(`deterministic` / `ite_lowered`, whose transient ITE choice point is cut before
return). A LEAF `clause_chain` (a fact table like `kind/2`) has no user callees,
so it stays dispatchable itself and an interpreted caller reaches its clean
cascade through the hook; but a predicate that *calls* a multi-clause predicate
is left interpreted. This is the same optimistic-then-swept whole-predicate
classification mprolog uses (`proposals/MPROLOG_MINING_NOTES.md`, F2), and F3's
"nondet-to-det crossing still pays dynamic dispatch" is exactly the boundary we
declined to cross here.

Two runtime correctness fixes fell out of wiring this, both latent before (the
lowered tier was dead code, so never exercised as a subroutine):

* **`get_constant`/`get_integer`/`get_nil` on an unbound argument register now
  bind the underlying variable**, not just overwrite the register
  (`WamState::head_constant`). Register-only was correct when a lowered function
  was the top-level entry (the caller reads the register), but wrong as a
  subroutine, where the caller passed its own variable expecting it bound — a
  fact-shaped `clause_chain` returned its result into a register the caller
  never read. This matches the interpreter's `GetConstant` exactly.

**Measurement (2a).** A det-heavy synthetic (a list folded through a
deterministic transform chain `step -> scale/bias/clampv`, driven by a
multi-clause `runl`), 50,000 elements, best of 4, this box:

| build | resolve ms |
| --- | ---: |
| interpreter (dispatch off) | ~1,010 |
| dispatched, with the soundness snapshot guard | ~1,000 |
| dispatched, guard removed (unsound, to price it) | ~965 |

Dispatch itself buys ~4% (1,010 → 965), but the per-call rollback snapshot
(`save_regs`, a clone of the live register file) costs ~3.5%, so the *sound*
tier nets roughly neutral on this workload. The honest reading: on the sound
intermediate, the hot code — the multi-clause recursion drivers — cannot be
dispatched, so the win is bounded by how much of the work is in cp-clean
leaves. The lever the profile actually points at (running the *nondeterministic*
predicates as native code) still needs item 2. Probes:
`tests/test_wam_rust_lowered_dispatch.pl` (dispatched answers == interpreted ==
SWI on every probe, including a `member/2` enumerator that must keep all three
solutions — proof the hook does not commit a nondet predicate).

### Item 2b (resumable nondeterminism): not attempted this round — verdict

Deferred deliberately, not started, given the round's budget and the priority
order (item 1 unify was the load-bearing deliverable). The design is scoped and
its precedent is solid (mprolog F3: a choice point is a saved resume label +
frame slot, the second solution is a JUMP not a re-call; the portable analogue
is a state-machine enum driven by a trampoline, resumed by restoring saved
locals and matching to the right arm). Two obstructions are already visible from
2a's work and should frame the 2b round:

1. **The cross-call convention.** 2a's lowered functions call each other through
   the *interpreter* (`run()`), which is what forced the cp-clean restriction. A
   resumable tier must make lowered-to-lowered calls direct (a
   `lowered_call`-style resume entry, not `run()`), or nondet callees will keep
   leaking choice points into their callers. mprolog's own compiler is the
   negative example here (F3: it goes zero-overhead nondet-to-nondet but pays
   full dispatch nondet-to-det).
2. **The snapshot cost.** 2a already shows the soundness snapshot (`save_regs`)
   costs about what dispatch saves. A resumable protocol pushes a choice point
   *per re-entry*; unless the saved-locals set is much smaller than the whole
   register file, the per-solution cost will not beat the interpreter's own
   choice-point machinery, which is already the thing being measured at 21%
   backtrack + 12% restore_regs. 2b must measure per-solution cost against that
   baseline before it lands, on the cut suite's sequence-oracle style (order +
   multiplicity + cut interaction, §9).

Recommendation: 2b is worth a dedicated round, but only with a cheaper
saved-state representation than `save_regs` and a direct lowered-to-lowered
resume path — otherwise it repeats 2a's neutral result at higher risk.

### Q2 (Rust half): is self-tail-recursion compiled to a loop today?

No — it is net-new. The emitter has no tail-recursion class. `wam_rust_lowerable`
classifies a predicate as `deterministic`, `clause_chain`, `multi_clause_n`,
`multi_clause_1`, or `ite_lowered`; none of these detects that a clause's last
goal is a self-call and rewrites it into a loop. A self-recursive predicate like
the resolver's list walkers lowers as `multi_clause_n` (each clause tried in
order, the recursive clause emitting a real `execute`/`call` that re-enters
through the interpreter or, post-2a, through a fresh function activation) — it
does not become mprolog's `tail` class (assign args once, `loopN:` label,
reassign registers, `goto loopN`, zero choice-point machinery, F11). Adding it
would be a new front-end class (`independ_head`-style check that the head has no
repeated argument variable, plus "single recursive clause, recursion in tail
position") emitting a `loop { }` in the lowered function; it is independent of
2b (a tail-recursive single-solution-path predicate has no choice point to
resume, so it is safe under the *existing* first-solution tier) and is the
lowest-risk next lowered-tier step.

## Path forward

0. **Make the lowered tier reachable** — the three-step scoping above.
   **Step 1 (the `lowered_call` call-site hook) and the sound intermediate
   LANDED 2026-09-05** (measured ~neutral; see the lowered-tier section). What
   remains: the resumable multi-clause protocol (item 2b — deferred with an
   obstruction write-up, needs a cheaper saved-state rep + direct
   lowered-to-lowered calls), a self-tail-recursion→loop class (Q2, net-new,
   lowest-risk next step), and ITE per clause + `(A ; B)`.
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
2026-09-05: the D59 `unify` quadratic **fixed** with shape-first dispatch (three
D59 probes go linear; the G1 index unblocked on the Rust lane, curve linear);
the D55 sound-intermediate lowered-tier dispatch (`lowered_call` hook) **landed**
and measured ~neutral, with a cp-clean eligibility fixpoint and a latent
`head_constant` subroutine-binding fix; item 2b (resumable nondeterminism)
deferred with an obstruction write-up; Q2 (self-tail-recursion loop) answered —
net-new. New probes: `test_wam_rust_unify_shape.pl` (9),
`test_wam_rust_lowered_dispatch.pl`.
