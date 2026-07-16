# WAM Lowering Taxonomy & Target Matrix

**Status:** working planning doc. Matrix cells for T1–T7, T9–T11 were verified
by reading each emitter's lowerability gate and emit code; the T8 (native
kernels) column is roadmap-derived and approximate (it's a curated library
feature, orthogonal to the generic-lowering plan). `~` marks a partial,
scaffold, or architecturally-distinct implementation (see the per-cell notes).

**Purpose.** The hybrid WAM targets each ship a "lowered" path: instead of
running every predicate through the bytecode step-loop interpreter, certain
predicate shapes are compiled to native per-predicate functions. Over time
each target grew a *different subset* of lowering strategies. This document
(1) names the strategies (a shared taxonomy), (2) records which target
implements each, and (3) gives a method for choosing which gaps to close.

---

## 1. The lowering types

Each "type" is a recognisable predicate/clause shape plus the native code
form it lowers to. They are roughly ordered from simplest to most advanced.

### T1 — Deterministic single-clause
One clause, no choice points inside it. Emitted as one native function that
inlines the instructions; on the last `proceed` it returns success. This is
the baseline every lowered emitter started from.

### T2 — If-then-else / negation / once  (`( C -> T ; E )`, `\+ G`, `once/1`)
The compiler lowers these to a soft-cut block
(`try_me_else … <cond> cut_ite <then> jump … ; <else>`). Two realisations:
- **T2a structurer** — fold the block into `ite(Cond,Then,Else)` via the
  shared `wam_ite_structurer` and emit native `if/else` with a trail
  rollback before the else. (Go, Rust, C++, Haskell, F#, Clojure, LLVM,
  Lua, Python, R.)
- **T2b choice-point** — keep the soft cut as a real choice point and
  implement `cut_ite`/`jump`/fall-through directly. (Elixir.)

### T3 — Multi-clause, clause-1 fast path  (`multi_clause_1` / `multi_clause_c1`)
Lower **clause 1** inline; on failure push a choice point and fall back to
the bytecode interpreter for clauses 2..n. Cheap win for first-arg-indexed
predicates that usually match clause 1.

### T4 — Multi-clause, all clauses  (`multi_clause_n`)
Lower **every** clause inline as sibling closures with an iter-style retry
CP; the interpreter is never entered for the predicate. Strictly more than
T3 (no interpreter hop on later clauses).

### T5 — Multi-clause as an if-then-else chain
Turn first-argument clause dispatch into a single `->` chain:
`p(a):-B1.  p(b):-B2.`  →  `( A1=a -> B1 ; A1=b -> B2 ; … )`.
Distinct from T3/T4 (which keep choice-point/closure-per-clause shape): the
clause heads become guards in one `->` cascade, which a host `if/elif/else`
maps onto directly. (Python's `is_ite_block_py` detection is this.)
**This is the "`->` form" suggested for Scala but not yet built there.**

### T6 — First-argument indexing
Native `switch`/dispatch on the first argument's principal functor so the
lowered entry jumps straight to the matching clause instead of trying
clauses in order. (The bytecode path uses `switch_on_constant/structure`;
lowered emitters currently *drop* those prefixes.)

### T7 — Parallel / Tier-2
Fan out independent work — `findall`/aggregate solutions, or independent
clause branches — across threads. Elixir uses `Task.async_stream`; the
generated `_branch` clause variants are the substrate.

### T8 — Native graph kernels
Hand-written specialised native code for whole recursion *patterns*
(`transitive_closure`, `category_ancestor`, `weighted_shortest_path`,
`astar_…`), opt-in via `kernel_dispatch(true)`. Not a generic lowering —
a curated fast path for the kernel library.

### T9 — Fact-table inline
Ground unit-clause predicates compiled to a data table (array/map) rather
than instruction sequences, with the lookup inlined.

### T10 — Mode-driven specialisation
Use `:- mode/1` + binding-state analysis to specialise head-match
instructions (e.g. inline `get_constant`/`get_value`/`is` when the target
register is provably bound, skipping the deref/unify dispatch). (R.)

### T11 — Last-call / tail-call optimisation (LCO)
Compile self-/mutual recursion in tail position to iteration (loop or
guaranteed tail call) instead of host recursion, bounding stack growth.
LLVM uses `musttail` for `execute` into lowered kernels; otherwise largely
unexplored.

---

## 2. Target × type matrix

Legend: ✓ present (verified in code) · `~` partial / scaffold / distinct
mechanism · ✗ absent. (T1–T7, T9–T11 verified by reading each emitter's
lowerability gate + emit; T8 depth is roadmap-derived — see notes.)

| Target  | T1 det | T2 ITE | T3 mc-1 | T4 mc-n | T5 mc→`->` | T6 idx | T7 par | T8 kernels | T9 facts | T10 mode | T11 LCO |
|---------|:------:|:------:|:-------:|:-------:|:----------:|:------:|:------:|:----------:|:--------:|:--------:|:-------:|
| scala   | ✓ | ✓ T2a | ✓ | ✓ | ✓ | `~` gated | ✗ | ✓ | ✓ | ✗ | ✗ |
| rust    | ✓ | ✓ T2a | ✓ | ✓ | ✓ | `~` gated | `~` gated | ✓ | ✓ capped | ✗ | ✗ |
| cpp     | ✓ | ✓ T2a | ✓ | ✓ | ✓ | `~` gated | ✗ | ✗ | ✗ | ✗ | ✗ |
| c       | ✓ | ✓ T2a | ✓ | ✓ | ✓ | `~` gated | ✗ | ✓ | ✓ capped | ✗ | ✗ |
| go      | ✓ | ✓ T2a | ✓ | ✓ | ✓ | `~` gated | ✗ | ✓ | ✗ | ✗ | ✗ |
| haskell | ✓ | ✓ T2a | ✓ | ✓ | ✓ | `~` gated | ✗ | ✓ | ✓ | ✗ | ✗ |
| fsharp  | ✓ | ✓ T2a | ✓ | ✓ | ✓ | `~` gated | ✗ | ~ | ✓ capped | ✗ | ✗ |
| clojure | ✓ | ✓ T2a | ✓ | ✓ | ✗ | ✗ | ~ | ~ | ✗ | ✗ | ✗ |
| llvm    | ✓ | ✓ T2a | ✓ (c1) | ✓ | ✓ | ✗ | ✗ | ~ | ✗ | ✗ | ~ |
| lua     | ✓ | ✓ T2a | ✓ | ✓ | ✓ | `~` gated | ✗ | ✗ | ✓ | ✗ | ✗ |
| python  | ✓ | ✓ T2a | ✓ | `~` hybrid | ✓ | `~` gated | ~ | ~ | ✗ | ✗ | ✗ |
| r       | ✓ | ✓ T2a | ✓ | **✓** | ✗ | ✗ | ✗ | ~ | ✓ | **✓** | ✗ |
| elixir  | ✓ | ✓ T2b | ✓ | ✓ | ✗ | ✗ | **✓** | ✓ | ✓ | ✗ | ✗ |
| wat     | ✓ | ✓ T2a | ✓ | ✓ | ✓ | `~` gated | ✗ | ✗ | ✗ | ✗ | ✗ |

Verification notes:
- **T3** confirmed ✓ for haskell and fsharp (both lower clause 1 and fall
  back to the interpreter for clauses 2+ — documented in their lowerability
  gates). **python T3 = ✓** (now): `py_multi_clause_1` extracts clause 1 of a
  multi-clause predicate to a lowered `pred_*` function; the registrar keeps
  the FULL bytecode but replaces clause 1's body with a `call_lowered`,
  retaining the leading `try_me_else` and clauses 2+ verbatim. On clause-1
  success the `call_lowered` falls through to `proceed` (the clause-2 choice
  point is left for backtracking); on failure the runtime's `call_lowered`
  handler calls `fail()`, popping that choice point (restoring trail + regs)
  and resuming the interpreter at clause 2 — an emitter-only change, no runtime
  modification. Because a T3 `pred_*` is clause-1-only (not the whole
  predicate), a predicate that *directly* calls a T3 predicate is kept in the
  interpreter (`whole_predicate_lowering/3` gate), so the T3 fallback is always
  reached through a bytecode call. Python keeps T5 too (its `is_ite_block_py`
  `if/elif/else` for the switch-indexed, no-`try_me_else` shape).
- **T5** (clause_chain) is now implemented across the hybrid targets via the
  shared `wam_clause_chain` front-end: scala, rust, cpp, go, haskell, fsharp,
  llvm, lua, r (plus python's original `is_ite_block_py` `if/elif/else` form).
  clojure declines T5 (no distinct-first-arg dispatch; it has T4 instead);
  elixir = ✗. **wat T5 = ✓** (now): the WAT lowered emitter strips the
  switch_on_* indexing prefix, runs the same `wam_clause_chain` front-end, and
  emits one WAT function with an unbound-A1 guard followed by a `do_get_constant`
  test per discriminator (a pure test when A1 is bound) wrapping each clause
  body inline — first-solution, matching WAT's lowered model (the exported entry
  replays the interpreter on failure; an unbound A1 is deferred there too). Each
  has a gated `*_lowered_t5` exec test; WAT's `test_wam_wat_lowered_t5` injects
  test-only exports that bind argument registers via `do_put_constant` and call
  the lowered function directly (WAT exports take no parameters), exercising
  real per-discriminator dispatch including non-first clauses through wat2wasm +
  node.
- **T7**: elixir is the only real implementation (`Task.async_stream` +
  `par_wrap_segment`); clojure/python have `_branch` scaffolds (counted `~`).
  go's clause-parallel goroutines live in the *non-WAM* `go_target.pl` direct
  compiler, not the WAM lowered emitter → go T7 = ✗ here. **rust T7 = `~` (built,
  gated, whole-body aggregates)**: a forkable aggregate that is a predicate's
  whole body compiles — behind `parallel_aggregates(true)` and a compile-time
  cost gate — to a generator/body split (`parallel_aggregate_transform`) plus a
  native `par_collect` wrapper that runs the body on a cloned WAM machine per
  input across threads, reducing by type (collect/count/sum/max/min/bag/set). The
  gate is mandatory: fan-out is 2–3.4× on 4 cores for expensive/recursive
  per-branch work (**3.39× measured end-to-end**) but a 5–200× *regression* on
  cheap branches (each branch clones its own machine — the cost backtracking
  avoids), so only expensive/recursive tiers fan out. `~` not ✓ because
  aggregates *embedded in a larger clause body* still compile sequentially (the
  `par_aggregate` WAM-instruction route, not yet built). Design/benchmark/handoff:
  `docs/reports/wam_rust_t7_parallel_perf.md`,
  `docs/reports/wam_rust_t7_speedup_benchmark.md`,
  `docs/reports/wam_rust_t7_RESUME.md`. (Like the LLVM T6 decline, the benchmark
  is what gated the decision — here it said "yes, but only behind the probe.")
- **T9**: scala's ✓ is the target-level fact-source backend (auto-inline ≤128
  rows, then CSV/TSV/LMDB) — a different mechanism than lua/r/haskell's
  emitter-level inline tables, but it is fact-table inlining, so ✓. **rust = ✓
  (default, capped):** an all-ground-facts predicate whose row count is in the
  inline window `[t9_min_rows, t9_max_rows]` (defaults 64..256) lowers by default
  to a static `OnceLock` row table + first-arg hash index + choice-point
  enumerator (`fact_table_attempt`), callable via WAM `call`/`execute`. Below the
  min, the negligible-cost T4 inline is used; above the cap, inlining is declined
  with a warning recommending an external fact source (LMDB/TSV), since inlining
  large fact sets bloats compile time / the binary. Opt out with
  `fact_table_inline(false)`. Tests: `tests/test_wam_rust_fact_table_exec.pl`,
  `tests/test_wam_rust_fact_table_emit.pl`,
  `tests/test_wam_rust_fact_table_callsite_exec.pl`,
  `tests/test_wam_rust_fact_table_throughput.pl`; benchmark
  `docs/reports/wam_rust_t9_fact_table_benchmark.md`. **fsharp = ✓ (default,
  capped):** same policy, emitted as a lowered fact-table predicate registered in
  `loweredPredicates` (so `call`/`execute` reach it like any T4/T5/T6 lowering); a
  `factTableAttempt` enumerator leaves a `FactTableRetry` choice point per
  remaining row, mirroring `select/3`. Tests:
  `tests/test_wam_fsharp_fact_table_exec.pl` (query-mode matrix),
  `tests/test_wam_fsharp_fact_table_emit.pl`. **c = ✓ (default, capped):** C
  already had a deterministic static-row-table + first-arg bucket scan; T9 makes
  the in-window fact lowering *backtrackable* — the handler drives a shared
  `wam_fact_table_scan` that leaves a `WAM_FACT_TABLE_RETRY` choice point
  (a `WamFactTableFrame` side-stack + resume function mirroring the runtime's
  disjunction CP), so every matching row is enumerated; registered as a foreign
  predicate, so it is reachable as a query and from another predicate unchanged.
  Below `t9_min_rows` keeps the cheap deterministic scanner; opt out with
  `fact_table_inline(false)`; above `t9_max_rows` declines + warns. Tests:
  `tests/test_wam_c_fact_table_exec.pl` (query-mode matrix),
  `tests/test_wam_c_fact_table_emit.pl`.
- **T8** (native kernels) is a curated library feature dispatched via shared
  `kernel_dispatch` plumbing, not a generic per-predicate lowering. ✓ marks
  the roadmap's validated full-parity set (Rust / Haskell / Elixir / Go /
  Scala); `~` marks targets with kernel references of unconfirmed depth
  (fsharp, clojure, llvm, python, r); ✗ where none found (cpp, wat, lua).
  This column is approximate and orthogonal to the T1–T6 plan below.
- **T11**: llvm uses `musttail` for `execute` into lowered kernels (`~`);
  no target does general recursion→loop.
- elixir's T3/T4 use the choice-point model (genuine CPs + cut barrier), not
  R's closure-per-clause shape — counted ✓ but architecturally distinct.
- **T4** (multi_clause_n) is now implemented across the hybrid targets:
  scala, rust, cpp, go, haskell, fsharp, clojure, llvm, lua, wat (plus r and
  elixir's prior CP-model versions; python's `~` hybrid). Every clause is
  lowered inline and tried in order; the interpreter is never entered for the
  predicate's own clause dispatch. Two families:
  - **imperative** (scala/rust/cpp/go/lua/llvm/wat): snapshot the registers +
    trail at entry, restore between clause attempts (e.g. `loRestoreClause`,
    `ClauseSnapshot`, a `[64 x %Value]` memcpy in LLVM IR; wat snapshots each
    argument-register cell + the trail top into locals and `val_store`s them
    back between per-clause blocks). These runtimes take a predicate's first
    solution (deterministic-prefix), so — unlike R/elixir — no retry/iter
    choice point is needed. (wat was unblocked here by the unify
    scalar-propagation fix, which a clause body threading a head variable into a
    goal would otherwise hit.)
  - **functional** (haskell/fsharp): immutability gives a free per-clause
    restore — each clause runs against the unchanged input state, chained
    with `mplus` / `Option.orElseWith`; no runtime change.
  clojure (immutable state-maps) gained its first multi-clause lowering here
  (previously a no-op stub → interpreter). Each has a gated `*_lowered_t4`
  exec test using a non-distinct-first-arg predicate that exercises the
  non-first clauses natively. T4 is gated below T5 (clause_chain) and above
  multi_clause_1/c1.
- **python T4 = `~` (hybrid).** Python's runtime is a genuine backtracking WAM
  (choice points + `fail()`), and `run_wam` backtracks *intra-query*, so the
  imperative first-solution shape above would diverge from the interpreter on a
  conjunction like `p(X), q(X)` (it would commit to clause 1) — failing the
  lowered-vs-interpreter parity standard. Python therefore lowers every clause
  *body* to a native `pred_*_cK` function but RETAINS the try/retry/trust
  dispatch scaffold in the bytecode (`py_multi_clause_n` + `multi_clause_n_registrar`),
  replacing only each clause body with a `call_lowered`. The runtime's proven
  choice-point machinery drives clause dispatch and backtracking unchanged, so
  every clause body is native while parity is preserved by construction. It is
  marked `~` rather than ✓ because the T4 headline ("interpreter never entered
  for clause dispatch") is not met — the O(n) dispatch stays in the (tiny)
  bytecode scaffold. T4 is tried before T3 and falls back to it when a later
  clause cannot be lowered (e.g. ends in a tail-call `execute`). A future
  full-native dispatch (R's genuine-CP closure model over the runtime's
  callable choice points) could slot in *front* of this hybrid for shapes that
  admit it. Gated exec test `test_wam_python_lowered_t4` covers clause hits,
  the backtracking-critical conjunctions (`color(X), want_green(X)` must redo
  into a non-first clause), and a lowered-vs-interpreter parity battery.

---

## 3. What the matrix shows (gaps)

Reading down the columns (after the T5 and T4 sweeps landed):

- **T5 (multi-clause → first-arg dispatch)** and **T4 (multi-clause all
  clauses)** are now ✓ across the hybrid targets (see the verification notes
  above). Remaining T4/T5 holes are intentional: clojure has no distinct
  first-arg dispatch (T4 only); python now has T3, T4 (`~` hybrid: native
  clause bodies over a retained bytecode dispatch scaffold, to preserve its
  backtracking-runtime parity) and T5 `if/elif/else`; wat now has T5
  (first-arg clause-chain dispatch) AND T4 (all clauses inline with
  argument-register + trail snapshot/restore between attempts, first-solution).
  So **every multi-clause column (T3/T4/T5) is now closed** across the targets
  that support each shape — no plain ✗ remains in T3/T4/T5.
- **T6 (first-arg indexing)** — **Rust, C++, Go, F#, Haskell, Scala, Lua,
  Python and WAT** now have a *gated* T6 (`~`); **LLVM declines on benchmark
  evidence**. All reuse the T5 `wam_clause_chain` / first-arg-index front-end,
  but when there are ≥ `t6_min_clauses` clauses (default 8) the back-end
  replaces the if-cascade with a native indexed dispatch. Three families:
  - **Atom-keyed** (atoms compared as *strings* at dispatch, so a string switch
    is a real win the host compiler does not already perform): Rust a two-stage
    `match` (string switch → integer jump table); C++ a static
    `std::unordered_map<std::string,int>` (no native string switch) → `switch`;
    **Go a native `switch t6atom.Name`**; **F# a native many-branch `match` on
    the atom's string** (`Atom of string`, lowered by the F# compiler to a
    hash/jump dispatch). Measured on generated code (tie at 4 clauses; growing
    with N): Rust 1.55×/5.7×/12.7×, C++ 2.1×/11.6×/40.8×
    (`docs/reports/wam_rust_dispatch_alloc_perf.md`); **Go 4.8×/31.7×/58.8×**
    (`docs/reports/wam_go_dispatch_t6_perf.md`) — Go's `valueEquals` cascade is
    an interface-call chain the compiler does not rewrite, so the switch wins
    big. (NOTE: Go was previously mislabelled int-interned — its atoms are
    `&Atom{Name string}` interned by name, i.e. string-keyed; F# likewise keeps
    atoms as strings, which is why both took the atom-keyed back-end cleanly.)
  - **Int-interned** (atoms become integers at codegen, so the discriminator is
    an integer-equality chain the host compiler *might* already switch-convert —
    the "lost to the compiler" question): haskell/scala/lua/llvm, each
    benchmarked (`docs/reports/wam_int_interned_t6_perf.md`). Verdict: **haskell,
    scala and lua now have a gated T6 too** — a `case` on the interned id (GHC →
    jump table), a `match` on it (scalac → JVM `tableswitch`), and a hash table
    of per-clause closures built once (interpreted Lua). Measured wins: Lua
    1.7×/8.2×/29.7×, Haskell 1.4×/2.5×/4.5×, Scala 1.3×/3.1×/≫ at N=8/64/256.
    **LLVM declines on benchmark evidence**: at `-O2`, SimplifyCFG already turns
    an int-equality if-chain into a `switch` (the if-chain and an explicit switch
    compile to identical assembly), so an explicit T6 there is redundant — the
    one genuine "lost to the compiler" case.
  - **VM-dispatched** (the lowered path keeps a bytecode/data-table dispatch a
    runtime instruction services, rather than a host `switch`): **Python** turns
    the compiler's dropped first-arg index back into a real
    `("switch_on_constant", {key: label})` instruction (the runtime already
    jumps O(1) on it for a bound first arg, skips to the try/retry chain for an
    unbound one, fails for a no-match) — emitter-only, with a fresh clause-1 body
    label so every key resolves; benchmarked 1.8×/9.3×/35.6× (interpreted dict
    vs linear `isinstance`+`name==`). **WAT** atoms are sparse i64 hashes (no
    dense `br_table`), so its lowered function does a **binary search** on the
    sorted clause hashes (each leaf still runs `do_get_constant` for
    collision/tag safety); benchmarked in V8 1.16×/1.79×/6.72× — V8 does not
    flatten the linear i64 chain. Both gated, both in
    `docs/reports/wam_python_wat_t6_perf.md`.

  This is the natural next advancement after T5 (same clause-head analysis,
  switch back-end): shared front-end, per-target back-end. The compiler flattens
  the cascade for few clauses, hence the gate. **The T6 column is now closed
  out: every T5 target has a gated T6, except LLVM which declines on a measured
  "the optimiser already does it" basis.** (Testing Python T6 surfaced a
  deeper, T6-independent arithmetic bug, since **fixed for the interpreter**:
  `put_structure`/`put_list` unconditionally bound the var the target register
  previously held, so for `X is Expr` — where the compiler leaves the result var
  aliased into A1 while A2 is overwritten with the expression structure — the
  result target was clobbered and **every `X is Expr` with an unbound X failed**
  in the bytecode interpreter. Fix: bind only X-register sub-term slots
  (`reg > _A_MAX`, created by `set_variable`/`unify_variable` for nested terms
  like `error(type_error(..),..)`); A-register call-output slots are overwritten
  without binding. The *lowered* Python emit was then reconciled with the same
  model: `parse_wam_text_py` now parses `set_*` (previously dropped), the
  structure read/write emit uses the runtime's read/write-ctx helpers instead of
  the old heap-consecutive `state.s` model, and compounds are built/matched with
  the runtime functor naming (`"+/2"`, not `"+"`), so inline native clause bodies
  build and unify terms identically to the interpreter. `emit_mode(lowered)`
  arithmetic-rule clauses (e.g. an `is`-accumulator) and term construction
  (`X = f(g(1),h(2,3))`) now run correctly; covered by
  `tests/test_wam_python_is_binding.pl` in both modes.)
- **T2 (ITE)** — now **complete across every target**, including WAT: the
  lowered emitter folds the soft-cut block with the shared `wam_ite_structurer`
  and emits native WAT (`(block $ite_condK (result i32) …)` for the condition
  with a `br` on first failure, a saved trail mark, and `$unwind_trail` before
  the else). `test_wam_wat_lowered_t2` exec-tests branch selection / negation /
  once / sequential+nested ITE via wat2wasm+node, and asserts the lowered fast
  path matches the bytecode interpreter on every case. (A previously documented
  WAT-runtime limitation — a condition's variable binding not propagating into
  the then-branch — is now **FIXED**: `unify_addrs` bound a variable to a Ref
  into a transient argument-register cell, so the variable silently changed when
  a later goal reused that register, breaking *every* arithmetic comparison in
  the `X = V, X <cmp> K` guard pattern, on both paths — some forms even spun a
  Ref cycle and hung. The fix copies scalars by value into the variable's heap
  cell (only compounds/cons/unbound cells, which live on the heap, are bound as
  a Ref). `test_wam_wat_unify_propagation` guards it, and the `cond_bind` cases
  in `test_wam_wat_lowered_t2` are now absolute-correctness assertions.)
- **python T3** — **DONE.** Python now lowers a multi-clause predicate's
  clause 1 (fast path) with interpreter fallback for clauses 2+ (see the T3
  verification note above), so the T3 column is ✓ for every target that has a
  clause-1 fast path. `test_wam_python_lowered_t3` exec-tests clause-1 hit,
  clause-2/3 fallback, and a lowered-vs-interpreter parity battery.
- **T10 (mode-driven specialisation)** and **T11 (LCO)** are essentially
  one-target experiments (R, and LLVM's `musttail`) that could generalise.
- **T7 (parallel)** is real only in Elixir; Clojure/Python have `_branch`
  scaffolds that are unfinished.

---

## 4. How to choose gaps to close

Score each candidate gap on four axes, then sequence:

1. **Portability of the mechanism.** A form that reuses a shared analyser
   (like `wam_ite_structurer` did for T2) lands across many targets cheaply.
   T5 and T6 both have a natural *shared front-end* (clause-head analysis)
   with per-target back-ends — high leverage.
2. **Soundness risk.** Perf-only enablements behind a gate (fallback to the
   interpreter when the gate declines) are low risk — the whole ITE sweep
   was this. Forms that change execution semantics (T7 parallel, T11 LCO)
   are higher risk and need stronger tests.
3. **Reachable test coverage.** Prefer forms we can exec-test end-to-end on
   installed toolchains (the ITE sweep's per-backend 15-case harness is the
   template). T8 kernels and T7 parallel need heavier harnesses.
4. **Breadth impact.** Scala is the classic-programs breadth-anchor; T4/T5
   on Scala directly widen "how much Prolog the target accepts" without the
   interpreter hop.

### Suggested sequencing

1. **T5 (multi-clause → `->` chain) — Scala first.** This is the start
   (the original ask). Build a shared clause-head→guard-chain analyser
   (a front-end mirroring how `wam_ite_structurer` is shared for T2):
   input a multi-clause predicate's clauses, output an
   `ite(Guard,Body,Rest)` cascade when the heads are a clean
   first-argument discrimination; decline (fall back) otherwise. Wire the
   Scala lowered emitter's back-end first, validate with a per-backend exec
   test (the ITE sweep's 15-case harness is the template), then port the
   back-end to the other structurer targets. Python already has a back-end
   (`is_ite_block_py`) to cross-check the shared front-end against. Lands
   behind the existing `emit_mode(functions)` gate with interpreter
   fallback → low risk.
2. **WAT T2 (ITE). — DONE.** Closed the last ITE cell; gated, exec-tested
   (`wat2wasm` + `node`) with a lowered-vs-interpreter parity check. The T2
   column is now fully ✓.
3. **T4 (multi-clause all-clauses)** for the structurer targets, reusing R's
   iter-CP shape as the reference. Removes the interpreter hop for fully
   supported predicates that aren't first-arg-discriminable (so don't get
   T5). **python DONE** (`~` hybrid — native clause bodies over a retained
   bytecode dispatch scaffold, since python's backtracking runtime makes the
   first-solution imperative shape unsound; a full-native R-style dispatch
   could layer in front later). **wat DONE** too: its lowered emitter splits the
   try/retry/trust chain into per-clause WAT slices via
   `wat_multi_clause_n_lowerable`, emits each as an inline block that snapshots
   and restores the argument registers + trail between attempts (first-solution,
   the public entry replays the interpreter on a 0 return). The **T4 column is
   now complete across every target.** Verified through real `wat2wasm`+`node`
   exec (`test_wam_wat_lowered_t4`), including arithmetic `is/2` *assignment*
   bodies — WAT's `put_structure` does not have the aliased-result-register bug
   that affected the Python interpreter, so an `is`-assignment that binds a local
   or the head's own arg evaluates correctly in lowered code.
4. **T6 (first-arg indexing)** — same clause-head front-end as T5 with a
   `switch` back-end instead of an `->` cascade. **DONE / closed out for every
   T5 target:** the atom-keyed set (rust/cpp/fsharp/go, string switch), the
   int-interned set that wins (haskell/scala/lua), and the VM-dispatched set
   (python's runtime `switch_on_constant`; wat's binary search on sparse atom
   hashes). **llvm declines** on benchmark evidence (its `-O2` already converts
   the int if-chain to a switch). See `docs/reports/wam_int_interned_t6_perf.md`
   and `docs/reports/wam_python_wat_t6_perf.md`.
5. Treat **T7/T10/T11** as research spikes (one target each) before any
   sweep.

Relationship of T4/T5/T6: all three are "do something smarter with a
multi-clause predicate." A shared clause-shape classifier could pick per
predicate — T5/T6 when the heads cleanly discriminate on arg 1, T4
otherwise — so the back-ends implement a *chosen* strategy rather than each
re-deciding (see §5).

---

## 5. Open questions

- Should T4 and T5 be *alternatives* (a predicate is either an `->` chain or
  closure-per-clause) or *layered* (try T5 when first-arg-indexable, else
  T4)? A shared clause-shape classifier could pick per predicate.
- Is there appetite for a single `lowering_strategy/3` selector shared
  across targets (input: clause shape; output: chosen type), so the matrix
  becomes "which back-ends implement the chosen type" rather than each
  target re-deciding?
