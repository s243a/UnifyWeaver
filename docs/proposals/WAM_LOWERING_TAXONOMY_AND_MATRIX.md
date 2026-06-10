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
| scala   | ✓ | ✓ T2a | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ |
| rust    | ✓ | ✓ T2a | ✓ | ✓ | ✓ | `~` gated | ✗ | ✓ | ✗ | ✗ | ✗ |
| cpp     | ✓ | ✓ T2a | ✓ | ✓ | ✓ | `~` gated | ✗ | ✗ | ✗ | ✗ | ✗ |
| go      | ✓ | ✓ T2a | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |
| haskell | ✓ | ✓ T2a | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ |
| fsharp  | ✓ | ✓ T2a | ✓ | ✓ | ✓ | ✗ | ✗ | ~ | ✗ | ✗ | ✗ |
| clojure | ✓ | ✓ T2a | ✓ | ✓ | ✗ | ✗ | ~ | ~ | ✗ | ✗ | ✗ |
| llvm    | ✓ | ✓ T2a | ✓ (c1) | ✓ | ✓ | ✗ | ✗ | ~ | ✗ | ✗ | ~ |
| lua     | ✓ | ✓ T2a | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |
| python  | ✓ | ✓ T2a | ✗ | ✗ | **✓** | ✗ | ~ | ~ | ✗ | ✗ | ✗ |
| r       | ✓ | ✓ T2a | ✓ | **✓** | ✗ | ✗ | ✗ | ~ | ✓ | **✓** | ✗ |
| elixir  | ✓ | ✓ T2b | ✓ | ✓ | ✗ | ✗ | **✓** | ✓ | ✓ | ✗ | ✗ |
| wat     | ✓ | **✗** | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

Verification notes:
- **T3** confirmed ✓ for haskell and fsharp (both lower clause 1 and fall
  back to the interpreter for clauses 2+ — documented in their lowerability
  gates). **python T3 = ✗**: `is_deterministic_pred_py` rejects *any*
  `try_me_else`, so a try-chain predicate stays in the interpreter; python's
  only multi-clause lowering is T5 (its `is_ite_block_py` detection of the
  switch-indexed, no-`try_me_else` shape → `if/elif/else`).
- **T5** (clause_chain) is now implemented across the hybrid targets via the
  shared `wam_clause_chain` front-end: scala, rust, cpp, go, haskell, fsharp,
  llvm, lua, r (plus python's original `is_ite_block_py` `if/elif/else` form).
  clojure declines T5 (no distinct-first-arg dispatch; it has T4 instead);
  elixir/wat = ✗. Each has a gated `*_lowered_t5` exec test.
- **T7**: elixir is the only real implementation (`Task.async_stream` +
  `par_wrap_segment`); clojure/python have `_branch` scaffolds (counted `~`).
  go's clause-parallel goroutines live in the *non-WAM* `go_target.pl` direct
  compiler, not the WAM lowered emitter → go T7 = ✗ here.
- **T9**: rust and scala's lowered *emitters* emit no fact tables; scala's
  ✓ is the target-level fact-source backend (auto-inline ≤128 rows, then
  CSV/TSV/LMDB) — a different mechanism than lua/r/haskell's emitter-level
  inline tables, but it is fact-table inlining, so ✓. rust = ✗.
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
  scala, rust, cpp, go, haskell, fsharp, clojure, llvm, lua (plus r and
  elixir's prior CP-model versions). Every clause is lowered inline and tried
  in order; the interpreter is never entered for the predicate's own clause
  dispatch. Two families:
  - **imperative** (scala/rust/cpp/go/lua/llvm): snapshot the registers +
    trail at entry, restore between clause attempts (e.g. `loRestoreClause`,
    `ClauseSnapshot`, a `[64 x %Value]` memcpy in LLVM IR). These runtimes
    take a predicate's first solution (deterministic-prefix), so — unlike
    R/elixir — no retry/iter choice point is needed.
  - **functional** (haskell/fsharp): immutability gives a free per-clause
    restore — each clause runs against the unchanged input state, chained
    with `mplus` / `Option.orElseWith`; no runtime change.
  clojure (immutable state-maps) gained its first multi-clause lowering here
  (previously a no-op stub → interpreter). Each has a gated `*_lowered_t4`
  exec test using a non-distinct-first-arg predicate that exercises the
  non-first clauses natively. T4 is gated below T5 (clause_chain) and above
  multi_clause_1/c1.

---

## 3. What the matrix shows (gaps)

Reading down the columns (after the T5 and T4 sweeps landed):

- **T5 (multi-clause → first-arg dispatch)** and **T4 (multi-clause all
  clauses)** are now ✓ across the hybrid targets (see the verification notes
  above). Remaining T4/T5 holes are intentional: clojure has no distinct
  first-arg dispatch (T4 only); python's multi-clause story is its T5
  `if/elif/else`; wat has neither yet.
- **T6 (first-arg indexing)** — **Rust and C++** now have a *gated* T6 (`~`):
  both reuse the T5 `wam_clause_chain` front-end, but when the discriminators
  are all atoms and there are ≥ `t6_min_clauses` of them (default 8) the
  back-end replaces the if-cascade with a native indexed dispatch — Rust a
  two-stage `match` (string switch → integer jump table); C++ a static
  `std::unordered_map<std::string,int>` (no native string switch) → `switch`.
  The other targets still drop the `switch_on_*` prefix and try clauses in
  order. This is the natural next advancement after T5 (same clause-head
  analysis, switch back-end): shared front-end, per-target back-end — high
  leverage, and a perf win that's sound behind a gate. Measured on generated
  code (tie at 4 clauses; then growing with N): Rust 1.55× at 8, 5.7× at 64,
  12.7× at 256; C++ 2.1× at 8, 11.6× at 64, 40.8× at 256 (see
  `docs/reports/wam_rust_dispatch_alloc_perf.md`). The compiler flattens the
  cascade for few clauses (matching the prior Go array-dispatch result), hence
  the gate. Int-interned targets (go/llvm/haskell/scala/lua) are the remaining
  candidates but carry the highest "lost to the compiler" risk (an int
  equality chain is already switch-converted), so each needs a per-target
  benchmark before shipping.
- **T2 (ITE)** — complete everywhere **except WAT** (the one remaining ITE
  gap; WAT has native `if/then/else` + `block`, so it's tractable). Closing
  it makes the T2 column fully ✓.
- **python T3** — python still lacks the clause-1 fast path (its only
  multi-clause lowering is T5); a small, contained gap.
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
2. **WAT T2 (ITE).** Closes the last ITE cell; small, gated, exec-testable
   (`wat2wasm` + `node`). Finishes the T2 column.
3. **T4 (multi-clause all-clauses)** for the structurer targets, reusing R's
   iter-CP shape as the reference. Removes the interpreter hop for fully
   supported predicates that aren't first-arg-discriminable (so don't get
   T5).
4. **T6 (first-arg indexing)** — same clause-head front-end as T5 with a
   `switch` back-end instead of an `->` cascade.
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
