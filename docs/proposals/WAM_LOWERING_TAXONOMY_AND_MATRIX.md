# WAM Lowering Taxonomy & Target Matrix

**Status:** working planning doc. The matrix cells were populated by surveying
the lowered emitters and targets (grep-level + spot reading), not exhaustive
per-cell verification — cells marked `~` or with `?` notes should be confirmed
against the code before committing to work on them.

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

Legend: ✓ implemented · `~` partial / scaffold / unverified · ✗ absent ·
n/a not applicable.

| Target  | T1 det | T2 ITE | T3 mc-1 | T4 mc-n | T5 mc→`->` | T6 idx | T7 par | T8 kernels | T9 facts | T10 mode | T11 LCO |
|---------|:------:|:------:|:-------:|:-------:|:----------:|:------:|:------:|:----------:|:--------:|:--------:|:-------:|
| scala   | ✓ | ✓ T2a | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ |
| rust    | ✓ | ✓ T2a | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | ~? | ✗ | ✗ |
| cpp     | ✓ | ✓ T2a | ✓ | ✗ | ✗ | ✗ | ✗ | ~? | ✗ | ✗ | ✗ |
| go      | ✓ | ✓ T2a | ✓ | ✗ | ✗ | ✗ | ~ clause-parallel | ✓ | ✗ | ✗ | ✗ |
| haskell | ✓ | ✓ T2a | ~? | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |
| fsharp  | ✓ | ✓ T2a | ~? | ✗ | ✗ | ✗ | ✗ | ~? | ✗ | ✗ | ✗ |
| clojure | ✓ | ✓ T2a | ✓ | ✗ | ✗ | ✗ | ~ `_branch` scaffold | ~? | ✗ | ✗ | ✗ |
| llvm    | ✓ | ✓ T2a | ✓ (c1) | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ~ `musttail` |
| lua     | ✓ | ✓ T2a | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ |
| python  | ✓ | ✓ T2a | ~? | ✗ | ✓ | ✗ | ~ `_branch` scaffold | ~? | ✗ | ✗ | ✗ |
| r       | ✓ | ✓ T2a | ✓ | **✓** | ✗ | ✗ | ✗ | ✓ | ✓ | **✓** | ✗ |
| elixir  | ✓ | ✓ T2b | ✓ (CP) | ✓ (CP) | ✗ | ✗ | **✓** | ✓ | ✓ | ✗ | ✗ |
| wat     | ✓ | **✗** | ✓ | ✗ | ✗ | ✗ | ✗ | ~? | ✗ | ✗ | ✗ |

Notes / things to verify before acting:
- T8 "kernels": `kernel_dispatch` plumbing appears in many targets, but full
  validated kernel parity (per `docs/WAM_TARGET_ROADMAP.md`) is Rust /
  Haskell / Elixir / Go / Scala. The `~?` cells are unconfirmed depth.
- T3 for haskell/fsharp/python marked `~?`: they lower clause 1 but the
  exact multi-clause-fallback shape wasn't re-verified here.
- elixir's T3/T4 are via the choice-point model (genuine CPs + cut barrier),
  not the closure-per-clause shape R uses — counted as ✓ but architecturally
  distinct.

---

## 3. What the matrix shows (gaps)

Reading down the columns:

- **T5 (multi-clause → `->` chain)** is implemented only by Python. This is
  the form flagged for Scala. It is a genuinely different lowering from
  T3/T4 and is portable to every structurer-style target.
- **T4 (multi-clause all-clauses)** exists only in R. Everyone else stops at
  T3 (clause-1 + interpreter fallback), so clauses 2..n always pay the
  interpreter hop.
- **T6 (first-arg indexing)** — nobody lowers it; all targets drop the
  `switch_on_*` prefix and try clauses in order.
- **T2 (ITE)** — complete everywhere **except WAT** (the one remaining ITE
  gap; WAT has native `if/then/else` + `block`, so it's tractable).
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

1. **WAT T2 (ITE).** Closes the last ITE cell; small, gated, exec-testable
   (`wat2wasm` + `node`). Finishes the column.
2. **T5 (multi-clause → `->` chain), shared front-end.** Build a shared
   clause-head→guard-chain analyser (mirroring the structurer), then wire
   Scala first (the original ask), then the other structurer targets. Each
   lands behind the existing `emit_mode(functions)` gate with interpreter
   fallback, so it's low-risk and reuses the 15-case exec harness pattern.
3. **T4 (multi-clause all-clauses)** for the structurer targets, reusing R's
   iter-CP shape as the reference. Removes the interpreter hop for fully
   supported predicates.
4. **T6 (first-arg indexing)** once T5's clause-head analyser exists — it's
   the same front-end with a `switch` back-end.
5. Treat **T7/T10/T11** as research spikes (one target each) before any
   sweep.

---

## 5. Open questions

- Should T4 and T5 be *alternatives* (a predicate is either an `->` chain or
  closure-per-clause) or *layered* (try T5 when first-arg-indexable, else
  T4)? A shared clause-shape classifier could pick per predicate.
- Is there appetite for a single `lowering_strategy/3` selector shared
  across targets (input: clause shape; output: chosen type), so the matrix
  becomes "which back-ends implement the chosen type" rather than each
  target re-deciding?
