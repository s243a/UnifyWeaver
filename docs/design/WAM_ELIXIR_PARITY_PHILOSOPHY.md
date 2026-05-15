# WAM Elixir Parity — Philosophy

How we plan to close the feature gap between `wam_elixir_target.pl`
and the C++/Rust/Haskell WAM targets, in what order, and why. This
doc is the rationale; the concrete inventory and per-PR phasing
live in `WAM_ELIXIR_GAPS_SPECIFICATION.md`.

Companion docs we lean on:
- `WAM_CPP_ISO_ERRORS_PHILOSOPHY.md` — the three-form ISO design
  Elixir will mirror.
- `WAM_ITEMS_API_PHILOSOPHY.md` — the cross-cutting refactor that
  may or may not land before the Elixir migration.
- `WAM_CROSS_TARGET_BENCHMARK_RESULTS.md` — the existing per-target
  benchmark comparison this work plugs into.

Related Elixir-specific docs (different scope):
- `WAM_ELIXIR_CORRECTNESS_GAPS.md` — running list of bug fixes
  found during benchmarking. About correctness issues in the
  *existing* Elixir target, not feature parity.
- `WAM_ELIXIR_PERF_PHASE_A_PLAN.md` — container/data-structure
  perf work (heap/trail/code as Map/tuple) to match Haskell/Rust
  baselines. Orthogonal to feature parity.

## 1. Where Elixir stands today

`wam_elixir_target.pl` is ~5000 LOC and ships:

- Core WAM: facts, head/body unification, indexing, `try_me_else` /
  `retry_me_else` / `trust_me`, allocate/deallocate, register
  binding, trail and choice points.
- Arithmetic: `is/2`, `>/2`, `</2`, `>=/2`, `=</2`, `=:=/2`.
- Negation as failure (`\+/1`, `not/1`) and a partial `findall/3`.
- BEAM-idiomatic dispatch: `step(state, instr)` returns
  `{:ok, state} | :fail`, with BEAM `try/catch` already used
  internally for `:fail` propagation across choice-point retry
  (see `wam_elixir_target.pl:423-426`).

What it lacks, vs the C++ reference:

- `catch/3` and `throw/1` — Prolog-level exception handling.
- ISO error machinery: `is_iso/2` / `is_lax/2`, the iso/lax
  variants of arithmetic compares, `succ_iso/2`, the per-predicate
  rewrite hook, the config loader, the audit predicate.
- A full meta-call surface: `call/N`, goal-term dispatch (`,/2`,
  `;/2`, `->/2` as data passed to `catch` / `findall`),
  `bagof/3`, `setof/3`.
- Items API consumption — Elixir is one of 11 targets that still
  parses WAM text.

The Rust and Haskell targets sit roughly where Elixir does on
these axes. C++ pulled ahead through the recent ISO + meta-call
PR series (#2079 → #2088 → #2106 → #2112). Closing the Elixir gap
puts it level with C++ and ahead of Rust/Haskell on exception
semantics.

## 2. Why now, and why Elixir

Three forces converging:

- **The ISO error stack landed cleanly for C++.** That work
  established a reusable shape — three-form keys (default / iso /
  lax), per-predicate config-driven rewrite, audit predicate. The
  next target to adopt this pattern proves it generalises and gives
  the design a second consumer that informs whether anything in the
  C++ shape was accidentally C++-specific.
- **Elixir is the natural second adopter.** Rust and Haskell
  haven't received recent feature investment; their generators are
  smaller and could fall further behind without surfacing pressure
  to cross-pollinate. Elixir is the most-actively-developed of the
  non-C++ targets (recent benchmark work on int-tuple storage,
  LMDB benchmarks, large-scale preprocess) and gets the most user
  attention, so parity gaps there bite hardest.
- **The cost of waiting compounds.** Each new ISO-aware builtin
  the C++ target adds is one more thing the Elixir target won't
  understand. The ISO key tables are static facts in the generator,
  copy-pasteable. The runtime body is the work. Easier to do the
  ports while the C++ shape is fresh than after another six months
  of C++-only feature additions.

## 3. What "parity" means here, and what it doesn't

**Parity means feature parity at the generator + runtime layer.**
A user writing `catch(Goal, error(type_error(_, _), _), Handler)`
should get the same observable behaviour from the Elixir-compiled
binary as from the C++-compiled one. Same with `is_iso/2`,
`succ_iso/2`, ISO arith compares.

**Parity does not mean runtime performance parity with Rust or
Haskell.** Three reasons we explicitly disclaim that goal:

- **Three different runtimes.** Elixir compiles to BEAM bytecode
  with JIT (OTP 26+). Rust is native AOT through LLVM. Haskell is
  native AOT through GHC with lazy evaluation and a different GC.
  The same WAM interpreter design lands on three completely
  different baselines.
- **The existing benchmark data already shows non-monotone gaps.**
  `WAM_CROSS_TARGET_BENCHMARK_RESULTS.md` shows Rust's atom
  interning gave a 7.9× speedup on the same WAM design that
  Haskell uses, and F# matched Rust on query time but lost
  80–185 ms to .NET startup. Treating any one runtime's number as
  a target the others must hit pushes us toward the wrong
  optimisations.
- **Cross-runtime tuning is an anti-goal.** Optimising Elixir to
  match Rust's L1-cache-hit dispatch loop would mean writing
  un-idiomatic BEAM code. The right comparison is
  Elixir-before-PR vs Elixir-after-PR — the delta of an individual
  change against its own baseline.

What we *do* commit to on perf: each new feature has a stated
cost on three call sites — happy path (no catch active, default
mode), feature-active path (catch frame on stack, no throw),
exceptional path (throw triggered). Happy path must be
byte-identical to today's dispatch. Feature-active path is
quantified per-PR. The methodology is in
`WAM_ELIXIR_GAPS_SPECIFICATION.md` §6.

## 4. Why catch/throw goes first

Six features in scope. The ordering isn't arbitrary.

`catch/3` + `throw/1` is the foundation. Without it:
- ISO errors have nowhere to throw to. `is_iso/2` would throw an
  `error/2` term and immediately crash the runtime instead of
  unwinding to a user catcher.
- The audit predicate has no story for "this site would change
  behaviour on flip" because both behaviours collapse to the same
  `:fail` return.
- Test coverage for ISO mode requires user code that catches and
  inspects the thrown term. Without `catch/3`, every ISO test is
  "binary crashed, that's all we can say."

In the C++ series, `catch/3` + `throw/1` shipped *before* the ISO
plumbing PR (commits `151c0178` → `f7d2c932` → `32567157` →
`0dda9d1b`) for exactly this reason. We mirror the order.

**Why not Items API first?** The Items API would simplify the
Elixir migration (no per-target text parser to maintain), but
Phase 1 hasn't landed in `wam_target.pl` yet — it's a ~1100 LOC
central refactor that's its own design decision. Bundling it with
the Elixir parity work would conflate two concerns and force
this work to wait on Phase 1's review cycle. Items API can land
in parallel; the Elixir migration to items becomes a follow-up
PR after both this work and Phase 1 are done.

**Why not the runtime parser first?** The runtime parser
(`prolog_term_parser.pl` transpiled to the target) only matters
when a target exposes `read/2` / `read_term_from_atom/2,3`. Elixir
doesn't expose those today. R is the only target that needs the
runtime parser; Elixir can adopt it later when its stdlib surface
grows.

## 5. The BEAM-native option for catch/throw

C++ implements `catch/3` via a side-stack `CatcherFrame` struct
walked manually, because C++'s native exceptions don't know about
the WAM trail / choice-point stack and would corrupt VM state if
allowed to unwind directly.

Elixir lives in a different runtime. BEAM has built-in
`throw/raise` + `try/catch/rescue` that the runtime's stack
already knows about. Today's Elixir target *already uses* this
internally for the `:fail` propagation pattern
(`wam_elixir_target.pl:423-426`):

```elixir
try do
  retry_clause(state, ...)
catch
  {:fail, thrown_state} -> backtrack(thrown_state)
end
```

Two options for `catch/3`:

- **Option A — Mirror C++ exactly:** side-stack
  `catcher_frames` list in the WAM state, manual unwinding on
  throw, synthetic `:catch_return` instruction. Most semantically
  predictable; the C++ design carries straight over.
- **Option B — Lift to BEAM `throw`:** raise a BEAM-level value
  (e.g. `{:wam_throw, prolog_term, frame_id}`) on Prolog `throw/1`,
  catch it in the dispatch loop, walk the side-stack catcher
  frames to find a match. State management still ours; *unwinding
  mechanics* delegated to BEAM.

The right answer is "build A first, measure, decide if B is worth
it." Option A is straightforward to implement (translate the C++
struct + helpers); Option B is an optimisation that may or may not
pay off given BEAM's exception cost.

We commit to Option A in the first PR and reserve Option B as a
follow-up if perf measurements warrant it. The decision becomes
informed once the catch-active hot-path overhead is measurable.

## 6. Cross-thread interactions

Two pieces of cross-cutting work could intersect with the parity
PRs.

- **Items API.** If Phase 1 lands in `wam_target.pl` before we're
  done, the parity PRs benefit: builtin rewrites become a typed
  `swap_key_in_item/3` walk over a structured items list instead
  of multi-shape text matching. Each ISO PR shrinks ~30%.
  Conversely, if the parity work finishes first, the eventual
  Elixir Items API migration (Phase 2 #6 in the migration plan)
  benefits from cleaner ISO-aware dispatch. Either order works.
- **Runtime parser.** Independent of this work; only relevant if
  Elixir grows `read/2` later.

We don't gate on the Items API. If it lands mid-stream, the
remaining PRs adopt the items-list shape; otherwise, parity
finishes first and Items API becomes its follow-up.

## 7. What's intentionally out of scope

- **Re-architecting the Elixir target.** No GenServer-based
  parallelism, no clustering, no hot-code-reload. This is a
  feature-coverage exercise, not a runtime redesign.
- **`bagof/3` and `setof/3`.** Phase 4 of the C++ series shipped
  these, but they sit on top of `findall/3` + meta-call dispatch
  + witness grouping. Worth a separate scoping pass after the
  catch/throw + ISO work lands. Not in the initial scope.
- **Performance tuning before correctness.** Each PR ships
  measured cost but doesn't optimise it. Optimisation passes
  are separate PRs after all features land, so we tune from a
  complete baseline.
- **Cross-target perf comparison as a success metric.** See §3.
  The bench surface is for catching regressions in our own
  baseline, not racing other targets.

## 8. Risks and mitigations

- **Risk:** The C++ catch/throw infrastructure is intertwined with
  the broader meta-call system (`\+/1`, `findall/3`, conj/disj/
  if-then-else as goal-terms, aggregate group iteration). Eight
  synthetic ops co-evolved. Lifting `catch/3` cleanly requires
  identifying which parts are catch-only vs shared.
  **Mitigation:** Phase 1 PR scope is `catch/3` + `throw/1` plus
  whatever subset of the meta-call surface those two require.
  `\+/1` already exists in Elixir, so the negation half is
  partially done. `call/N` may need to ship as part of the same
  PR if `catch/3`'s goal dispatch needs it.

- **Risk:** Elixir uses BEAM `throw` internally for `:fail`. A
  Prolog-level `throw/1` lifted to BEAM `throw` (option B in §5)
  could collide with internal control-flow throws.
  **Mitigation:** Distinct tag. Internal throws use `{:fail,
  state}`; user throws would use `{:wam_throw, term, frame_id}`.
  The catch arms that handle `:fail` won't match `:wam_throw`,
  so the two paths stay separate. We prove this in a unit test
  before committing to option B.

- **Risk:** BEAM JIT (OTP 26+) optimises hot paths differently
  than the AOT compilers Rust/Haskell use. A change that's
  free on Rust's hot path could regress Elixir's.
  **Mitigation:** Bench-before / bench-after gating per PR
  (§ specifications doc 6). We measure on Elixir, not on
  C++/Rust/Haskell.

## 9. Open questions

Flagged for the implementation PR review:

- **Should the audit predicate be Elixir-specific or shared?**
  C++ ships `wam_cpp_iso_audit/3`. The shape is generic enough
  that we could extract a shared `wam_iso_audit/4` taking a
  target-specific config. Open until the second adopter.
- **Should ISO arithmetic compares share runtime bodies between
  default and `_lax`?** C++ does this (one body, two dispatch
  keys). Worth confirming the same trick works on BEAM where
  function clauses have their own pattern-match cost.
- **`Context` slot of `error/2` — bind to a fresh var (C++ v1
  choice) or to the predicate indicator?** C++ chose unbound for
  simplicity. We can match or improve when implementing.
