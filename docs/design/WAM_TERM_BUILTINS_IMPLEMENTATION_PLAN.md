# WAM Term Builtins: Implementation Plan

This document records the phased rollout plan for adding the four Group A
term inspection builtins (`functor/3`, `arg/3`, `=../2`, `copy_term/2`) to
the WAM target layer and the three hybrid backends.

For the *why*, see `WAM_TERM_BUILTINS_PHILOSOPHY.md`. For the *what*, see
`WAM_TERM_BUILTINS_SPECIFICATION.md`.

The plan is structured to **fail fast** if the underlying hypothesis — that
these builtins are load-bearing for transpiling reach and possibly for
performance — turns out to be wrong. Phase 0 validates the premise before
any runtime code is written. Phase 2 implements in one target only before
committing to a multi-target rollout. Phase 6 explicitly tests the perf
hypothesis and is willing to report a negative result.

## 0. Starting point

As of 2026-04-10, none of the three hybrid WAM targets (WAM-WAT, WAM-Rust,
WAM-Haskell) implements any of `functor/3`, `arg/3`, `=../2`, `copy_term/2`,
`assert/1`, `retract/1`. The canonical `is_builtin_pred/2` table in
`wam_target.pl:606-619` does not list them either, which means predicates
using them currently emit `call` to a non-existent predicate and fail at
the WAM-generation layer (or fall through to host SWI-Prolog interop, the
slowest path).

The three most recent WAM targets' builtin sets are nearly identical:
cut, arithmetic, type checks, `\+/1`, list operations, I/O. Rust's
`category_ancestor` performance does *not* come from term inspection; it
comes from specialized WAM instructions bypassing the general dispatch.

## Phase 0 — Audit the premise

**Goal:** Before any runtime code is written, confirm that Group A builtins
are actually load-bearing for any Prolog code we care about.

**Tasks:**

1. Grep the full test corpus (`tests/`, `examples/`, any sample Prolog in
   `docs/`) for literal uses of `functor(`, `arg(`, `=..`, `copy_term(`.
2. For each hit, classify:
   - Is it in **generated** code (e.g. `wam_target.pl` uses `=..` internally
     for goal decomposition at compile time — not relevant)?
   - Is it in a **test predicate** that exercises Prolog-level features?
   - Is it in an **example** of user-facing Prolog that UnifyWeaver is
     supposed to transpile?
3. For each user-facing hit, determine what happens today: does the test
   pass via host SWI fallback? Does it skip? Does it error?
4. Write up findings in a short note, tentatively placed at
   `docs/design/WAM_TERM_BUILTINS_PHASE_0_AUDIT.md`, covering:
   - Total hits, classified
   - Which predicates are actively blocked on this feature
   - Whether the transpiling-reach argument survives first contact with
     the test corpus

**Exit criterion:** Either (a) at least one user-facing predicate is
demonstrably blocked by the gap, justifying Phases 1–6, or (b) the corpus
shows no real demand, in which case the plan is either shelved or narrowed
to "future-proofing only, lowest priority."

**Deliverable:** 1 commit — the audit note. No code changes.

## Phase 1 — Canonical WAM layer

**Goal:** Teach the WAM compiler (`wam_target.pl`) to recognize the four
builtins so it emits `builtin_call functor/3, 3` etc. instead of a `call`
to a non-existent predicate. No backend yet handles the new IDs; calls
will fail loudly at runtime with "unknown builtin", which is correct
intermediate behavior.

**Tasks:**

1. Extend `is_builtin_pred/2` in `wam_target.pl` (around line 619) with:
   ```prolog
   is_builtin_pred(functor, 3).
   is_builtin_pred(arg, 3).
   is_builtin_pred((=..), 2).
   is_builtin_pred(copy_term, 2).
   ```
2. Verify: run the existing test suite. Nothing should regress — the only
   change is that previously-unknown builtins now compile to `builtin_call`
   instructions, which backends will refuse with "unknown builtin ID" at
   runtime.
3. Decide whether any currently-passing tests now *fail* because a
   predicate was silently doing something wrong (e.g. routing to host
   SWI) and is now routed to a backend that doesn't support it. If so,
   mark those tests pending on Phase 2+ completion.

**Exit criterion:** `is_builtin_pred/2` lists all four; existing tests
either pass or are marked pending with a clear reason.

**Deliverable:** 1 commit — the `is_builtin_pred/2` extension plus a note
in the commit message about any tests that moved to pending.

## Phase 2 — WAM-WAT implementation (smallest blast radius)

**Goal:** Implement all four builtins in WAM-WAT first. This target has
the most context from recent work (PR #1224), the fewest downstream
consumers, and the simplest test surface (codegen assertions only, no
end-to-end execution yet — that comes in Phase 3).

The implementation order matches difficulty: start with the easiest,
build confidence, then tackle `copy_term`.

### 2.1 `arg/3` (easiest)

`arg(+N, +T, ?A)` is O(1) heap access — read the Nth argument cell of a
compound.

**Tasks:**

1. Add `builtin_id('arg/3', 19).` to `wam_wat_target.pl`
2. Add `$builtin_arg` helper to `compile_wam_helpers_to_wat/2`:
   - Read A1 (N), A2 (T), A3 (A) via `$get_reg_*`
   - Dereference
   - If T is not compound, return 0 (fail)
   - If N is out of range, return 0
   - Load the Nth argument cell from T's heap offset
   - Unify with A via `$unify_regs` (or direct set if A is unbound)
3. Add case `18` to `$execute_builtin` dispatch
4. Add a generation test: compile `test :- arg(2, foo(a,b,c), X), X == b.`,
   assert the emitted `.wat` contains `$builtin_arg`
5. Commit.

**Deliverable:** 1 commit.

### 2.2 `functor/3`

Both modes — read and construct. Read mode is trivial; construct mode
needs heap allocation for the fresh arg cells.

**Tasks:**

1. Add `builtin_id('functor/3', 18).`
2. Add `$builtin_functor`:
   - Read A1 (T), A2 (N), A3 (A), deref
   - Branch on `T`'s tag: if unbound → construct mode, else read mode
   - Read mode: extract name and arity, unify into A2 and A3
   - Construct mode: N must be atomic, A must be a non-negative integer
     - If A = 0: bind T to N
     - Else: allocate compound header + A fresh unbound cells on the heap
       via `$heap_alloc`, bind T to the compound
3. Add dispatch case
4. Add generation tests for both modes (read + construct)
5. Commit.

**Deliverable:** 1 commit.

### 2.3 `=../2` (univ)

Both modes. Decompose uses `functor/3` + `arg/3` patterns internally.
Compose uses `functor/3` construct mode + iterative unification of args
from a list.

**Tasks:**

1. Add `builtin_id('=../2', 20).`
2. Add `$builtin_univ`:
   - Decompose: walk T, emit `[F, a1, ..., an]` as a list on the heap,
     bind to L
   - Compose: walk L, extract head (functor) and remaining args, call
     into the same construct-compound logic used by `$builtin_functor`
3. Dispatch case 20
4. Generation tests for both modes
5. Commit.

**Deliverable:** 1 commit.

### 2.4 `copy_term/2` (hardest)

The only builtin that needs a scratch var map. Implementation follows
§3.1.1 of the spec.

**Tasks:**

1. Add `builtin_id('copy_term/2', 21).`
2. Reserve a 64-entry scratch area in the WAM state layout for the var
   map (update `wam_state.wat.mustache` and the state-base constants)
3. Add `$builtin_copy_term`:
   - Initialize the scratch var map
   - Push A1's heap offset onto a work stack (reuse env/choice stack area)
   - Iterate: pop, copy the cell (with var-map lookup for Unbound/Ref
     cases), push child offsets for Compound/List
   - Bind A2 to the root of the copied term
4. Dispatch case 21
5. Generation tests **including the sharing test** — this is the single
   most important test. Compile `test :- copy_term(f(X,X), C), C = f(A,B),
   A == B.` and verify generation.
6. Commit.

**Deliverable:** 1 commit.

**Exit criterion for Phase 2:** All four builtins have WAM-WAT helpers
and generation tests. The `.wat` files parse with `wat2wasm`. No runtime
execution is validated yet — that is Phase 3's job.

## Phase 3 — WAM-WAT functional execution test harness

**Goal:** Stand up the first-ever functional execution test for WAM-WAT,
so Phase 2's builtins can be validated against actual runtime behavior
(not just codegen shape). This phase is also the prerequisite for the
functional-test follow-up identified in the previous conversation, and
doing it here amortizes the infrastructure cost across both workstreams.

**Tasks:**

1. Verify `wat2wasm` and `wasmtime` (or equivalent) are available in the
   Termux environment. If not, document the requirement and skip this
   phase with tests marked pending.
2. Add a `record_result(reg, tag, payload)` host import to the WAT module
   template (`module.wat.mustache`) and the state initialization code.
3. Write a Prolog helper `run_wam_wat_module/3` that:
   - Takes a predicate and test input
   - Generates the `.wat`, compiles with `wat2wasm`
   - Runs the `.wasm` via `wasmtime` with the `record_result` host import
     connected to a temp file
   - Reads back the results and returns them to the test
4. Write functional tests for each Phase 2 builtin:
   - `test_functor_read_runtime` — compile + run, assert the recorded output
   - `test_functor_construct_runtime`
   - `test_arg_runtime`
   - `test_univ_decompose_runtime` and `test_univ_compose_runtime`
   - `test_copy_term_ground_runtime`
   - `test_copy_term_sharing_runtime` — the critical one
5. Commit per-builtin as tests come online.

**Exit criterion:** At least one functional test per builtin passes
end-to-end. The `copy_term` sharing test explicitly passes.

**Deliverable:** 3–5 commits — harness infrastructure + per-builtin
runtime tests.

**Risk:** If `wasmtime` is not available in Termux or the harness hits
subtle WAT codegen bugs that were masked by "parses with wat2wasm", this
phase could balloon. Budget for 1–2 debugging commits fixing real WAM-WAT
runtime bugs surfaced by the new tests. This is a *good* outcome — it is
exactly the reason to have functional tests at all.

## Phase 4 — Port to WAM-Rust

**Goal:** Same four builtins in the Rust backend. The Rust target already
has functional execution tests (`test_wam_rust_runtime.pl`), so validation
is easier than WAM-WAT.

**Tasks:**

1. Add `arg/3`, `functor/3`, `=../2`, `copy_term/2` handlers to the WAM
   interpreter loop in `wam_rust_target.pl` (around line 195)
2. Extend `Value` impl in `templates/targets/rust_wam/value.rs.mustache`
   with any missing helpers (likely `functor_and_arity()` and a
   `copy_fresh(&mut VarIdSource, &mut HashMap<VarId, VarId>)` method)
3. Add integration tests to `test_wam_rust_runtime.pl` that compile a
   predicate using each builtin, invoke via `cargo test`, assert on bindings
4. Include the `copy_term` sharing test
5. Commit per-builtin or bundle — Rust tends to be clean enough to bundle

**Exit criterion:** All four builtins work in Rust end-to-end via
`cargo test`.

**Deliverable:** 2–3 commits.

## Phase 5 — Port to WAM-Haskell

**Goal:** Same four builtins in the Haskell backend.

**Tasks:**

1. Add cases to the builtin dispatch in `wam_haskell_target.pl`'s inline
   Haskell code generator (around line 195–245)
2. Leverage Haskell's pattern matching and persistent data structures —
   `functor/3`, `arg/3`, `=../2` should each be a few lines of generated
   Haskell
3. `copy_term` uses `IntMap` as the var map, state-threaded through the
   recursive walker
4. Add codegen tests to `test_haskell_target.pl` (string-presence
   assertions, matching the existing Haskell test style — no execution)
5. Commit per-builtin or bundle

**Exit criterion:** All four builtins are emitted in the generated
Haskell for test predicates. Codegen tests pass.

**Note:** Haskell does not get functional execution tests as part of this
plan. That gap is not this plan's responsibility — it is a known
architectural asymmetry (Haskell tests are syntactic, Rust tests are
functional). Adding Haskell functional tests is a separate, larger
workstream.

**Deliverable:** 1–2 commits.

## Phase 6 — Perf investigation

**Goal:** Test the performance hypothesis from the philosophy doc with a
real predicate and a real measurement. Be willing to report a negative
result honestly.

**Tasks:**

1. Identify a predicate (from the Phase 0 audit, or construct one) that
   currently falls back to host SWI-Prolog interop because it uses one
   of the new builtins. Something in the neighborhood of:
   - A small meta-interpreter that uses `=..` and `call`
   - A term-walker that uses `functor` + `arg`
   - A memo predicate that uses `copy_term`
2. Establish a baseline: measure wall time on host SWI interop
3. Measure: wall time when the same predicate runs through each of
   WAM-Rust, WAM-Haskell, WAM-WAT using the new builtins
4. Compare. Four outcomes are all acceptable:
   - **Clear win**: native builtin is significantly faster than host
     interop. Write up and move the predicate class to the "definitely
     transpile via WAM" list.
   - **Modest win**: native is faster but not dramatically so. Document
     the factor and the workload characteristics.
   - **No measurable difference**: host interop was fine all along for
     this workload. Document that the transpiling-reach argument is the
     sole justification.
   - **Native is slower**: unlikely but possible (e.g., if the predicate
     is dominated by a fast-path in SWI that the WAM reimplements
     inefficiently). Investigate the hot spot, decide whether it's
     fixable within this plan or requires a follow-up.
5. Write up results in a new `docs/design/WAM_TERM_BUILTINS_PHASE_6_PERF.md`
   (mirroring the structure of the wam-haskell perf implementation plan
   retrospectives).

**Exit criterion:** A written perf writeup with numbers, regardless of
whether the numbers are flattering. A negative result is a valid outcome.

**Deliverable:** 1–2 commits — benchmark harness additions + the writeup.

## Phase sequencing summary

| Phase | Content                                        | Est. commits | Blocks next? |
|-------|------------------------------------------------|-------------|--------------|
| 0     | Audit: are these builtins actually needed?     | 1           | Yes          |
| 1     | Canonical WAM layer (`is_builtin_pred/2`)      | 1           | Yes          |
| 2     | WAM-WAT implementation (4 builtins)            | 4           | No (3 runs in parallel) |
| 3     | WAM-WAT functional test harness                | 3–5         | No           |
| 4     | WAM-Rust port                                  | 2–3         | No           |
| 5     | WAM-Haskell port                               | 1–2         | No           |
| 6     | Perf investigation + writeup                   | 1–2         | — (terminal) |

**Total:** ~13–18 commits across ~5 PRs.

**Suggested PR grouping:**

- **PR 1:** Phase 0 + Phase 1 — audit note + `is_builtin_pred/2` extension.
  Small, low-risk, establishes the workstream.
- **PR 2:** Phase 2 — WAM-WAT codegen for all four builtins.
- **PR 3:** Phase 3 — WAM-WAT functional test harness + runtime validation.
  This is where real runtime bugs may surface.
- **PR 4:** Phase 4 + Phase 5 — WAM-Rust and WAM-Haskell ports. Could be
  split into two PRs if the Rust change is large.
- **PR 5:** Phase 6 — perf writeup.

Phase 3 runs in parallel with Phase 2 only if the functional harness is
designed independently of specific builtins. In practice, because Phase 3
depends on having Phase 2 code to execute, sequential is cleaner.

## Risks and open questions

### Risk: Phase 0 finds no real demand

If the audit shows no test predicates currently use these builtins, the
transpiling-reach argument becomes speculative. Two responses:

1. **Shelve the plan** — document the finding, revisit when a real demand
   appears.
2. **Construct demand** — write a new example that demonstrates where the
   gap bites (e.g., a tiny meta-interpreter that can't currently be
   transpiled), and use that as the justification.

The philosophy doc leans toward option 2 if the gap is small, and
option 1 if the audit is genuinely bare.

### Risk: `copy_term` sharing bug

The most likely source of silent correctness failure across all three
backends. The fix for this risk is in the spec: the sharing test
(`test_copy_term_sharing`) is mandatory and runs in every backend that
has functional tests. Rust's test harness makes it easy. WAM-WAT's new
harness (Phase 3) must include it. Haskell only gets codegen verification,
which is weaker — explicitly note this in the per-builtin commit.

### Risk: WAM-WAT functional harness balloons

Phase 3 is the most speculative part of the plan. If `wasmtime` isn't
available in the test environment, or if subtle runtime bugs in the
existing WAM-WAT code surface under real execution, Phase 3 could grow
significantly. **Budget for it.** If it blocks progress, split off the
harness work as its own PR and continue Phases 4–5 using codegen-only
tests for WAM-WAT.

### Open question: arity limit

The WAM-WAT spec assumes `A1-A32` as argument registers. `functor/3`
construct mode with `A > 32` is an edge case. For v1, document the limit
and fail on over-sized compounds. Real code rarely needs arity > 8.

### Open question: scratch var map size

The spec fixes the `copy_term` scratch var map at 64 entries. For most
realistic terms this is sufficient. If a test predicate needs more, the
easy fix is to double the constant. The long-term fix is to grow the
scratch area dynamically, which is a Phase-7+ refinement.

### Open question: should `assert`/`retract` be scoped in?

Explicitly no. Group B (dynamic database) is architecturally different
(mutable clause store, dispatch plumbing, indexing) and deserves its own
philosophy/spec/plan triple. Noted in the spec as deferred.
