# WAM Scala Target — Status

Living summary of the hybrid WAM-Scala backend
(`wam_scala_target.pl` + `wam_scala_lowered_emitter.pl`). Distinct
from the **non-WAM** direct Scala compiler (`scala_target.pl`). Usage
guide: [`WAM_SCALA_TARGET.md`](WAM_SCALA_TARGET.md).

Companion docs:

- [`WAM_SCALA_TARGET.md`](WAM_SCALA_TARGET.md) — usage guide.
- [`WAM_CROSS_TARGET_CONFORMANCE.md`](WAM_CROSS_TARGET_CONFORMANCE.md).
- [`WAM_BACKEND_CONVENTIONS.md`](WAM_BACKEND_CONVENTIONS.md).
- [`WAM_HYBRID_TARGETS_COMPARISON.md`](WAM_HYBRID_TARGETS_COMPARISON.md).

## Role

**Generalization anchor + default conformance backend.** Carries the
classic-program suite (n-queens, Ackermann, Fibonacci, …) that sets
the transpiler's generalisation upper bound, and runs as a **default
CI** conformance target alongside Elixir.

## Codegen surface

| Module | Approx. lines |
|---|---:|
| `src/unifyweaver/targets/wam_scala_target.pl` | ~1.4k |
| `src/unifyweaver/targets/wam_scala_lowered_emitter.pl` | ~0.8k |
| Dedicated tests | ~11 files (~137 plunit cases) |

## What's shipped

**Dual lowering.** WAM instruction VM plus per-predicate emitter
(`emit_mode(functions)`) — clause-1 fast path with interpreter
fallback.

**All 7 shared kernels.** Opt-in via `kernel_dispatch(true)`. An
intra-Scala mode bench (`benchmarks/wam_scala_mode_bench.md`, cited —
not re-run here) shows kernel dispatch ~4×@depth100, ~9×@depth300;
shallow queries can regress.

**Four fact backends.** Inline / file CSV / grouped TSV / **arity-N
LMDB** (validated end-to-end); auto-inline ≤128 rows.

**Atom interning.** Aggressive compile-time atom interning.

**Conformance.** `ct_default_target(scala)` — one of only two
backends (with Elixir) that run conformance by default, not opt-in.

**Nested head terms + strict (in)equality (fixed 2026-08-09,
CONF-FIX-SCALA-NESTED / CONF-FIX-SCALA-EQ).** Two defects that the
six classic conformance programs could not see, because every query in
them is ground — a head is always *matched*, never used to *bind*:

- The runtime carried a single `unifyQueue`, but the compiler emits a
  nested term interleaved with the enclosing term's arguments, so
  `get_structure` on a list element discarded the pending cons tail
  and the following `unify_variable` failed the clause. A
  `unifyStack` of enclosing queues now restores them when the nested
  term's arguments run out — the counterpart of the C runtime's
  `WamArgCtx` stack, and applied to both the interpreter and the
  lowered `lo*` helpers.
- `==/2` and `\==/2` were absent from the builtin table entirely, so
  the shared compiler's `builtin_call` fell through and failed
  closed. Added via a read-only `termIdentical` traversal (never
  binds; two distinct unbound variables are not identical). Same
  class-7 gap the C and Haskell runtimes carried.

**Head-constructed output arguments (fixed 2026-08-10,
CONF-FIX-SCALA-BUILD).** `test(nqueens)` had never terminated on a
*failing* query — `queens_q(4, [1,2,3,4])` ran past 600 s where the
search is 24 permutations. It was not a search problem; every failing
`queens_q` hung, down to N=2. Two independent defects in the write path:

1. **`finalizeBuild` skipped the bind-through for A registers.** That
   rule is right for `put_structure`/`put_list` (body staging — the
   M139/M140 cyclic-term fix) and wrong for `get_structure`/`get_list`
   in write mode, where the A register holds the *caller's* output
   variable. `select_uw(H, [X|T], [X|T2])` therefore never bound the
   caller's `Rest`, so `permutation_uw` recursed with an unbound list
   and `select_uw` became an infinite generator. `BuildFrame` now
   carries a `bindThrough` flag set by the get-family only.
2. **`GetStructure` had no arity and guessed it** by counting the
   `unify_*` instructions that follow. That over-counts exactly when a
   nested term is interleaved with the enclosing term's arguments: in
   `cbuild_one(X, [tk(X)|[]])` the `tk/1` frame counted 2, swallowed the
   cons tail, and the cons frame never completed — so `finalizeBuild`
   never ran for it. The instruction now carries `sArity` (the emitter
   was already parsing it and throwing it away), and the read branch
   checks arity too, since the interned functor id is the name only.

Both were invisible to the conformance suite because its one write-mode
check used `=/2`, which re-unifies and so succeeds whether or not the
head did the construction. It now uses `==/2`; see the `buildnest`
program.

## Gaps (relative to Rust / Haskell / F#)

- **No ISO three-form contract adoption** (low ISO surface in source).
- **No runtime-parser capability entry** in
  `wam_runtime_parser_capability.pl`.
- **No two-level lazy/cached LMDB policies** — the LMDB backend is
  arity-N but flat, without F#-style eager/lazy/cached tiers.
- Cross-target effective-distance bench vs Elixir/Haskell still open.
- ~~`test(nqueens)` does not terminate~~ — **fixed 2026-08-10**, see
  below. The whole classic-programs suite now runs to completion.

## Whole-program exercise (A2, 2026-09): known / suspected deficiencies

The peerhailer CLI-parser exercise (see
[`WAM_FLEET_GAPS.md`](WAM_FLEET_GAPS.md)) found three JS-WAM runtime bug
classes that are fleet-wide suspects. Scala's audit:

| # | Deficiency | Status | Evidence / reason |
|---|---|---|---|
| A1 | `sub_string/5` builtin missing | **verified missing** | `sub_atom/5` is in the intercept table (`runtime.scala.mustache:1868`); `sub_string/5` is not |
| A2 | Y-register clobber across `Call` of a no-`Allocate` fact | **verified (structural)** | encoding `X→+100 / Y→+200` (`wam_scala_target.pl:106-107`), so X101 ≡ Y1; registers are one flat `s.regs` array. `Allocate` *copies* the current Y range 201..299 into the new frame (`runtime.scala.mustache:1185-1192`) but `Deallocate` drops the frame **without restoring** (`:1194-1196`) and `Call` takes no snapshot — a no-`Allocate` fact with >99 X placeholders overwrites the caller's live Y values |
| A3 | `Execute` of a builtin doesn't return to the continuation | **handled for its intercept set** | `interceptedExecuteBuiltin` (14 builtins incl. `call/N`, `sub_atom/5`, `between/3`, sort/bag/set, `format/2`) is routed from *both* the `Call` and `Execute` arms with `withReturn` plumbing (`:1158-1175`, `:1846-1870`); a builtin outside the set with no label → silent `backtrack` |
| A4 | String fidelity | **rung 0** | no string term type in `WamTerm`; D37's double-quoted literals intern as atoms |

Note for the "default conformance backend" role: being green on the
classic programs did not protect the JS runtime from any of these three
classes — they only fire on shapes (huge ground facts, last-goal string
builtins) the classic suite never emits. See Class C in the fleet doc.

Pattern lane: `scala_target.pl` compiles facts from
`clause(Head, true)` — the G-A3-8 execute-at-compile-time hazard is
absent; the G-A3 machinery has no analogue there.

## Path forward

1. ISO three-form adoption.
2. Cross-target effective-distance benchmark vs Elixir/Haskell.
3. Richer LMDB policy tiers.
4. Runtime-parser capability entry if term IO is wanted.

## Document status

Fleet-aligned snapshot; source-verified line/kernel/LMDB facts and the
`ct_default_target(scala)` default-CI registration against
`wam_scala_target.pl` and the conformance harness (2026-07-11). Perf
figures cited from the mode bench, not re-run. Refreshed 2026-08-09
after CONF-FIX-SCALA-NESTED / CONF-FIX-SCALA-EQ; note that the emitted
runtime needs **Scala 2.13+** (it uses `String.toIntOption`), so a 2.11
toolchain fails to compile the project rather than diverging.
2026-09-01: added the whole-program (A2) deficiency audit; see
[`WAM_FLEET_GAPS.md`](WAM_FLEET_GAPS.md).
