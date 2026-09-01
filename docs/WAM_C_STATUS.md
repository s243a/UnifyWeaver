# WAM C Target — Status

Living summary of the hybrid WAM-C backend
(`wam_c_target.pl` + `wam_c_runtime/`). Distinct from the **non-WAM**
direct C compiler (`c_target.pl`). The running checklist that has
historically functioned as this target's status is
[`WAM_C_TARGET_NEXT_STEPS.md`](../WAM_C_TARGET_NEXT_STEPS.md); this doc
is the fleet-aligned snapshot.

Companion docs:

- [`WAM_C_TARGET_NEXT_STEPS.md`](../WAM_C_TARGET_NEXT_STEPS.md) — living checklist.
- [`WAM_CROSS_TARGET_CONFORMANCE.md`](WAM_CROSS_TARGET_CONFORMANCE.md).
- [`WAM_BACKEND_CONVENTIONS.md`](WAM_BACKEND_CONVENTIONS.md).
- [`WAM_HYBRID_TARGETS_COMPARISON.md`](WAM_HYBRID_TARGETS_COMPARISON.md).

## Role

**Portable C ABI / FFI-glue substrate.** Small-footprint native
runtime that doubles as shared C glue other systems targets (Rust/Go
FFI kernels) can lean on. Historically undercounted as "907 lines" —
the current codegen is far larger.

## Codegen surface

| Module | Approx. lines |
|---|---:|
| `src/unifyweaver/targets/wam_c_target.pl` | ~6.6k |
| `src/unifyweaver/targets/wam_c_runtime/` (header + runtime) | ~1k header |
| Lowered emitter | none as a separate module — lowered **helpers** prototype inside the target |
| Dedicated tests | ~9 files |

## What's shipped

**All 7 shared kernels + `bidirectional_ancestor`.** Full shared
detector set plus the bidirectional ancestor kernel, with reverse-CSR
child-index paths for reverse traversal.

**Fact sources.** TSV plus an LMDB FactSource — `lmdb` is a heavy
theme in the target (mmap-backed fact storage, not TSV-only).

**Meta / aggregates.** Aggregates and `bagof`/`setof` meta-goals.

**Lowered helpers prototype.** No standalone
`wam_c_lowered_emitter.pl`; deterministic helper emission lives in the
target module.

**Conformance.** Registered as `conformance_target(c)` and passes the
whole spec with no `ct_xfail`/`ct_skip`; stays opt-in (needs a `gcc`
per-program build) rather than default CI.

**Nested-term unification (fixed 2026-08-06, CONF-FIX-C-NESTED).** The
runtime carried a single `S` register, but the compiler emits a nested
term *interleaved* with the enclosing term's arguments, so a head like
`p([tk(X)|R], X, R)` lost the pointer to the cons tail: the last
`unify_variable` read past the heap top — a wrong answer at `-O0`, a
SIGSEGV at `-O1`. The same defect existed in the write path, where the
cons tail slot was simply never written, so any constructor of the shape
`foo(X, [tag(X)|Rest])` called with an unbound output silently failed.
Both are fixed by a `WamArgCtx` stack that saves and restores the
enclosing term's argument pointer around nested terms; the depth is
restored on backtracking. Found by the `nested` conformance program.

**Read-mode variable identity (fixed 2026-08-09, CONF-FIX-C-EQ-DEREF).**
`unify_variable` in READ mode copied the heap cell into the register by
value. That is correct for a bound cell — `VAL_REF`/`VAL_ATOM`/`VAL_STR`/
`VAL_LIST` each carry their own identity — but `VAL_UNBOUND` carries no
address, so the register ended up holding its own detached variable. A
head with a repeated variable inside a structure (`csame(p(X, X), X)`)
then aliased `X` with the *second* slot and bound only that one, leaving
the caller's first slot unbound. Read mode now installs a `VAL_REF` at
`S` when the cell is unbound, mirroring the write branch. Found by the
`repeatvar` conformance program; `=/2` hides this class of bug because it
re-unifies, so only an identity comparison exposes it.

## Gaps (relative to Rust / Haskell / F#)

- **No ISO three-form contract adoption** — not a reference adopter
  alongside C++/Elixir (see
  [`design/WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md`](design/WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md)).
- **No runtime-parser capability entry** in
  `wam_runtime_parser_capability.pl` — no source-term parsing path.
- **No two-level lazy/cached LMDB policies** (F#/Haskell tier); the
  FactSource is present but the policy surface is thinner.
- Effective-distance cross-target matrix presence is thinner than the
  Tier-A kernel benches.
- **Two pre-existing failures in `tests/test_wam_c_target.pl`**
  (`transitive_closure2` / `transitive_distance3` native kernel helpers
  reported missing). Present on an unmodified tree as of 2026-08-09 and
  unrelated to the unification fixes (re-confirmed against a stashed
  baseline); not yet triaged.

## Whole-program exercise (A2, 2026-09): known / suspected deficiencies

The peerhailer CLI-parser exercise (see
[`WAM_FLEET_GAPS.md`](WAM_FLEET_GAPS.md)) found three JS-WAM runtime bug
classes that are fleet-wide suspects. C's audit:

| # | Deficiency | Status | Evidence / reason |
|---|---|---|---|
| A1 | `sub_string/5` builtin missing | **verified missing** | not in `wam_execute_builtin` or the `Execute` special cases (the C++ sibling has the alias; C does not) |
| A2 | Y-register clobber across `Call` of a no-`Allocate` fact | **absent (aliasing form) / suspected (overflow form)** | C segregates A/X/Y into distinct banks with an `is_y` flag per operand (`wam_c_target.pl:1684-1710`), so X registers cannot alias Y. But the banks are fixed 256-slot arrays (`wam_runtime.h.mustache:11,81-82`) and `_XT` temps map to `128 + N` inside the X bank (`:1705-1708`); a ground fact needing more X registers than the bank holds would index **out of bounds** — no bounds check was found. Suspected memory-corruption hazard on very large facts |
| A3 | `Execute` of a builtin doesn't return to the continuation | **verified** | `INSTR_EXECUTE` special-cases only findall/bagof/setof, then label lookup, then `return false` (`wam_c_target.pl:2257-2270`) — a last-goal builtin outside `is_builtin_pred/2` silently fails |
| A4 | String fidelity | **rung 0** | no string term tag; D37's double-quoted literals intern as atoms |

Pattern lane: `c_target.pl` compiles facts from `clause(Head, true)` —
the G-A3-8 execute-at-compile-time hazard is absent.

## Path forward

1. Adopt the ISO three-form error contract (C++ is the reference).
2. Add a runtime-parser capability entry if compiled/native term IO is
   wanted.
3. Promote the lowered-helpers prototype into a first-class lowered
   emitter if C moves off "substrate" duty.
4. Richer LMDB lazy/cached policy tiers.
5. Publish a dedicated effective-distance bench row.

## Document status

Fleet-aligned snapshot; source-verified line/kernel/LMDB/conformance
facts against `wam_c_target.pl`, the conformance harness, and the
parser-capability module (2026-07-11). Refreshed 2026-08-09 after
CONF-FIX-C-EQ-DEREF; the C arm now carries no `ct_xfail` at all. Update
the living checklist first when milestones land, then refresh here.
2026-09-01: added the whole-program (A2) deficiency audit; see
[`WAM_FLEET_GAPS.md`](WAM_FLEET_GAPS.md).
