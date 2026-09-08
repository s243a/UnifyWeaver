<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (s243a)
-->

# WAM Elixir Target — Status

Living summary of the hybrid WAM-Elixir backend
(`wam_elixir_target.pl` + `wam_elixir_lowered_emitter.pl` +
`wam_elixir_utils.pl`). Distinct from the **non-WAM** direct Elixir
compiler documented in [`ELIXIR_TARGET.md`](ELIXIR_TARGET.md)
(`elixir_target.pl`). Created 2026-09-01 as part of the whole-program
deficiency audit — this backend previously had no `WAM_*_STATUS.md` of
its own despite being a default-CI conformance arm.

Companion docs:

- [`WAM_BACKEND_CONVENTIONS.md`](WAM_BACKEND_CONVENTIONS.md)
- [`WAM_CROSS_TARGET_CONFORMANCE.md`](WAM_CROSS_TARGET_CONFORMANCE.md)
- [`WAM_HYBRID_TARGETS_COMPARISON.md`](WAM_HYBRID_TARGETS_COMPARISON.md)
- [`WAM_FLEET_GAPS.md`](WAM_FLEET_GAPS.md)

## Role

**Default-CI conformance arm + ISO reference adopter (with C++), with a
full lowered default path and actor/Task-based parallelism.**

## Codegen surface

| Module | Approx. lines |
|---|---:|
| `src/unifyweaver/targets/wam_elixir_target.pl` | ~7.0k |
| `src/unifyweaver/targets/wam_elixir_lowered_emitter.pl` | ~2.3k |
| Dedicated tests | `test_wam_elixir_*` (classic programs, lowered ITE / phase-3 / phase-4c/4d, utils) + `tests/elixir_e2e/` |

## What's shipped

- **Dual lowering** — WAM instruction interpreter plus a lowered
  emitter used as the default path in tests (broader lowered coverage
  than the det-centric Rust/Haskell emitters).
- **ISO errors** — reference adopter alongside C++ (see
  `WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md`); `catch/3` + `throw/1`
  routed through the `Call`/`Execute` arms.
- **Default-CI conformance** — runs on every CI pass (with Scala);
  CONF-FIX-ELIXIR-REPEATVAR / -BUILD / -BUILDNEST landed 2026-08 (see
  [`WAM_FLEET_GAP_TASKS.md`](WAM_FLEET_GAP_TASKS.md)).
- **Parallelism** — actor/Task-based parallel path (one of only three
  backends with a parallelism mechanism, per the parity punchlist).
- **Cons-cell aliasing precedent** — `step_get_structure_matches?/2`
  aliases `./2`↔`[|]/2` (cited in `WAM_BACKEND_CONVENTIONS.md` §1).

## Whole-program exercise (A2, 2026-09): known / suspected deficiencies

The peerhailer CLI-parser exercise (see
[`WAM_FLEET_GAPS.md`](WAM_FLEET_GAPS.md)) found three JS-WAM runtime bug
classes that are fleet-wide suspects. Elixir's audit:

| # | Deficiency | Status | Evidence / reason |
|---|---|---|---|
| A1 | `sub_string/5` builtin missing | **verified missing** | not in the step dispatch |
| A2 | Y-register clobber across `Call` of a no-`Allocate` fact | **verified (structural)** | `reg_id/2` maps `X→+100 / Y→+200` (`wam_elixir_utils.pl:46-52`) — its comment says the ranges "avoid aliasing", but with a 100-slot X window **X101 ≡ Y1**. `state.y_regs` is swapped out only at `Allocate` (`wam_elixir_target.pl:376-382`, "so the callee can freely use Y-reg slots… without clobbering the caller"), so a no-`Allocate` ground fact with >99 X placeholders writes ids ≥ 201 straight into the *caller's live* `y_regs` |
| A3 | `Execute` of a builtin doesn't return to the continuation | **verified partial** | the `execute` arm routes tail-position `call/N`, `catch/3` and `throw/1` correctly (`wam_elixir_target.pl:346-364`); any other unlabelled name → `:fail` (`:366-370`) — a last-goal builtin outside `is_builtin_pred/2` silently fails |
| A4 | String fidelity | **rung 0** | no string term tag; D37's double-quoted WAM string tokens intern as atoms |

Any A2/A3 fix must land in **both** the interpreter and the lowered
emitter (the F# CONF-FIX-FSHARP-LOWERED-GETVALUE episode is the
standing lesson).

Pattern lane: `elixir_target.pl` compiles facts from
`clause(Head, true)` (`:240`) — the G-A3-8 execute-at-compile-time
hazard is absent; the G-A3 machinery (cross-predicate calls by output
count, multi-output tuples, semidet sentinel, compound runtime data)
has no analogue there.

## Path forward

1. Close A2 with either a non-aliasing register encoding or a JS-style
   Y-range snapshot on `Call` (see `WAM_BACKEND_CONVENTIONS.md` §8).
2. Extend the `execute` arm's builtin routing beyond call/catch/throw
   (§7); mirror in the lowered emitter.
3. Attempt the `examples/cli_args/` whole-program benchmark (Class C in
   the fleet doc) — Elixir's lowered-default path makes it the best
   BEAM candidate.

## Document status

Created 2026-09-01 during the whole-program (A2) deficiency audit; A2
and A3 rows source-verified against `wam_elixir_target.pl` /
`wam_elixir_utils.pl` at that date. Role/shipped facts summarized from
the fleet gap-task cards and the parity punchlist, not re-run.
