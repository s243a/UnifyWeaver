<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM lowered tier — Stage 0 census + Stage 1 (F11) result

Implements Stage 0 and Stage 1 of
[`../proposals/WAM_RUST_LOWERED_TIER_THROUGHPUT_PLAN.md`](../proposals/WAM_RUST_LOWERED_TIER_THROUGHPUT_PLAN.md).
Stage 2 (the resumable trampoline / item 2b) is **out of scope** and was not
started. All numbers below are measured on the build box, `LC_ALL=C.UTF-8`,
release builds, against the frozen `examples/pkg_resolver/resolver.pl` and
`resolver_store.pl` (neither modified).

## TL;DR verdict

- **Census:** F11-shaped predicates account for a **large** share of the B2
  differential's interpreter work — **46.6%** of all `call`/`execute` dispatches
  (matching_deps/4 15.2%, matching_versions/4 11.1% lead). So on paper Stage 1
  could move B2.
- **But the SAFE first-solution subset is small and shallow.** The subset that
  can be dispatched as a *sound* first-solution native loop — mutually-exclusive
  clauses (determinism) **and** a body that makes no user call that could reach a
  nondeterministic callee (a store seek, `satisfies/2`, …) — is dominated by
  **shallow catalog accessors** (`provides_list/2`, `conflicts_list/2`, …), plus
  a couple of genuine list walkers (`names_of/2`, `key_pkg_rows/3`). That subset
  is **~12%** of B2 dispatches.
- **F11 measured net-negative.** For those shallow accessors the two per-call
  register-file snapshots the sound-dispatch tier takes (the `lowered_dispatch`
  guard's `save_regs` + the loop's `lo_clause_snapshot`) cost about what the
  removed interpreter dispatch saves — the **O2** obstruction in the plan, §2.
  A drift-cancelling interleaved A/B put **B2 at +7% with F11 on** (≈24.77 s vs
  ≈23.09 s baseline, 4 rounds, highly consistent). B3 was neutral-to-slightly
  negative.
- **Decision:** per the plan's "bank it if it helps", F11 is **implemented,
  gated, and DEFAULT-OFF**. Both lanes stay green *with it on* (term differential
  2600/0/0, store 503/0) and *with it off* (the committed default). Enable with
  the `f11_tail_loop(true)` compile option or `UW_F11_ON=1`.
- **Bearing on Stage 2:** the census confirms the plan's framing. The hot path
  is genuinely nondeterministic multi-clause drivers, and even the *deterministic*
  F11 slice does not beat the interpreter under the current
  full-register-file snapshot machinery. The real lever is Stage 2's **cheaper
  saved-locals representation (P2, clears O2)** plus **direct lowered→lowered
  calls (P1, clears O1)** — not F11.

## Stage 0 — census

Method. A per-predicate `call`/`execute` dispatch counter was compiled into a
throwaway instrumented copy of the term crate (env-gated, not committed) and run
over the exact B2 differential corpus (2600 cases) and the B3 5000-package
`resolve_layered`. Dispatch count is the hotness proxy (it tracks the
interpreter's step/backtrack machinery — the 65%/21%/12% profile the status doc
records). Structural F11-eligibility was computed from each predicate's clauses
(single tail-recursive clause + mutually-exclusive heads + deterministic body).

### B2 differential — top contributors, class, share (of 743,562 dispatches)

| predicate | dispatches | share | class | in shipped F11 bank? |
|---|---:|---:|---|---|
| matching_deps/4 | 112,758 | 15.2% | F11-shaped, **2b-ish body** (calls dep_to_req/3) | no (body calls user pred) |
| matching_versions/4 | 82,786 | 11.1% | F11-shaped, **2b-ish body** (calls satisfies/2) | no |
| lookup_held/3 | 53,712 | 7.2% | recursive, non-tail | no |
| item_ver/3 | 46,827 | 6.3% | non-recursive leaf | no |
| long_enough/2 | 38,877 | 5.2% | recursive, non-tail | no |
| dep_breaks/5 | 36,504 | 4.9% | non-tail recursion | no |
| provides_list/2 | 22,504 | 3.0% | **F11 pure-body** (catalog accessor) | **yes** |
| selected_ver/3 | 19,016 | 2.6% | recursive, non-tail | no |
| provides_sat/5 | 18,620 | 2.5% | non-recursive | no |
| scan_base_holds/3 | 18,065 | 2.4% | has a non-tail self-call | no |
| conflicts_list/2 | 16,707 | 2.2% | **F11 pure-body** | **yes** |
| direct_on/4 | 14,243 | 1.9% | F11-shaped, body calls dep_mentions/2 | no |
| dep_mentions/2 | 13,723 | 1.8% | non-recursive (2 clauses) | no |
| conflicts_in/4 | 13,347 | 1.8% | non-recursive | no |
| build_tree/4 | 12,002 | 1.6% | non-tail recursion (2b) | no |
| base_list/2 | 11,911 | 1.6% | **F11 pure-body** | **yes** |
| depends_list/2 | 11,027 | 1.5% | **F11 pure-body** | **yes** |
| layers_list/2 | 10,225 | 1.4% | **F11 pure-body** | **yes** |
| no_acc_conflicts/4 | 9,367 | 1.3% | F11-shaped, body calls conflicts_in/4 | no |
| same_key/4 | 8,708 | 1.2% | recursive, non-tail | no |

Aggregate shares of B2 dispatches:

| bucket | share |
|---|---:|
| **F11-eligible (structural, 27 preds)** | **46.6%** |
| — of which SAFE to dispatch (pure body, shipped bank, 12 preds) | ~12% |
| — F11-shaped but body reaches a nondet callee (2b-only in effect) | ~34% |
| 2b-only drivers (real backtracking: pick/7, blocked_from/4, dep_breaks/5, build_tree/4, …) | remainder |
| already-lowered / leaf (fact tables, non-recursive) | remainder |

The shipped F11 bank (pure body, mutually-exclusive heads): `provides_list/2`,
`conflicts_list/2`, `base_list/2`, `depends_list/2`, `layers_list/2`,
`packages/2`, `excluded_list/2`, `alias_list/2`, `requested_list/2`,
`installed_list/2` (catalog accessors), plus `names_of/2` and `key_pkg_rows/3`
(list walkers).

### B3 (`resolve_layered`, 5000 packages) — top contributors (of 91,559)

| predicate | dispatches | share | note |
|---|---:|---:|---|
| same_key/4 | 22,525 | 24.6% | index-build compare (2b, non-tail) |
| build_tree/4 | 20,020 | 21.9% | index tree build (2b, non-tail) |
| key_dep_rows/3 | 15,004 | 16.4% | F11-shaped, body calls dep_to_req/3 |
| dep_to_req/3 | 15,003 | 16.4% | non-recursive |
| group_keyed/2 | 10,011 | 10.9% | F11-shaped, body calls same_key/4 |
| key_pkg_rows/3 | 7,523 | 8.2% | **F11 pure-body** (in shipped bank) |

B3 F11-eligible (structural) = 36.1% of dispatches, but B3's top two
(`same_key`, `build_tree`, ~46%) are non-tail index builders, not F11.

**Decision-gate reading.** A large share is F11-*shaped*, so Stage 1 was worth
doing. But the share that is F11 *and* safe to commit to first-solution *and*
deep enough to amortise the dispatch cost is small; the rest is effectively
2b-only (the body re-enters the interpreter for a nondet/opaque callee). This is
the census result the plan asked for: it tells us F11 alone will not move B2 and
that the real lever is Stage 2.

## Stage 1 — F11 implementation

Eligibility (`wam_rust_f11_lowerable/4` + `rust_pred_heads_exclusive/3` in the
emitter/target). A predicate is F11-lowered only when **all** hold:

1. Exactly one clause is recursive and its recursion is the clause's tail call
   (`execute self`); no other clause references self, and the recursive clause
   has no non-tail self reference.
2. Clause heads are **pairwise non-unifiable** (`rust_heads_mutually_exclusive/1`)
   — the determinism carrier: at most one clause matches any goal, so the loop's
   first solution is the predicate's only solution and committing to it drops no
   answer an interpreted caller would find by backtracking.
3. The body is **pure**: head match + pure builtins + the loop, with no user
   `call`/`execute` other than the tail self-call (`rust_f11_body_pure/2`). This
   keeps the loop off the interpreter's `run()` re-entry entirely, so no callee
   can leak a choice point or (on the store lane) reach a nondeterministic seek.

Emission (`emit_f11_tail_loop/7`): a `loop { }` that snapshots clause-entry
state, tries the single recursive clause **first** (it matches every iteration
but the last, so this skips a failed base-clause head-match plus its
`restore_regs` per iteration), rebinds the argument registers via the clause's
own `put_*` and `continue`s instead of re-dispatching; the mutually-exclusive
base clauses follow and return on their terminal. Dispatched from an interpreted
`call`/`execute` through the existing `lowered_call` → `WamState::lowered_dispatch`
guard, which declines (rolls back to the interpreter) if any call leaves a choice
point. No `Allocate`/`Deallocate` amortisation beyond one frame per iteration
(Y-registers still live in an env frame), no per-recursion choice point, no
`run()` round-trip on the recursion edge.

Correctness invariants preserved (plan §6): trail record + truncation via
`lo_clause_snapshot`/`lo_restore_clause`; the `backtrack_floor` (D71) and cut
barriers are set by `lowered_dispatch` at entry depth; Y-frame discipline via the
`Arc`-cloned stack in the snapshot; sentinels (`cp==0`) via `lowered_dispatch`;
solution order/multiplicity guaranteed by the mutual-exclusivity + pure-body
gates and checked by the differentials.

A latent runtime bug was fixed alongside: `deref_match_atom` (the fast path
behind `match_reg_atom`, used by `get_constant []`/`get_nil` in every lowered
emitter) did not alias an empty `Value::List` with the atom `[]`, so a lowered
clause discriminating `[]` vs `[_|_]` wrongly rejected a genuinely-empty list
delivered as a `List`. Fixed in `state.rs.mustache`; it is a general correctness
fix for the lowered tier (inert in the committed F11-off build).

Attribution. The F11 recognition idea (single tail-recursive clause +
independent/mutually-exclusive heads → one flat loop, zero choice-point
machinery) is adapted from Kenichi Sasagawa's M-Prolog / N-Prolog
`tail_recursive/6` + `independ_head/1` (Modified BSD; see
[`../proposals/MPROLOG_MINING_NOTES.md`](../proposals/MPROLOG_MINING_NOTES.md)
finding F11). Idea only — no mprolog code was copied. The Modified BSD notice
sits alongside the adapted logic in `wam_rust_lowered_emitter.pl`.

## Gates (this box, LC_ALL=C.UTF-8)

| gate | committed (F11 off) | F11 on (`UW_F11_ON=1`) |
|---|---|---|
| term corpus (B1) | 51/51 | 51/51 |
| term differential (B2) | 2600 / 0 / 0 | 2600 / 0 / 0 |
| store corpus | 51/51 (identical to term) | 51/51 (identical to term) |
| store differential | 503 / 0 | 503 / 0 |

## Measurement — before (D74 / F11 off) vs after (F11 on)

Interleaved, drift-cancelling A/B (same box, same cases file, alternating
binaries) is the reliable comparison; single-shot absolute times drifted with
box load.

**B2 (2600-case differential, rust leg, interleaved base vs F11):**

| round | baseline (F11 off) | F11 on | delta |
|---:|---:|---:|---:|
| 1 | 23,093 ms | 24,779 ms | +1,686 ms |
| 2 | 23,084 ms | 24,769 ms | +1,685 ms |
| 3 | 23,094 ms | 24,724 ms | +1,630 ms |
| 4 | 24,235 ms | 24,869 ms | +634 ms |

Median ≈ **+1.65 s (+7%)** with F11 on. SWI on the same box runs B2 in ≈2.5–2.6 s,
so the **target/SWI ratio is ≈9.0× (F11 off) → ≈9.6× (F11 on)** — i.e. F11 makes
the gap slightly *worse*, it does not move B2.

**B1 corpus wall time:** ≈58–61 ms either way (neutral).

**B3 resolve_layered (best of 3):** 5000 packages ≈1,945 ms (off) vs ≈1,911 ms
(on) — neutral within noise; 500 packages ≈216–234 ms — neutral.

**Verdict: F11 did NOT move B2 — it regresses it ≈7% and is neutral on B1/B3.**
The cause is the O2 snapshot cost applied to shallow accessors: each dispatched
call pays two register-file snapshots, which for a 1–2 iteration accessor
outweighs the interpreter dispatch removed. Reported honestly and banked OFF.

## What this means for Stage 2

- The census confirms the hot path is 2b-shaped, and even the deterministic F11
  slice loses to the interpreter under full-register-file snapshots. So Stage 2
  must land **P2 (a minimal saved-locals representation, clearing O2)** for any
  native tier to pay — measure per-solution/per-call cost against the 21%
  backtrack + 12% restore_regs baseline *before* committing (plan §5).
- The ~34% of B2 that is F11-*shaped but 2b-in-effect* (matching_deps/4,
  matching_versions/4, direct_on/4, no_acc_conflicts/4, group_keyed/2,
  key_dep_rows/3) becomes cleanly lowerable once **P1 (direct lowered→lowered
  deterministic calls, clearing O1)** lets the body call `satisfies/2` /
  `dep_to_req/3` as native deterministic functions instead of re-entering
  `run()`. That is where the F11 idea and Stage 2 converge — but it needs P1+P2
  first, exactly as the plan sequences it.
