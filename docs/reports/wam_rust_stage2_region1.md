<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM lowered tier — Stage 2 region 1 build + real A/B (matching_deps ⊕ dep_to_req)

Implements region 1 of Stage 2 of
[`../proposals/WAM_RUST_LOWERED_TIER_THROUGHPUT_PLAN.md`](../proposals/WAM_RUST_LOWERED_TIER_THROUGHPUT_PLAN.md)
§5 (P1 direct native calls, P2 minimal-locals snapshot) — the ONE region the
measurement spike ([`wam_rust_stage2_spike.md`](wam_rust_stage2_spike.md))
prototyped as a cost model. This round turns that cost model into a **real,
wired, wall-clock-validated** native region and reports whether reality confirms
or refutes the projection. Scope is exactly this one region; no other region and
no nondet resume extension were touched. All numbers on the build box,
`LC_ALL=C.UTF-8`, release + LTO, against the frozen `resolver.pl` /
`resolver_store.pl` (neither modified).

## TL;DR — CONFIRMED, banked ON

The real interleaved A/B **confirms** the spike's ~6% machinery-floor
projection: fusing `matching_deps/4 ⊕ dep_to_req/3` into one native region
removes a **median 6.98% of B2 wall time** (all six rounds negative), moving the
SWI ratio from ~9.0× to ~8.3×, with every correctness gate green in both
configs. Per the plan's "bank it if it helps", the region is **default ON**
(gated; disable with `region1(false)` / `UW_REGION1_OFF=1`).

## The region and how it is fused

```prolog
matching_deps([], _Name, _Ver, []).
matching_deps([depends(N,V,D,C)|Rest], Name, Ver, Out) :-
    ( N==Name, V==Ver -> dep_to_req(D,C,Req), Out=[Req|Rs] ; Out=Rs ),
    matching_deps(Rest, Name, Ver, Rs).
dep_to_req(alternatives(Alts), _C, req(alternatives(Alts), any)) :- !.
dep_to_req(D, C, req(D, C)).
```

`matching_deps/4` walks a dependency list; per element it calls `dep_to_req/3`
(a pure 2-clause committed rewrite, called only from `matching_deps` in the term
resolver). The region compiles both into ONE native function entered once,
`WamState::region_matching_deps_dispatch`:

- **P1 (direct native call).** `dep_to_req` is inlined as `region_dep_to_req`, a
  plain Rust call — never `vm.run()`. The per-element `execute` re-dispatch and
  the `Call dep_to_req` are gone from the hot path.
- **P2 (minimal-locals snapshot).** Rollback saves only three scalars
  (`trail_len`, `heap_len`, `var_counter`) — NOT a `save_regs` full-register
  clone. This is the exact cost the F11 round (Stage 1) lost to; the minimal
  snapshot is what flips it.
- **Fusion.** The `matching_deps` recursion becomes a native `loop`; the
  self-call never happens at runtime — no `Allocate`/`Deallocate`, no
  per-recursion choice point, no re-dispatch.

The region reuses existing runtime machinery for correctness: `terms_identical`
(the exact `==/2` comparison) for the `N==Name, V==Ver` guards, `deref_shallow`
for list/element inspection, and `unify` for the single output binding.

Wiring: `lowered_call` (the same crate-level hook the Call/Execute arms already
consult before label lookup) gets a `"matching_deps/4"` arm that routes to the
region, emitted only when the flag is on AND the frozen resolver shape is present
(`rust_region1_applicable/1`, a structural `=@=` variant check on the clauses).
The store lane's `matching_deps_store`/`dep_to_req_store` never match, so the
store path is untouched by the region.

## G-gates (Kimi K2 soundness review) — how each is satisfied

| gate | applies? | how satisfied (machinery reused) |
|---|---|---|
| **G-1** minimal saved set from backward liveness + all arg registers forced live; frame pinned while a CP references it | yes | The region has a single rollback point (entry). Backward liveness from it reduces to "A1..A4 at entry", which are **forced live by never being clobbered** — the walk reads them into Rust locals and mutates no register (verified by the `frame_pinning_*` test). The minimal snapshot records `trail_len`/`heap_len`/`var_counter`. No frame is dropped because none is allocated and no CP is created. |
| **G-2** replacement determinism guard (static det lattice + runtime post-call CP-depth check; deopt else) | yes | Static: at-most-one-solution proven below (G-4). Runtime: the region pushes NO choice point and calls NO interpreted code, so CP-depth is unchanged **by construction**; it is asserted after the walk (`choice_points.len() != cp_depth → decline`) — the generalised form of `lowered_dispatch`'s existing post-call guard. Any off-shape input DECLINES (`None`) and the interpreter runs the predicate. |
| **G-3** cut across tiers passed explicitly, saved-in-CP/restored | **no cut crosses a boundary** | `matching_deps` has no cut. `dep_to_req`'s first-clause cut is realised as a Rust `if/else` in `region_dep_to_req` (there is no choice point for it to prune), and no interpreted callee runs — so no cut barrier ever crosses a tier boundary. Nothing to pass or restore. |
| **G-4** at-most-one-solution proof (NOT just non-unifiable heads) | yes | `matching_deps` selects on `[]`/`[_|_]` (mutually exclusive) with a deterministic committed (`->`) body; `dep_to_req` is single-solution because its clause-1 cut commits and clause-2 is the catch-all — **its heads are NOT non-unifiable**, so the proof rests on the cut + catch-all, exactly G-4's requirement. Transitively the fused region has at most one solution, so committing drops no answer. |
| **G-5** activation handle on resume-state CP; cut invalidates own CP | **vacuous** | The recursion is compiled to a loop, so no resume-state CP is ever created — there is no activation to confuse and no own-CP to invalidate on cut. (A future nondet extension would re-introduce this obligation.) |

The region additionally DECLINES to the interpreter (never a wrong commit) for
anything outside the recognised shape: an unbound/open list tail, a non-`depends/4`
element, an unbound dependency (whose interpreter clause-1 head would *bind* it),
or an already-bound output — each covered by a stress test.

## Gates — both configs

Region ON is the committed default; region OFF via `UW_REGION1_OFF=1`.

| gate | region OFF | region ON (committed default) |
|---|---|---|
| term corpus (B1) | 51 / 51 | 51 / 51 |
| term differential (B2) | 2600 / 0 / 0 | 2600 / 0 / 0 |
| store corpus | 51 / 51 (identical to term) | 51 / 51 |
| store differential | 503 / 0 | 503 / 0 |

Region-on B2 answers were also checked **byte-identical** to region-off over the
full 2600-case corpus (`diff` of the two binaries' output), not merely
divergence-free vs SWI.

### Region-specific stress tests (Kimi scenarios)

`#[cfg(test)] mod region1_matching_deps_tests` in
`templates/targets/rust_wam/state.rs.mustache` (runs via `cargo test` in every
generated crate; 8/8 pass):

- `fusion_walk_and_dep_to_req_cut_decision` — the fused walk + the **determinism
  decision** (`alternatives(_) → req(_,any)` via the cut vs the catch-all).
- `compound_version_structural_equality` — `==` is structural (compound version).
- `empty_list_yields_empty_output` — base clause + `[]`/empty-list aliasing.
- `frame_pinning_argument_and_temporary_registers_preserved` — **G-1** arg-register
  + temporary-register liveness; **G-2** no CP left; output binding trailed for a
  caller's backtrack (**recursive-lowered-backtrack safety**).
- `decline_open_tail_restores_minimal_snapshot` — off-shape → decline with the P2
  minimal snapshot rolled back (no binding/trail/CP leak).
- `decline_bound_output`, `decline_non_depends_element`,
  `decline_unbound_dep_in_matching_element` — the determinism-decision decline paths.

## THE REAL A/B — wall-clock, drift-cancelling, interleaved

Two release binaries built from the same codegen: region-OFF (no
`matching_deps/4` arm in `lowered_call`) and region-ON. **Binary identity
verified** — distinct `sha256` and the OFF crate's `lib.rs` has no region arm
(heeding the F11 report's warning that `build.sh`'s `cargo | tail` can fail to
relink: each binary was copied out and hashed, confirming a genuine relink):

```
OFF sha256 64b5602ea0423af1c64919f91f76407ae92812f6ee03f2a053823f8a0aa9080a
ON  sha256 844decac7fa06a6b5d631b0efe01b61e262cbfba9ce58b08b79f1ed59a2e8207
```

Interleaved (OFF then ON each round) on the same 2600-case `cases.jsonl`, timing
the Rust leg only (the drift-cancelling method the F11 round used):

| round | OFF (ms) | ON (ms) | delta (ms) | delta (%) |
|---:|---:|---:|---:|---:|
| 1 | 23,075 | 21,694 | −1,381 | −6.0% |
| 2 | 23,260 | 21,538 | −1,722 | −7.4% |
| 3 | 23,173 | 21,740 | −1,433 | −6.2% |
| 4 | 23,447 | 22,288 | −1,159 | −4.9% |
| 5 | 24,493 | 21,510 | −2,983 | −12.2% |
| 6 | 23,283 | 21,601 | −1,682 | −7.2% |

**OFF median 23,272 ms; ON median 21,648 ms; delta median −1,624 ms = −6.98%.**
All six rounds negative. SWI on this box runs B2 in ≈2.6 s, so the target/SWI
ratio moves **≈9.0× (OFF) → ≈8.3× (ON)**.

### Confirm / refute verdict

**CONFIRMED.** The spike projected a **~6% B2 floor** from this family's
choice-point machinery share alone (18.7% of the 33% backtrack+restore_regs
envelope), with a realistic single-family gain of ~8–12% once P1's removed
step-dispatch is counted. The real measurement lands at **−7.0% median** — at/
just above the 6% machinery floor and inside the projected band. The cost model
was faithful; reality agrees. The minimal-snapshot mechanism (P2) is what makes
it a win where Stage 1's F11 (full `save_regs`) was net −7%.

## Default decision

**Default ON**, gated. The A/B is a clear, consistent win and every gate is
green in both configs, so the plan's "bank it if it helps" makes ON the default.
It remains disable-able (`region1(false)` option or `UW_REGION1_OFF=1`) and is
shape-gated, so non-resolver projects are unaffected regardless.

## Files changed (owned surface only)

- `src/unifyweaver/targets/wam_rust_target.pl` — `rust_region1_enabled/1`
  (default ON), `rust_region1_applicable/1` + the structural recognizer, and the
  `matching_deps/4` arm wired into `lowered_call` (deduped against the other
  banks).
- `templates/targets/rust_wam/state.rs.mustache` — `region_matching_deps_dispatch`
  / `region_dep_to_req` / `region_decline` (P1+P2+fusion, the G-gate argument in
  comments) and the `region1_matching_deps_tests` stress module.
- `examples/pkg_resolver/rust/uw_resolve_wam/**` — the committed term crate,
  regenerated via `build.sh` (default ON).
- `docs/reports/wam_rust_stage2_region1.md` — this report.

## Recommendation on the remaining regions — **GO**

The real measurement validates the mechanism end-to-end, not just the cost model:
one deterministic region, cleanly fused with a minimal snapshot and direct native
call, delivers its projected share of B2 with zero correctness cost and every
G-gate satisfiable by construction. The spike's coverage table puts the full
deterministic class at ~65–70% of B2 dispatches; the next regions
(`matching_versions/4 ⊕ satisfies/2`, the B3 index-build `key_dep_rows`/
`group_keyed` regions reusing `dep_to_req`/`same_key`, and re-enabling the F11
accessor bank under P2) are the same shape and should be built next, each behind
its own flag and its own real A/B, on the same prove-then-commit discipline. The
nondet resume extension (§7) remains a separate later round.

## Reproduce

```
# committed default (region ON):
bash examples/pkg_resolver/rust/build.sh
bash examples/pkg_resolver/rust/run_corpus_rust.sh          # 51/51
bash examples/pkg_resolver/rust/run_differential_rust.sh    # 2600/0
cd examples/pkg_resolver/rust/uw_resolve_wam && cargo test --release --lib region1   # 8/8

# region OFF:
UW_REGION1_OFF=1 bash examples/pkg_resolver/rust/build.sh
bash examples/pkg_resolver/rust/run_differential_rust.sh    # 2600/0

# store lane (both configs): 503/0
bash examples/pkg_resolver/rust_store/run_differential_rust_store.sh
UW_REGION1_OFF=1 bash examples/pkg_resolver/rust_store/run_differential_rust_store.sh
```
