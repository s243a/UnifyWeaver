<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM lowered tier — Stage 2 region 2 build + real A/B (matching_versions ⊕ satisfies ⊕ version_lt)

Implements region 2 of Stage 2 of
[`../proposals/WAM_RUST_LOWERED_TIER_THROUGHPUT_PLAN.md`](../proposals/WAM_RUST_LOWERED_TIER_THROUGHPUT_PLAN.md)
§5 (P1 direct native calls, P2 minimal-locals snapshot), following the exact
recipe [region 1](wam_rust_stage2_region1.md) validated. It fuses a **3-deep**
predicate chain — `matching_versions/4 → satisfies/2 → version_lt/2` (and
`version_lt`'s whole Debian §5.6.12 `segs_lt`/`pad_head`/`segs_lt_1`/`order_lt`/
`order_val` transitive chain) — into one native region, gated behind its own flag
and shape recognizer, with its OWN methods and stress tests. Scope is exactly
this one region; region 1 stays enabled and untouched throughout; no other region
and no nondet resume extension were touched. All numbers on the build box,
`LC_ALL=C.UTF-8`, release + LTO, against the frozen `resolver.pl` /
`resolver_store.pl` (neither modified).

## TL;DR — CONFIRMED, banked ON

The real interleaved A/B **confirms** the spike's projection for this family
(11.1% of B2 dispatches): fusing `matching_versions/4 ⊕ satisfies/2 ⊕
version_lt/2` removes a **median 7.06% of B2 wall time** (all six rounds
negative), moving the SWI ratio from ~8.9× to ~8.2×, with every correctness gate
green in both configs AND region-on output byte-identical to region-off over the
full 2600-case corpus. Per the plan's "bank it if it helps", the region is
**default ON** (gated; disable with `region2(false)` / `UW_REGION2_OFF=1`).

## The region and how a 3-deep chain is fused

```prolog
matching_versions([], _Name, _C, []).
matching_versions([package(N, V)|Rest], Name, C, Out) :-
    ( N == Name, satisfies(V, C) -> Out = [V|Vs] ; Out = Vs ),
    matching_versions(Rest, Name, C, Vs).

satisfies(_Ver, any).
satisfies(Ver, eq(E))         :- Ver = E.
satisfies(Ver, gte(G))        :- \+ version_lt(Ver, G).
satisfies(Ver, lte(G))        :- \+ version_lt(G, Ver).
satisfies(Ver, lt(H))         :- version_lt(Ver, H).
satisfies(Ver, gt(H))         :- version_lt(H, Ver).
satisfies(Ver, range(Lo, Hi)) :- \+ version_lt(Ver, Lo), version_lt(Ver, Hi).

version_lt(v(A,B,C), v(D,E,F))          :- ( A<D -> true ; ... ).
version_lt(deb(E1,U1,R1), deb(E2,U2,R2)) :- ( E1<E2 -> true ; E1=:=E2, segs_lt(U1,U2) -> true ; ... ).
% + segs_lt/2, pad_head/2, segs_lt_1/2, order_lt/2, order_val/2 (Debian §5.6.12)
```

`matching_versions/4` walks a package list; per element it runs the
`N==Name, satisfies(V,C)` guard. The region compiles the whole chain into ONE
native function entered once, `WamState::region_matching_versions_dispatch`:

- **P1 (direct native call).** `satisfies`, `version_lt`, `segs_lt`, `order_lt`
  and `order_val` are inlined as `region_satisfies` / `region_version_lt` /
  `region_segs_lt` / `region_segs_lt_1` / `region_order_lt` / `region_order_val`
  — plain Rust, never `vm.run()`. The per-element `execute` re-dispatch and the
  `Call satisfies` / `Call version_lt` / … are gone from the hot path.
- **P2 (minimal-locals snapshot).** Rollback saves only three scalars
  (`trail_len`, `heap_len`, `var_counter`) via the shared `region_decline` — NOT
  a `save_regs` register-file clone (the exact cost Stage 1's F11 lost to).
- **Fusion.** The `matching_versions` recursion becomes a native `loop`; the
  self-call never happens — no `Allocate`/`Deallocate`, no per-recursion choice
  point, no re-dispatch. Because the whole `satisfies`/`version_lt` chain is
  inlined too, fusing removes those dispatches as well, not just
  `matching_versions`' own.

The region reuses existing runtime machinery for correctness: `terms_identical`
(the exact `==/2`) for the `N==Name` guard and for `Ver = E` on ground terms and
for `O1==O2` on order-code lists; `deref_shallow` for list/element/segment
inspection.

### The shared callee — `satisfies/2` (and `version_lt/2`) — is inlined privately

Region 1's `dep_to_req/3` was called only from `matching_deps`. This region's
`satisfies/2` and `version_lt/2` are **shared** across many callers
(`filter_satisfies`, `layer_satisfies`, `provide_satisfies`, `cmp_ver`, …).
**Choice made: inline a PRIVATE native copy for the fused region only, leaving
the interpreted `satisfies`/`version_lt` fully intact.** Why this over a shared
lowering: it is behavior-identical at every other call site *by construction* —
the interpreted predicate is untouched, and only the `matching_versions/4`
dispatch arm routes to the native region — so no analysis of the other call
sites' modes or determinism is needed, and there is zero risk of changing them.
A dedicated stress test (`shared_satisfies_callee_is_pure_no_state_mutation`)
proves the private copies mutate no machine state (trail / choice points / heap /
var counter), so nothing can leak to the interpreted callers even indirectly.

### `version_lt` — the highest correctness risk — matched to SWI exactly

The native `version_lt` reproduces the two clauses (v/3 and deb/3) and the whole
Debian Policy §5.6.12 segment walk: epoch compared first; then upstream then
revision segment lists; each segment `s(OrderCodes, DigitInt)` compared
order-run-then-number; `~` (code 126) ranked `-1` so it sorts before everything
including end-of-string; letters ranked as themselves; every other byte ranked
`+256` so it sorts after letters; the missing-part `s([],0)` pad. It reuses the
runtime's `terms_identical` for the `O1==O2` order-code equality rather than
re-deriving it. It is validated against SWI's own `version_lt` over the full
`DEB_POOL` **10×10 truth table** (`debian_version_lt_truth_table_matches_swi`),
which covers `~rc`, `+dfsg`, epochs, and numeric-vs-alpha runs.

## G-gates (Kimi K2 soundness review) — how each is satisfied

| gate | applies? | how satisfied |
|---|---|---|
| **G-1** minimal saved set from backward liveness + all arg registers forced live | yes | Single rollback point (entry). Backward liveness reduces to "A1..A4 at entry", forced live by never being clobbered (the walk reads them into Rust locals, mutates no register — `determinism_no_choice_point_and_registers_preserved`). Minimal snapshot records `trail_len`/`heap_len`/`var_counter`; no frame allocated, no CP created, so none to drop. |
| **G-2** determinism guard (no CP / no interpreted code; assert CP-depth; deopt else) | yes | The region pushes NO choice point and calls NO interpreted code, so CP-depth is unchanged **by construction**; it is asserted after the walk (`choice_points.len() != cp_depth → decline`). Every off-shape input DECLINES (`None`): open tail, non-`package/2` element, bound output, unbound version/constraint, an unbound or **mixed v/deb** `version_lt` operand — each covered by a decline test. |
| **G-3** cut across tiers passed explicitly | **no cut crosses a boundary** | `matching_versions` and `satisfies` have no cut. The cuts in the `version_lt` chain (`segs_lt`/`pad_head`/`order_lt`/`order_val`) are realised as Rust `if/else`/`match` (no choice point to prune), and no interpreted callee runs — so no cut barrier crosses a tier boundary. Nothing to pass or restore. |
| **G-4** at-most-one-solution proof (NOT just non-unifiable heads) | yes | `matching_versions` selects on `[]`/`[_|_]` (exclusive) with a committed (`->`) body. **`satisfies` is single-solution**: its 7 clauses are mutually exclusive on the ground constraint functor (any/eq/gte/lte/lt/gt/range) and each body is semidet (`Ver=E` on ground terms; `\+ version_lt(...)`, negation semidet; conjunctions of semidet goals) — the proof rests on this, NOT on head non-unifiability (the heads share the first arg and are NOT non-unifiable). **`version_lt` is single-solution**: its two heads ARE non-unifiable (v/3 vs deb/3), AND each body is a committed if-then-else chain; `segs_lt`/`order_lt` likewise commit via `->`/cuts on mutually-exclusive list-shape heads. Transitively the fused region has at most one solution, so committing drops no answer. This is exactly G-4's requirement for `satisfies`/`version_lt` genuinely being single-solution for the call patterns used. |
| **G-5** activation handle on resume-state CP; cut invalidates own CP | **vacuous** | The recursion is compiled to a loop, so no resume-state CP is ever created — there is no activation to confuse and no own-CP to invalidate on cut. (Same as region 1; a future nondet extension would re-introduce this obligation.) |

## Gates — both configs

Region ON is the committed default; region OFF via `UW_REGION2_OFF=1`. Region 1
stays enabled in both configs (the OFF build isolates region 2 only).

| gate | region OFF | region ON (committed default) |
|---|---|---|
| term corpus (B1) | 51 / 51 | 51 / 51 |
| term differential (B2) | 2600 / 0 / 0 | 2600 / 0 / 0 |
| store corpus | 51 / 51 | 51 / 51 |
| store differential | 503 / 0 | 503 / 0 |

Region-on B2 answers were also checked **byte-identical** to region-off over the
full 2600-case corpus — the two binaries' output has the **same sha256**
(`8b635990ae7cb47d502b15fb6be369f6d4deb992d78ea9e485708aa3d34364b1`), not merely
divergence-free vs SWI. The store lane is untouched by the region (its
`matching_versions_store`/renamed callees never match the recognizer), so store
ON ≡ store OFF by construction.

### Region-specific stress tests (Kimi scenarios)

`#[cfg(test)] mod region2_matching_versions_tests` in
`templates/targets/rust_wam/state.rs.mustache` (runs via `cargo test` in every
generated crate; **15/15 pass**, region 1's 8/8 unaffected, full lib 157/157):

- `debian_version_lt_truth_table_matches_swi` — the full DEB_POOL **10×10** truth
  table vs SWI (epoch, `~rc`, `+dfsg`, numeric runs).
- `version_lt_specific_debian_edges` — named edges: `~rc1 < 1.0`, `1.0 < 1.0+dfsg1`,
  epoch dominance (`1:1.0 < 2:0.1`), revision compare (`-1 < -2`), numeric-not-lexical
  (`36<38`, `2<10`).
- `version_lt_v3_triples` — the v/3 path.
- `satisfies_all_seven_clauses` — the determinism decision across all 7 constraint
  functors.
- `fused_walk_keeps_only_matching_and_satisfying`, `fused_walk_with_debian_versions_and_any`,
  `empty_list_yields_empty_output` — the fused walk semantics.
- `determinism_no_choice_point_and_registers_preserved` — G-1/G-2/G-4 + trailed output.
- `shared_satisfies_callee_is_pure_no_state_mutation` — the **shared-callee invariance**.
- `decline_open_tail_restores_minimal_snapshot`, `decline_non_package_element`,
  `decline_bound_output`, `decline_unbound_version_in_matching_element`,
  `decline_unbound_constraint`, `decline_mixed_v_deb_operand` — the decline paths.

## THE REAL A/B — wall-clock, drift-cancelling, interleaved

Two release binaries built from the same codegen: region-OFF (no
`matching_versions/4` arm in `lowered_call`, region 1 still ON) and region-ON.
**Binary identity verified** — distinct `sha256`, and the OFF crate's `lib.rs`
has no `region_matching_versions_dispatch` arm (heeding the F11 relink caveat:
each binary was copied out and hashed, confirming a genuine relink, and the
OFF/ON codegen was diffed to confirm the arm is the only routing difference):

```
OFF sha256 f9f141604757efe3ae76c35f531e4b189e85a2cb13ffced10f1182faa39a041f
ON  sha256 755e520982105bc16b3e58e6f0165239d123625ee2a8d7bbbb84a39c9ad57b2f
```

Interleaved (OFF then ON each round) on the same 2600-case `cases.jsonl`, timing
the Rust leg only (the drift-cancelling method regions used before):

| round | OFF (ms) | ON (ms) | delta (ms) | delta (%) |
|---:|---:|---:|---:|---:|
| 1 | 21,869 | 20,595 | −1,274 | −5.83% |
| 2 | 22,327 | 20,483 | −1,844 | −8.26% |
| 3 | 21,996 | 20,383 | −1,613 | −7.33% |
| 4 | 21,929 | 20,441 | −1,488 | −6.79% |
| 5 | 21,572 | 20,322 | −1,250 | −5.79% |
| 6 | 22,409 | 20,272 | −2,137 | −9.54% |

**OFF median 21,962.5 ms; ON median 20,412.0 ms; delta median −1,550.5 ms =
−7.06%** (mean −7.26%). All six rounds negative. SWI on this box runs B2 in
≈2.48 s, so the target/SWI ratio moves **≈8.9× (OFF) → ≈8.2× (ON)**.

### Confirm / refute verdict

**CONFIRMED.** The spike put this family at **11.1% of B2 dispatches** (vs region
1's 15.2%, which measured −6.98%). A naïve proportional projection would put
region 2 near −5%, but the real measurement lands at **−7.06%** — *better* than
proportional, because fusing the whole `satisfies`/`version_lt`/`segs_lt` chain
removes those dispatches too, not just `matching_versions`' own. The cost model
held; reality agrees, and the minimal-snapshot mechanism (P2) is again what makes
a deterministic region a clear win.

## Default decision

**Default ON**, gated. The A/B is a clear, consistent win (all six rounds
negative, −7% median) and every gate is green in both configs with byte-identical
output, so the plan's "bank it if it helps" makes ON the default. It remains
disable-able (`region2(false)` option or `UW_REGION2_OFF=1`) and is shape-gated
(the frozen shape of ALL of matching_versions/satisfies/version_lt and the whole
version_lt chain is verified with `=@=`), so non-resolver projects are unaffected
regardless.

## Files changed (owned surface only)

- `src/unifyweaver/targets/wam_rust_target.pl` — `rust_region2_enabled/1`
  (default ON), `rust_region2_applicable/1` + the eight structural recognizers
  (matching_versions, satisfies, version_lt, segs_lt, pad_head, segs_lt_1,
  order_lt, order_val), and the `matching_versions/4` arm wired into
  `lowered_call` alongside region 1's.
- `templates/targets/rust_wam/state.rs.mustache` —
  `region_matching_versions_dispatch` / `region_satisfies` / `region_version_lt`
  / `region_segs_lt` / `region_segs_lt_1` / `region_seg_parts` /
  `region_order_lt` / `region_order_val` / `region_int` /
  `region_codes_identical` / `region_is_ground` / `region_proper_list` (P1+P2+
  fusion, the G-gate argument in comments) and the
  `region2_matching_versions_tests` stress module.
- `examples/pkg_resolver/rust/uw_resolve_wam/**` — the committed term crate,
  regenerated via `build.sh` (default ON).
- `docs/reports/wam_rust_stage2_region2.md` — this report.

## Recommendation on the remaining regions — **GO**

Two deterministic regions now validate the mechanism end-to-end, and region 2
proves the recipe extends cleanly to (a) a **3-deep** chain and (b) a **shared**
callee (inline-a-private-copy is the low-risk answer) and (c) **non-trivial
arithmetic** (the Debian segment walk matched SWI exactly via a truth-table
gate). The next regions are the same shape:

- **B3 index build — `key_dep_rows`/`group_keyed`** (reusing `dep_to_req`/
  `same_key`): **GO.** Same det-body/det-callee fusion; `dep_to_req` is already
  a validated native (region 1), so this is largely wiring a second entry point.
- **Re-enabling the F11 accessor bank under P2**: **ADJUST, measure first.** F11
  was banked OFF because its per-call *full* register snapshot cost about what it
  saved on shallow accessors. The P2 minimal snapshot is exactly the lever that
  flipped regions 1 and 2; applying it to the accessor bank is the right next
  experiment, but accessors are shallower than these fused walks, so it must be
  its own real A/B before banking — do not assume the region wins carry over.

Each remaining region stays behind its own flag and its own real A/B, on the same
prove-then-commit discipline. The nondet resume extension (§7) remains a separate
later round.

## Reproduce

```
# committed default (region 2 ON, region 1 ON):
bash examples/pkg_resolver/rust/build.sh
bash examples/pkg_resolver/rust/run_corpus_rust.sh          # 51/51
bash examples/pkg_resolver/rust/run_differential_rust.sh    # 2600/0
cd examples/pkg_resolver/rust/uw_resolve_wam && cargo test --release --lib region2   # 15/15

# region 2 OFF (region 1 stays ON):
UW_REGION2_OFF=1 bash examples/pkg_resolver/rust/build.sh
bash examples/pkg_resolver/rust/run_differential_rust.sh    # 2600/0

# store lane (region-independent): 503/0, 51/51
bash examples/pkg_resolver/rust_store/run_differential_rust_store.sh
bash examples/pkg_resolver/rust_store/run_corpus_rust_store.sh
```
