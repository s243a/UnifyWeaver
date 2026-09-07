<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM lowered tier — Stage 2 region 3 build + real B3 A/B (the two B3 index builders)

Implements the B3 index-builder regions of Stage 2 of
[`../proposals/WAM_RUST_LOWERED_TIER_THROUGHPUT_PLAN.md`](../proposals/WAM_RUST_LOWERED_TIER_THROUGHPUT_PLAN.md)
§5 (P1 direct native calls, P2 minimal-locals snapshot), following the exact
recipe validated in [region 1](wam_rust_stage2_region1.md) and
[region 2](wam_rust_stage2_region2.md). Two separate fusions, each behind its own
flag / shape recognizer / methods / tests:

- **Region 3a: `key_dep_rows/3 ⊕ dep_to_req/3`** — reuses region 1's already-validated
  `region_dep_to_req` native.
- **Region 3b: `group_keyed/2 ⊕ same_key/4`** — a new native inline of `same_key`.

Scope is exactly these two regions; regions 1 and 2 stay enabled and untouched
throughout; no other region and no nondet resume extension were touched. All
numbers on the build box, `LC_ALL=C.UTF-8`, release + LTO, against the frozen
`resolver.pl` / `resolver_store.pl` (neither modified).

## TL;DR — CONFIRMED, both banked ON

Unlike regions 1/2 (B2 walks, ~7% each), these are the **B3 index builders**,
and B3's `resolve_layered` is dominated by them: the census puts `same_key`
(24.6%), `key_dep_rows` (16.4%) + `dep_to_req` (16.4%), and `group_keyed`
(10.9%) at **~68% of B3 dispatches**. Fusing both pairs into native regions
removes essentially that whole share. The real interleaved A/B on B3 (the 5000-
package `resolve_layered`) measures a **median −56% of B3 resolve wall time**
(≈1996 ms → ≈872 ms, all six rounds negative), with **byte-identical** output
and every correctness gate green in both configs on both lanes. Per the plan's
"bank it if it helps", **both regions are default ON** (gated; disable with
`region3a(false)`/`UW_REGION3A_OFF=1` and `region3b(false)`/`UW_REGION3B_OFF=1`).
A B2 spot-check shows no regression (slight improvement). The store lane's
`resolve_layered_store` does not call the index builders, so the regions are
**inert there** — neutral perf, identical correctness.

## The two regions and how each is fused

### Region 3a — `key_dep_rows/3 ⊕ dep_to_req/3`

```prolog
key_dep_rows([], _I, []).
key_dep_rows([depends(N, V, D, C)|Rest], I, [(N-V)-I-Req|Ks]) :-
    dep_to_req(D, C, Req),
    I1 is I + 1,
    key_dep_rows(Rest, I1, Ks).
```

This is region 1's `matching_deps/4` shape — a list walk whose recursion is the
last goal and whose output list is built top-down in the clause head — **minus**
the `N==Name, V==Ver` filter (EVERY element yields a row) and **plus** a monotone
position counter `I` (`I1 is I+1`). It compiles to one native
`WamState::region_key_dep_rows_dispatch`:

- **P1 (direct native call).** `dep_to_req` is inlined as `region_dep_to_req` —
  region 1's already-validated native, reused verbatim (a plain Rust call, never
  `vm.run()`).
- **P2 (minimal-locals snapshot).** Rollback saves only three scalars
  (`trail_len`, `heap_len`, `var_counter`) via the shared `region_decline`.
- **Fusion.** The recursion becomes a native `loop`; the self-call never happens
  — no `Allocate`/`Deallocate`, no per-recursion choice point, no re-dispatch.
  The integer position is a plain Rust `i64` local.

The output row `(N-V)-I-Req` = `-(-(-(N,V),I),Req)` is built with the bare-`-`
functor form (the same convention region 1's `req(...)` uses; `functor_of`
normalises it everywhere the interpreter's `"-/2"` is compared/sorted/serialised).

### Region 3b — `group_keyed/2 ⊕ same_key/4`

```prolog
group_keyed([], []).
group_keyed([K-_-X|Rest], [K-[X|Xs]|Gs]) :-
    same_key(Rest, K, Xs, Rest1),
    group_keyed(Rest1, Gs).

same_key([], _K, [], []).
same_key([K2-I-X|Rest], K, Xs, Rest1) :-
    ( K2 == K -> Xs = [X|Xs1], same_key(Rest, K, Xs1, Rest1)
    ;           Xs = [], Rest1 = [K2-I-X|Rest] ).
```

`group_keyed/2` walks a SORTED keyed-row list and groups consecutive rows sharing
a key, using `same_key/4` (called only from here) to consume each run. It compiles
to one native `WamState::region_group_keyed_dispatch`:

- **P1 (direct native).** `same_key` is inlined as the region's inner loop — no
  `vm.run()`, no per-run dispatch.
- **P2 (minimal snapshot).** The same three-scalar `region_decline` rollback.
- **Fusion + non-tail handling** (see below).

**Genericity.** The row shape `K-_-X` = `-(-(K,_),X)` matches BOTH of
`group_keyed`'s call sites — the dep rows `(N-V)-I-Req` (K = N-V, a compound key)
and the pkg rows `N-I-V` (K = N, an atom key) — so one region serves both, via the
shared `region_key_val` destructurer.

## Non-tail handling (the census called these "non-tail")

Both `key_dep_rows` and `group_keyed` have their **own** recursion as a last-call
with the output cons cell built in the clause head (a fresh tail variable filled
by the recursion). That is exactly regions 1/2's "native loop building the output
vector top-down" shape — no explicit stack needed.

The genuinely non-tail structure is `group_keyed` calling the helper walk
`same_key` and then resuming at the leftover `Rest1`. This is handled with an
**explicit native accumulator over the materialised input** (option (a) of the
task): an outer index `i` marks the start of each group and an inner index `j`
scans the same-key run; the outer loop resumes at `i = j` — precisely same_key's
`Rest1 = [K2-I-X|Rest]`. Solution order is preserved (groups in first-seen key
order, each group's values in list order) and the P2 minimal-snapshot rollback
still bounds the commit. **No resume-state choice point is created**, so this
stays a deterministic region — the nondet resume round is not entered.

## G-gates (Kimi K2 soundness review) — how each is satisfied

Both regions share the same argument; per-region specifics noted.

| gate | applies? | how satisfied |
|---|---|---|
| **G-1** minimal saved set from backward liveness + all arg registers forced live; frame pinned | yes | Single rollback point (entry). Backward liveness reduces to the argument registers at entry (A1..A3 for 3a, A1..A2 for 3b), **forced live by never being clobbered** — the walk reads them into Rust locals and mutates no register (`frame_pinning_registers_preserved` in each module). Minimal snapshot records `trail_len`/`heap_len`/`var_counter`; no frame allocated, no CP created. |
| **G-2** determinism guard (no CP / no interpreted code; assert CP-depth; deopt else) | yes | Each region pushes NO choice point and calls NO interpreted code, so CP-depth is unchanged **by construction** and asserted after the walk (`choice_points.len() != cp_depth → decline`). Every off-shape input DECLINES (`None`): 3a — open tail, non-`depends/4` element, unbound dep (dep_to_req clause-1 would bind it), bound output, non-integer index; 3b — open tail, non-`-(-(_,_),_)` element, bound output. Each covered by a decline test. |
| **G-3** cut across tiers passed explicitly | **no cut crosses a boundary** | 3a: `key_dep_rows` has no cut; `dep_to_req`'s clause-1 cut is a Rust `if/else` in `region_dep_to_req`. 3b: `group_keyed` has no cut; `same_key`'s `K2==K ->` is a Rust `if/else` (no choice point to prune). No interpreted callee runs, so no cut barrier crosses a tier boundary. |
| **G-4** at-most-one-solution proof (NOT just non-unifiable heads) | yes | 3a: `key_dep_rows` selects on []/[_|_] (exclusive) with a committed body (`dep_to_req` det, `is/2` functional, then the tail recursion); `dep_to_req` is single-solution via its clause-1 cut + clause-2 catch-all — its heads are **NOT** non-unifiable, so the proof rests on the cut (region 1's proof, reused). 3b: both `group_keyed` and `same_key` select on []/[_|_] (exclusive) with committed bodies; `same_key`'s `K2==K` is `==/2` (structural, **never binds**), so its if-then-else is genuinely semidet, and grouping is a deterministic function of the (ground, sorted) input. Both transitively at most one solution — committing drops no answer. |
| **G-5** activation identity / cut-invalidation on a resume-state CP | **vacuous** | Every recursion (3a's loop, 3b's outer+inner loops) is compiled to a native loop, so no resume-state CP is ever created — there is no activation to confuse and no own-CP to invalidate on cut. (A future nondet extension would re-introduce this obligation.) |

## Gates — both configs, both lanes

Regions ON is the committed default; OFF via `UW_REGION3A_OFF=1 UW_REGION3B_OFF=1`
(regions 1+2 stay ON in both configs — the OFF build isolates region 3 only).

| gate | region 3 OFF | region 3 ON (committed default) |
|---|---|---|
| term corpus (B1) | 51 / 51 | 51 / 51 |
| term differential (B2) | 2600 / 0 / 0 | 2600 / 0 / 0 |
| store corpus | 51 / 51 | 51 / 51 |
| store differential | 503 / 0 | 503 / 0 |

The B3 term-lane `resolve_layered` (5000 packages) output was checked
**byte-identical** OFF vs ON (`diff` of the two binaries' output), selection_size
10 both. The store-lane scale probe (via `run_scale_rust_store.sh`, which builds
the store-embedding binary against the matching scale store) returns the identical
correct selection (10) in both configs.

### Region-specific stress tests (Kimi scenarios)

`#[cfg(test)] mod region3a_key_dep_rows_tests` (9/9) and
`mod region3b_group_keyed_tests` (9/9) in
`templates/targets/rust_wam/state.rs.mustache` (via `cargo test`; full lib
**175/175**, regions 1/2's 8+15 unaffected):

- **3a:** `fusion_walk_index_and_dep_to_req_cut` (the fused walk + monotone index
  + dep_to_req's alternatives→any cut), `nonzero_start_index`,
  `empty_list_yields_empty_output`, `frame_pinning_registers_preserved` (G-1/G-2 +
  trailed output), and the decline paths `decline_open_tail_restores_minimal_snapshot`,
  `decline_bound_output`, `decline_non_depends_element`, `decline_unbound_dep`,
  `decline_non_integer_index`.
- **3b:** `groups_dep_row_shape_with_compound_keys`,
  `groups_pkg_row_shape_with_atom_keys` (both call-site shapes),
  `all_distinct_keys_are_singleton_groups`, `key_equality_is_structural`
  (structural `==` over compound keys), `empty_list_yields_empty_output`,
  `frame_pinning_registers_preserved`, and the decline paths
  `decline_open_tail_restores_minimal_snapshot`, `decline_bound_output`,
  `decline_non_pair_element`.

## THE REAL A/B — wall-clock, drift-cancelling, interleaved, on B3

Two release binaries from the same codegen: region-3-OFF (no `key_dep_rows/3` or
`group_keyed/2` arm in `lowered_call`; regions 1+2 still ON) and region-3-ON.
**Binary identity verified** — distinct `sha256`, and the OFF crate's `lib.rs`
has zero `region_key_dep_rows_dispatch`/`region_group_keyed_dispatch` arms
(genuine relink; heeding the F11 relink caveat, each binary was copied out and
hashed):

```
OFF sha256 6e7e80ac914fcaac2f0478244c60e00f7b225b3587dc60a86eed9d5a4f9af057
ON  sha256 fd171003a273c57f2e2caaf19aa4d5f3d87002be5240777de233cd4eaabcd232
```

Interleaved (OFF then ON each round) on the same 5000-package `resolve_layered`
(`case_5000.json`), timing the `resolve_ms` leg (the drift-cancelling method the
region rounds used):

| round | OFF (ms) | ON (ms) | delta (ms) | delta (%) |
|---:|---:|---:|---:|---:|
| 1 | 1,993.3 | 867.1 | −1,126.3 | −56% |
| 2 | 2,018.2 | 853.6 | −1,164.6 | −57% |
| 3 | 1,986.1 | 876.1 | −1,110.1 | −55% |
| 4 | 2,022.7 | 893.4 | −1,129.3 | −55% |
| 5 | 1,999.7 | 841.7 | −1,158.0 | −57% |
| 6 | 1,914.3 | 891.9 | −1,022.5 | −53% |

**OFF median ≈1,996 ms; ON median ≈872 ms; delta median ≈−1,124 ms = −56%.**
All six rounds negative. `load_ms` (JSON→term, not the index path) is unchanged
(~22–25 ms either way).

### Confirm / refute verdict

**CONFIRMED — and larger than regions 1/2 by design.** Regions 1/2 each removed
~7% of B2 because they are one B2 family apiece. Regions 3a+3b remove the B3
index-build bottleneck, which the census puts at ~68% of B3 `resolve_layered`
dispatches (same_key + key_dep_rows + dep_to_req + group_keyed). The real
measurement — **−56% of B3 resolve wall time** — matches that: fusing both pairs
into deterministic native loops with the P2 minimal snapshot turns the
interpreter's dominant B3 work into a pair of tight Rust loops. The
minimal-snapshot mechanism (P2) is again what makes deterministic regions a clear
win.

### B2 spot-check (no regression)

Term-lane B2 (2600-case rust leg), interleaved OFF vs ON, 3 rounds:

| round | OFF (ms) | ON (ms) |
|---:|---:|---:|
| 1 | 20,090 | 19,705 |
| 2 | 20,089 | 19,747 |
| 3 | 20,096 | 19,822 |

ON is consistently ≤ OFF (~1.7% faster) — **no regression**, a slight gain
because the subset of B2 cases whose catalogs exceed the `index_threshold(64)`
also build the index (the census records `same_key` at 8,708 B2 dispatches).

### Store lane

The store lane's `resolve_layered_store/3` does not call `index_catalog` /
`key_dep_rows` / `group_keyed` (it is a store-backed lazy path using
`collect_deps_store` / `pick_need_store`), so the regions — although present in the
store crate (which compiles `resolver.pl` alongside `resolver_store.pl`, and their
predicates are **not** `_store`-renamed, unlike regions 1/2's) — are **never
invoked at runtime by the store resolve**. The store B3 scale probe is neutral
(OFF 0.038 s / ON 0.040 s, selection 10 both) and the store differential is 503/0
in both configs. (Note: the store `--scale-probe` binary EMBEDS a specific store,
so it must be built via `run_scale_rust_store.sh`, which sets `STORE_DIR` to the
matching scale store; a bare `build.sh` builds against a different store and the
probe then fails — a pre-existing quirk that the pristine regions-1+2 baseline
reproduces identically, unrelated to region 3.)

## Default decision

**Both regions default ON**, gated. The B3 A/B is a large, consistent win (all
six rounds negative, −56% median) with byte-identical output and every gate green
in both configs on both lanes, so the plan's "bank it if it helps" makes ON the
default. Each remains independently disable-able (`region3a(false)`/
`UW_REGION3A_OFF=1`, `region3b(false)`/`UW_REGION3B_OFF=1`) and each is
shape-gated (the frozen shape of `key_dep_rows`/`dep_to_req` for 3a and
`group_keyed`/`same_key` for 3b is verified with `=@=`), so non-resolver projects
are unaffected regardless.

## Files changed (owned surface only)

- `src/unifyweaver/targets/wam_rust_target.pl` — `rust_region3a_enabled/1` +
  `rust_region3a_applicable/1` + `rust_region3a_key_dep_rows_ok` (reusing region
  1's `rust_region1_dep_to_req_ok`); `rust_region3b_enabled/1` +
  `rust_region3b_applicable/1` + `rust_region3b_group_keyed_ok` +
  `rust_region3b_same_key_ok`; and the `key_dep_rows/3` and `group_keyed/2` arms
  wired into `lowered_call` alongside regions 1/2's (each deduped against the
  other banks).
- `templates/targets/rust_wam/state.rs.mustache` —
  `region_key_dep_rows_dispatch` (region 3a, reusing `region_dep_to_req`) and
  `region_group_keyed_dispatch` + `region_key_val` (region 3b), both with the
  G-gate argument in comments, and the `region3a_key_dep_rows_tests` /
  `region3b_group_keyed_tests` stress modules.
- `examples/pkg_resolver/rust/uw_resolve_wam/**` — the committed term crate,
  regenerated via `build.sh` (default ON, all four region arms).
- `docs/reports/wam_rust_stage2_region3.md` — this report.

## Recommendation on what remains

- **F11 accessor bank re-enable under P2** — **ADJUST, measure first** (unchanged
  from region 2's recommendation). F11 was banked OFF because its per-call *full*
  register snapshot cost about what it saved on shallow accessors; the P2 minimal
  snapshot is the lever that flipped regions 1/2/3, and applying it to the
  accessor bank is the right next experiment — but accessors are shallower than
  these fused walks, so it needs its own real A/B before banking.
- **The nondet resume round (§7)** — still a separate later round. B3's remaining
  top contributor is `build_tree/4` (21.9% of B3 dispatches), a non-tail binary
  tree builder that threads a difference-list `Rest` accumulator; it is
  deterministic-in-effect for the frozen shape and may be lowerable as a native
  explicit-stack region like 3b — but it should be assessed carefully for whether
  a resume-state CP is required (if so, it belongs in the nondet round). The
  genuine backtracking drivers (`pick/7`, `blocked_from/4`, `dep_breaks/5`) remain
  the nondet round's target.

## Reproduce

```
# committed default (regions 3a+3b ON, regions 1+2 ON):
bash examples/pkg_resolver/rust/build.sh
bash examples/pkg_resolver/rust/run_corpus_rust.sh          # 51/51
bash examples/pkg_resolver/rust/run_differential_rust.sh    # 2600/0
cd examples/pkg_resolver/rust/uw_resolve_wam && cargo test --release --lib region3   # 18/18
bash examples/pkg_resolver/rust/run_scale_rust.sh 5000      # resolve_ms ~840, selection 10

# region 3 OFF (regions 1+2 stay ON):
UW_REGION3A_OFF=1 UW_REGION3B_OFF=1 bash examples/pkg_resolver/rust/build.sh
bash examples/pkg_resolver/rust/run_differential_rust.sh    # 2600/0
bash examples/pkg_resolver/rust/run_scale_rust.sh 5000      # resolve_ms ~2000, selection 10

# store lane (both configs): 503/0, 51/51; scale probe via the official script
bash examples/pkg_resolver/rust_store/run_differential_rust_store.sh
bash examples/pkg_resolver/rust_store/run_scale_rust_store.sh   # selection 10, ~0.04 s
```
