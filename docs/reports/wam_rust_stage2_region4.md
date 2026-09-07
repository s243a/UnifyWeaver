<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM lowered tier — Stage 2 region 4 build + real B3 A/B (`build_tree/4`)

Implements the last B3 index builder of Stage 2 of
[`../proposals/WAM_RUST_LOWERED_TIER_THROUGHPUT_PLAN.md`](../proposals/WAM_RUST_LOWERED_TIER_THROUGHPUT_PLAN.md)
§5 (P1 direct native calls, P2 minimal-locals snapshot), following the recipe
validated in [region 1](wam_rust_stage2_region1.md), [region 2](wam_rust_stage2_region2.md)
and [region 3](wam_rust_stage2_region3.md). One fusion, behind its own flag /
shape recognizer / method / tests:

- **Region 4: `build_tree/4`** — the balanced-BST builder over a difference list
  that `list_to_tree/2` drives, native inlined as a bounded (log-depth) explicit
  recursion over the materialised input.

Scope is exactly this region; regions 1/2/3 stay enabled and untouched
throughout; no nondet resume extension was touched. All numbers on the build box,
`LC_ALL=C.UTF-8`, release + LTO, against the frozen `resolver.pl` /
`resolver_store.pl` (neither modified).

## TL;DR — CONFIRMED, banked ON

`build_tree/4` is the B3 census's remaining top single contributor at **~21.9% of
B3 `resolve_layered` dispatches** — the piece regions 3a/3b (which removed the
`same_key`/`key_dep_rows`/`dep_to_req`/`group_keyed` ~68%) left on the table. It
is a divide-and-conquer balanced-tree builder, and the shape assessment (below)
is that it is **deterministic — at most one solution, no backtracking, no
resume-state choice point** — so it belongs to the deterministic tier, not the
nondet round. It is fused into one native `WamState::region_build_tree_dispatch`.

The real interleaved A/B on B3 (the 5000-package `resolve_layered`), **on top of
regions 1/2/3 which stay ON in both legs**, measures a **median −41% of B3
resolve wall time** (≈911 ms → ≈539 ms, all six rounds negative), with
**byte-identical** output (selection_size 10 both) and every correctness gate
green in both configs on both lanes. Per the plan's "bank it if it helps",
**region 4 is default ON** (gated; disable with `region4(false)` /
`UW_REGION4_OFF=1`). The store lane's `resolve_layered_store` does not call
`build_tree` on its resolve path, so the region is **inert there** — neutral
perf, identical correctness.

## The region and how it is fused

```prolog
list_to_tree(Pairs, Tree) :-
    length(Pairs, N),
    build_tree(N, Pairs, Tree, []).

build_tree(N, Pairs, Tree, Rest) :-
    (   N =:= 0
    ->  Tree = t,
        Rest = Pairs
    ;   NL is (N - 1) // 2,
        NR is N - 1 - NL,
        build_tree(NL, Pairs, L, [K-V|Mid]),
        build_tree(NR, Mid,   R, Rest),
        Tree = t(L, K, V, R)
    ).
```

`build_tree/4` builds a balanced binary search tree from a sorted list by
divide-and-conquer: it consumes the first `N` elements of `Pairs` in list order —
each a `K-V` pair (region 3b's grouped rows `K-[X|Xs]` = `-(K, Group)`) that
becomes exactly one tree node — and threads the leftover suffix out through the
difference-list argument `Rest`. It compiles to one native
`WamState::region_build_tree_dispatch`:

- **P1 (direct native).** No `vm.run()`, no per-recursion dispatch: the whole
  divide-and-conquer runs in Rust.
- **P2 (minimal-locals snapshot).** Rollback saves only three scalars
  (`trail_len`, `heap_len`, `var_counter`) via the shared `region_decline`.
- **Fusion + non-tail handling** (see below).

### Non-tail handling (build_tree is genuinely non-tail)

`build_tree/4` is NOT last-call: it makes two self-calls and *then* builds the
node `t(L, K, V, R)`. That is a genuine tree recursion, not regions 1/2/3a's
"native loop building the output vector top-down". It is handled with a
**bounded native recursion over the materialised input** (region 3b's option-(a)
explicit-accumulator idea generalised to a tree): the input list is walked once
into a `Vec` of the first `N` element terms (capturing the raw list tail at
position `N` as the difference-list `Rest`), and a small associated recursion
`region_build_tree_rec(elems, start, n)` produces `(Tree, next_index)` by exactly
`build_tree`'s split. Because the split is **balanced** — `NL = (N-1)//2` and
`NR = N-1-NL` differ by at most one — the recursion depth is **O(log N)** (≈30
for a billion elements), so it is safe as a native recursion (equivalently an
explicit stack of depth O(log N)); there is no interpreter recursion, no
per-call frame, and **no resume-state choice point** — so this stays a
deterministic region and the nondet round is not entered.

The difference-list `Rest` is resolved by capturing the raw list suffix at
position `N` during the input walk and unifying `A4` with it: when `N` equals the
list length (the `list_to_tree` call, `Rest = []`) the suffix is `[]`; the
`N =:= 0` base binds `Rest = Pairs` (the whole, unconsumed input); an internal
partial consume leaves the correct suffix. All three are covered by tests.

## G-gates (Kimi K2 soundness review) — how each is satisfied

| gate | applies? | how satisfied |
|---|---|---|
| **G-1** minimal saved set from backward liveness + all arg registers forced live; frame pinned | yes | Single rollback point (entry). Backward liveness reduces to the argument registers at entry (A1 = N, A2 = Pairs, A4 = Rest), **forced live by never being clobbered** — the walk reads them into Rust locals and mutates no register before the two final unifies (`frame_pinning_registers_preserved`). Minimal snapshot records `trail_len`/`heap_len`/`var_counter`; no frame allocated, no CP created. |
| **G-2** determinism guard (no CP / no interpreted code; assert CP-depth; deopt else) | yes | The region pushes NO choice point and calls NO interpreted code, so CP-depth is unchanged **by construction** and asserted after the build (`choice_points.len() != cp_depth → decline`). Every off-shape input DECLINES (`None`): non-integer / negative `N`, bound `Tree` output, open/improper tail, a consumed element that is not a `-/2` pair, or a list shorter than `N` (which the interpreter would *fail* on — declining lets it produce that failure). Each covered by a decline test. |
| **G-3** cut across tiers passed explicitly | **no cut crosses a boundary** | `build_tree/4` has no cut (the `->` is realised as a Rust `if`), and no interpreted callee runs, so no cut barrier crosses a tier boundary. |
| **G-4** at-most-one-solution proof (NOT just non-unifiable heads) | yes | **One clause**, so clause selection is trivially deterministic. The body's `N =:= 0 ->` is `=:=/2` (arithmetic, never binds, semidet — a hard commit leaving no CP); the two `is/2` goals are functional; and both self-calls are at-most-one by induction on `N` (the head-unification `[K-V|Mid]` on the leftover can only *fail*, never branch). So the region has at most one solution and committing drops no answer. Head non-unifiability is not even in play (single clause); the proof rests on the deterministic body. |
| **G-5** activation identity / cut-invalidation on a resume-state CP | **vacuous** | The recursion is a native (bounded, log-depth) recursion with no resume-state CP, so there is no activation to confuse and no own-CP to invalidate on cut. (A future nondet extension would re-introduce this obligation — but build_tree is deterministic, so it never needs one.) |

## Gates — both configs, both lanes

Region 4 ON is the committed default; OFF via `UW_REGION4_OFF=1` (regions 1/2/3
stay ON in both configs — the OFF build isolates region 4 only).

| gate | region 4 OFF | region 4 ON (committed default) |
|---|---|---|
| term corpus (B1) | 51 / 51 | 51 / 51 |
| term differential (B2) | 2600 / 0 / 0 | 2600 / 0 / 0 |
| store corpus | 51 / 51 | 51 / 51 |
| store differential | 503 / 0 | 503 / 0 |

The B3 term-lane `resolve_layered` (5000 packages) output was checked
**byte-identical** OFF vs ON (`diff` of the two binaries' stdout), selection_size
10 both.

### Region-specific stress tests

`#[cfg(test)] mod region4_build_tree_tests` (11/11) in
`templates/targets/rust_wam/state.rs.mustache` (via `cargo test`; full lib
**186/186**, regions 1/2/3's modules unaffected):

- `builds_balanced_tree_matching_oracle` — N ∈ {1,2,3,4,5,7,8,15,16,31,64,100},
  each tree checked against an **independent** divide-and-conquer oracle, and
  `Rest` binding to `[]` when the whole list is consumed;
- `zero_nodes_yields_atom_t_and_rest_is_whole_list` (the `N =:= 0` base: `Tree =
  t`, `Rest = Pairs`);
- `partial_consume_leaves_correct_difference_list_tail` (N < length: the internal
  recursive-call shape — `Rest` is the correct suffix);
- `single_node_tree` (`t(t, K, V, t)`);
- `frame_pinning_registers_preserved` (G-1/G-2 + trailed output);
- and the decline paths `decline_bound_tree_output`, `decline_non_integer_count`,
  `decline_negative_count`, `decline_list_shorter_than_count_restores_snapshot`,
  `decline_open_tail_restores_snapshot`, `decline_non_pair_element`.

## THE REAL A/B — wall-clock, drift-cancelling, interleaved, on B3

Two release binaries from the same codegen: region-4-OFF (no `build_tree/4` arm
in `lowered_call`; **regions 1/2/3 still ON**) and region-4-ON. **Binary identity
verified** — distinct `sha256`, and the OFF crate's `lib.rs` has zero
`region_build_tree_dispatch` arms (genuine relink; each binary was copied out and
hashed):

```
OFF sha256 a24503738f6cbe5c34d14d4e280219a719371a8a1772bb8fa339d475d3ac54cf
ON  sha256 fae79a29ea11835c6b97ab883cb14f196685744a6c3ed8a3fb6112337036ce42
```

Interleaved (OFF then ON each round) on the same 5000-package `resolve_layered`
(`case_5000.json`), timing the `resolve_ms` leg (the drift-cancelling method the
region rounds used):

| round | OFF (ms) | ON (ms) | delta (ms) | delta (%) |
|---:|---:|---:|---:|---:|
| 1 | 966.0 | 567.8 | −398.2 | −41.2 |
| 2 | 988.0 | 552.4 | −435.6 | −44.1 |
| 3 | 911.0 | 542.5 | −368.6 | −40.5 |
| 4 | 910.7 | 532.7 | −377.9 | −41.5 |
| 5 | 876.6 | 522.4 | −354.2 | −40.4 |
| 6 | 876.8 | 535.9 | −340.9 | −38.9 |

**OFF median ≈911 ms; ON median ≈539 ms; delta median ≈−372 ms = −41%.**
All six rounds negative. `load_ms` (JSON→term, not the index path) is unchanged
(~22–25 ms either way).

Note this OFF leg (≈911 ms) is much faster than region 3's OFF leg (≈2000 ms)
because here **regions 1/2/3 stay ON in both legs** — the −41% is the *marginal*
gain from adding region 4 on top of the already-lowered index builders. Combined
with the region-3 result, the full deterministic index-builder stack (regions
1/2/3/4 ON) takes B3 `resolve_layered` from the pristine interpreter's ≈2000 ms
to ≈539 ms — a cumulative **≈−73%**.

### Confirm / refute verdict

**CONFIRMED.** `build_tree/4` was the census's remaining ~21.9% of B3 dispatches,
and fusing it into a deterministic native (log-depth) recursion with the P2
minimal snapshot removes essentially that whole share — the measured −41% on top
of regions 1/2/3 matches. The P2 minimal-snapshot mechanism is again what makes a
deterministic region a clear win.

### Store lane

The store lane's `resolve_layered_store/3` does not call `build_tree` on its
resolve path (it is a store-backed lazy path), so the region — although present
in the store crate (which compiles `resolver.pl` alongside `resolver_store.pl`,
and `build_tree/4` is **not** `_store`-renamed) — is **never invoked at runtime by
the store resolve**. The store corpus is 51/51 and the store differential is
503/0 in both configs.

## Default decision

**Region 4 default ON**, gated. The B3 A/B is a large, consistent win (all six
rounds negative, −41% median on top of regions 1/2/3) with byte-identical output
and every gate green in both configs on both lanes, so the plan's "bank it if it
helps" makes ON the default. It remains disable-able (`region4(false)` /
`UW_REGION4_OFF=1`) and is shape-gated (the frozen shape of `build_tree/4` is
verified with `=@=`), so non-resolver projects are unaffected.

## Files changed (owned surface only)

- `src/unifyweaver/targets/wam_rust_target.pl` — `rust_region4_enabled/1` +
  `rust_region4_applicable/1` + `rust_region4_build_tree_ok` +
  `rust_region4_is_build_tree/1`; and the `build_tree/4` arm wired into
  `lowered_call` alongside regions 1/2/3a/3b's (deduped against the other banks).
- `templates/targets/rust_wam/state.rs.mustache` —
  `region_build_tree_dispatch` + the `region_build_tree_rec` helper (with the
  G-gate argument in comments) and the `region4_build_tree_tests` stress module.
- `examples/pkg_resolver/rust/uw_resolve_wam/**` — the committed term crate,
  regenerated via `build.sh` (default ON, all five region arms).
- `docs/reports/wam_rust_stage2_region4.md` — this report.

## Recommendation on what remains

- **F11 accessor bank re-enable under P2** — measured in this same round (see the
  F11 section of the deterministic-cleanup verdict). Bounded attempt, honest
  verdict recorded there.
- **The nondet resume round (§7)** — still a separate later round. With
  `build_tree/4` now lowered deterministically, the remaining B3 targets are the
  genuine backtracking drivers `pick/7`, `blocked_from/4`, `dep_breaks/5`.

## Reproduce

```
# committed default (regions 1/2/3/4 ON):
bash examples/pkg_resolver/rust/build.sh
bash examples/pkg_resolver/rust/run_corpus_rust.sh          # 51/51
bash examples/pkg_resolver/rust/run_differential_rust.sh    # 2600/0
cd examples/pkg_resolver/rust/uw_resolve_wam && cargo test --release --lib region4   # 11/11
bash examples/pkg_resolver/rust/run_scale_rust.sh 5000      # resolve_ms ~540, selection 10

# region 4 OFF (regions 1/2/3 stay ON):
UW_REGION4_OFF=1 bash examples/pkg_resolver/rust/build.sh
bash examples/pkg_resolver/rust/run_differential_rust.sh    # 2600/0
bash examples/pkg_resolver/rust/run_scale_rust.sh 5000      # resolve_ms ~910, selection 10

# store lane (both configs): 503/0, 51/51
bash examples/pkg_resolver/rust_store/run_corpus_rust_store.sh
bash examples/pkg_resolver/rust_store/run_differential_rust_store.sh
```
