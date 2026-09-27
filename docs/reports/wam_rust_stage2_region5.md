<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Rust WAM lowered tier — Stage 2 region 5 build + real B2 A/B (`dep_breaks/5`)

Lowers `dep_breaks/5` — the one already-deterministic member of the three
"nondet drivers" — as **deterministic region 5** of the Rust WAM lowered tier,
following the P1/P2 recipe validated in
[region 1](wam_rust_stage2_region1.md) … [region 4](wam_rust_stage2_region4.md)
and the classification in
[the nondet-driver classification](wam_rust_nondet_driver_classification.md) (D85).

This is a **recognizer widening**, not a resolver rewrite: `resolver.pl` /
`resolver_store.pl` are **not modified**. It is also NOT a nondet resume round —
`dep_breaks/5` is committed-choice (class (a)), so it lowers into the plain
deterministic tier with the soundness gates G-4/G-5 **vacuous**.

Scope is exactly this region; regions 1/2/3/4 stay enabled and untouched
throughout. All numbers on the build box, `LC_ALL=C.UTF-8`, release + LTO,
against the frozen `resolver.pl` / `resolver_store.pl`.

## TL;DR — CONFIRMED, banked ON

`dep_breaks/5` is ~4.9% of B2 dispatches (the `pick`/`blocked_from`/`dep_breaks`
group per the census, of which `dep_breaks` is the only lowerable member). The
D85 classification established it is **class (a) — committed-per-iteration
recursion — ALREADY deterministic**: the per-step nondet goal `dep_breaks_need`
(a `member/2` search over alternatives) sits **inside the `->` condition**, which
commits its first solution **before** the sole tail self-call in the else-branch,
so the whole predicate has **at most one solution and leaves no choice point**. It
is fused into one native `WamState::region_dep_breaks_dispatch`.

The real interleaved A/B on B2 (the 2600-case term differential corpus), **on top
of regions 1/2/3/4 which stay ON in both legs**, measures a **median −3.15% of
B2 wall time** over 16 drift-cancelling rounds (13/16 negative; mean −2.73%,
stdev 2.38%, standard error 0.6% → t ≈ −4.6), with **byte-identical** output
(all 2600 cases) and every correctness gate green in both configs on both lanes.
Per the plan's "bank it if it helps", **region 5 is default ON** (gated; disable
with `region5(false)` / `UW_REGION5_OFF=1`). The store lane's
`resolve_layered_store` does not reach `dep_breaks/5` on its resolve path, so the
region is **inert there** — neutral perf, identical correctness.

## The region and how it is fused

```prolog
dep_breaks([depends(HN, HV, D, C)|Rest], N, V, Acc, COut) :-
    (   HN == N,
        HV == V,
        dep_breaks_need(Acc, D, C, CBroken)   % NONDET (member/2 inside)
    ->  COut = CBroken                          % COMMIT — first solution only
    ;   dep_breaks(Rest, N, V, Acc, COut)       % sole tail self-call (else)
    ).

dep_breaks_need(Acc, alternatives(Alts), _C, COut) :- !,
    member(dep(D0, COut), Alts), selected_ver(Acc, D0, MV),
    \+ satisfies(MV, COut),
    \+ (member(dep(D2, C2), Alts), selected_ver(Acc, D2, MV2), satisfies(MV2, C2)).
dep_breaks_need(Acc, D, C, C) :-
    selected_ver(Acc, D, MV), \+ satisfies(MV, C).

selected_ver([H|Rest], Name, Ver) :-
    ( H = Name-Ver -> true ; selected_ver(Rest, Name, Ver) ).
```

`dep_breaks/5` is a **single clause** (no base clause: on `[]` there is no
matching clause, so it FAILS). Its body is a committing `->` whose **condition**
holds the per-iteration goals — including the nondeterministic `dep_breaks_need`
— and whose **else-branch is the sole tail self-call**. The `->` commits
`dep_breaks_need`'s first solution before the recursion, so no alternative
survives past the iteration boundary and there is no second solution. It is
additionally consumed **first-solution** by its caller chain
(`dep_breaks_moving/5` under `first_broken/4`'s `dep_breaks_moving(...) -> ...`),
doubly confirming the commit.

It compiles to one native `WamState::region_dep_breaks_dispatch`:

- **P1 (direct native).** No `vm.run()`, no per-recursion dispatch: the whole
  depends-list walk and the committing condition run in Rust.
  `region_dep_breaks_need` inlines the frozen `dep_breaks_need/4`;
  `region_selected_ver` inlines `selected_ver/3`; both reuse region 2's
  already-validated `region_satisfies` / `region_version_lt`. All read-only.
- **P2 (minimal-locals snapshot).** Rollback saves only three scalars
  (`trail_len`, `heap_len`, `var_counter`) via the shared `region_decline`.
- **Committed-choice handling** (see below).

### The recognizer widening (region-4 family → committing condition with nondet)

Region 4's `build_tree/4` special-cases an **arithmetic** committing test
(`N =:= 0`, which is semidet and never binds). Region 5 **widens the same
committing-`->` family** to accept a committing condition whose per-iteration goal
**contains a nondeterministic sub-goal** (`dep_breaks_need`, `member/2` inside),
because the `->` makes it single-solution. Concretely the recognizer requires:

1. `dep_breaks/5` is a single clause whose body is `( Cond -> Then ; Self(...) )`,
   the else-branch being the sole tail self-call, `Cond` containing no recursive
   call (`rust_region5_dep_breaks_ok`, `=@=`);
2. the committed sub-predicates `dep_breaks_need/4` (2 clauses) and
   `selected_ver/3` (1 clause) are present in the frozen shape
   (`rust_region5_dep_breaks_need_ok` / `rust_region5_selected_ver_ok`, `=@=`);
3. the condition's `satisfies/2` + `version_lt/2` chain is present (reuses region
   2's `rust_region2_{satisfies,version_lt,segs_lt,pad_head,segs_lt_1}_ok`).

The det-lattice treats `Cond` as "at-most-one-solution **after** the commit"
rather than requiring `Cond` itself to be det — the runtime realises exactly that
first-solution semantics, and the emitted region is still a plain native loop with
the P2 minimal snapshot and **no** resume-state CP.

### Committed-choice handling (native loop, first-solution semantics)

`region_dep_breaks_dispatch` walks the depends list in Rust. For each
`depends(HN, HV, D, C)` it tests `HN == N, HV == V` (`terms_identical` = the
`==/2` builtin, never binds), and on a match evaluates `region_dep_breaks_need`:

- **`Some(Some(cbroken))`** — the `->` condition is TRUE: commit `COut = cbroken`,
  stop the walk, advance `pc`. (`region_dep_breaks_need` reproduces the
  interpreter's left-to-right first solution: for `alternatives(Alts)`, the
  clause-1 negation guard `\+ (…)` is invariant across the outer `member(D0)`, so
  it decides the clause wholesale — if **any** alt's selected version satisfies
  its constraint the clause fails; otherwise the **first** alt (list order) whose
  selected version does not satisfy its constraint commits. For a plain dep it is
  `selected_ver(Acc,D)=MV ∧ ¬satisfies(MV,C) ⇒ COut = C`.)
- **`Some(None)`** — the condition is FALSE: the `->` takes the else branch =
  tail self-call, i.e. continue the walk.
- **`None`** — an off-shape operand the interpreter must handle (unbound dep,
  improper `Alts`/`Acc`, a `satisfies`/`selected_ver` that itself declines): the
  region **declines** (`region_decline`) and the interpreter runs the predicate.

On list exhaustion with no commit, `dep_breaks/5` has no clause for `[]`, so the
region returns **`Some(false)`** — a genuine deterministic failure, not a decline
(the read-only walk made no bindings, so nothing to unwind). The interpreter
consumes that failure exactly as it would the interpreted clause.

## G-gates (Kimi K2 soundness review) — how each is satisfied

| gate | applies? | how satisfied |
|---|---|---|
| **G-1** minimal saved set from backward liveness + all arg registers forced live; frame pinned | yes | Single rollback point (entry). Backward liveness reduces to the argument registers at entry (A1 = depends list, A2 = N, A3 = V, A4 = Acc, A5 = COut), **forced live by never being clobbered** — the walk reads them into Rust locals and mutates no register before the sole `COut` unify (`frame_pinning_registers_preserved_and_no_cp`). Minimal snapshot records `trail_len`/`heap_len`/`var_counter`; no frame allocated, no CP created. |
| **G-2** determinism guard (no CP / no interpreted code; assert CP-depth; deopt else) | yes | The region pushes NO choice point and calls NO interpreted code (`region_dep_breaks_need`/`region_selected_ver`/`region_satisfies` are read-only Rust), so CP-depth is unchanged **by construction** and asserted after the walk (`choice_points.len() != cp_depth → decline`). Every off-shape input DECLINES (`None`): bound `COut` output, open/improper list tail, a non-`depends/4` element, an unbound dep in the condition, an improper `Alts`/`Acc`, or any `satisfies`/`selected_ver` that itself declines. Each covered by a decline test. |
| **G-3** cut across tiers passed explicitly | **no cut crosses a boundary** | `dep_breaks/5` has no cut (its `->` is a Rust `if`); `dep_breaks_need/4`'s clause-1 cut is realised as a Rust `if` (D = `alternatives(_)` selects clause 1, else clause 2 — there is no choice point for the cut to prune) and `selected_ver/3`'s inner `->` likewise. No interpreted callee runs, so no cut barrier crosses a tier boundary. Nothing to pass or restore. |
| **G-4** at-most-one-solution proof (NOT just non-unifiable heads) | yes | **One clause**, so clause selection is trivially deterministic. The body's `Cond -> Then ; Self` commits `Cond`'s first solution: `HN == N`/`HV == V` are `==/2` (semidet, never bind), and `dep_breaks_need`'s member/2 nondeterminism is **discarded by the `->`** before the else tail. So the region has at most one solution and committing drops no answer — proven from the **commit**, not head non-unifiability (single clause). Additionally consumed first-solution by `first_broken/4`. |
| **G-5** activation identity / cut-invalidation on a resume-state CP | **vacuous** | The walk is a native loop with no resume-state CP, so there is no activation to confuse and no own-CP to invalidate on cut. (This is the whole point of lowering a committed-choice recursion the deterministic way rather than as a nondet resume round.) |

## Gates — both configs, both lanes

Region 5 ON is the committed default; OFF via `UW_REGION5_OFF=1` (regions 1/2/3/4
stay ON in both configs — the OFF build isolates region 5 only, verified: the OFF
crate's `lib.rs` has **zero** `region_dep_breaks_dispatch` arms).

| gate | region 5 OFF | region 5 ON (committed default) |
|---|---|---|
| term corpus (B1) | 51 / 51 | 51 / 51 |
| term differential (B2) | 2600 / 0 / 0 | 2600 / 0 / 0 |
| store corpus | 51 / 51 | 51 / 51 |
| store differential | 503 / 0 | 503 / 0 |

The full 2600-case B2 term-lane output was checked **byte-identical** OFF vs ON
(`diff` of the two binaries' stdout over the shared `cases.jsonl`, 165 842 bytes,
0 differences).

### Region-specific stress tests

`#[cfg(test)] mod region5_dep_breaks_tests` (13/13) in
`templates/targets/rust_wam/state.rs.mustache` (via `cargo test`; full lib
**199/199**, regions 1/2/3/4's modules unaffected — 186 → 199, +13):

- `commits_on_matching_broken_plain_dep`, `walks_past_non_matching_then_commits`
  (the committing walk + the else-branch tail continue);
- `alternatives_all_broken_commits_first`, `alternatives_one_satisfies_guard_fails`,
  `alternatives_missing_selection_is_skipped` (the clause-1 `alternatives(_)`
  path: first-broken commit, the invariant negation guard, and `member` skipping
  an alt whose `selected_ver` fails);
- `fails_when_selected_version_satisfies`, `empty_depends_list_fails`,
  `non_matching_head_never_evaluates_condition` (the deterministic `Some(false)`
  failure: no clause for `[]`, condition-false paths);
- `frame_pinning_registers_preserved_and_no_cp` (G-1/G-2 + trailed output);
- and the decline paths `decline_bound_output`, `decline_open_tail_restores_snapshot`,
  `decline_unbound_dep_in_condition`, `decline_non_depends_element`.

## THE REAL A/B — wall-clock, drift-cancelling, interleaved, on B2

Two release binaries from the same codegen: region-5-OFF (no `dep_breaks/5` arm
in `lowered_call`; **regions 1/2/3/4 still ON**) and region-5-ON. **Binary
identity verified** — distinct `sha256`, and the OFF crate's `lib.rs` has zero
`region_dep_breaks_dispatch` arms (genuine relink; each binary was copied out and
hashed):

```
OFF sha256 726d14e5ac517f6aa33361373bd619aec9be53c8e95ccd70d64ddf3d59aa4500
ON  sha256 efc10fb38158ea25309ce05eee07d82270ea7a128476efae093a05f2e17397f1
```

Interleaved (OFF then ON each round) over the same 2600-case B2 differential
corpus (`cases.jsonl`), timing the binary run only (warm start; the
drift-cancelling method the region rounds use):

| round | OFF (ms) | ON (ms) | delta (ms) | delta (%) |
|---:|---:|---:|---:|---:|
| 1 | 18,371 | 17,223 | −1,148 | −6.2 |
| 2 | 17,590 | 17,299 | −291 | −1.7 |
| 3 | 18,059 | 17,254 | −805 | −4.5 |
| 4 | 17,625 | 17,088 | −537 | −3.0 |
| 5 | 17,603 | 17,724 | +121 | +0.7 |
| 6 | 17,414 | 17,188 | −226 | −1.3 |
| 7 | 17,631 | 17,131 | −500 | −2.8 |
| 8 | 17,501 | 17,343 | −158 | −0.9 |
| 9 | 17,713 | 16,853 | −860 | −4.9 |
| 10 | 17,381 | 17,777 | +396 | +2.3 |
| 11 | 17,865 | 16,872 | −993 | −5.6 |
| 12 | 17,774 | 16,921 | −853 | −4.8 |
| 13 | 18,058 | 18,152 | +94 | +0.5 |
| 14 | 17,904 | 17,238 | −666 | −3.7 |
| 15 | 18,131 | 17,524 | −607 | −3.3 |
| 16 | 18,032 | 17,247 | −785 | −4.4 |

**Median ≈ −3.15%; mean ≈ −2.73%; stdev 2.38% (standard error 0.6% over 16
rounds → t ≈ −4.6). 13/16 rounds negative** (the three positives all within
+2.3%). `load` (JSON→term) is unchanged.

### Confirm / refute verdict

**CONFIRMED.** `dep_breaks/5` was ~4.9% of B2 dispatches, and fusing it into a
deterministic native loop with the P2 minimal snapshot removes the interpretation
overhead of that share (the native still does the per-element work, so the gain
is a fraction of the dispatch share, not the whole of it). The measured −3.15%
median on top of regions 1/2/3/4 is a real, statistically significant signal
(t ≈ −4.6) — smaller and noisier than the index-builder regions because the
depends-list walk is shorter per call and B2 has higher run-to-run variance, but
consistently negative. The P2 minimal-snapshot mechanism is again what makes a
deterministic region a clean win. Combined with the classification's core point:
**a committed-choice recursion lowers exactly like the deterministic regions — no
resume-state trampoline needed.**

### Store lane

The store lane's `resolve_layered_store/3` does not reach `dep_breaks/5` on its
resolve path, so the region — although present in the store crate (which compiles
`resolver.pl` alongside `resolver_store.pl`, and `dep_breaks/5` is **not**
`_store`-renamed) — is **never invoked at runtime by the store resolve**. The
store corpus is 51/51 and the store differential is 503/0 in both configs.

## Default decision

**Region 5 default ON**, gated. The B2 A/B is a modest but consistent and
statistically significant win (13/16 rounds negative, −3.15% median on top of
regions 1/2/3/4) with byte-identical output and every gate green in both configs
on both lanes, so the plan's "bank it if it helps" makes ON the default. It
remains disable-able (`region5(false)` / `UW_REGION5_OFF=1`) and is shape-gated
(the frozen shapes of `dep_breaks/5`, `dep_breaks_need/4`, `selected_ver/3` are
verified with `=@=`), so non-resolver projects are unaffected.

## Files changed (owned surface only)

- `src/unifyweaver/targets/wam_rust_target.pl` — `rust_region5_enabled/1` +
  `rust_region5_applicable/1` + `rust_region5_{dep_breaks,dep_breaks_need,
  selected_ver}_ok` + `rust_region5_is_dep_breaks/1`; and the `dep_breaks/5` arm
  wired into `lowered_call` after regions 1/2/3a/3b/4 (deduped against the other
  banks; the region-4 `Eligible` chaining variable renamed to `Eligible4` to
  thread region 5's exclusion — a mechanical change, region 4's behaviour
  unchanged).
- `templates/targets/rust_wam/state.rs.mustache` — `region_dep_breaks_dispatch`
  + the `region_dep_breaks_need` / `region_selected_ver` / `region_dep_pair`
  helpers (with the G-gate argument in comments) and the `region5_dep_breaks_tests`
  stress module.
- `examples/pkg_resolver/rust/uw_resolve_wam/**` — the committed term crate,
  regenerated via `build.sh` (default ON, all six region arms).
- `docs/reports/wam_rust_stage2_region5.md` — this report.

`examples/pkg_resolver/resolver.pl` and `resolver_store.pl` are **unmodified**
(verified `git diff` against the base is empty). `pick/7` and `blocked_from/4`
are untouched (separate future work per D85: `pick/7` is dead code, `blocked_from/4`
genuinely exposes alternatives consumed by `findall`).

## Reproduce

```
# committed default (regions 1/2/3/4/5 ON):
bash examples/pkg_resolver/rust/build.sh
bash examples/pkg_resolver/rust/run_corpus_rust.sh          # 51/51
bash examples/pkg_resolver/rust/run_differential_rust.sh    # 2600/0
cd examples/pkg_resolver/rust/uw_resolve_wam && cargo test --release --lib region5   # 13/13

# region 5 OFF (regions 1/2/3/4 stay ON):
UW_REGION5_OFF=1 bash examples/pkg_resolver/rust/build.sh
bash examples/pkg_resolver/rust/run_differential_rust.sh    # 2600/0

# store lane (both configs): 503/0, 51/51
bash examples/pkg_resolver/rust_store/build.sh
bash examples/pkg_resolver/rust_store/run_corpus_rust_store.sh
bash examples/pkg_resolver/rust_store/run_differential_rust_store.sh
```

## Attribution

The committed-choice loop shape and the deterministic-lowering discipline are
standard patterns (also used by this project's PLAWK native codegen); the
resume-state choice-point concept they deliberately AVOID here is from Kenichi
Sasagawa's M-Prolog / N-Prolog (`https://github.com/sasagawa888/mprolog`,
Modified BSD), concepts only, never code, per `MPROLOG_MINING_NOTES.md`.
