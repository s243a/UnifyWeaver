<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# uw-resolve on wam_rust

`examples/pkg_resolver/resolver.pl` (frozen, P0.5, SWI-oracled) compiled
through the **wam_rust** target and driven by a thin term↔JSON shim.

Counterpart to [`../wamjs/`](../wamjs/) (the JavaScript build). Same
resolver source, same corpus, same seeded differential generator — so the
two builds are directly comparable and any divergence is the *backend's*,
never the program's.

```
rust/
  build.pl               swipl entry point: resolver.pl -> Rust WAM project
  build.sh               build.pl + shim copy + cargo build --release
  shim/main.rs           the EDGE: JSON <-> WAM terms, nothing else
  shim/json.rs           dependency-free JSON reader/writer
  uw_resolve_wam/        the generated Cargo crate (committed, regenerable)
  compare_corpus.mjs     corpus comparison (SWI expected vs Rust got)
  run_corpus_rust.sh     39 contract scenarios vs SWI  (+ B1 timing)
  run_differential_rust.sh  2400 seeded catalogs, SWI leg vs Rust leg
  scale_to_case.mjs      5k scale catalog -> one shim case (optional cap)
  swi_scale_ref.pl       SWI reference leg for B3, same JSON input
  run_scale_rust.sh      B3 sweep: load vs resolve, per catalog size
```

Gates for this build also include `tests/test_wam_rust_term_sharing.pl`
(nine probes pinning the structural-sharing representation) alongside
`tests/test_wam_rust_builtin_probes.pl` and
`tests/test_wam_rust_cut_semantics.pl`.

## Build and run

```bash
bash examples/pkg_resolver/rust/build.sh              # ~54 s (release)
bash examples/pkg_resolver/rust/run_corpus_rust.sh    # 39/39 vs SWI
bash examples/pkg_resolver/rust/run_differential_rust.sh   # 2400 cases
bash examples/pkg_resolver/rust/run_scale_rust.sh     # B3 sweep
```

`build.sh` regenerates `uw_resolve_wam/` from `../resolver.pl` every time;
the crate is committed so the build is inspectable without a toolchain.

## What is in the shim (and what is not)

`shim/main.rs` converts JSON catalogs / requests into `Value` terms, calls
one entry predicate per query (`resolve/3`, `resolve_layered/3`,
`explain_blocked_list/3`, `layer_closure/3`, `removal_orphans/3`,
`safe_upgrade/4`, `upgrade_set_result/4`, `freeze_audit/2`, `dependents/3`,
`dependents_installed/3`), and converts the answer term back to JSON —
including the `blocked/3`, `audit/2` and verdict **explanation terms**, not
just selections. There is no resolver logic in Rust: no candidate order, no
constraint arithmetic, no layer walk, no topological sort. Those all live in
`uw_resolve_wam/src/lib.rs`, which is compiler output.

The shim **panics** if a query "succeeds" with its output variable still
unbound, so a runtime that returns early can never launder itself into a
plausible-looking empty answer. (That guard is how the premature-halt bug
below stopped hiding.)

## Results

| gate | result |
|---|---|
| contract corpus | **39 / 39** match SWI (selections *and* explanation terms) |
| seeded differential | **2400 cases, 0 divergences, 0 crashes**, all ten queries |
| cut-semantics probes | **35 / 35** in each of four configurations (`tests/test_wam_rust_cut_semantics.pl`) |
| term-sharing probes | **9 / 9** (`tests/test_wam_rust_term_sharing.pl`) |
| B3 on the 5000-package catalog | **completes** — 4.73 s, 680 MB peak RSS |

## wam_rust defects this program forced out

Every one produced a *silent wrong answer* or a silent failure — none
crashed. Probes live in `tests/test_wam_rust_builtin_probes.pl` (`bNN`) and
`tests/test_wam_rust_cut_semantics.pl` (`pNN`).

| # | Symptom | Cause | Fix | Probe |
|---|---|---|---|---|
| 1 | `resolve(Cat, [], S)` failed instead of returning `[]` (5 corpus scenarios) | list builtins matched only `Value::List`; `put_constant []` delivers the **atom** `[]` (CONVENTIONS §1) | `deref_list_arg/1` routed through the existing `value_as_list`, applied at all 22 list-argument sites | `b01` |
| 2 | `findall(X, Goal, L)` with **zero** solutions reported SUCCESS and dropped every later goal in the clause | the continuation PC was recorded by `EndAggregate`, which never runs when the goal has no solutions; the stale value was `0` = HALT | `BeginAggregate` scans forward for its matching `EndAggregate` and carries the continuation on the aggregate frame | `b02` |
| 3 | `L == []` false when `L` came back from a builtin as the native empty list | `==/2` / `\==/2` used raw `Value: PartialEq`, without the list aliasing `unify` and `term_compare` already apply | `terms_identical/2` (empty-list, cons-spelling and `f/2`↔`f` aliasing) | `b03` |
| 4 | a clause whose **last** goal is a runtime builtin bound its outputs and then reported failure (gap **A3**) | `Execute`'s arm routed only `catch/3`, `throw/1`, `succ/2` and a hand-listed `atomic/1`; everything else fell off `else { false }` | class fix (CONVENTIONS §7, Go's `BuiltinExecute` as the model): `Execute` falls back to the **whole** builtin table, then takes `Proceed`'s return path | `b04` |
| 5 | same class in non-last position (`call <name>`) | same hand-listed gate on the `Call` arm | `Call` falls back to the whole builtin table (`execute_builtin` advances the PC itself, as `BuiltinCall` already relies on) | `b05` |
| 6 | `pick/7` returned `Ver = 2` — a **choice-point depth** — instead of `v(0,1,0)`; 15 / 2400 differential divergences | `wam_target.pl` reserves a permanent `Y` register for an if-then-else barrier *after* deciding the clause needs no environment, so it emits `get_level Y1` in clauses with **no `Allocate`**; this runtime routes `Y` to the topmost env frame — the **caller's** — and overwrote the caller's `Y1` | keep the barrier level on the ITE's own choice point (`ChoicePoint::levels`, looked up by name innermost-first) and never touch a register | differential; `pNN` probes cover the ITE contexts |
| 7 | `!` inside a `findall`/`bagof`/`setof`/`aggregate_all` inner goal destroyed the aggregate frame, so the whole clause silently vanished | the aggregate choice point sat **below** the clause's cut barrier | `BeginAggregate` raises `cut_barrier` above its own choice point (CONVENTIONS §9, barrier-raising contexts) | `p12`–`p16`, `p29` |
| 8 | `!` in a single-clause callee cut past a *preceding* fact predicate's choice point (`findall(Y, (e(_), h(Y)), L)` collected 1 instead of 2) | `pending_cut_barrier` parked by a fact predicate's `TryMeElse` was never consumed (facts have no `Allocate`) and leaked into the next unrelated `Allocate` | pair the pending barrier with the PC that must consume it | `p28` |
| 9 | every clause ending in `call/1` was dead | `call/N` had **no implementation at all**; `execute call/1` found no label and failed | `call/1`…`call/8` as meta builtins routed through `call_goal_once`, plus `,/2`, `;/2`, `->/2`, `\+/1` and `!` in the meta-call walker | `b06`, `p09`, `p10` |
| 10 | a `!` inside a meta-called goal escaped into the enclosing clause | `call_goal_once` truncated leftover choice points but never raised the barrier | `call_goal_once` sets `cut_barrier` to the scope entry depth (§9: `call/1` is an opaque cut scope) | `p09`, `p10` |
| 11 | `bagof/3` and `setof/3` always failed | wam_rust never passed `inline_bagof_setof(true)`, so they arrived as `execute bagof/3` — no label, no builtin; and the 4-operand `begin_aggregate` they compile to was emitted as `NoOp` | inline them by default; accept the 4-operand form; implement `bagof`/`setof` in the aggregate finalisation (fail on empty; `setof` sorts + dedups) | `b07`, `p14`, `p15` |

Defect 6 is an upstream `wam_target.pl` bug (a permanent register reserved
without forcing an environment) that every backend routing `Y` to a frame is
exposed to — per CONVENTIONS §8's own list that is Haskell, Python, R and F#
as well as Rust. wam_target.pl is frozen for this work, so the fix is a
runtime defence; the upstream fix would be to make the barrier reservation
force `HasEnv`.

### Refusals rather than wrong answers

* **`bagof`/`setof` with a non-empty ISO witness list** (free variables in
  the inner goal outside the template and outside `^/2`) needs grouping —
  one solution per distinct witness binding. That is not implemented, so
  `wam_line_to_rust_instr/4` **throws** `unsupported_wam_instruction(
  bagof_setof_witness_grouping(...))` at compile time rather than emitting an
  ungrouped answer. The witness-free form (which the resolver and all 35
  probes use) is implemented.
* **`call/1` over a conjunction whose left conjunct is nondeterministic and
  whose right conjunct fails** cannot be retried by the structural meta-call
  walker (the alternatives live in the interpreter, not in the walk). The
  runtime prints a diagnostic on stderr and fails, rather than reporting a
  first-solution answer as if it were the only one.
* An `execute`/`call` of a name with no label, no dynamic clauses, no foreign
  registration and no builtin can be reported with `UW_WAM_WARN_UNKNOWN=1`.
  It is opt-in because at runtime it is indistinguishable from a builtin that
  merely failed.

## Benchmarks

Single machine, 4 cores, `--release` for every timed run. SWI-Prolog 9.0.4.

All timings below are **after** the structural-sharing change (D52); the
"was" column is the same box before it.

### B1 — one full contract-corpus run

| | now | was |
|---|---:|---:|
| Rust binary, 39 scenarios, single process | **32–36 ms** | 93–103 ms |
| compile time (swipl codegen + `cargo build --release`) | ~54 s | ~44 s |

### B2 — seeded differential

Generator: `examples/pkg_resolver/gen_catalogs.mjs`, seed `0xa5b6c7d8`,
**2400 cases**, all ten queries.

| leg | now | was |
|---|---:|---:|
| SWI oracle (`diff_runner.pl`) | 2.12 s | 1.88 s |
| wam_rust binary | **16.5 s** | 306.9 s |
| divergences | **0** | 0 |
| crashes | **0** | 0 |

≈ **7.8× slower than SWI** on the same catalogs (was ≈163×), in a single
process with no per-case start-up cost on either side. **20× faster than
before** on identical input.

### B3 — `resolve_layered` on the scale catalog

Catalog: `examples/pkg_resolver/store/gen_scale_catalog.mjs`, seed
`0xc0ffee01` (5000 packages / 15000 dependency edges). The probe is the same
bound `resolve_layered` request `run_scale_demo.sh` uses (`p30`, base
`p0-0.0.0`). Both legs read the *same* JSON case file, so the catalogs are
byte-identical at every truncation point. Load = term construction; resolve =
the query.

| package cap | packages | depends | Rust load | Rust resolve | Rust peak RSS | *was* (resolve) |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | 54 | 153 | 0.21 ms | **62 ms** | 13.8 MB | 3 561 ms |
| 50 | 68 | 193 | 0.29 ms | **74 ms** | 15.5 MB | 5 133 ms |
| 60 | 83 | 231 | 0.32 ms | **78 ms** | 17.3 MB | 7 020 ms |
| 100 | 134 | 368 | 0.48 ms | **115 ms** | 22.8 MB | 16 441 ms |
| 150 | 212 | 518 | 0.75 ms | **171 ms** | 29.6 MB | 32 843 ms |
| 250 | 363 | 811 | 1.19 ms | **274 ms** | 43.5 MB | 79 228 ms |
| 500 | 736 | 1549 | 2.60 ms | **503 ms** | 77.6 MB | — |
| 1000 | 1503 | 3036 | 5.64 ms | **940 ms** | 148.0 MB | — |
| 5000 (full) | 7514 | 15000 | 22.8 ms | **4 725 ms** | **680 MB** | OOM at 8.5 GB |

SWI on the same box, same JSON case: 5k load 31.1 ms, resolve 16.9 ms.
Selection size is 10 at every point that contains `p30`; the answers match.
The *was* column is this file's previous ladder; the 40/50/60/100 points were
re-measured on the pre-change build before the fix and reproduced it to
within 1% (3 561 / 5 133 / 7 020 / 16 441 ms), so the two columns are
comparable.

**Both curves are now linear in catalog size** — 7514/363 ≈ 20.7× the
packages costs 17.2× the resolve time and 15.6× the memory. They were
quadratic: 363 packages used to cost 79 s, and 5k could not run at all.

The fix was structural sharing in `Value`
([`value.rs.mustache`](../../../templates/targets/rust_wam/value.rs.mustache)):
a compound's arguments and a list's elements are one refcounted `Arc` spine,
and a list **tail is that same spine one element along**. Two costs
collapsed:

* `Value::clone` became O(1), so `save_regs()` no longer deep-copies the
  catalog into every choice point (the recorded blocker);
* `get_list` stopped copying the remaining tail, so walking an N-element
  list is O(N) instead of allocating — and *retaining*, via choice points —
  N copies of the rest of the list. That second one was the larger of the
  two: with only the cheap-clone half in place the 5k case still OOM-killed
  (at 13.6 GB — further along than the original 8.5 GB, because it now got
  there faster) and memory was still quadratic.

Sharing needed no trail cooperation. Terms in this runtime are immutable —
a variable binding lives in `WamState::bindings` keyed by the variable's
name, never inside the term cell, and the trail restores bindings, not cells
— so two terms may share a spine with no aliasing hazard. `Arc` rather than
`Rc` because `WamState` must stay `Send` for the T7 parallel-aggregate
substrate. `tests/test_wam_rust_term_sharing.pl` pins the nine ways a
program can observe whether two terms secretly share storage.

**What is left is interpreter machinery, not copying.** A callgrind profile
of the 363-package resolve attributes the run to `step` (65%), `backtrack`
(21%), `restore_regs` (12%), `get_reg` (11%), `trail_binding` (8%),
`unify` (7%), `deref_var`/`deref_heap` (12%), `put_reg` (6%) — with
malloc/free ~31% of all instructions and the binding/Y-register hashing
~6.5%. There is no arithmetic or constraint hot spot to capture: the cost is
spread across the register file, the trail and the string-keyed binding
table. See `docs/WAM_RUST_STATUS.md` for what that implies for the lowered
tier.

## Residuals

1. **Performance**: B3 now completes at every size and both curves are
   linear, but the 5k resolve is still ~280× SWI (4.73 s vs 16.9 ms). The
   remaining cost is interpreter machinery, and the lever for it — running
   predicates as direct Rust functions — is **not wired up**: see residual 6.
   `run_scale_rust.sh` keeps its `ulimit -v` cap, but as a regression guard
   rather than the binding constraint it used to be: the 5k point fits in
   680 MB.
2. **`call/1` is first-solution.** It commits to the goal's first answer and
   drops leftover choice points (the `call_goal_once` contract, shared with
   `maplist`/`include`/`foldl`). Nondeterministic `call/1` therefore loses
   solutions; the one case that would be silently wrong (a conjunction with a
   nondeterministic left conjunct) refuses loudly instead. None of the 35
   probes and none of the 2400 differential cases need more.
3. **`bagof`/`setof` ISO grouping** is a compile-time refusal (see above).
4. **Lowered tier coverage**: Rust keeps a lowered predicate's WAM entry in
   the shared table, so an interpreted caller still goes through the
   interpreter. The lowered functions are exercised by calling them directly
   (`cut_probe --lowered pNN`, oracled against `once/1`); a lowered *caller*
   dispatching to a lowered *callee* is not reachable from a label-entry
   probe. Measured on this program: see residual 6.
5. Two pre-existing wam_rust test failures are unrelated to this work and
   fail identically on the base commit: `test_wam_rust_runtime`
   (`astar_weighted_path` integration case) and
   `test_wam_rust_parallel_aggregate_gate`
   (`project_mode_shared_path_annotated`).
6. **The lowered tier is dead code on this program.** Rebuilt with
   `emit_mode(functions)`, **74 of the resolver's 79 predicates lower**
   (27 deterministic, 20 T4 multi-clause-n, 15 multi-clause-1, 12 ITE) and
   the crate compiles — but the resulting binary produces byte-identical
   answers in the same time (363-package resolve 267 ms vs 244 ms
   interpreted, peak RSS 43.4 MB both), because nothing dispatches to the
   lowered functions: `collect_wam_entries` emits a WAM label for lowered
   predicates too, and the only name-keyed call-site hook in the generated
   `lib.rs` is `fact_table_call`, which is empty here. See
   `docs/WAM_RUST_STATUS.md` for the scoping of what an Execute-of-user
   protocol would need, including the fact that T4's contract is
   first-solution.
7. **`arg/3` over a body-constructed compound is broken** (found by the new
   sharing probes; reproduces identically on the pre-sharing runtime, so it
   is pre-existing and unrelated). `T = k(5,2), arg(1,T,V)` leaves `V`
   unbound — Rust prints the construction-time placeholder `_H1` where SWI
   prints `5`. Head unification (`T = k(V,_)`) reads the same argument
   correctly, which is why the resolver and the 2400-case differential never
   see it: the resolver does not use `arg/3`. Minimal reproduction and the
   two-runtime comparison are in `docs/WAM_RUST_STATUS.md`.
