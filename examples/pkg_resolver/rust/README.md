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

## Build and run

```bash
bash examples/pkg_resolver/rust/build.sh              # ~44 s (release)
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

### B1 — one full contract-corpus run

| | |
|---|---|
| Rust binary, 39 scenarios, single process | **93–103 ms** |
| compile time (swipl codegen + `cargo build --release`) | **~44 s** (reported separately by `build.sh`) |

### B2 — seeded differential

Generator: `examples/pkg_resolver/gen_catalogs.mjs`, seed `0xa5b6c7d8`,
**2400 cases**, all ten queries.

| leg | wall time |
|---|---|
| SWI oracle (`diff_runner.pl`) | **1.88 s** |
| wam_rust binary | **306.9 s** |
| divergences | **0** |
| crashes | **0** |

≈ **163× slower than SWI** on the same catalogs, in a single process with
no per-case start-up cost on either side.

### B3 — `resolve_layered` on the scale catalog

Catalog: `examples/pkg_resolver/store/gen_scale_catalog.mjs`, seed
`0xc0ffee01` (5000 packages / 15000 dependency edges). The probe is the same
bound `resolve_layered` request `run_scale_demo.sh` uses (`p30`, base
`p0-0.0.0`). Both legs read the *same* JSON case file, so the catalogs are
byte-identical at every truncation point. Load = term construction; resolve =
the query.

| package cap | packages | depends | Rust load | Rust resolve | SWI load | SWI resolve |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | 54 | 153 | 0.26 ms | **3 533 ms** | 2.37 ms | 0.22 ms |
| 50 | 68 | 193 | 0.21 ms | **5 069 ms** | 2.72 ms | 0.29 ms |
| 60 | 83 | 231 | 0.25 ms | **6 517 ms** | 2.56 ms | 0.28 ms |
| 100 | 134 | 368 | 0.38 ms | **15 348 ms** | 2.97 ms | 0.39 ms |
| 150 | 212 | 518 | 0.77 ms | **32 843 ms** | 3.24 ms | 0.50 ms |
| 250 | 363 | 811 | 1.06 ms | **79 228 ms** | 2.99 ms | 0.83 ms |
| 5000 (full) | 7514 | 15000 | — | **did not finish** — 8.5 GB RSS and still running after 4 min, then OOM-killed; aborts on allocation failure under a 2 GiB cap | 23.98 ms | **9.40 ms** |

Selection size is 10 at every point that contains `p30`; the answers match.

**Loading is fine — Rust builds the term 2–8× faster than SWI. Resolving is
the problem, and it is a memory problem first.** `save_regs()` clones every
live register into every choice point, and `Value` has no structural sharing
(`Value::List(Vec<Value>)`, `Value::Str(String, Vec<Value>)` — every clone is
a deep copy). Register `A1` holds the *whole catalog*, so each choice point
deep-copies the entire catalog: cost and memory grow as
(choice points × catalog size). That is the quadratic in the table and the
OOM at 5k.

Fixing it means giving `Value` structural sharing (`Rc`/`Arc` around the
argument vectors) — a change across the whole target, out of scope here and
recorded in [`../../../docs/WAM_RUST_STATUS.md`](../../../docs/WAM_RUST_STATUS.md).

**The speed hope did not materialise: on this workload wam_rust is two
orders of magnitude slower than SWI, and slower than the JS build.** The
value delivered here is correctness — the differential is clean and eight
silent-wrong-answer classes are gone.

## Residuals

1. **Performance / memory**: see B3. wam_rust cannot run the 5k-package
   catalog at all. Root cause is the deep-copy `Value` representation in
   `save_regs`, not the resolver.
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
   probe.
5. Two pre-existing wam_rust test failures are unrelated to this work and
   fail identically on the base commit: `test_wam_rust_runtime`
   (`astar_weighted_path` integration case) and
   `test_wam_rust_parallel_aggregate_gate`
   (`project_mode_shared_path_annotated`).
