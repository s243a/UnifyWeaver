# T7 parallel aggregates (Rust WAM target) — resume / handoff

**Last updated:** 2026-06-14. **Purpose:** single resume doc so a fresh (post-compact)
session can continue without re-deriving anything. Read this first, then the three
companion reports in `docs/reports/`: `cost_analysis_machinery.md`,
`wam_rust_t7_parallel_perf.md`, `wam_rust_t7_2b2_fork_analysis.md`.

## Goal

T7 = "Tier-2 / parallel" lowering: fan out the independent branches of a forkable
aggregate (`findall`/`aggregate_all`/`bagof`/`setof`) across threads on the Rust
WAM target. Performance is real **only for expensive per-branch work** (cheap
branches regress 5–200× because each parallel worker must clone its own WAM
machine), so it is **cost-gated**.

## Status (what is DONE + where)

| Piece | PR | State |
|---|---|---|
| Gated parallel substrate `gated_collect` (generic, adaptive probe) | #3025 | merged |
| Cost machinery `cost_analysis.pl` + T7 gate `parallel_gate.pl` | #3034 | merged |
| 2a: cost gate drives codegen annotation (both standalone + shared-table paths) | #3039 | merged |
| 2b.1: `WamState: Clone + Send` (forkable machine) | #3041 | merged |
| Exec acceptance harness + fork analysis report | #3051 | open |
| Route-1: split + transform + **parallel runtime orchestrator (exec-verified)** | #3054 | open |

Local branch with the route-1 work: `claude/t7-2b2-split-analysis` (clean tree).
PRs are merged by the user; create PRs freely when ready.

## Architecture (cost gate → transform → runtime), key files

1. **Cost machinery** — `src/unifyweaver/core/cost_analysis.pl`.
   `build_cost_model(+Module,-Model)`, `goal_cost_tier(+Goal,+Model,-Tier)` →
   `trivial|cheap|moderate|expensive|recursive`. Recursion (call-graph
   reachable-from-self) ⇒ `unbounded` ⇒ tier `recursive`.

2. **T7 gate + split + transform** — `src/unifyweaver/core/parallel_gate.pl`:
   - `forkable_aggregate(+Goal,-Template,-Generator)` — recognises the shapes.
   - `aggregate_parallel_decision(+Goal,+Model,-parallel|sequential)` — from the
     generator's cost tier (only `expensive`/`recursive` go parallel; override via
     `parallel_worthy_tier/1`).
   - `split_aggregate_generator(+InnerGoal,+Model,-Enum,-Body,-Frontier)` — cuts a
     conjunction at the first non-cheap goal; soundness gate refuses cut, control
     (`;`/`->`/`\+`), side-effects, no-fan-out, no-body.
   - `parallel_aggregate_transform(+AggGoal,+Model,+Seed,-Helpers,-Plan)`:
     - `Helpers = [ (__par_enum_Seed(Input):-Enum), (__par_body_Seed(Input,Value):-Body) ]`
     - `Plan = par_aggregate(AggType, EnumName/1, BodyName/2, Result)`
     - `Input` packs `vars(Enum) ∩ (vars(Body) ∪ vars(Value))` (value-aware
       frontier — needed when the collected template uses an enum-bound var).
     - **Proven result-preserving in pure Prolog** (running the two helpers
       sequentially collects exactly the original aggregate's value sequence).

3. **Generic substrate** — `src/unifyweaver/targets/rust_runtime/par_aggregate.rs`
   (merged): `gated_collect` with adaptive probe. NOT used by generated projects
   directly (projects use the orchestrator below); kept as the documented adaptive
   mechanism.

4. **Runtime orchestrator (generated into every project)** —
   `templates/targets/rust_wam/par_aggregate.rs.mustache`:
   - `par_collect(base, enum_fn, body_fn) -> Vec<Value>` and `seq_collect(...)`.
   - `collect_inputs` runs the enumerator to gather input tuples; `map_bodies`
     runs the body on a **cloned machine per input** across `thread::scope`
     workers (chunked, order-preserved); `run_body` resets the clone per branch.
   - Exec-verified `par_collect == seq_collect` on real generated predicate
     functions (`tests/test_wam_rust_par_aggregate_exec.pl`).

## Tests (all green)

- `tests/test_cost_analysis.pl` (16) — cost model.
- `tests/test_parallel_gate.pl` (27) — gate + split + transform (incl. transform
  result-preservation, enum-bound value-var case).
- `tests/test_wam_rust_parallel_aggregate_gate.pl` — 2a annotation (both paths).
- `tests/test_wam_rust_machine_forkable.pl` — `Clone+Send` (cargo-gated).
- `tests/test_wam_rust_aggregate_exec.pl` — sequential aggregate exec baseline.
- `tests/test_wam_rust_par_aggregate_exec.pl` — **parallel orchestrator exec
  (par==seq)**, cargo-gated.
- `tests/test_wam_rust_par_aggregate.pl` — substrate unit tests (merged).

Run a Prolog test: `swipl -q -g "run_tests(UNIT),halt" -t "halt(1)" tests/FILE.pl`
(exec tests need `cargo`; fresh project builds are slow — minutes).

## Gotchas learned (do not re-discover these)

- **lib.rs is NOT rendered from `lib.rs.mustache`.** It uses the inline
  `template(rust_wam_lib, ...)` fact in `src/unifyweaver/core/template_system.pl`
  (`render_named_template(rust_wam_lib,...)`). The `pub mod par_aggregate;` line
  lives there. (The `.mustache` copy is kept in sync but unused.)
- **Two predicate-codegen paths in `wam_rust_target.pl`:**
  `compile_wam_predicate_to_rust/4` (standalone) and
  `compile_wam_predicate_to_rust_shared/4` (the shared-table path that real
  projects use). Anything that must reach generated projects has to be in the
  shared path. `Options` is threaded via `generate_predicate_codes/4`.
- **WAM solution enumeration from the orchestrator:** a WAM predicate yields its
  next solution only by `backtrack()` AND `run()` — `backtrack()` alone restores
  a regular choice point forever (it is popped by `trust_me` during `run()`),
  causing an infinite hang. (The `tc_ancestor` enumeration example in
  `test_wam_rust_runtime.pl` is a NATIVE KERNEL and enumerates differently —
  don't copy it for WAM predicates.)
- **Collected bindings are raw heap `Ref`s** — resolve with
  `deref_var(deref_heap(&v))` (as `EndAggregate` does); `deref_heap` is recursive
  so nested lists/structs resolve.
- **`assertion/1` runs its goal under `\+ \+`** (undoes bindings) — never use it to
  bind a variable you then inspect.
- `WamState` is `Clone+Send+Sync` and `Value` is `Send+Sync`, compile-asserted in
  `state.rs.mustache` (`_wam_state_is_forkable`). `stack` is `Arc<...>` (not `Rc`)
  to stay `Send`.
- Aggregate WAM lowering lives in `compile_aggregate_all/6` in
  `src/unifyweaver/targets/wam_target.pl` (SHARED front-end → all targets;
  changing it is risky). It emits `begin_aggregate Type,V,R` … `end_aggregate V`.

## THE REMAINING LAST-MILE: compile-pipeline injection

Everything above is built, tested, and proven. What's missing is the wiring that
makes a *user's* `findall`/`aggregate_all` automatically use the parallel runtime.

**Recommended design — a `par_aggregate` WAM instruction (Rust-target-local, do
NOT change the shared `compile_aggregate_all`):**

1. During Rust project compilation, for each predicate detect forkable aggregates
   in clause bodies (reuse `parallel_gate`), and when
   `parallel_aggregates(true)` + `parallel_aggregate_transform` succeeds:
   - add the two helper predicates (`__par_enum_*`/1, `__par_body_*`/2) to the
     set compiled into the shared WAM table (they become normal entry PCs);
   - in the containing predicate, replace the `begin_aggregate…end_aggregate`
     block with a single new instruction
     `par_aggregate(AggType, EnumLabel, BodyLabel, ResultReg)`.
2. Add `Instruction::ParAggregate(...)` + a runtime handler in the generated
   interpreter (`wam_rust_target.pl` instruction arms) that:
   - runs the enum predicate from `EnumLabel` collecting input tuples (the
     `collect_inputs` logic, but via the machine: clone self, set pc=EnumLabel,
     A1=Unbound, run, deref-collect, backtrack+run loop);
   - parallel-maps the body (clone per input, pc=BodyLabel, A1=input,
     A2=Unbound, run, deref-collect) — i.e. a PC/label-based variant of
     `par_collect` (the fn-pointer `par_collect` proves the mechanism; the
     instruction handler needs the label-based form, using `self.run()`);
   - reduces by `AggType` (reuse the `aggregate_frame` finalisation logic:
     collect→List, count→len, sum/max/min) and binds `ResultReg`.
3. Acceptance: extend/parallel the existing exec harness so a *user* `findall`
   over a recursive/heavy body compiles and runs, asserting result == sequential
   (and, on a big workload, a speedup). The harness pattern is in
   `tests/test_wam_rust_par_aggregate_exec.pl`.

Gate the whole path behind `parallel_aggregates(true)` so default output is
unchanged (as 2a already is).

## One-line resume

"Continue T7 route-1: implement the compile-pipeline injection — emit the
`__par_enum`/`__par_body` helpers into the shared WAM table and a new
`par_aggregate(AggType,EnumLabel,BodyLabel,ResultReg)` instruction whose runtime
handler runs the enum then label-based parallel-maps the body (clone per input)
and reduces by AggType; gate behind `parallel_aggregates(true)`; validate with an
exec test that a user `findall` == sequential."
