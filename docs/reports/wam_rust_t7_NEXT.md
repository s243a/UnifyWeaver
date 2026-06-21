# T7-on-Rust — what to do next

Concise, action-oriented next-steps note. For the full handoff (architecture,
gotchas, all tests) see `docs/reports/wam_rust_t7_RESUME.md`.

## Where we are (done + on `main`)

Whole-body parallel aggregates are **complete and perf-proven**:
`cost gate → split → transform → generated `__par_enum`/`__par_body` helpers →
native `par_collect` wrapper` running the body on a cloned WAM machine per input
across threads. All common reduce types: **collect / count / sum / max / min /
bag / set**. **3.39× measured** end-to-end (4 cores). Cost-gated so cheap
aggregates stay sequential (no fork regression).

## Immediate (housekeeping)

- **Merge PR #3138** — matrix doc update (`rust T7 ✗ → ~ gated`). The code PRs
  (#3123 route-1, #3135 reduce types) are already merged.

## The one substantial T7 item left — DONE (2026-06-14)

**Aggregates embedded in a *larger* clause body** (not the whole body), via the
`par_aggregate` WAM-instruction route below, are now **implemented and
exec-verified**. A clause with a guard goal before an embedded `findall`
compiles, with `parallel_aggregates(true)`, to synthesised `__par_enum`/
`__par_body` helpers (ordinary shared-table WAM predicates) plus a single
`par_aggregate(AggType, EnumLabel, BodyLabel, ResultReg)` instruction spliced
over the `begin_aggregate..end_aggregate` block. The handler drives the helpers
by entry-PC label (`par_collect_labels`, the label-based sibling of
`par_collect`), reduces by agg type (collect/count/sum/max/min — mirrors the
sequential `aggregate_frame`), and binds the result in place. Verified by
`tests/test_wam_rust_par_aggregate_embedded_exec.pl` (parallel result == known
sequential answer) with the full `test_wam_rust_target.pl` (174) and
`test_parallel_gate.pl` (25) regressions green and route-1 unaffected.

The original design is retained below for reference.

### Plan — the `par_aggregate` WAM-instruction route

1. **Detect + rewrite (compile-time).** In the Rust target, when a clause body
   contains a forkable, parallel-eligible aggregate (reuse `parallel_gate`):
   synthesise its `__par_enum`/`__par_body` helpers (add to the shared-table
   compile set, as `rust_inject_parallel_aggregates` already does), and replace
   that aggregate's `begin_aggregate … end_aggregate` block with a single new
   WAM line `par_aggregate(AggType, EnumLabel, BodyLabel, ResultReg)`.
2. **New instruction.** Add `Instruction::ParAggregate(...)` + its text parse
   (`wam_line_to_rust_instr`, ~`wam_rust_target.pl:3232`) + an interpreter arm
   (near the `BeginAggregate` arm, ~`wam_rust_target.pl:763`).
3. **Handler = label-based `par_collect`.** Clone `self`; run the enum from
   `EnumLabel` collecting input tuples (set `pc`, `A1=Unbound`, `run()`, deref,
   `backtrack()+run()` loop); parallel-map the body from `BodyLabel` per input
   (clone, `pc=BodyLabel`, `A1=input`, `A2=Unbound`, `run()`, deref); reduce by
   `AggType`; bind `ResultReg`; `pc += 1`. Add a `par_collect_labels` variant to
   `templates/targets/rust_wam/par_aggregate.rs.mustache` (the fn-pointer
   `par_collect` proves the mechanism; this one uses `self.run()` + labels).
4. **Exec test.** A clause with an *embedded* `findall` (other goals before/after)
   compiles + runs == sequential. Pattern: `tests/test_wam_rust_parallel_injection_exec.pl`.

Gate behind `parallel_aggregates(true)` (default output unchanged), as the
whole-body path already is.

### Gotchas (already learned — don't rediscover)

- WAM solution enumeration needs **`backtrack()` AND `run()`** (backtrack alone
  loops forever; `tc_ancestor` is a native kernel, enumerates differently).
- Collected bindings are raw heap `Ref`s → resolve with `deref_var(deref_heap(_))`.
- `set` ordering uses the runtime's `term_compare` (pub) + `dedup_by`.
- `lib.rs` is rendered from the **inline `template(rust_wam_lib, …)`** in
  `template_system.pl`, NOT `lib.rs.mustache`.
- Real projects use the **shared-table** path (`compile_wam_predicate_to_rust_shared`),
  not the standalone one.
- `WamState` is `Clone+Send+Sync`, `Value` is `Send+Sync` (compile-asserted).

## Alternatives to the above

- **Pivot to another matrix gap** (`WAM_LOWERING_TAXONOMY_AND_MATRIX.md`) — e.g.
  rust **T9 fact-table inline** (✗ today), or **T10/T11**.
- **Call T7 done** — it's a clean, complete, proven milestone for whole-body
  aggregates; the embedded case is an enhancement, not a gap in correctness.

## Reference docs

- `docs/reports/wam_rust_t7_RESUME.md` — full handoff (architecture + gotchas).
- `docs/reports/wam_rust_t7_2b2_fork_analysis.md` — the instruction-route design.
- `docs/reports/wam_rust_t7_speedup_benchmark.md` — the 3.39× measurement.
- `docs/reports/wam_rust_t7_parallel_perf.md` — original benchmark + gate design.
- `docs/reports/cost_analysis_machinery.md` — the cost model the gate uses.
