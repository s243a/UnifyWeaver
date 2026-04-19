# WAM Fact Shape Implementation Plan

## Goal

Move the WAM lowered emitters from "every fact is a function" toward
"facts are data." Start with Elixir, where the scaling failure is
already observable (a 6009-clause `category_parent/2` that the host
compiler cannot load in ten minutes). Generalise to other WAM targets
once the contract stabilises.

The phased rollout is designed so each phase is shippable on its own
and produces observable results.

## Phase A: Classification infrastructure (no behaviour change)

Status: not started.

Work:

- Implement `fact_only/1`, `clause_count/2`, `first_arg_groundness/2`
  as emitter-internal predicates operating on the already-loaded clause
  list.
- Add option plumbing: `fact_layout/2`, `fact_count_threshold/1`,
  `fact_layout_defaults/1`.
- At `lower_predicate_to_elixir/4` entry, compute the layout for each
  predicate. Store it on a new fact that later phases consume.
- Emit layout choice as a comment in the generated module so it is
  observable without a debugger.

Acceptance:

- New unit tests assert correct classification for: empty predicate,
  one-clause rule, 100-clause facts, 1000-clause facts, user override.
- All existing WAM-Elixir tests still pass (behaviour unchanged —
  everything still emits as `compiled`).

## Phase B: `inline_data` emission for Elixir

Status: not started.

Work:

- In `wam_elixir_lowered_emitter`, add an `emit_inline_data/4` branch
  selected by the Phase-A layout choice.
- Emit `@facts` module attribute as an Elixir list of tuples. Convert
  atoms/strings/numbers to host literals; variable head args become a
  sentinel. Structure/list args are deferred to Phase C.
- Emit a `run/1` that delegates to `WamRuntime.stream_facts/3`, and a
  `run(args)` wrapper that mirrors the current arg-prep code.
- In `wam_elixir_target`, implement `stream_facts/3`,
  `resume_fact_stream/2`, and make `backtrack/1` recognise the
  fact-stream CP shape.

Acceptance:

- `examples/debug_wam_elixir_ancestor.pl` still produces the expected
  4/1/4 solution counts with correct N values.
- `examples/benchmark/benchmark_wam_elixir_effective_distance.py`
  at dev scale completes without timing out (driver issue
  notwithstanding — a separate arithmetic fix may be needed).
- `category_parent/2` at scale 300 emits a module < 100 KB with one
  `defp run/1` instead of 6009 `defp clause_*/1`; the Elixir compiler
  loads it in seconds, not minutes.
- A new unit test asserts the shape of a generated `inline_data`
  module for a small fact set.

## Phase C: First-argument indexing

Status: not started.

Work:

- When `first_arg_groundness = all_ground` and `fact_index_policy` is
  `first_arg` or `auto`, emit an additional `@facts_by_arg1`
  attribute: an Elixir map grouping facts by their first-arg value.
- Update `stream_facts/3` to accept an optional index hint and
  dispatch to the indexed bucket when `regs[1]` is bound at call
  time.
- Keep the flat-scan fallback as the default and as the
  mixed-groundness path.

Acceptance:

- A new unit test exercises a seeded call (first arg bound) on a
  >1000-clause fact-only predicate and measures a significant
  speed-up over the flat-scan path.
- The reproducer and ancestor-scale benchmark still produce correct
  answers.

## Phase D: `external_source` layout and `FactSource` behaviour

Status: not started.

Work:

- Define the `FactSource` behaviour in `wam_elixir_target.pl`:
  `open/3`, `next/2`, `close/2`, `lookup_by_first_arg/3`.
- Ship one concrete implementation: `WamRuntime.FactSource.Tsv`
  backed by two-column TSV files, matching the formats already in
  `data/benchmark/*/`.
- Add the emitter path for `fact_layout(P/A, external_source(SourceSpec))`.
- Update the benchmark generator to route very large fact-only
  predicates through `external_source(tsv(...))` rather than
  inlining 6 MB of literals.

Acceptance:

- `benchmark_wam_elixir_effective_distance.py` at scale 300 runs
  end-to-end, producing output the reference TSV can be diffed
  against.
- An explicit `fact_layout(cat_parent/2, external_source(tsv(...)))`
  declaration in the benchmark generator overrides the default and
  is exercised by tests.

## Phase E: Cost-based default selection

Status: not started.

Work:

- Move the Phase-A threshold policy behind a pluggable selector in
  the emitter.
- Measure compile-time and size cost for each layout across a
  representative corpus.
- Tune defaults so that the chosen layout minimises host-compile
  time while keeping small predicates in `compiled` form.

Acceptance:

- Default behaviour across the existing benchmark corpus is
  measurably better than any fixed threshold.
- Selection is overridable per-predicate and per-module, and is
  observable in the generated module's layout-comment.

## Phase F: Port to other WAM targets

Status: not started.

Targets in rough priority order:

1. `wam_clojure_target` — Clojure is next most likely to see the same
   pathology at scale.
2. `wam_go_target`, `wam_rust_target` — both compile facts to
   functions today.
3. `wam_haskell_target` — has its own fact-heavy fast paths; fit may
   already be acceptable, worth measuring first.
4. `wam_csharp_native_target` — the C# side is already on the
   materialization-aware runtime, but the WAM-generated-code side may
   still compile facts as methods.
5. Other targets as measurement warrants.

For each target:

- Implement the `inline_data` path using the host's most natural
  literal shape.
- Implement `stream_facts` / `resume_fact_stream` / fact-stream CP
  handling in the target's runtime.
- Add a regression test against a fact-only predicate that would
  previously have triggered the host compiler limit.

Acceptance:

- Each ported target compiles a >5000-clause fact-only predicate in
  seconds.
- The target's existing correctness tests still pass.

## Risks and mitigations

- **Risk: correctness drift between `compiled` and `inline_data`.**
  Mitigation: a golden-output test suite that exercises a predicate
  with each layout and diffs the solution stream.
- **Risk: indexed and flat-scan paths diverge semantically when arg 1
  starts bound and becomes unbound mid-backtrack.**
  Mitigation: the CP snapshot records the list being scanned, so a
  restore always resumes the same list; indexed lookup only happens
  at the entry call, never mid-stream.
- **Risk: third-party drivers depend on the per-clause `defp`
  naming.**
  Mitigation: drivers talk to `run/1` and `next_solution/1` only; the
  spec explicitly keeps those stable. A note in the CHANGELOG flags
  the private-function-name change.

## Exit criteria

The project is considered done when:

1. `category_parent/2` at scale 300 compiles and runs through the
   effective-distance benchmark to a correct reference-row count on
   Elixir.
2. At least one other WAM target has the same layout available.
3. Documentation in `docs/design/WAM_ELIXIR_CORRECTNESS_GAPS.md` and
   the parallel target-design docs reflect the new default shape.
