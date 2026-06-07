# Recurrence Evaluation Strategy — Implementation Plan

This document is the work plan for implementing the recurrence-evaluation-strategy selector. For the why, see [`RECURRENCE_EVALUATION_STRATEGY_PHILOSOPHY.md`](RECURRENCE_EVALUATION_STRATEGY_PHILOSOPHY.md). For the API and behaviour, see [`RECURRENCE_EVALUATION_STRATEGY_SPECIFICATION.md`](RECURRENCE_EVALUATION_STRATEGY_SPECIFICATION.md).

## Sequencing summary

The work breaks into seven phases. Phases 0–4 build the module itself; phase 5 wires it into the F# WAM target as the first consumer; phase 6 adds tests; phase 7 captures the result back into book-18.

Phases 0–4 are sequential. Phase 5 depends on 0–4. Phase 6 can begin in parallel with phase 5 (tests for individual phases can be written as those phases land). Phase 7 follows.

Total estimated effort: **~1.5 days of focused Prolog + tests + docs**. The biggest contributor is the conflict-resolution machine in phase 3.

| Phase | Deliverable | Estimated effort |
|-------|-------------|------------------|
| 0 | Module skeleton + data types | 1 hour |
| 1 | Signal classification (intent vs data) | 1 hour |
| 2 | Cost-model rules for per-query branch | 2 hours |
| 3 | Conflict-resolution machine | 4 hours |
| 4 | Decision-trace emission (stderr + comment) | 2 hours |
| 5 | F# WAM target integration | 2 hours |
| 6 | Tests (unit + integration) | 2 hours |
| 7 | Book-18 updates (chapters 9, 10) | 1 hour |

## Phase 0 — Module skeleton + data types

### Goal

Create `src/unifyweaver/core/recurrence_evaluation_strategy.pl` with the public module declaration, type-checking helpers, and stub predicates for the public API. All predicates compile and return placeholder values; tests can target the stubs.

### Deliverables

- New file `src/unifyweaver/core/recurrence_evaluation_strategy.pl` with module declaration matching the [Specification §Module](RECURRENCE_EVALUATION_STRATEGY_SPECIFICATION.md#module).
- Type-checking helpers: `valid_recurrence/1`, `valid_workload/1`, `valid_strategy/1`. Use `library(error)` `must_be/2` and `domain_error/2`.
- Stub implementations of every public predicate that return a `not_yet_implemented(Phase)` tagged term. Calls to stubs do not throw; they return a sentinel so downstream code can detect partial implementation.
- A short module-level docstring at the top of the file pointing to the philosophy/spec/impl-plan triple.

### Scope notes

- No real logic. Stubs only.
- The stub for `select_evaluation_strategy/3` returns `strategy_choice(strategy(per_query(unidirectional)), trace([step(stub, not_yet_implemented(phase_0), [])]))` — a working baseline that the F# WAM integration can call against from day one.

### Success criteria

- Module loads without error.
- Every exported predicate exists and is type-checked.
- Calls to `select_evaluation_strategy/3` return the baseline strategy.
- A smoke test (`tests/core/test_res_skeleton.pl`) verifies the module loads, the baseline strategy is returned, and the trace contains the `stub` step.

### Estimated effort

1 hour.

## Phase 1 — Signal classification

### Goal

Implement `classify_signals/3` and the underlying intent/data dispatch table. Workload terms get sorted into `IntentSignals` and `DataSignals` according to the [Specification §Workload signals](RECURRENCE_EVALUATION_STRATEGY_SPECIFICATION.md#workload-signals).

### Deliverables

- `classify_signals/3` implementation reading the dispatch table.
- A static dispatch table (Prolog facts) mapping signal functors to `intent` or `data` plus the signal's source name (caller/manifest/policy/inferred).
- Forward-compat behaviour: unknown signal functors go to `DataSignals` with a stderr warning. Tests cover this.
- Unit tests in `tests/core/test_res_signals.pl` covering: every known signal type, a mix, an unknown signal, empty workload, only-intent, only-data.

### Scope notes

- The dispatch table lives in the module file as ordinary facts (no JSON config). Adding a signal type means adding a fact.
- The classification is *static* — it doesn't depend on the recurrence or graph state. Same workload always classifies the same way.

### Success criteria

- All unit tests pass.
- `classify_signals([], I, D)` returns `I = [], D = []`.
- Manually-constructed mixed workloads return the expected partition.
- Unknown signal warning is emitted exactly once per unknown functor per test run.

### Estimated effort

1 hour.

## Phase 2 — Cost-model rules for per-query branch

### Goal

Implement the six initial cost-model rules from the [Specification §Phase B](RECURRENCE_EVALUATION_STRATEGY_SPECIFICATION.md#phase-b--cost-model-evaluation) plus the scoring/tiebreak logic. `apply_cost_model/3` returns a `cost_model_choice(Strategy, Score, DecidingRule)` term plus a flat list of all rules that fired.

### Deliverables

- The six rules as clauses of a multifile-style `cost_model_rule/5` predicate (so future targets can add rules without editing the core module — though for now all rules are in-module).
- The scoring aggregator: collect all firing rules, score the cumulative preference per candidate strategy, pick the highest-scoring strategy; tiebreak by declaration order.
- `apply_cost_model/3` wrapping the above.
- Unit tests in `tests/core/test_res_cost_model.pl` covering each rule individually plus rule-interaction cases (multiple rules firing; tie-breaks).

### Scope notes

- The rule API is `cost_model_rule(RuleName, Recurrence, DataSignals, Score, ChosenStrategy)`. Future rules add a new clause.
- The deciding-rule trace step names the rule that *won* the scoring, not every rule that fired.
- Numeric contraction rate `r` is read from `Recurrence` but is not used by any Phase 2 rule. The plumbing is there for Phase 8+ (fixed-point branch).

### Success criteria

- All six rules return correct strategies for their preconditions.
- Cardinality(large) + csr_available triggers `prefer_bidirectional_csr_present`.
- Empty data signals trigger `default_fallback` and return `per_query(unidirectional)`.
- Two rules with the same score resolve to the earlier-declared one with the tie noted in the trace.

### Estimated effort

2 hours.

## Phase 3 — Conflict-resolution machine

### Goal

Implement `resolve_against_intent/5` with the six-step hierarchy from the [Specification §Phase C](RECURRENCE_EVALUATION_STRATEGY_SPECIFICATION.md#phase-c--conflict-resolution). This is the most involved phase; budget extra time.

### Deliverables

- `resolve_against_intent/5` with each step as a separate helper predicate (`step1_no_intent/4`, `step2_intent_matches/4`, `step3_third_option/4`, `step4_scope_disambiguation/4`, `step5_satisfiability_check/4`, `step6_caller_wins/4`). Each step takes the same inputs and produces either `resolved(Strategy, TraceEntry)` or `next_step`.
- The `admissible_strategies/2` helper — for a given `Recurrence`, returns the list of strategies whose `KernelKind` admits them. Reads a static table mapping kernel kinds to admissible strategy lists.
- An intent-compatibility matcher (`intent_compatible_with_strategy/2`) that says whether a given intent signal is satisfied by a candidate strategy.
- The satisfiability adjuster — handles `build_csr_at_compile_time`, `degrade_to_compatible`, `degrade_with_warning`.
- Unit tests in `tests/core/test_res_conflict.pl` covering each step in isolation, the full hierarchy walked through, every step exiting both via `resolved` and via `next_step`.

### Scope notes

- Each step is small and individually testable.
- The full hierarchy is just a fold over the steps — `resolve_against_intent/5` walks the list of steps in order, taking the first `resolved`.
- The trace entry for each step records what fired and why; `next_step` returns an empty trace entry.
- "Build CSR at compile time" emits a step into the generated program but is not done in the strategy module — the module just *decides* the build should happen; the target executes the decision.

### Success criteria

- Each step's individual tests pass.
- Integration tests walk realistic decision scenarios end-to-end, including the conflict + override case.
- The trace for each decision is complete and accurate.

### Estimated effort

4 hours.

## Phase 4 — Decision-trace emission

### Goal

Implement `emit_reasoning_trace/1` (stderr renderer) and `format_trace_for_comment/2` (generated-code comment renderer). Both render the same structured trace into different surface forms.

### Deliverables

- `emit_reasoning_trace/1` — multi-line stderr output, one line per step, indented for readability. Critical signals named.
- `format_trace_for_comment/2` — multi-line comment string for the generated code header. Uses configurable comment syntax (`Prefix` argument; defaults to `//` for C-family).
- Pretty-printer helpers: `strategy_pretty/2`, `signal_pretty/2`, `step_pretty/2`.
- Unit tests in `tests/core/test_res_trace.pl` covering rendering of each step type plus end-to-end rendering of full traces from the Phase 3 tests.

### Scope notes

- The structured trace is the canonical form; renderers are derived. This is what guarantees consistency between stderr and code-comment renderings.
- Comment syntax is parameterised so multiple targets can use the same renderer. F# uses `//`; Haskell uses `--`; Prolog uses `%`.
- The renderers should be idempotent (rendering the same trace twice produces identical output).

### Success criteria

- Sample traces render as expected in both forms.
- Comment format is valid syntax in F#, Haskell, and Prolog (each tested with a smoke parse).
- The stderr output is reasonably tight (no excessive line breaks, indentation consistent).

### Estimated effort

2 hours.

## Phase 5 — F# WAM target integration

### Goal

Wire `select_evaluation_strategy/3` into `src/unifyweaver/targets/wam_fsharp_target.pl` as the first consumer. The existing `maybe_upgrade_bidirectional/2` becomes one of several upgrade paths driven by the strategy decision.

### Deliverables

- New helper predicate `build_recurrence_term/3` in `wam_fsharp_target.pl` (or in a shared utility module) that constructs the `Recurrence` term from a detected kernel + manifest + relation-policy lookups.
- New helper predicate `build_workload_signals/2` that gathers caller options + manifest entries + relation-policy declarations + graph stats into a workload list.
- The existing `maybe_upgrade_bidirectional/2` call site replaced with a `select_evaluation_strategy/3` call, followed by dispatch on the returned `Strategy`.
- Generated-code comment insertion: `format_trace_for_comment(Trace, Comment)` + comment-string into the F# project's kernel call site.
- Backwards-compat: existing tests (`tests/core/test_wam_fsharp_bidirectional_e2e.pl`) pass without modification because explicit `kernel_mode(bidirectional)` is honoured by the conflict-resolution hierarchy.

### Scope notes

- Where the build_recurrence_term and build_workload_signals helpers live is a small design question. Probably better as a shared `src/unifyweaver/core/recurrence_inputs.pl` module so future Haskell/C WAM targets reuse them. Decide during implementation.
- The integration changes the F# target file in only a small number of places (the upgrade call site, the comment insertion point). The rest of the F# target is unchanged.

### Success criteria

- `tests/core/test_wam_fsharp_bidirectional_e2e.pl` passes unchanged.
- A new variant of the test that calls without explicit `kernel_mode(bidirectional)` (relying on cost-model selection) also produces bidirectional code when the workload signals support it.
- A unit test verifies the generated F# file includes a comment header with the trace.

### Estimated effort

2 hours.

## Phase 6 — Tests

### Goal

Comprehensive unit and integration tests. Some are scoped into Phases 1–5 above; this phase backfills coverage and adds end-to-end integration tests.

### Deliverables

- `tests/core/test_recurrence_evaluation_strategy.pl` — integration test running the full pipeline end-to-end with several realistic workload + recurrence combinations:
  - Bidirectional auto-selected from CSR + cardinality
  - Unidirectional auto-selected from missing CSR
  - Manifest forces strategy; cost model disagrees; manifest wins via scope
  - Caller forces strategy; manifest disagrees; caller wins via step 6 (with warning verified)
  - CSR missing but buildable; selector chooses build-CSR + bidirectional
  - Inadmissible intent triggers degradation
- The Phase 0–5 unit tests collected and linked from the integration test.
- A test-only helper `tests/core/test_res_helpers.pl` for constructing mock recurrences and workloads.

### Scope notes

- Tests should not require a real LMDB or graph — mock data is sufficient for strategy selection logic.
- The existing F# WAM e2e test continues to require LMDB/CSR setup (that's its job); the new strategy tests are pure Prolog.

### Success criteria

- All tests pass on a clean checkout.
- Code coverage of the new module is at least 80% (track with a simple `cov_helper` or just by visual inspection — the module is small).

### Estimated effort

2 hours.

## Phase 7 — Book-18 updates

### Goal

Update [`book-18-graph-algorithms`](../../education/book-18-graph-algorithms/) chapters 9 and 10 to reflect that the work named there as "open" has been done (in part). Update [appendix B](../../education/book-18-graph-algorithms/13_appendix_b_internal_theory.md) if any new internal-theory entries are warranted.

### Deliverables

- Edit `education/book-18-graph-algorithms/09_constraint_hint_predicates.md`:
  - Update §Hints-that-exist-today to include the new strategy-class and search-algorithm hints.
  - Update §Hints-that-don't-yet-exist to remove anything we just shipped.
- Edit `education/book-18-graph-algorithms/10_pattern_detection.md`:
  - Update §What-exists-today to include the strategy selector as a now-shipped piece of infrastructure.
  - Update §What-is-missing to reflect the narrower remaining gaps (mostly the fixed-point branch and ML-based detection).
  - Update §Concrete-prototyped-detection-bidirectional-ancestor — note that auto-selection now works.
- Possibly add an entry to appendix B summarising the strategy-selector design and pointing to the three design docs. Decide during implementation.
- Bump the "Status" line of book-18's README.md if appropriate (currently says "Initial — this book is a starting point").

### Scope notes

- The chapter edits are surgical. Aim for minimal text changes that update the empirical reality without rewriting the chapter's arc.
- Book-18 is in the *education* repo (separate from the main UnifyWeaver repo). Branch + commit + PR there separately from the main-repo work.

### Success criteria

- The chapter text reflects the shipped state of the strategy selector.
- Cross-references between the new design docs and the book remain consistent.

### Estimated effort

1 hour.

## Out-of-scope for this iteration

The philosophy doc names several gaps the design accommodates but the implementation does not address in this round. Reiterating them here for clarity:

- **Fixed-point compilation for F# WAM.** The strategy selector's `fixed_point(...)` branch is structurally present in the API and stubbed in admissibility checks; no rules in Phase 2 select it. Adding fixed-point evaluation for F# WAM is a separate (larger) piece of work that book-18 ch7 leaves as future direction.
- **Cached / lookup-table strategy.** Same situation.
- **Hybrid strategies** (magic-set transformation, demand-driven Datalog, seed-and-refine). Named in the design space; not implemented.
- **`unifyweaver explain <pred>` command.** The structured trace is emitted; a nice renderer command is not yet built. The plain stderr rendering is the temporary substitute.
- **Numeric-loop-breaking detection.** The infrastructure passes `numeric_contraction_rate(R)` through to the selector, but the *detector* that infers `r` from clause structure for a PageRank-style predicate is future work. For now, `r` is supplied via algorithm-manifest hints or omitted.
- **ML-based pattern-to-strategy classifier.** Explicitly rejected for this iteration (see philosophy §Alternatives). Revisit if the pattern space grows beyond hand-enumeration.

Each item is independently work-itemisable. None blocks this iteration.

## Risk and mitigation

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Conflict-resolution hierarchy has unintended interactions between steps | Medium | Step-by-step testing in Phase 3; each step's tests run before the integration. |
| F# WAM integration breaks existing tests | Low | Backwards-compat by-design: explicit options bypass the cost model via step 6. Run `tests/core/test_wam_fsharp_bidirectional_e2e.pl` early. |
| The Recurrence and Workload terms become unwieldy | Medium | Keep them as flat lists of structured terms; resist nesting. Add helper constructors as needed. |
| Decision traces become too verbose at scale | Low | The trace is *structured*; renderers can filter or summarise. Add a verbosity knob to `emit_reasoning_trace/1` if needed. |
| Cost-model rules require refactoring as we add more | Medium | The scoring system is the explicit abstraction. Adding rules means adding clauses; refactoring means refactoring the scoring weights. The risk is in tuning, not in structure. |

## Definition of done

This work is done when:

- All seven phases above are complete with their stated deliverables.
- The full test suite passes (existing + new).
- The F# WAM target's existing explicit-option behaviour is preserved.
- The cost-model-driven auto-selection produces correct strategies for the realistic workload combinations the tests cover.
- The decision trace appears in stderr at compile time and as a comment in generated F# code.
- Book-18 chapters 9 and 10 reflect the shipped state.
- All design docs are merged into the main UnifyWeaver repo.

## Sequencing with parallel work

This work does not block:

- Empirical task #14 (routing-correction redundancy) — independent.
- Empirical task #10 (synthetic non-tree-like graph) — independent.
- Other book-18 expansions — independent.

This work *does* benefit later from:

- Fixed-point compilation for F# WAM landing — would populate the `fixed_point` branch with real selections.
- A `unifyweaver explain <pred>` command — would render the trace more discoverably.
- A numeric-loop-breaking detector — would let auto-inferred `r` drive convergence-rate-based rules.

But none of those need to wait for this work to merge.

## See also

- [`RECURRENCE_EVALUATION_STRATEGY_PHILOSOPHY.md`](RECURRENCE_EVALUATION_STRATEGY_PHILOSOPHY.md) — why this exists.
- [`RECURRENCE_EVALUATION_STRATEGY_SPECIFICATION.md`](RECURRENCE_EVALUATION_STRATEGY_SPECIFICATION.md) — what the module does.
- [`SCAN_STRATEGY_IMPLEMENTATION_PLAN.md`](SCAN_STRATEGY_IMPLEMENTATION_PLAN.md) — closest sibling implementation plan in shape.
- [`KERNEL_SHAPE_RECOGNITION.md`](KERNEL_SHAPE_RECOGNITION.md) — upstream detection layer.
- [`book-18-graph-algorithms`](../../education/book-18-graph-algorithms/) chapters 9 and 10.
