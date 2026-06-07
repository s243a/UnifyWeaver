# Recurrence Evaluation Strategy — Implementation Plan

This document is the work plan for implementing the recurrence-evaluation-strategy selector. For the why, see [`RECURRENCE_EVALUATION_STRATEGY_PHILOSOPHY.md`](RECURRENCE_EVALUATION_STRATEGY_PHILOSOPHY.md). For the API and behaviour, see [`RECURRENCE_EVALUATION_STRATEGY_SPECIFICATION.md`](RECURRENCE_EVALUATION_STRATEGY_SPECIFICATION.md).

## Sequencing summary

The work breaks into seven phases. Phases 0–4 build the module itself; phase 5 wires it into the F# WAM target as the first consumer; phase 6 adds tests; phase 7 captures the result back into book-18.

Phases 0–4 are sequential. Phase 5 depends on 0–4. Phase 6 can begin in parallel with phase 5 (tests for individual phases can be written as those phases land). Phase 7 follows.

Total estimated effort: **~2 days of focused Prolog + tests + docs**. The biggest contributors are the conflict-resolution machine in phase 3 and the helper module in phase 5.

| Phase | Deliverable | Estimated effort |
|-------|-------------|------------------|
| 0 | Module skeleton + data types | 1 hour |
| 1 | Three-tier signal classification | 1.5 hours |
| 2 | Cost-model rules with explicit priority + admissibility | 2.5 hours |
| 3 | Conflict-resolution machine (semantic step names) | 4 hours |
| 4 | Pure-functional trace renderers | 2 hours |
| 5 | F# WAM target integration + shared input-construction helpers | 3 hours |
| 6 | Tests (unit + integration + trace-inspection) | 2.5 hours |
| 7 | Book-18 updates + companion education-repo issue | 1.5 hours |

## Phase 0 — Module skeleton + data types

### Goal

Create `src/unifyweaver/core/recurrence_evaluation_strategy.pl` with the public module declaration, type-checking helpers, and stub predicates for the public API. All predicates compile and return placeholder values; tests can target the stubs.

### Deliverables

- New file `src/unifyweaver/core/recurrence_evaluation_strategy.pl` with module declaration matching the [Specification §Module](RECURRENCE_EVALUATION_STRATEGY_SPECIFICATION.md#module).
- Type-checking helpers: `valid_recurrence/1`, `valid_workload/1`, `valid_strategy/1`. Use `library(error)` `must_be/2` and `domain_error/2`.
- Stub implementations of every public predicate that return a `not_yet_implemented(Phase)` tagged term. Calls to stubs do not throw; they return a sentinel so downstream code can detect partial implementation.
- A short module-level docstring at the top of the file pointing to the philosophy/spec/impl-plan triple plus the determinism contract.

### Scope notes

- No real logic. Stubs only.
- The stub for `select_evaluation_strategy/3` returns `strategy_choice(strategy(per_query(unidirectional)), trace([step(stub, not_yet_implemented(phase_0), [])]))` — a working baseline that the F# WAM integration can call against from day one.
- The stub trace step uses the explicit atom `stub` as its name, so test assertions can detect "stub steps leaked into a later phase's behaviour" (see Phase 5 success criteria).

### Success criteria

- Module loads without error.
- Every exported predicate exists and is type-checked.
- Calls to `select_evaluation_strategy/3` return the baseline strategy.
- A smoke test (`tests/core/test_res_skeleton.pl`) verifies the module loads, the baseline strategy is returned, the trace contains the `stub` step, and the determinism contract holds (called inside `if-then-else` succeeds deterministically; called inside `findall/3` returns exactly one solution).
- **`valid_strategy/1` must check the outer `strategy/1` wrapper, not just the inner `Mode`.** A `Mode` term passed without the outer wrapper (e.g. `per_query(bidirectional)` directly, not `strategy(per_query(bidirectional))`) should fail validation. The test for `valid_strategy/1` includes both positive cases (well-formed `strategy(_)` terms) and negative cases (bare `Mode` terms, malformed wrappers).

### Estimated effort

1 hour.

## Phase 1 — Three-tier signal classification

### Goal

Implement `classify_signals/4` and the underlying intent / declared-data / inferred-data dispatch tables. Workload terms get sorted into three lists according to the [Specification §Workload signals](RECURRENCE_EVALUATION_STRATEGY_SPECIFICATION.md#workload-signals).

### Deliverables

- `classify_signals/4` implementation reading the dispatch tables.
- A static dispatch table (Prolog facts) mapping signal functors to one of `intent` / `declared_data` / `inferred_data` plus the signal's source name (caller / manifest / policy / inferred-by-X).
- Forward-compat behaviour: unknown signal functors go to `InferredDataSignals` with a trace warning (a `step(classify_signals, ..., [unknown_signal(F)])` entry). The selector module does *not* write to stderr; the trace renderer surfaces the warning.
- **Concatenation of declared-data + inferred-data before calling `apply_cost_model/3`.** The cost-model's data-signals input is the simple list concatenation of declared-data and inferred-data (declared first, then inferred). This concat happens at the call site in `select_evaluation_strategy/3`, not inside `apply_cost_model/3` (which sees a flat list and treats it as its own input). Rules that need to distinguish the two tiers do so by checking individual signal terms against the dispatch table or by reading the signal's `source` attribute if needed; for Phase 2 rules, the confidence-weighting in the scoring sees per-signal tier directly.
- Unit tests in `tests/core/test_res_signals.pl` covering: every known signal type in each tier, a mix, an unknown signal (verified to land in `inferred_data` with a trace warning), empty workload, only-intent, only-declared-data, only-inferred-data.

### Scope notes

- The dispatch table lives in the module file as ordinary facts (no JSON config). Adding a signal type means adding a fact.
- The classification is *static* — it doesn't depend on the recurrence or graph state. Same workload always classifies the same way.
- The graph_stats(b_eff, D, r) signal from earlier drafts is split into three independent signals (`b_eff(F)`, `branching_d(F)`, `contraction_r(F)`) per the SPEC; each is classified as `declared_data` separately. Partial availability is natural.

### Success criteria

- All unit tests pass.
- `classify_signals([], I, D, F)` returns `I = [], D = [], F = []`.
- Manually-constructed mixed workloads return the expected partition across all three tiers.
- Unknown-signal warning appears in the trace exactly once per unknown functor per test run; no stderr writes from the module.

### Estimated effort

1.5 hours.

## Phase 2 — Cost-model rules with explicit priority + admissibility

### Goal

Implement the six initial cost-model rules from the [Specification §Phase B](RECURRENCE_EVALUATION_STRATEGY_SPECIFICATION.md#phase-b--cost-model-evaluation) plus the scoring/tiebreak logic with explicit Priority argument. Implement `admissible_strategies/2` with the convergence-gating and monotonicity-gating from the SPEC. `apply_cost_model/3` returns a `cost_model_choice(Strategy, Score, DecidingRule)` term.

### Deliverables

- The six rules as clauses of a `cost_model_rule/6` predicate (signature: `+RuleName, +Priority, +Recurrence, +DataSignals, -Score, -ChosenStrategy`). The rules are *not multifile* — clause-order-dependent tiebreaking across load orders is fragile; the explicit `Priority` argument replaces it. Future rules add a new in-module clause with their own priority.
- The scoring aggregator: collect all firing rules, score the cumulative preference per candidate strategy (with confidence weighting for inferred signals — see SPEC), pick the highest-scoring strategy. Tiebreak by `Priority` (higher wins); priority-tie broken by lexicographic `RuleName`.
- `admissible_strategies/2` with the *two-condition* admissibility test from the SPEC: KernelKind permits + termination-guarantee permits. Termination-guarantee logic splits by `value_domain(combinatorial|numeric)` (combinatorial+monotone is sufficient; numeric needs contraction-rate < 1). The monotonicity-cross-class restriction is *not* applied here — it depends on intent context and is applied in Phase C resolution.
- A kernel-kind-to-strategies static table (Prolog facts) mapping each `KernelKind` to its admissible strategy list. *Maintenance note*: this table must be kept in sync with `recursive_kernel_detection.pl`'s detector registry. Add a TODO comment in both files referencing the other.
- `apply_cost_model/3` wrapping the above.
- Unit tests in `tests/core/test_res_cost_model.pl` covering: each rule individually; the SPEC worked-scoring example; rule-interaction cases (multiple rules firing; priority tiebreaks; confidence weighting for inferred-only rules); the contraction-rate-gating of `fixed_point` strategies for numeric recurrences. (The `monotone(false)` cross-class restriction is tested in Phase 3 with the conflict-resolution machine — *not* here, because admissibility deliberately does not apply that restriction.)

### Scope notes

- The `Priority` argument explicit replaces multifile load-order. If a future contributor wants to add a rule, they pick a priority and add a clause — the behaviour is data, not load-order-dependent.
- The deciding-rule trace step names the rule that *won* the scoring, not every rule that fired. The full firing-rule list is in a separate trace detail.
- Numeric contraction rate `r` is read from `Recurrence` and used by `admissible_strategies/2` to gate fixed_point. None of the Phase 2 *rules* select fixed_point (that's stubbed), but admissibility correctly excludes fixed_point when `r` is none or ≥ 1.

### Success criteria

- All six rules return correct strategies for their preconditions.
- The worked scoring example from the SPEC produces the exact score and winning rule documented there.
- `cardinality(large)` + `csr_available(true)` triggers `prefer_bidirectional_csr_present`.
- Empty data signals trigger `default_fallback` and return `per_query(unidirectional)`.
- Two rules with the same score resolve to the higher-Priority one with the tie noted in the trace.
- For `value_domain(numeric)` recurrences, `numeric_contraction_rate(none)` causes `admissible_strategies/2` to omit all `fixed_point(_)` strategies regardless of KernelKind.
- For `value_domain(combinatorial)` recurrences, `fixed_point` strategies remain admissible regardless of `numeric_contraction_rate` value (the contraction rate is not consulted for combinatorial).
- `monotone(false)` does *not* affect `admissible_strategies/2` output — the monotonicity-cross-class restriction is applied in Phase 3's conflict-resolution machine where intent is in scope. A unit test verifies that `admissible_strategies/2` returns the *same* list for `monotone(true)` and `monotone(false)` recurrences with otherwise-identical properties.

### Estimated effort

2.5 hours.

## Phase 3 — Conflict-resolution machine (semantic step names)

### Goal

Implement `resolve_against_intent/5` with the six-step hierarchy from the [Specification §Phase C](RECURRENCE_EVALUATION_STRATEGY_SPECIFICATION.md#phase-c--conflict-resolution), using **semantic step names** (not ordinals). This is the most involved phase; budget extra time.

### Deliverables

- `resolve_against_intent/5` with each step as a separate helper predicate using semantic names: `step_no_intent/4`, `step_intent_matches/4`, `step_third_option/4`, `step_scope_disambiguation/4`, `step_satisfiability/4`, `step_caller_wins/4`. Each step takes the same inputs and produces either `resolved(Strategy, TraceEntry)` or `next_step`.
- The `intent_compatible_with_strategy/2` predicate implementing the [intent-compatibility matrix](RECURRENCE_EVALUATION_STRATEGY_SPECIFICATION.md#intent-compatibility-matrix) from the SPEC. Each row of the matrix is a clause; adding intent types means adding clauses.
- The satisfiability adjuster — handles `build_csr_at_compile_time`, `degrade_to_compatible`, `degrade_with_warning`. The adjuster records the chosen adjustment as a **detail field** of the `step_satisfiability` trace entry (e.g. `step(satisfiability, adjusted, [adjustment(build_csr_at_compile_time), unmet_intent(...), reason(...)])`), NOT as a separate `step(adjustment, ...)` entry — the per-step single-entry fold contract requires each step to contribute exactly one trace entry. The strategy term remains the post-adjustment destination.
- Unit tests in `tests/core/test_res_conflict.pl` covering each step in isolation, the full hierarchy walked through, every step exiting both via `resolved` and via `next_step`, and the SPEC's full scoping-example.

### Scope notes

- Each step is small and individually testable.
- The full hierarchy is just a fold over the steps — `resolve_against_intent/5` walks the list of steps in order, taking the first `resolved`.
- The trace entry for each step records what fired and why; `next_step` returns an empty trace entry (the step is skipped but does not appear in the trace).
- "Build CSR at compile time" is recorded as `adjustment(build_csr_at_compile_time)` in the **detail field** of whatever step caused it (typically `step_satisfiability` for unmet-intent adjustments, or `step_cost_model_choice` for rule-level adjustments — see [SPEC §scoring-example](RECURRENCE_EVALUATION_STRATEGY_SPECIFICATION.md#scoring-example)). The strategy module does *not* execute the build; the target adapter walks the trace looking for `adjustment(...)` detail fields on any step and emits the build-CSR step in its generated code.
- The intent-compatibility matrix is a finite table. Adding intent signal types means adding rows. Document the matrix's existence in the module file's header comments so future contributors know to update it.

### Hidden-scope notes (flagged by review)

Two undocumented maintenance surfaces this phase creates:

1. **Kernel-kind-to-strategies table** (from Phase 2) — when `recursive_kernel_detection.pl` adds a new detector, the table here must be updated. Each phase adds a TODO comment in both files cross-referencing the other.
2. **Intent-compatibility matrix** — when a new intent signal type is added to the SPEC, the matrix here must be updated. Comments in the module file flag this.

Both are documented in this implementation plan, in the SPEC, and as in-code comments at the maintenance points themselves. Triple coverage is intentional — these are the most likely "I added X and forgot Y" bugs.

### Success criteria

- Each step's individual tests pass.
- Integration tests walk realistic decision scenarios end-to-end, including the conflict + override case.
- The trace for each decision is complete and accurate (matches the SPEC's example outputs to the structural level — actual atom values may differ for non-stable rules).

### Estimated effort

4 hours.

## Phase 4 — Pure-functional trace renderers

### Goal

Implement `render_trace_for_stderr/2` (produces list of strings) and `format_trace_for_comment/3` (produces multi-line comment string with per-line prefix). Both render the same structured trace into different surface forms. **Neither writes to any stream** — the selector module remains pure-functional.

### Deliverables

- `render_trace_for_stderr/2` — returns a list of strings, one per trace step, indented for readability. Critical signals named. Target adapter writes the strings to stderr; the renderer does not.
- `format_trace_for_comment/3` — returns a multi-line comment string. Takes a `CommentPrefix` argument (`"// "`, `"-- "`, `"% "`, `"# "`, etc.) and prefixes *every* line, not just the first. Critical: `%` (Prolog) and `#` (Python) are line-level comment syntaxes; prefixing only the first line would produce invalid code.
- Pretty-printer helpers: `strategy_pretty/2`, `signal_pretty/2`, `step_pretty/2`.
- Unit tests in `tests/core/test_res_trace.pl` covering rendering of each step type plus end-to-end rendering of full traces from the Phase 3 tests.
- **Round-trip parse test**: for each supported `CommentPrefix`, render a sample trace as a comment header followed by a minimal valid source body for that language, then feed the full prefixed output to the language's actual parser and verify zero parse errors. Specifically:
  - **Prolog**: write the rendered comment + a minimal clause (`test_compiled :- true.`) to a `.pl` file; invoke `swipl --halt -g test_compiled -t halt -s <file>` and assert exit code 0. This catches the failure mode where `%` is not prefixed on every line (Prolog `%` is line-level; missing prefix causes a syntax error on the next line).
  - F# / Haskell / Python: pattern-match assertions on the rendered string (every non-blank line begins with the prefix; no embedded delimiters from the trace text break out of the comment). Full language-parser round-trips are not assumed here because those toolchains aren't available in the Prolog test environment; the Prolog round-trip catches the most failure-prone case (the line-level-comment-syntax languages).
- **(Earlier-draft note)** A previous version of this test specification incorrectly proposed using `read_term/2` on the *stripped* content of the comment. That doesn't work — the comment body is human-readable text, not Prolog terms; `read_term/2` would fail to parse it. The correct test is the one above: feed the full prefixed output (with comment prefixes intact) to `swipl --halt` as source code, where Prolog's lexer correctly skips comments and the embedded clause body is what gets parsed.

### Scope notes

- The structured trace is the canonical form; renderers are derived. This is what guarantees consistency between stderr and code-comment renderings.
- Comment prefix is a string argument, fully parameterised. F# uses `"// "`; Haskell uses `"-- "`; Prolog uses `"% "`; Python uses `"# "`; SQL uses `"-- "`.
- The renderers should be idempotent (rendering the same trace twice produces identical output).
- The selector module's exports include the renderer predicates but the selector itself does not call them. Target adapters call the renderers.

### Success criteria

- Sample traces render as expected in both forms.
- The Prolog round-trip test (feed prefixed output + minimal clause body to `swipl --halt`, assert exit 0) catches the missing-line-prefix failure mode.
- Pattern-match assertions cover F# / Haskell / Python (no language tooling assumed in the test env).
- The selector module test suite contains zero `format/2` or `write/1` calls outside the explicit renderers — verified by grep in CI.

### Estimated effort

2 hours.

## Phase 5 — F# WAM target integration + shared input-construction helpers

### Goal

Wire `select_evaluation_strategy/3` into `src/unifyweaver/targets/wam_fsharp_target.pl` as the first consumer. Create the shared `recurrence_inputs.pl` module for target-agnostic input construction.

### Deliverables

- New module `src/unifyweaver/core/recurrence_inputs.pl` exporting:
  - `build_recurrence_term/3` — constructs the `Recurrence` term from a detected kernel + manifest + relation-policy lookups.
  - `build_workload_signals/2` — gathers caller options + manifest entries + relation-policy declarations + graph stats into a workload list.
  - **Constraint**: the helper module must remain *target-agnostic*. No F#-specific (Haskell-specific, C-specific) logic. If a target needs target-specific input transformation, it does so before calling the helpers.
  - **Enforcement (two layers)**:
    1. **Load-isolation test** — `tests/core/test_recurrence_inputs_isolated.pl` loads `recurrence_inputs.pl` with no target modules loaded (only the core dependencies `recursive_kernel_detection`, `algorithm_manifest`, `relation_policy`, `cost_model`, `cost_function`). The module must load cleanly. If it depends on a target module, loading without target modules will fail. This is a stronger test than grep because it catches transitive dependencies grep misses (e.g. a helper-of-a-helper that imports F#-specific code).
    2. **Grep check (secondary)** — `tests/core/test_recurrence_inputs_grep.pl` greps for target atoms (`fsharp`, `haskell`, `csharp`, etc.) in the module body. Faster than load-isolation but doesn't catch transitive deps. Kept as a tripwire because grep is cheap.
- The existing `maybe_upgrade_bidirectional/2` call site in `wam_fsharp_target.pl` replaced with:
  1. Build inputs via `recurrence_inputs:build_recurrence_term/3` and `recurrence_inputs:build_workload_signals/2`.
  2. Call `recurrence_evaluation_strategy:select_evaluation_strategy/3`.
  3. Dispatch on the returned `Strategy`. The existing upgrade path becomes one branch of the dispatch.
  4. Walk the trace looking for `adjustment(...)` detail fields on any step (typically on `step_satisfiability` or `step_cost_model_choice`); execute each adjustment by emitting the corresponding code into the generated F# project (e.g. a build-CSR step).
  5. Call `recurrence_evaluation_strategy:format_trace_for_comment(Trace, "// ", Comment)` and insert at the kernel call site.
  6. Optionally call `recurrence_evaluation_strategy:render_trace_for_stderr/2` and write to stderr; quiet-mode skips this.
- Backwards-compat: existing tests (`tests/core/test_wam_fsharp_bidirectional_e2e.pl`) pass without modification because explicit `kernel_mode(bidirectional)` is honoured by the conflict-resolution hierarchy.

### Scope notes

- The integration changes `wam_fsharp_target.pl` in a small number of places (the upgrade call site, the comment insertion point). The rest of the F# target is unchanged.
- `recurrence_inputs.pl` is the canonical shared helper module. Haskell and C WAM targets will reuse it in their own future integration work.

### Success criteria

- `tests/core/test_wam_fsharp_bidirectional_e2e.pl` passes unchanged.
- A new variant of the test (`tests/core/test_wam_fsharp_strategy_autoselect_e2e.pl`) calls without explicit `kernel_mode(bidirectional)`, relying on cost-model selection, and produces bidirectional code when the workload signals support it.
- **Trace-inspection assertion (positive)**: the new test reads the generated F# file, parses the comment header, and asserts the trace's `final_decision` step contains `per_query(bidirectional)` and was decided by `prefer_bidirectional_csr_present` (or equivalent cost-model rule). This proves the selector was actually invoked, not bypassed.
- **Trace-inspection assertion (negative)**: the same test asserts the trace contains **no** `step(caller_wins, ...)` entry, and the `final_decision`'s `decided_by` field is NOT `caller_wins`. A resolution machine that incorrectly walks all the way to the caller-wins fallback when it should have resolved earlier (e.g. via `step_intent_matches` or `step_scope_disambiguation`) would produce a `caller_wins` step in the trace; the negative assertion catches this. Without this assertion, a buggy resolution machine that always falls through to caller-wins would pass the positive trace-inspection test (the right strategy gets chosen, but for the wrong reason).
- **Stub-leak assertion**: same test asserts that *no* trace step has `Name = stub`. Stub steps from Phase 0 should never appear in real selections; if they do, Phases 1–4 have an incomplete implementation.
- The load-isolation test (`tests/core/test_recurrence_inputs_isolated.pl`) loads `recurrence_inputs.pl` without any target modules and confirms it loads cleanly. Catches transitive target dependencies.
- The grep tripwire test (`tests/core/test_recurrence_inputs_grep.pl`) verifies `recurrence_inputs.pl` has no target-specific atoms in its body. Faster than load-isolation; kept as a tripwire for the cases grep catches.
- **Mock-second-caller test** (`tests/core/test_recurrence_inputs_mock_caller.pl`): a mock caller imitating a hypothetical Haskell-WAM integration exercises `build_recurrence_term/3` and `build_workload_signals/2` with workload signals that have no F# semantics. Catches the failure mode where the helpers were silently shaped by F# as the only first-iteration caller, freezing the API with implicit F#-specific assumptions before the second target validates it. The mock-second-caller doesn't have to *produce* working Haskell code; it just has to exercise the helpers in a way that proves they're not F#-shaped.

### Estimated effort

3 hours.

## Phase 6 — Tests (unit + integration + trace-inspection)

### Goal

Comprehensive unit and integration tests. Some are scoped into Phases 1–5 above; this phase backfills coverage and adds end-to-end integration tests.

### Deliverables

- `tests/core/test_recurrence_evaluation_strategy.pl` — integration test running the full pipeline end-to-end with several realistic workload + recurrence combinations:
  - Bidirectional auto-selected from CSR + cardinality
  - Unidirectional auto-selected from missing CSR (declared + inferred)
  - Manifest forces strategy; cost model disagrees; manifest wins via scope_disambiguation
  - Caller forces strategy; manifest disagrees; caller wins via step_caller_wins (with warning verified in trace, not stderr)
  - CSR missing but buildable; selector chooses build-CSR + bidirectional (adjustment step verified)
  - Inadmissible intent triggers degradation
  - `numeric_contraction_rate(none)` + explicit fixed-point intent throws `no_admissible_strategy`
- The Phase 0–5 unit tests collected and linked from the integration test.
- A test-only helper `tests/core/test_res_helpers.pl` for constructing mock recurrences, workloads, and trace-inspection assertions.

### Scope notes

- Tests should not require a real LMDB or graph — mock data is sufficient for strategy selection logic.
- The existing F# WAM e2e test continues to require LMDB/CSR setup (that's its job); the new strategy tests are pure Prolog.
- The trace-inspection helper provides predicates like `trace_step_present(+Trace, +StepName, +ExpectedOutcome)` and `trace_has_no_stubs(+Trace)` for use in higher-level tests.

### Success criteria

- All tests pass on a clean checkout.
- Code coverage of the new module is at least 80% (track with a simple `cov_helper` or just by visual inspection — the module is small).
- The trace-inspection helpers are reused in `test_wam_fsharp_strategy_autoselect_e2e.pl` (verified by import).

### Estimated effort

2.5 hours.

## Phase 7 — Book-18 updates + companion education-repo issue

### Goal

Update [`book-18-graph-algorithms`](https://github.com/s243a/UnifyWeaver_Education/tree/main/book-18-graph-algorithms) chapters 9 and 10 to reflect that the work named there as "open" has been done (in part). Update [appendix B](https://github.com/s243a/UnifyWeaver_Education/tree/main/book-18-graph-algorithms/13_appendix_b_internal_theory.md) if any new internal-theory entries are warranted. Open a companion issue in the `UnifyWeaver_Education` repo to track these updates explicitly.

### Deliverables

- **Companion issue** in `UnifyWeaver_Education` repo (opened when this PR merges in the main repo). The issue title: "Book-18 updates from recurrence-evaluation-strategy implementation". The issue body links the main-repo PR (#2864), names the chapters/sections affected, and tracks the cross-repo update as a unit of work.
- Edit `book-18-graph-algorithms/09_constraint_hint_predicates.md`:
  - Update §Hints-that-exist-today to include the new strategy-class and search-algorithm hints (`strategy(...)`, `kernel_mode(...)`, `force_search_algorithm(...)`).
  - Update §Hints-that-don't-yet-exist to remove anything we just shipped.
  - Add a note about the three-tier signal classification (intent / declared-data / inferred-data) and how relation-policy declarations vs static-analysis inferences differ in confidence.
- Edit `book-18-graph-algorithms/10_pattern_detection.md`:
  - Update §What-exists-today to include the strategy selector as a now-shipped piece of infrastructure.
  - Update §What-is-missing to reflect the narrower remaining gaps (mostly the fixed-point branch, the C# query runtime as second consumer, ML-based detection).
  - Update §Concrete-prototyped-detection-bidirectional-ancestor — note that auto-selection now works.
- New entry in `book-18-graph-algorithms/13_appendix_b_internal_theory.md`:
  - `B.13 — The r-as-contraction-rate conjecture`. Records that the design depends on identifying `r = b'/(b_eff·D)` with the spectral contraction rate of the linearised `d_wPow` iteration operator; documents that this is a conjecture pending verification, points to the philosophy doc's hedge, names the rigorous identification as future theory work.
- Bump the "Status" line of book-18's README.md if appropriate (currently says "Initial — this book is a starting point").

### Scope notes

- The chapter edits are surgical. Aim for minimal text changes that update the empirical reality without rewriting the chapter's arc.
- Book-18 is in the *education* repo (separate from the main UnifyWeaver repo). Branch + commit + PR there separately from the main-repo work. The companion issue ties the two PRs together for tracking purposes.
- The appendix B entry is the proof-of-acknowledgement that the design depends on a conjecture; the entry persists across book revisions until the conjecture is rigorously proved (or refuted).

### Success criteria

- The chapter text reflects the shipped state of the strategy selector.
- Cross-references between the new design docs and the book remain consistent.
- The companion `UnifyWeaver_Education` issue is open and linked from this PR's discussion.
- The new appendix B entry exists and is reachable from the appendix B index.

### Estimated effort

1.5 hours.

## Out-of-scope for this iteration

The philosophy doc names several gaps the design accommodates but the implementation does not address in this round. Reiterating them here for clarity:

- **Fixed-point compilation for F# WAM.** The strategy selector's `fixed_point(...)` branch is structurally present in the API and stubbed in admissibility checks; no rules in Phase 2 select it. Adding fixed-point evaluation for F# WAM is a separate (larger) piece of work that book-18 ch7 leaves as future direction. The C# parameterised query target is the existing realisation of bottom-up fixed-point and would be the natural second consumer of the selector when fixed-point lands.
- **Cached / lookup-table strategy.** Same situation.
- **Hybrid strategies** (magic-set transformation, demand-driven Datalog, seed-and-refine). Named in the design space; not implemented.
- **`unifyweaver explain <pred>` command.** The structured trace is emitted; a nice renderer command is not yet built. The plain stderr rendering (via target adapter) is the temporary substitute.
- **Numeric-loop-breaking detection.** The infrastructure passes `numeric_contraction_rate(R)` through to the selector, but the *detector* that infers `r` from clause structure for a PageRank-style predicate is future work. For now, `r` is supplied via algorithm-manifest hints or omitted.
- **ML-based pattern-to-strategy classifier.** Explicitly rejected for this iteration (see philosophy §Alternatives). Revisit if the pattern space grows beyond hand-enumeration.
- **Rigorous identification of `r` with the spectral contraction rate.** The philosophy doc hedges this as a conjecture; appendix B.13 in book-18 names it as tracked theory work. Not blocking this iteration.

Each item is independently work-itemisable. None blocks this iteration.

## Risk and mitigation

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Conflict-resolution hierarchy has unintended interactions between steps | Medium | Step-by-step testing in Phase 3; each step's tests run before the integration. |
| F# WAM integration breaks existing tests | Low | Backwards-compat by-design: explicit options bypass the cost model via `step_caller_wins`. Run `tests/core/test_wam_fsharp_bidirectional_e2e.pl` early. |
| The Recurrence and Workload terms become unwieldy | Medium | Keep them as flat lists of structured terms; resist nesting. Add helper constructors as needed. |
| Decision traces become too verbose at scale | Low | The trace is *structured*; renderers can filter or summarise. Add a verbosity knob to `render_trace_for_stderr/2` if needed. |
| Cost-model rules require refactoring as we add more | Medium | The scoring system + explicit `Priority` argument is the abstraction. Adding rules means adding clauses; refactoring means refactoring the scoring weights. The risk is in tuning, not in structure. |
| Stub trace entries leak into post-skeleton code paths | Medium | Phase 5 success criteria includes a stub-leak assertion in the integration test. Catches the leak before users see it. |
| Kernel-kind-to-strategies table drifts from `recursive_kernel_detection.pl` registry | Medium | TODO comments in both files; Phase 3 documents the maintenance surface explicitly. Future kernel-addition PRs are expected to touch both. |
| Intent-compatibility matrix drifts from SPEC's matrix table | Low | Phase 3 documents the matrix as the single source of truth; in-code comment points to the SPEC section. |
| Cross-repo update for book-18 gets forgotten after main-repo PR merges | Medium | Phase 7 opens a companion issue in `UnifyWeaver_Education` immediately on merge, creating the cross-repo tracking link. |
| The r = diagonal-dominance conjecture turns out to be wrong in a substantive way | Low | The selector treats `r` as an upper-bound estimator; cost rules are robust to small estimator error. If the conjecture is wrong, the cost rules need recalibration, not rearchitecting. Appendix B.13 tracks the theory work. |
| `recurrence_inputs.pl` shaped by F# as its only first-iteration caller, freezing the API with implicit F#-specific assumptions before the Haskell or C WAM target validates it | Medium | The Phase 5 mock-second-caller test (`test_recurrence_inputs_mock_caller.pl`) exercises the helpers with non-F# workload signals before Haskell-WAM exists as a real consumer. Catches premature F#-shape calcification. The load-isolation test catches a related failure mode (transitive deps on F# modules). |

## Definition of done

This work is done when:

- All seven phases above are complete with their stated deliverables.
- The full test suite passes (existing + new), including:
  - Backwards-compat: `tests/core/test_wam_fsharp_bidirectional_e2e.pl` passes unchanged.
  - Auto-select: `tests/core/test_wam_fsharp_strategy_autoselect_e2e.pl` passes (verifies selector was invoked via trace inspection).
  - Stub-leak: no trace contains stub steps after Phase 5.
- The cost-model-driven auto-selection produces correct strategies for the realistic workload combinations the tests cover.
- The decision trace appears as a comment in generated F# code, and (optionally) as stderr output when the target adapter renders it.
- The selector module itself does *not* write to any stream — verified by grep in CI.
- Book-18 chapters 9 and 10 reflect the shipped state.
- Companion issue in `UnifyWeaver_Education` is open and tracks the book-update PR.
- Appendix B.13 in book-18 records the `r = contraction-rate` conjecture and its tracked-theory-work status.
- All design docs (philosophy, spec, implementation plan) are merged into the main UnifyWeaver repo.

## Sequencing with parallel work

This work does not block:

- Empirical task #14 (routing-correction redundancy) — independent.
- Empirical task #10 (synthetic non-tree-like graph) — independent.
- Other book-18 expansions — independent.

This work *does* benefit later from:

- Fixed-point compilation for F# WAM landing — would populate the `fixed_point` branch with real selections.
- C# query runtime integration as second consumer — would validate the cross-target reuse claim.
- A `unifyweaver explain <pred>` command — would render the trace more discoverably.
- A numeric-loop-breaking detector — would let auto-inferred `r` drive convergence-rate-based rules.
- Rigorous identification of `r` with the spectral contraction rate (appendix B.13) — would tighten the cost-model rules' theoretical foundation.

But none of those need to wait for this work to merge.

## See also

- [`RECURRENCE_EVALUATION_STRATEGY_PHILOSOPHY.md`](RECURRENCE_EVALUATION_STRATEGY_PHILOSOPHY.md) — why this exists.
- [`RECURRENCE_EVALUATION_STRATEGY_SPECIFICATION.md`](RECURRENCE_EVALUATION_STRATEGY_SPECIFICATION.md) — what the module does.
- [`SCAN_STRATEGY_IMPLEMENTATION_PLAN.md`](SCAN_STRATEGY_IMPLEMENTATION_PLAN.md) — closest sibling implementation plan in shape.
- [`KERNEL_SHAPE_RECOGNITION.md`](KERNEL_SHAPE_RECOGNITION.md) — upstream detection layer.
- `book-18-graph-algorithms` chapters 9 and 10 + appendix B (in the separate `UnifyWeaver_Education` repo).
