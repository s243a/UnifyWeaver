# Recurrence Evaluation Strategy — Specification

> *Design-document-set: recurrence-evaluation-strategy v1 (philosophy / specification / implementation-plan).*

This document specifies the API, data structures, decision logic, and outputs of the recurrence-evaluation-strategy selector. For the why, see [`RECURRENCE_EVALUATION_STRATEGY_PHILOSOPHY.md`](RECURRENCE_EVALUATION_STRATEGY_PHILOSOPHY.md). For the work plan, see [`RECURRENCE_EVALUATION_STRATEGY_IMPLEMENTATION_PLAN.md`](RECURRENCE_EVALUATION_STRATEGY_IMPLEMENTATION_PLAN.md).

## Module

`src/unifyweaver/core/recurrence_evaluation_strategy.pl`

```prolog
:- module(recurrence_evaluation_strategy, [
    %% Main entry point
    select_evaluation_strategy/3,        % +Recurrence, +Workload, -StrategyAndTrace

    %% Helper exports for target consumers and tests
    classify_signals/4,                  % +Workload, -Intent, -DeclaredData, -InferredData
    apply_cost_model/3,                  % +Recurrence, +DataSignals, -CostModelChoice
    resolve_against_intent/5,            % +IntentSignals, +CostModelChoice, +Recurrence,
                                         %   -Strategy, -Trace

    %% Trace renderers (called by target adapters, not by the selector itself)
    render_trace_for_stderr/2,           % +Trace, -StderrLines
    format_trace_for_comment/3,          % +Trace, +CommentPrefix, -CommentString

    %% Introspection
    admissible_strategies/2,             % +Recurrence, -List
    strategy_pretty/2                    % +Strategy, -String
]).
```

### Determinism contracts (per predicate)

Each exported predicate has its own determinism contract. These matter for callers using the predicates inside `if-then-else`, `findall/3`, or other constructs sensitive to determinism.

- **`select_evaluation_strategy/3`** — `det`. Always succeeds for a well-formed `Recurrence` and `Workload`. Never fails. Throws `error(no_admissible_strategy(_), _)` on the structural failure case (no strategy is admissible for the given recurrence). Always binds `StrategyAndTrace` to a ground term.

- **`classify_signals/4`** — `det`. Always succeeds; partitions the input into three lists; never fails. Returns three lists (possibly empty) for any input list (including empty).

- **`apply_cost_model/3`** — `det`. Always succeeds when given a well-formed `Recurrence` and data-signals list. Returns a `cost_model_choice(Strategy, Score, DecidingRule)` term; if no rules fire, returns the `default_fallback` choice. Never fails.

- **`resolve_against_intent/5`** — `det`. Always resolves to one of the six steps; if all earlier steps pass, `step_caller_wins` always resolves. Never fails. Throws only if a step's internal logic encounters a malformed intent (e.g. an intent with no matcher in the intent-compatibility matrix beyond the "unknown intent" fallback).

- **`render_trace_for_stderr/2`** — `det`. Always succeeds for any well-formed trace. Returns a list of strings.

- **`format_trace_for_comment/3`** — `det`. Always succeeds for any well-formed trace and comment prefix. Returns a string.

- **`admissible_strategies/2`** — `det`. May return `[]` (which then causes `select_evaluation_strategy/3` to throw `no_admissible_strategy`). Never fails on a well-formed `Recurrence`.

- **`strategy_pretty/2`** — `det`. Always succeeds for any well-formed `Strategy` term.

### Trace emission is pure-functional

The selector module **does not write to any stream**. It returns the structured trace as part of the `StrategyAndTrace` return value. Target adapters (or thin wrappers around them) call the renderer helpers (`render_trace_for_stderr/2`, `format_trace_for_comment/3`) and decide whether to emit. This keeps the selector pure-functional and testable; quiet-mode is the default for library and test use.

## API

### `select_evaluation_strategy(+Recurrence, +Workload, -StrategyAndTrace)`

The main entry point. Combines signal classification, cost-model evaluation, and conflict resolution into one call.

- `Recurrence` — a term describing properties of the recurrence the user's predicate defines (see [Recurrence properties](#recurrence-properties)). Constructed by the caller from the detected kernel + clause analysis + algorithm-manifest declarations.
- `Workload` — a list of signal terms describing the workload (see [Workload signals](#workload-signals)). Includes caller options, manifest entries, relation-policy declarations, and graph statistics.
- `StrategyAndTrace` — the term `strategy_choice(Strategy, Trace)` where `Strategy` is the chosen evaluation strategy and `Trace` is a structured list of decision steps.

### Recurrence properties

```prolog
recurrence(KernelKind, Pred/Arity, Properties)
```

where `KernelKind` is the kernel atom from `recursive_kernel_detection` (e.g. `bidirectional_ancestor`, `transitive_closure2`, `category_ancestor`), `Pred/Arity` is the user predicate, and `Properties` is a list of:

| Property | Type | Meaning |
|----------|------|---------|
| `value_domain(D)` | atom | `combinatorial` (boolean / finite Herbrand-base lattice — Datalog case) or `numeric` (continuous real-valued lattice — `d_wPow`, PageRank case). Drives whether `fixed_point` admissibility needs a contraction rate (see [Admissibility](#admissibility)) |
| `has_combinatorial_loop_break(B)` | boolean | `true` if the user's clauses include `\+ member` or equivalent visited-set tracking. This is a **per-query traversal safety** signal, *not* a fixed_point enabler. **Absent vs `false`**: a recurrence that omits this property entirely (e.g. Dijkstra-family kernels whose per-query safety comes from priority-queue monotonicity rather than from a visited-set) is treated as "not applicable" — the selector neither requires nor benefits from the signal for that kernel. A recurrence that explicitly declares `has_combinatorial_loop_break(false)` is treated as "user-stated unsafe on cyclic input" — the selector emits a warning if it would auto-select a per-query strategy that traverses cycles. The two cases are semantically distinct |
| `numeric_contraction_rate(R)` | maybe Float | An upper bound on the iteration contraction rate. `none` if unknown or non-numeric; a float `< 1` if known to converge. Only consulted when `value_domain(numeric)` — combinatorial recurrences don't need it |
| `monotone(B)` | boolean | `true` if the recurrence operator is monotone over its solution lattice (default `true` for Datalog-shaped recurrences). `false` disables cross-class strategy auto-selection (applied in Phase C, not in admissibility) |
| `result_layout(L)` | term | The output-tuple layout from `recursive_kernel_detection:kernel_result_layout/2` |
| `result_mode(M)` | atom | `stream` / `deterministic` / `deterministic_collection` — from `kernel_result_mode/2` |

The `Recurrence` term is constructed once per `select_evaluation_strategy/3` call. The caller's responsibility is to populate it from the detected kernel and clause analysis. The module does not re-detect; it consumes.

**Note on removed properties.**

- `directions_admissible(Dirs)` was removed because it can disagree silently with `admissible_strategies/2` for the same `KernelKind`. The selector now *derives* admissible directions internally from the kernel-kind-to-strategies table — single source of truth, no inconsistency possible.
- `expected_cardinality(C)` was removed because it duplicates the `cardinality(C)` workload signal (which comes from the same `relation_policy` declaration). Two sources for the same fact would create a silent sync requirement between `build_recurrence_term/3` and `build_workload_signals/2`. The cost model reads `cardinality` from the workload signals only; recurrence properties hold only what is intrinsic to the recurrence's *structure* (loop-break, contraction rate, value domain, monotonicity, result shape), not properties of the *data* the recurrence runs over.

### Workload signals

A list of signal terms. The classifier (`classify_signals/4`) sorts them into three tiers: intent, declared-data, and inferred-data.

#### Intent signals (override cost model)

| Signal | Source | Meaning |
|--------|--------|---------|
| `kernel_mode(Mode)` | caller option | Mode: `bidirectional`, `unidirectional`, `astar`, `dijkstra` |
| `strategy(S)` | manifest `decl_algorithm_optimization` | S: `per_query(_)`, `fixed_point(_)`, `cached`, `hybrid(_)` |
| `force_search_algorithm(A)` | caller option | A: `bfs`, `dfs`, `bidirectional`, `astar`, `dijkstra` |

Intent signals express the user's required outcome. They bypass the cost model when they are unambiguous.

#### Declared-data signals (high-confidence input to cost model)

| Signal | Source | Meaning |
|--------|--------|---------|
| `csr_path(_)` | caller option | CSR (child-direction index) is available at this path. **Most commonly used by the F# WAM target** (which consumes the on-disk CSR via LightningDB cursors); other targets may use `csr_available(true)` without a path, or extend with target-specific path-resolution. Targets that don't need an on-disk CSR path should use `csr_available(true)` instead of `csr_path(_)` — the implementation plan's Phase 5 mock-second-caller test (`tests/core/test_recurrence_inputs_mock_caller.pl`) deliberately exercises the no-`csr_path` route to verify the helpers don't assume an F#-WAM-specific shape |
| `csr_available(true)` | caller option | Explicit assertion of CSR availability |
| `cardinality(C)` | `relation_policy` declaration | `small` / `medium` / `large` / `unknown` for the edge predicate |
| `determinism(D)` | `relation_policy` | `det` / `semidet` / `nondet` / `multi` |
| `unique(B)` | `relation_policy` | `true` / `false` for the relation |
| `query_pattern(P)` | manifest | `single_pair` / `all_pairs` / `all_from_source` / `sample` |
| `query_frequency(F)` | manifest | `low` / `high` / `sustained` |
| `graph_mutability(M)` | manifest | `static` / `append_only` / `mutable` |
| `heuristic_predicate_available(B)` | manifest / caller option | `true` if an admissible heuristic predicate is declared. The corresponding caller-option key is `heuristic_predicate(+Pred/Arity)` — when this option is passed, `build_workload_signals/2` derives `heuristic_predicate_available(true)` and records the predicate indicator in a separate `heuristic_predicate(...)` workload signal so cost-model rules can name it. Precondition for `prefer_astar_heuristic_available` rule |
| `heuristic_predicate(Pred/Arity)` | derived from caller option `heuristic_predicate(Pred/Arity)` | The specific heuristic predicate to use; carried alongside `heuristic_predicate_available(true)` for downstream code-generation |
| `b_eff(F)` | calibration declaration | Friendship-paradox-corrected effective branching factor |
| `branching_d(F)` | calibration declaration | Mean child branching factor `D` |
| `contraction_r(F)` | calibration declaration | Convergence ratio `r = b'/(b_eff·D)` |
| `distribution_fit_policy(P)` | manifest / caller option | User-selected or default policy for representing finite path-statistic distributions. The policy controls when exact histograms switch to hybrid or parametric finite-support approximations; see `DISTRIBUTIONAL_FIT_POLICY.md` |

The three calibration signals (`b_eff`, `branching_d`, `contraction_r`) are *independent* — partial availability (e.g. `b_eff` and `branching_d` known but `r` not yet computed) is expressed naturally by including only the available signals.

Declared-data has high confidence; the cost model treats it as fact.

#### Inferred-data signals (lower-confidence input to cost model)

| Signal | Source | Meaning |
|--------|--------|---------|
| `csr_available(false)` | inferred (no `csr_path` supplied) | Default inference: no CSR |
| `csr_buildable(B)` | inferred from edge-predicate analysis | `true` if CSR can be built in O(\|E\|) from the existing parent-direction edge predicate |
| `query_pattern(P)` | inferred from mode declarations or call-site analysis | When not declared in manifest |
| `query_frequency(F)` | inferred from call-site count | When not declared in manifest |

Inferred-data carries explicit uncertainty; cost-model rules can downweight it via confidence-adjusted scoring (see [Phase B](#phase-b--cost-model-evaluation)).

### Admissibility

`admissible_strategies(+Recurrence, -List)` returns the strategies the selector may pick from, based on properties of the recurrence alone (no intent context). A strategy is admissible iff *both* of the following hold:

1. **KernelKind permits it** — read from a kernel-kind-to-strategies static table. Concrete examples:
   - `bidirectional_ancestor` — `value_domain(combinatorial)` — permits `per_query(bidirectional)` and `per_query(astar)`. fixed_point not in this kernel's permitted set. *Rationale*: the kernel's design intent is per-pair single-query lookup (bidirectional A\* between a source and a target); the existing F# WAM template (`templates/targets/fsharp_wam/kernel_bidirectional_ancestor.fs.mustache`) emits per-query code with no provision for batch fixed-point evaluation. Adding fixed_point to this kernel's permitted set would be a structural change (new template + cost-model rules); it's a registry gap by deliberate choice, not by oversight.
   - `transitive_closure2` — `value_domain(combinatorial)` — permits `per_query(unidirectional)`, `per_query(bidirectional)` (via upgrade), and `fixed_point(semi_naive)`. The fixed_point branch is structurally permitted; whether it's auto-selected depends on cost-model rules and is currently stubbed.
   - `weighted_shortest_path3` — `value_domain(numeric)` — permits `per_query(dijkstra)`. fixed_point not in this kernel's permitted set in the current registry. *Rationale*: the kernel's existing template is per-query Dijkstra with a priority-queue traversal; Dijkstra's correctness depends on non-negative edge weights, which the kernel's admissibility precondition enforces. A fixed_point variant would be Bellman-Ford (numeric + iteration-count bound from graph diameter — see philosophy doc's Bellman-Ford gap note) or a value-iteration template; neither template exists yet. This is a *different* gap from `bidirectional_ancestor`'s per-query-only nature: `bidirectional_ancestor` is structurally pair-oriented (source + target are both inputs), while `weighted_shortest_path3` is single-source-many-targets — different algorithmic shapes, both currently per-query-only by template design.

   *Loop-breaking note for `weighted_shortest_path3`*: Dijkstra is *inherently* safe on non-negative-weight graphs without an explicit visited-set — the priority-queue's monotonic-extraction property guarantees each node is finalised once. So `has_combinatorial_loop_break` is *not applicable* for this kernel rather than `false`: the per-query traversal safety comes from a different mechanism (priority-queue monotonicity + non-negative weights). The selector treats absence of `has_combinatorial_loop_break` as "not applicable; safety guaranteed by other kernel-specific means" for kernels in the Dijkstra family, distinct from `has_combinatorial_loop_break(false)` which would indicate "unsafe on cyclic input." Future framing may introduce a `loop_break_mechanism(Mechanism)` property to capture this distinction explicitly.

2. **Termination guarantee permits it** — for any `fixed_point(...)` strategy, the recurrence must have a termination guarantee. Two cases by value domain:
   - `value_domain(combinatorial)` — finite-state iteration always halts (state-space is finite; iteration either reaches a fixed point or enters a detectable cycle in bounded time). **No contraction-rate guarantee needed.** Datalog-shape recurrences and Bellman-Ford-with-visited-set fall in this case.
   - `value_domain(numeric)` AND `numeric_contraction_rate(R)` with `R < 1.0` — continuous lattice needs contraction. `d_wPow` and PageRank-style recurrences fall in this case.
   - If neither condition holds (e.g. `value_domain(numeric)` with `numeric_contraction_rate(none)`), all `fixed_point(_)` strategies are removed from admissibility regardless of what KernelKind would permit.

   For `per_query(...)` strategies, no convergence guarantee is required from the recurrence — per-query traversal terminates by combinatorial visited-set tracking (`has_combinatorial_loop_break(true)` is the relevant safety property, separate from fixed_point admissibility).

If the resulting list is empty, `select_evaluation_strategy/3` throws `error(no_admissible_strategy(Recurrence), _)`.

**Note: `monotone(true)` is NOT in the admissibility test.** Admissibility is about termination; the iteration of a combinatorial recurrence over a finite state space terminates whether or not the recurrence is monotone (non-monotone iterations may oscillate, but they oscillate *in a detectable cycle* on a finite state space, so the evaluator can stop them). Monotonicity is what guarantees the iteration converges to the *least fixed point* — a semantic question about whether the answer is the one the user wants, not a termination question. Implementations of fixed_point evaluation for combinatorial recurrences are expected to detect cycles and either return a meaningful intermediate result or surface an error; UnifyWeaver's intended kernels are all monotone in practice, so the cycle-detection path is theoretical for now.

**Why `monotone(false)` cross-class restriction is NOT applied here.** A separate concern: when `monotone(false)`, the selector should not auto-select *across* strategy classes — picking `per_query` when `fixed_point` was the cost-model default could change observable behaviour because non-monotone recurrences can be evaluation-order-sensitive. But this restriction depends on the user's intent (which class they're asking for), and `admissible_strategies/2` sees only the recurrence — no intent context. The cross-class restriction is therefore applied in Phase C (conflict resolution), where intent is in scope. Admissibility returns the full kernel+termination admissible set; Phase C narrows it under `monotone(false)` + cross-class intent.

This split (kernel+termination in admissibility; monotonicity-cross-class in resolution) keeps `admissible_strategies/2` a pure function of the recurrence — testable in isolation, no intent dependency. The cost the split pays: there is a brief window inside `step_satisfiability` where a strategy is admissible-by-recurrence but inadmissible-by-monotonicity-cross-class-restriction. The step's satisfiability-adjuster respects the cross-class restriction explicitly to avoid this phantom window.

### Strategy

The output of strategy selection:

```prolog
strategy(Mode)

where Mode is one of:
    per_query(SearchAlgo)
    fixed_point(IterAlgo)        % stubbed; not auto-selectable in this iteration
    cached                       % stubbed
    hybrid(Components)           % stubbed

SearchAlgo: unidirectional | bidirectional | astar | dijkstra
IterAlgo:   semi_naive | naive | top_down | bottom_up
```

The `Strategy` term represents only the *destination* — what the target will emit code for. Any pre-strategy adjustments (e.g. building CSR at compile time, building a calibration cache) are tracked in the `Trace` as `adjustment(...)` **detail fields** of whichever step caused them (cost-model rule or satisfiability resolution), not as separate step entries and not folded into the strategy term.

This split is deliberate: it keeps the `Strategy` type clean and machine-checkable, and it lets the trace fully describe the side-effects the target must perform before the strategy code runs — without breaking the per-step single-entry fold contract that the trace renderers rely on.

### Decision trace

```prolog
trace(Steps)
```

`Steps` is a list of `step/3` terms. Each step has `step(Name, Outcome, Details)` where `Name` is one of the semantic step names, `Outcome` records what happened, and `Details` is a list of supporting facts:

```prolog
step(classify_signals, classified(Intent, Declared, Inferred), [resolved_contradiction(...)?])
step(cost_model_choice, chosen(Strategy, Score, DecidingRule), [signal(...), reason(...), adjustment(...)?])
step(no_intent, applied | passed, [no_intent_signals?])           % applied if Intent = []; passed otherwise
step(intent_matches, applied | passed, [matched_class(...)?])     % applied if all intents match CM
step(third_option, found(Alt) | not_found | passed, [signals_considered(...)?])
step(scope_disambiguation, resolved(Strategy, by(...)) | no_scope_overlap | passed, [intent_a(...)?, intent_b(...)?])
step(satisfiability, satisfiable | adjusted | degraded(Action) | passed, [unmet_intent(...)?, adjustment(...)?])
step(caller_wins, applied | passed, [caller_intent(...)?, overridden_manifest(...)?])
step(final_decision, Strategy, [decided_by(Name), overridden(_), trace_summary(...)])
```

The `passed` outcome is added to every resolution step (no_intent, intent_matches, third_option, scope_disambiguation, satisfiability, caller_wins). It records that the step was reached during the hierarchy walk but did not fire — typically because its precondition didn't hold. Including `passed` steps in the trace preserves the complete walk history (which steps were considered before the resolving step fired). A trace renderer can elide `passed` steps for brevity if desired; the structured trace itself records them all.

**Relationship between `passed` and the step-specific non-firing atoms (`no_scope_overlap`, `not_found`).** The schema includes both generic `passed` and step-specific named non-firing outcomes (`no_scope_overlap` for `step_scope_disambiguation`, `not_found` for `step_third_option`). These are **named-reason variants of `passed` for specific steps**, not a separate resolution category:

- `step_scope_disambiguation` returns `no_scope_overlap` when it ran but neither intent refines the other — semantically equivalent to "passed with reason = no refinement relation"
- `step_third_option` returns `not_found` when it ran a search but found no compatible third option — semantically equivalent to "passed with reason = search exhausted"
- `passed` (without a step-specific reason) is for steps whose precondition wasn't met, so they didn't run at all

In all three cases, the hierarchy walks to the next step. The distinction is just *whether the step ran and exhausted its search* (`no_scope_overlap`, `not_found`) versus *whether the step was skipped because its trigger wasn't present* (`passed`). Trace renderers may treat all three equivalently for brevity; the structured trace distinguishes them for debuggability.

**Important note on adjustments.** There is no standalone `step(adjustment, ...)` entry type. Adjustments (`build_csr_at_compile_time`, `degrade_to_compatible`, etc.) are recorded as `adjustment(...)` **detail fields** of whichever step caused them — typically `step_cost_model_choice` for rule-level adjustments (a cost-model rule whose preferred strategy includes an adjustment), or `step_satisfiability` for resolution-driven adjustments (an unmet intent that the satisfiability adjuster resolves by building CSR at compile time). The `adjustment(...)?` annotation on those step lines marks the field as optional. This preserves the per-step single-entry fold contract: each resolution step contributes exactly one trace entry.

Target adapters that need to execute adjustments walk the trace looking for `adjustment(...)` detail fields on *any* step, not for a separate step type.

Each step's `Outcome` is one of a small enumeration; the renderers (below) know how to display each.

**On `conflict_detection`**: earlier drafts had a generic `conflict_detection: none | conflict(...)` step that overloaded two distinct situations (no intent at all vs intent matches cost model). The new design splits these into `step(no_intent, applied, ...)` and `step(intent_matches, applied, ...)` — distinct steps that record distinct reasons. The reader of a trace can now tell *which* of the two no-conflict situations occurred.

The trace is consumed by renderer helpers (not by the selector itself):

1. **`render_trace_for_stderr(+Trace, -Lines)`** produces a list of strings, one per step. The target adapter (or a wrapper) writes these to stderr.
2. **`format_trace_for_comment(+Trace, +CommentPrefix, -CommentString)`** produces a multi-line comment string the target inserts at the kernel call site. `CommentPrefix` is the target's per-line comment syntax (`//` for F#/C, `--` for Haskell, `%` for Prolog).

Both renderings are derived from the same structured trace, so they are guaranteed consistent.

## Decision logic

### Phase A — classify signals

```
classify_signals(Workload, IntentSignals, DeclaredDataSignals, InferredDataSignals)
```

Each workload term is matched against the intent / declared-data / inferred-data dispatch table. The result is three lists. Anything unrecognised goes into `InferredDataSignals` with a warning logged via the trace (forward-compat: new signal types should be acknowledged, not silently dropped). The selector itself does not write the warning to stderr; the trace renderer does.

**Deduplication of contradictory same-key signals.** When the workload contains two signals with the same key but contradictory values from different tiers — e.g. caller passes `csr_available(true)` (declared) AND static analysis infers `csr_available(false)` (inferred) — `classify_signals/4` resolves the conflict in favour of the declared signal and drops the inferred one. The trace records this with a `step(classify_signals, ..., [resolved_contradiction(csr_available, declared(true), inferred(false), dropped(inferred))])` detail entry so the user can see what happened.

Rationale: declared signals are user-stated facts; inferred signals are compiler estimates. When the user explicitly disagrees with the compiler's inference, the user wins by definition. Without this resolution, downstream rule-firing would see both signals and produce ambiguous behaviour (depending on which the rule matches first). Same-key contradictions *between two declared signals*, or *between two inferred signals*, are not currently expected (the producers — `relation_policy` declarations, static analysis passes — should be internally consistent within each tier); if observed, the trace records a `step(classify_signals, ..., [ambiguous_signal(...)])` warning and the first occurrence wins as a last-resort tiebreak.

### Phase B — cost model evaluation

```
apply_cost_model(Recurrence, DataSignals, CostModelChoice)
```

`DataSignals` is the concatenation of declared-data and inferred-data signals; rules can inspect each tier separately if they want to.

The cost model is a set of rules that, given recurrence properties and data signals, produce partial preferences (strategy + score). Rules compose by summing scores per candidate strategy; the highest-scoring strategy wins.

#### Rule registry

Rules are declared via a non-multifile predicate `cost_model_rule/6`:

```prolog
cost_model_rule(+RuleName, +Priority, +Recurrence, +DataSignals, -Score, -ChosenStrategy)
```

`Priority` is an integer used for tiebreaking (higher = preferred). Rules live in `recurrence_evaluation_strategy.pl` itself — they are not multifile. (An earlier draft considered making them multifile to allow target adapters to register rules; this was rejected because clause-order-dependent tiebreaking is fragile across load orders. If extensibility is later needed, the `Priority` argument is the explicit knob.)

#### Initial rule set (Phase 1 of the implementation plan)

| Rule | Priority | When (signal-value preconditions; tier is irrelevant for firing) | Output | Score |
|------|----------|------|--------|-------|
| `prefer_bidirectional_csr_present` | 100 | `csr_available(true)`, `query_pattern(single_pair)`, `cardinality(large)` | `per_query(bidirectional)` | +3 |
| `prefer_bidirectional_csr_buildable` | 90 | `csr_buildable(true)`, `cardinality(large)`, `query_frequency(high)` | `per_query(bidirectional)` + `adjustment(build_csr_at_compile_time)` | +2 |
| `prefer_unidirectional_no_csr` | 80 | `csr_available(false)` AND `csr_buildable(false)` | `per_query(unidirectional)` | +2 |
| `prefer_unidirectional_small` | 70 | `cardinality(small)`, no contradicting signal | `per_query(unidirectional)` | +1 |
| `prefer_astar_heuristic_available` | 60 | recurrence admits `per_query(astar)`, `heuristic_predicate_available(true)` | `per_query(astar)` | +1 |
| `default_fallback` | 1 | always | `per_query(unidirectional)` | +0 |

**Note on rule preconditions and signal tiers.** A rule's *preconditions* are predicates over signal *values*; they fire on the value regardless of whether the signal arrived as declared-data or inferred-data. The *tier* of each matching signal is consulted only by the confidence-weighting layer (next subsection) — declared signals contribute weight 1.0, inferred signals contribute weight 0.8. Earlier drafts of this table annotated each precondition with its expected typical tier (e.g. "declared: csr_available(false)"); those annotations were misleading because they suggested rules wouldn't fire on the other tier. Removed.

#### Confidence weighting for inferred signals

The confidence-weighting formula is a single expression that subsumes both the all-inferred and mixed cases:

```
weighted_score = raw_score × (sum of per-signal weights) / (number of matching signals)

where each signal contributes:
  - 1.0 if declared
  - 0.8 if inferred
```

Equivalently: the multiplier is the *mean* per-signal weight across the rule's matching signals. The all-inferred case reduces to `raw × (0.8 + ... + 0.8) / N = raw × 0.8`; the mixed case averages weights as shown. The single-formula presentation prevents an implementer from applying the averaged form to all-inferred (numerically equivalent here, but fragile if the constants change). The constants (`1.0`, `0.8`) are soft and future tuning may adjust them.

#### Scoring example

A worked example to anchor future rule additions. Suppose the workload is:

- Declared: `cardinality(large)`, `query_pattern(single_pair)`, **`query_frequency(high)`**
- Inferred: `csr_buildable(true)`, `csr_available(false)`
- (No intent signals)

Rules that fire (signal-value preconditions evaluated regardless of tier; tier consulted only by confidence weighting):

- `prefer_bidirectional_csr_buildable` — all preconditions match: `csr_buildable(true)` ✓, `cardinality(large)` ✓, `query_frequency(high)` ✓. Raw score +2. Confidence per signal: `csr_buildable(true)` is inferred (0.8), `cardinality(large)` is declared (1.0), `query_frequency(high)` is declared (1.0). Weighted score: 2 × (0.8 + 1.0 + 1.0) / 3 ≈ 1.87.
- `prefer_unidirectional_no_csr` — `csr_available(false)` precondition matches (the signal is in the workload, tier-irrelevant for firing). But `csr_buildable(false)` does *not* match — we have `csr_buildable(true)` in the workload. Rule does not fire on value mismatch.

  *Reader note:* this rule's non-firing in the worked example is the common case **in graph configurations where the edge predicate has a buildable inverse** (the static analyser typically infers `csr_buildable(true)` whenever this holds, which is most of the time on Wikipedia-style categorisation graphs). So `prefer_unidirectional_no_csr` typically fires when the system is told "no CSR, and don't try to build one either" — a relatively rare combination in those configurations. The `default_fallback` rule plus `prefer_bidirectional_csr_buildable` cover the more typical scenarios there.

  For graph configurations where the edge predicate has *no* buildable inverse — e.g. embedded environments with restricted schemas, streaming sources where CSR construction is infeasible, or graphs where the inverse would be too large to materialise — `prefer_unidirectional_no_csr` becomes the typical firing rule, not the rare one. The "common case" characterisation depends on the workload's CSR-buildability profile.
- `default_fallback` — fires. Raw score +0.

Winner: `per_query(bidirectional)`, weighted score ≈ 1.87, deciding rule `prefer_bidirectional_csr_buildable`. The build-CSR adjustment is recorded as a detail field of the cost-model-choice trace entry (rule-level adjustments live where the rule fired, not as a separate step); the trace looks like `step(cost_model_choice, chosen(per_query(bidirectional), 1.87, prefer_bidirectional_csr_buildable), [adjustment(build_csr_at_compile_time), ...])`. Likewise, if a `step_satisfiability` resolution had triggered the adjustment (different scenario), the adjustment would live as a detail field of the satisfiability step. **Adjustments are detail fields of whatever step caused them, not separate trace steps**, so the per-step single-entry fold contract is preserved.

Either way, the `Strategy` term is `per_query(bidirectional)` — the adjustment is trace metadata, not part of the strategy term.

(*Earlier draft note:* a previous version of this example omitted `query_frequency(high)` from the workload, which would have prevented `prefer_bidirectional_csr_buildable` from firing because its precondition list requires it. Fixed in this revision; the example is now self-consistent.)

If a contradiction had appeared — e.g. `prefer_unidirectional_small` also firing with declared `cardinality(small)` (which contradicts `cardinality(large)`) — the cost model would treat this as malformed input and let the conflict-resolution phase deal with it. (In practice the declared-data tier should be internally consistent because relation-policy declarations are deduplicated; if the inferred-data tier produces a contradiction with declared-data, the inferred value loses.)

#### Composition with other cost models

This cost model is deliberately coarser than the per-operation cost models in [`COST_FUNCTION_PHILOSOPHY.md`](COST_FUNCTION_PHILOSOPHY.md) and [`CACHE_COST_MODEL_PHILOSOPHY.md`](CACHE_COST_MODEL_PHILOSOPHY.md). It reasons about strategy-class fit, not about per-operation cost. The two compose: a rule in `cost_model_rule/6` can *call* the cache-cost or scan-strategy model as a helper when their inputs are available. The composition is one-directional: cache-cost and scan-strategy do not call back.

### Phase C — conflict resolution

```
resolve_against_intent(IntentSignals, CostModelChoice, Recurrence, Strategy, Trace)
```

The hierarchy from the philosophy doc, with semantic step names. Each step is implemented as a helper predicate `step_<name>/4` that either resolves (returning `resolved(Strategy, TraceEntry)`) or passes (`next_step`). `resolve_against_intent/5` walks the steps in order and takes the first `resolved`.

#### `step_no_intent` — no intent signals at all

If `IntentSignals = []`, return `CostModelChoice` immediately. Trace step: `step(no_intent, applied, [no_intent_signals])`.

#### `step_intent_matches` — intent matches cost-model choice

If every intent signal is satisfied by `CostModelChoice` (using the [intent-compatibility matrix](#intent-compatibility-matrix)), return that strategy. Trace step: `step(intent_matches, applied, [matched_signals(...)])`.

#### `step_third_option` — search for a strategy that satisfies all intents

Iterate over `admissible_strategies(Recurrence, Admissible)` looking for any strategy that satisfies *all* intent signals (using the intent-compatibility matrix). If found, return it. Trace step: `step(third_option, found(Strategy) | not_found, [candidates_considered(...)])`.

**`monotone(false)` cross-class restriction is applied here too.** When `monotone(Recurrence, false)`, `step_third_option` narrows its candidate set to strategies in the same class as `CostModelChoice` before searching, mirroring the cross-class restriction applied in `step_satisfiability`. Without this, the third-option search could find a cross-class strategy and return it before `step_satisfiability` ever ran — bypassing the cross-class refusal. Applying the restriction here (and at any other resolution step that could pick from `admissible_strategies` directly) ensures the restriction is enforced uniformly across the hierarchy. The narrowed candidate list is recorded in the trace: `step(third_option, ..., [monotone_false_narrowed_candidates(...)])`.

#### `step_scope_disambiguation` — refinement (narrower strategy-set wins)

If two intent signals come from different scopes (manifest vs caller) and one *refines* the other in strategy-space, the refined (narrower) intent wins without it being treated as conflict. This is normal scoping rules, not an override.

**Formal definition.** Intent A **refines** intent B iff A's strategy-set is a (non-empty) **proper** subset of B's strategy-set — i.e. A ⊊ B (A is contained in B and A ≠ B). A is the more specific intent; A wins. The relation is asymmetric — if neither A refines B nor B refines A (e.g. their strategy-sets are *disjoint*, *partially overlapping*, or *equal*), this step does *not* fire, and resolution falls through to the next steps.

**Why proper-subset and not loose subset.** The equal-set case (e.g. manifest says `strategy(per_query(unidirectional))` and caller says `kernel_mode(unidirectional)` — both resolve to `{per_query(unidirectional)}`) should *not* fire `step_scope_disambiguation` because there's nothing to disambiguate; the two intents agree exactly and the resolution belongs at `step_intent_matches` (which fires earlier in the hierarchy). The proper-subset definition makes equal-set a no-op for this step explicitly.

**Open-ended strategy-sets from unbound terms.** Intents with unbound variables produce conceptually-infinite strategy-sets — e.g. `strategy(per_query(_))` denotes the set of all valid concrete groundings of `per_query(X)`. The `refines` check on such intents uses *symbolic* subset testing rather than enumeration: A's set is a proper subset of B's iff every ground term matching A also matches B AND there exists some ground term matching B that does not match A. For the common cases — `strategy(per_query(X))` (unbound) refined by `kernel_mode(bidirectional)` (concrete `{per_query(bidirectional), per_query(astar)}`) — the symbolic check resolves cleanly. For pathological cases (intents referencing arbitrary predicates over strategies), the implementation should fail-safe by treating the relation as `unknown` and falling through to the next step rather than committing to a refinement it can't verify.

(Terminology note: earlier drafts used "subsumes" — switched to "refines" because the natural reading of "A subsumes B" varies between type-theory convention and natural-language convention. "A refines B" is unambiguous: A is more specific.)

(Terminology note: earlier drafts used "subsumes" — switched to "refines" because the natural reading of "A subsumes B" varies between type-theory convention and natural-language convention. "A refines B" is unambiguous: A is more specific.)

**Correct worked example.** Suppose:
- Manifest intent: `strategy(per_query(_))` — broad, matches every per_query strategy (set: `{per_query(unidirectional), per_query(bidirectional), per_query(astar), per_query(dijkstra)}`)
- Caller intent: `kernel_mode(bidirectional)` — narrow, matches `{per_query(bidirectional), per_query(astar)}` per the [intent-compatibility matrix](#intent-compatibility-matrix)

Caller's strategy-set is a subset of manifest's. Caller **refines** manifest. Caller wins by specificity. The cost model picks within caller's set (probably `per_query(bidirectional)` if CSR is available, else `per_query(astar)` if a heuristic is available, etc.).

Trace step: `step(scope_disambiguation, resolved(per_query(bidirectional), by(caller_refines_manifest)), [refined_intent(kernel_mode(bidirectional)), broader_intent(strategy(per_query(_)))])`.

**Disjoint-intents counter-example (this step does NOT apply).** If instead:
- Manifest intent: `strategy(per_query(unidirectional))` (set: `{per_query(unidirectional)}`)
- Caller intent: `kernel_mode(bidirectional)` (set: `{per_query(bidirectional), per_query(astar)}`)

The two sets are disjoint — neither refines the other. `step_scope_disambiguation` does not fire. Resolution falls through to `step_satisfiability` (if adjustments could help) and then to `step_caller_wins` (caller's intent wins by fallback, with a loud warning recorded in the trace).

(*Earlier draft note:* a previous version of this section presented the disjoint-intents case as if it resolved via scope_disambiguation, which contradicted the step's own rule. The fix is to use the subset example above for the rule's worked illustration, and to name the disjoint case as a fall-through.)

#### `step_satisfiability` — adjust the unsatisfiable

If the intent is structurally unmet (e.g. caller wants `bidirectional` but declared `csr_available(false)` AND inferred `csr_buildable(false)`):

- Determine if an adjustment makes it satisfiable. Concrete adjustments in priority order, each with an explicit precondition that distinguishes when it applies:
  - `build_csr_at_compile_time` — applicable when `csr_buildable(true)` (inferred or declared) **AND** the workload signals satisfy the same preconditions as `prefer_bidirectional_csr_buildable` (specifically `cardinality(large)` and `query_frequency(high)`). If those preconditions are unmet, building CSR isn't cost-justified — fall through to the next adjustment.
  - `degrade_to_compatible` — applicable when there exists an admissible strategy in the *same strategy class* as the caller's intent that differs only in algorithm (e.g. caller asked for `per_query(bidirectional)` and `per_query(astar)` is admissible — both are `per_query`-class). The adjuster picks the closest same-class alternative and emits a warning. If no same-class compatible strategy exists, fall through.
  - `degrade_with_warning` — applicable as a last-resort fallback when neither of the above applies. Falls back to a strategy that contradicts the intent's class outright (e.g. `fixed_point(semi_naive)` when caller asked for `per_query(bidirectional)`) and warns loud in the trace.

- Selector picks the first adjustment whose precondition holds.

**`monotone(false)` cross-class restriction in the adjuster.** Under `monotone(false)`, the adjuster explicitly *refuses* any adjustment that would cross strategy classes. Concrete mechanism: before committing an adjustment, the adjuster checks whether the candidate post-adjustment strategy and the caller's stated intent are in the same strategy class (both `per_query(...)`, or both `fixed_point(...)`, etc.). If they're not, the adjustment is rejected and the adjuster falls through to the next adjustment (or to `step_caller_wins` if none remain). This is mechanism (b) — *refuse cross-class adjustment* — rather than (a) (post-hoc re-check). Refusal is preferred because it short-circuits earlier: the user gets a clearer trace ("adjustment rejected: would cross strategy class under monotone(false)") and the resolution moves on without the wasted-work appearance of an adjustment that was computed and then discarded.

Trace step (single entry per resolution, per the fold contract): `step(satisfiability, adjusted, [adjustment(build_csr_at_compile_time), unmet_intent(kernel_mode(bidirectional)), reason(no_csr_present_but_buildable)])`. The adjustment is recorded as a *detail field* of the satisfiability step, not as a separate step — this preserves the invariant that each resolution step contributes exactly one trace entry. The target adapter reads the adjustment detail and executes the side-effect (e.g. emitting a build-CSR step into the generated code) before the kernel call.

The chosen strategy is the post-adjustment one (e.g. `per_query(bidirectional)`); the adjustment lives in the trace as detail metadata, *not* folded into the strategy term.

#### `step_caller_wins` — fallback

Genuine conflict that none of the previous steps resolved. Caller's intent wins; manifest's intent is overridden. Trace step: `step(caller_wins, applied, [caller_intent(kernel_mode(bidirectional)), overridden_manifest(strategy(unidirectional)), reason(unknown_consider_reconciling)])`. The warning is not emitted directly by the selector; it appears in the trace and the renderer surfaces it.

### Intent-compatibility matrix

The matrix used by `step_intent_matches` and `step_third_option` to decide whether a given intent signal is satisfied by a candidate strategy. Indexed by intent type:

| Intent signal | Satisfied by strategy |
|---------------|----------------------|
| `kernel_mode(bidirectional)` | `per_query(bidirectional)`, `per_query(astar)` (astar is bidirectional-flavour: see asymmetry note below) |
| `kernel_mode(unidirectional)` | `per_query(unidirectional)`, `per_query(bfs)`, `per_query(dfs)` |
| `kernel_mode(astar)` | `per_query(astar)` **only**. **Does NOT match `per_query(bidirectional)`**. If a user passes `kernel_mode(astar)` and the cost model prefers `per_query(bidirectional)`, `step_intent_matches` does *not* fire (the intent isn't satisfied) and resolution falls through to `step_third_option` (looking for any admissible `per_query(astar)`) or `step_caller_wins` (if astar is not admissible for the kernel). This asymmetry is the most common confusion when extending the matrix; check the negative case carefully when adding a new intent type |
| `kernel_mode(dijkstra)` | `per_query(dijkstra)` |
| `strategy(per_query(X))` | `per_query(X)` if X is ground; any `per_query(_)` if X is unbound |
| `strategy(fixed_point(X))` | `fixed_point(X)` if X is ground; any `fixed_point(_)` if X is unbound |
| `strategy(cached)` | `cached` |
| `strategy(hybrid(_))` | any `hybrid(_)` |
| `force_search_algorithm(A)` | any strategy whose inner algorithm matches A |

**Asymmetry note on bidirectional vs astar.** The matrix is deliberately asymmetric: `kernel_mode(bidirectional)` is *broader* and matches both plain bidirectional A\* search and `per_query(astar)` (since A\* with admissible heuristic is a bidirectional-flavour algorithm); `kernel_mode(astar)` is *narrower* and matches only `per_query(astar)`. The asymmetry reflects user intent: someone asking for `kernel_mode(bidirectional)` is expressing a class preference (bidirectional-flavour search), so A\* counts; someone asking for `kernel_mode(astar)` is specifically requesting A\* (with its heuristic-driven behaviour), so plain bidirectional doesn't count. If a user wants either bidirectional-or-astar, they should use `kernel_mode(bidirectional)`.

**Trace snippet illustrating the asymmetric match.** A user passing `kernel_mode(bidirectional)` who ends up with `per_query(astar)` (e.g. when no CSR is present but a heuristic predicate is available) might be surprised. The reasoning trace makes the asymmetric match explicit:

```
[evaluation-strategy] selecting for category_ancestor/4
[evaluation-strategy]   classify_signals: intent=[kernel_mode(bidirectional)], declared=[cardinality(large), heuristic_predicate_available(true), heuristic_predicate(my_heuristic/2)], inferred=[csr_available(false), csr_buildable(false)]
[evaluation-strategy]   cost_model_choice: per_query(astar), score=1.0, deciding-rule=prefer_astar_heuristic_available
[evaluation-strategy]   intent_matches: applied (kernel_mode(bidirectional) is satisfied by per_query(astar) per intent-compatibility matrix: A* counts as bidirectional-flavour)
[evaluation-strategy]   final_decision: per_query(astar), decided_by=intent_matches
```

The generated-code comment will include the same `intent_matches` line, so a reader of the generated F# source can see that `kernel_mode(bidirectional)` was satisfied by `per_query(astar)` and why.

This matrix is defined as a predicate `intent_compatible_with_strategy/2` in the module. Adding a new intent signal type means adding entries to the matrix.

## Trace renderers

### `render_trace_for_stderr(+Trace, -Lines)`

Produces a list of strings, one per step in the trace. The target adapter writes these to stderr.

Example output (as the list of strings the renderer produces):

```
[evaluation-strategy] selecting for category_ancestor/4
[evaluation-strategy]   classify_signals: intent=[kernel_mode(bidirectional)], declared=[csr_available(true), cardinality(large), query_pattern(single_pair)], inferred=[]
[evaluation-strategy]   cost_model_choice: per_query(bidirectional), score=3.0, deciding-rule=prefer_bidirectional_csr_present
[evaluation-strategy]   intent_matches: applied (intent satisfied by cost-model choice)
[evaluation-strategy]   final_decision: per_query(bidirectional), decided_by=intent_matches
```

Conflict example — refinement-based scope disambiguation (caller refines manifest; subset relationship; caller wins by specificity):

```
[evaluation-strategy] selecting for category_ancestor/4
[evaluation-strategy]   classify_signals: intent=[kernel_mode(bidirectional), strategy(per_query(_))], declared=[csr_available(true)], inferred=[]
[evaluation-strategy]   cost_model_choice: per_query(bidirectional), score=3.0, deciding-rule=prefer_bidirectional_csr_present
[evaluation-strategy]   scope_disambiguation: resolved(per_query(bidirectional), by(caller_refines_manifest))
[evaluation-strategy]   final_decision: per_query(bidirectional), decided_by=scope_disambiguation, overridden=[]
```

Note: this scenario has caller's `kernel_mode(bidirectional)` (strategy-set `{per_query(bidirectional), per_query(astar)}`) being a subset of manifest's `strategy(per_query(_))` (strategy-set `{per_query(unidirectional), per_query(bidirectional), per_query(astar), per_query(dijkstra)}`). Caller refines manifest; `step_scope_disambiguation` fires.

Conflict example — disjoint intents, fall through to caller-wins:

```
[evaluation-strategy] selecting for category_ancestor/4
[evaluation-strategy]   classify_signals: intent=[kernel_mode(bidirectional), strategy(per_query(unidirectional))], declared=[csr_available(true)], inferred=[]
[evaluation-strategy]   cost_model_choice: per_query(bidirectional), score=3.0, deciding-rule=prefer_bidirectional_csr_present
[evaluation-strategy]   no_intent: passed (intent signals present)
[evaluation-strategy]   intent_matches: passed (cost-model choice satisfies kernel_mode(bidirectional) but not strategy(per_query(unidirectional)))
[evaluation-strategy]   third_option: not_found (no admissible strategy satisfies both intent signals — intents are mutually exclusive)
[evaluation-strategy]   scope_disambiguation: no_scope_overlap (intents have disjoint strategy-sets; neither refines the other)
[evaluation-strategy]   satisfiability: passed (no structurally-unmet intent — both are achievable individually, just not jointly)
[evaluation-strategy] WARNING: caller_wins applied — caller's kernel_mode(bidirectional) overrode manifest's strategy(per_query(unidirectional)); reason for override unknown; consider reconciling
[evaluation-strategy]   final_decision: per_query(bidirectional), decided_by=caller_wins, overridden=[manifest:strategy(per_query(unidirectional))]
```

Note: this scenario has the two intents pointing at disjoint strategy-sets (`{per_query(bidirectional), per_query(astar)}` vs `{per_query(unidirectional)}`). Neither refines the other, so `step_scope_disambiguation` does NOT resolve. Resolution falls through to `step_caller_wins` with a loud warning.

(*Earlier-draft note:* a previous version of this conflict example used `by(narrower_subsumes_broader)` (stale post-rename to `refines`) AND showed the disjoint-intents scenario resolving via `step_scope_disambiguation`, which contradicted its own rule. Fixed in this revision: the first example now uses a true refinement scenario where the step actually fires; the second example shows the disjoint case correctly falling through to caller-wins.)

### `format_trace_for_comment(+Trace, +CommentPrefix, -CommentString)`

Produces a multi-line comment string the target inserts at the kernel call site. The `CommentPrefix` argument is the per-line comment syntax for the target language:

- F# / C / C++ / Rust / Java: `"// "`
- Haskell / SQL: `"-- "`
- Prolog: `"% "`
- Python / Ruby / Shell: `"# "`

Critical: the renderer emits the prefix on *every* line of the comment. Prolog `%` and Python `#` are line-level comment syntaxes; a renderer that only prefixes the first line would produce invalid code. The implementation plan calls for a parse-round-trip test rather than visual inspection.

Example output (F# prefix):

```fsharp
// =========================================================================
// Evaluation strategy: per_query(bidirectional)
// Decided by: intent_matches (cost-model preferred + caller agreed)
// Cost model: rule=prefer_bidirectional_csr_present, score=3.0
// Deciding signals: csr_available(true) [declared] + cardinality(large) [declared] + query_pattern(single_pair) [declared]
// Alternatives considered: per_query(astar) [score 1.0]
// Adjustments: none
// =========================================================================
```

## Integration with existing modules

### Upstream: `recursive_kernel_detection.pl`

The detector produces `recursive_kernel(KernelKind, Pred/Arity, ConfigOps)`. The strategy selector reads:

- `KernelKind` → maps to admissible strategies (via the internal kernel-kind-to-strategies table — see [Admissibility](#admissibility))
- `Pred/Arity` → passes through to the Recurrence term
- `ConfigOps` → fed into `Recurrence` properties (selectively — see below)

The selector does not call the detector; the caller (target adapter) is responsible for invoking detection first and passing the result.

**Note on ConfigOps → Recurrence properties mapping.** The caller (typically `build_recurrence_term/3` in `recurrence_inputs.pl`) maps detector-emitted `ConfigOps` keys into the `Recurrence` properties list. **Callers must NOT populate `directions_admissible` in the Recurrence properties list** — it was removed from the user-facing properties in an earlier revision because it could disagree silently with the kernel-kind-to-strategies table maintained inside the selector. If a `ConfigOps` entry uses the key `directions_admissible`, `build_recurrence_term/3` drops it (and emits a `step(classify_signals, ..., [dropped_property(directions_admissible, reason(removed_from_spec))])` trace entry — not a stderr write, since the selector module is pure-functional) rather than passing it through. The selector derives admissible directions from KernelKind alone — single source of truth.

The full signature of `build_recurrence_term/3` and the wider `recurrence_inputs.pl` module is specified in the [implementation plan §Phase 5](RECURRENCE_EVALUATION_STRATEGY_IMPLEMENTATION_PLAN.md#phase-5--f-wam-target-integration--shared-input-construction-helpers); the SPEC mentions it here as the canonical site where the `directions_admissible` drop happens.

**Maintenance note**: when a new kernel detector is added to `recursive_kernel_detection.pl`, the corresponding entry in the kernel-kind-to-strategies table inside `recurrence_evaluation_strategy.pl` must also be added. This is an undocumented maintenance surface in the current codebase; the implementation plan adds a `TODO` comment in both files referencing the other.

### Upstream: `algorithm_manifest.pl`

The selector reads `manifest_optimization_options(Name, Opts)` to populate the intent signals. The `decl_algorithm` declaration's `kernel(Pred/Arity)` field tells the caller which predicate to query for.

### Upstream: `relation_policy.pl`

The selector reads `get_relation_policy(EdgePred/2, cardinality, Card, unknown)` and similar accessors to populate declared-data signals. The relation-policy module is the canonical source for `cardinality`, `determinism`, `unique`, `on_duplicate`.

### Adjacent: `cost_model.pl` and `cost_function.pl`

The selector uses these as helpers when its rules need per-operation cost estimates. For Phase 1, the rules above are coarse enough that they don't need detailed cost estimation; later rules may. The dependency is one-directional: the selector calls into cost-model; cost-model never calls into the selector.

### Downstream: `wam_fsharp_target.pl`

The F# WAM target is the first consumer. Integration shape:

1. Target runs `recursive_kernel_detection:detect_recursive_kernel(...)` as before.
2. **NEW**: target builds the `Recurrence` term and `Workload` signal list (see [Helper module](#helper-module-for-input-construction) below).
3. **NEW**: target calls `select_evaluation_strategy(Recurrence, Workload, strategy_choice(Strategy, Trace))`.
4. **NEW**: target reads the `Strategy` and `Trace` terms, dispatches accordingly (existing `maybe_upgrade_bidirectional/2` becomes one of several upgrade paths driven by the strategy), and executes any `adjustment(...)` steps from the trace (e.g. emitting a build-CSR step).
5. **NEW**: target calls `format_trace_for_comment(Trace, "// ", Comment)` and inserts the comment header at the kernel call site in the generated F# file.
6. **NEW**: target optionally calls `render_trace_for_stderr(Trace, Lines)` and writes the lines to stderr at compile time. Quiet-mode tests skip this step.

The existing explicit-option-driven path (`kernel_mode(bidirectional)` from caller) continues to work — the explicit option becomes an intent signal that the selector honours.

### Helper module for input construction

Construction of the `Recurrence` term and the `Workload` signal list is target-agnostic. To avoid each WAM target duplicating the construction logic, the helpers `build_recurrence_term/3` and `build_workload_signals/2` live in a separate shared module `src/unifyweaver/core/recurrence_inputs.pl` (created as part of the implementation work).

**Constraint**: the helper module must remain target-agnostic — no F#-specific (or Haskell-specific, C-specific) logic in `recurrence_inputs.pl`. If a target needs target-specific input transformation, it should do that before calling `build_*` and pass already-transformed input. This keeps the helper reusable across WAM targets and the eventual C# query target.

### Downstream: future WAM targets

`wam_haskell_target.pl` and `wam_c_target.pl` consume the same `select_evaluation_strategy/3` predicate. Their integration follows the same six-step shape; comment rendering uses the appropriate `CommentPrefix` for each target.

### Downstream: planned future consumer — C# query runtime

`csharp_query_target.pl` is the existing realisation of bottom-up fixed-point compilation. When the level-1 decision tree's `fixed_point(...)` branch gets a real chooser, the C# query target becomes the natural second consumer of `select_evaluation_strategy/3`. The selector's API accommodates this; the integration is a future work item.

## Edge cases and error handling

### Recurrence with no admissible strategy

If `admissible_strategies(Recurrence, [])`, `select_evaluation_strategy/3` throws `error(no_admissible_strategy(Recurrence), context(select_evaluation_strategy/3, ...))`. The target adapter catches and emits a compile-time error pointing at the recurrence shape. This case arises when, for example, a `bidirectional_ancestor` kernel is paired with `numeric_contraction_rate(none)` *and* the caller insists on `strategy(fixed_point(_))` — neither per-query (which is what bidirectional_ancestor is for) nor fixed-point (which is disqualified by the missing contraction rate) is admissible.

### Intent signals reference an unknown strategy

If `kernel_mode(quantum)` is passed, the selector treats it as if no `kernel_mode` signal was present (i.e. skips it) and records the skip via *two* trace mechanisms:

1. A `step(classify_signals, ..., [unknown_intent(kernel_mode(quantum))])` entry in the trace, for the machine-readable record.
2. A *prominent* unknown-intent surfacing in the trace renderers: `render_trace_for_stderr/2` emits the unknown-intent warning as a top-level line prefixed `[evaluation-strategy] WARNING:`, *not* buried in a step's details. `format_trace_for_comment/3` includes an `Unknown intents skipped:` line in the comment header when any unknown intents were classified. A typo-caused skip should be impossible to miss when reading either rendering.

Forward-compat principle: unknown intent should not break compilation, only be ignored with prominent warnings so the user notices a typo before relying on the wrong strategy.

### Multiple cost-model rules tie at the same score

Resolved by the explicit `Priority` argument of `cost_model_rule/6` (higher wins). If priorities also tie, the rule with the lexicographically earliest `RuleName` wins. Trace records the tie and the tiebreak: `step(cost_model_choice, chosen(...), [tied_with([...]), tiebreak_by(priority)])`.

### Recurrence properties inconsistent

If `value_domain(numeric)` AND `numeric_contraction_rate(R)` with `R >= 1.0` (or `R = none`), the selector removes all `fixed_point(...)` strategies from admissibility (as specified above) and proceeds with the remaining admissible strategies. If no strategies remain admissible after this filtering, the previous case ("no admissible strategy") applies.

If `value_domain(combinatorial)` AND `monotone(true)`, `fixed_point` strategies are admissible regardless of contraction rate — combinatorial monotone recurrences over finite lattices terminate by Tarski's theorem.

If `monotone(false)` is combined with intent that requests cross-class auto-selection, the conflict-resolution phase narrows the candidate set to the kernel's primary strategy class (no cross-class auto-selection under non-monotonicity). The trace records the narrowing in the relevant step's details. If the user's intent forces a different class explicitly via `step_caller_wins`, the caller wins but the trace records a warning that auto-selection was disabled and the chosen strategy may not have its convergence properties verified for this recurrence.

## Versioning and compatibility

This module is new; there is no prior version to be compatible with. The API contract is:

- `select_evaluation_strategy/3` is the stable entry point.
- The `Strategy` term shape is stable; new branches (e.g. populating `fixed_point` for F# WAM) may be added; existing branches do not change semantically.
- The `Trace` term shape is stable in its top-level structure; new step types may be added; existing step types do not change semantics.
- Cost-model rules can be added freely; the scoring system + Priority argument is the abstraction.
- Workload signal types can be added; classification falls back to `inferred` with a trace warning for unknowns.
- The intent-compatibility matrix can be extended; existing entries do not change.

## What is *not* in the API

- Re-running kernel detection. The caller already detected; the selector consumes.
- Emitting target-language code. The selector picks a strategy; the target emits.
- Writing to stderr or any other stream. The selector returns the structured trace; renderers and target adapters decide what to emit.
- Decoding the trace structure into JSON or SARIF. The trace is Prolog terms; the stderr and comment renderers exist. Other renderings are future work.
- Persisting decisions across compilation runs (caching, learning). Each call is independent.

## See also

- [`RECURRENCE_EVALUATION_STRATEGY_PHILOSOPHY.md`](RECURRENCE_EVALUATION_STRATEGY_PHILOSOPHY.md) — the why.
- [`RECURRENCE_EVALUATION_STRATEGY_IMPLEMENTATION_PLAN.md`](RECURRENCE_EVALUATION_STRATEGY_IMPLEMENTATION_PLAN.md) — the work plan.
- [`KERNEL_SHAPE_RECOGNITION.md`](KERNEL_SHAPE_RECOGNITION.md) — the upstream detector layer.
- [`ALGORITHM_MANIFEST_SPECIFICATION.md`](ALGORITHM_MANIFEST_SPECIFICATION.md) — the manifest spec.
- [`RELATION_POLICY_DECLARATIONS.md`](RELATION_POLICY_DECLARATIONS.md) — relation-policy hints.
- [`SCAN_STRATEGY_SPECIFICATION.md`](SCAN_STRATEGY_SPECIFICATION.md) — closest-sibling spec in shape.
- [`COST_FUNCTION_PHILOSOPHY.md`](COST_FUNCTION_PHILOSOPHY.md) and [`CACHE_COST_MODEL_PHILOSOPHY.md`](CACHE_COST_MODEL_PHILOSOPHY.md) — adjacent cost-model layers.
