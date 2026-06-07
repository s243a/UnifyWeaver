# Recurrence Evaluation Strategy — Specification

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

### Determinism contract

`select_evaluation_strategy/3` is **deterministic (`det`)**:
- It always succeeds for a well-formed `Recurrence` and `Workload`.
- It never fails.
- It throws on structural error only — `error(no_admissible_strategy(_), _)` is the canonical case.
- It always binds `StrategyAndTrace` to a ground term.

This matters for callers using the selector inside `if-then-else`, `findall/3`, or other constructs sensitive to determinism. The helper predicates `classify_signals/4`, `apply_cost_model/3`, and `resolve_against_intent/5` are also `det` under the same conditions.

`admissible_strategies/2` is `det` and may return `[]` (which then causes `select_evaluation_strategy/3` to throw). `strategy_pretty/2` is `det`.

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
| `has_combinatorial_loop_break(B)` | boolean | `true` if the user's clauses include `\+ member` or equivalent visited-set tracking |
| `numeric_contraction_rate(R)` | maybe Float | An upper bound on the iteration contraction rate. `none` if unknown or non-numeric; a float `< 1` if known to converge. Values `>= 1.0` or `none` disqualify all `fixed_point` strategies (see [Admissibility](#admissibility)) |
| `monotone(B)` | boolean | `true` if the recurrence operator is monotone over its solution lattice (default `true` for Datalog-shaped recurrences). `false` disables cross-class strategy auto-selection |
| `result_layout(L)` | term | The output-tuple layout from `recursive_kernel_detection:kernel_result_layout/2` |
| `result_mode(M)` | atom | `stream` / `deterministic` / `deterministic_collection` — from `kernel_result_mode/2` |
| `expected_cardinality(C)` | atom | `small` / `medium` / `large` / `unknown` — from relation-policy if declared, else `unknown` |

The `Recurrence` term is constructed once per `select_evaluation_strategy/3` call. The caller's responsibility is to populate it from the detected kernel and clause analysis. The module does not re-detect; it consumes.

**Note on `directions_admissible`.** Earlier drafts of this spec exposed `directions_admissible(Dirs)` as a user-supplied recurrence property. It has been removed from the user-facing properties because it can disagree silently with `admissible_strategies/2` for the same `KernelKind`. The selector now *derives* admissible directions internally from the kernel-kind-to-strategies table — single source of truth, no inconsistency possible. If a future kernel-kind needs per-instance variation in admissible directions (e.g. a kernel where the direction depends on the underlying graph's properties), this property can be reintroduced with an assertion that it intersects with the kernel-kind table.

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
| `csr_path(_)` | caller option | CSR (child-direction index) is available at this path |
| `csr_available(true)` | caller option | Explicit assertion of CSR availability |
| `cardinality(C)` | `relation_policy` declaration | `small` / `medium` / `large` / `unknown` for the edge predicate |
| `determinism(D)` | `relation_policy` | `det` / `semidet` / `nondet` / `multi` |
| `unique(B)` | `relation_policy` | `true` / `false` for the relation |
| `query_pattern(P)` | manifest | `single_pair` / `all_pairs` / `all_from_source` / `sample` |
| `query_frequency(F)` | manifest | `low` / `high` / `sustained` |
| `graph_mutability(M)` | manifest | `static` / `append_only` / `mutable` |
| `b_eff(F)` | calibration declaration | Friendship-paradox-corrected effective branching factor |
| `branching_d(F)` | calibration declaration | Mean child branching factor `D` |
| `contraction_r(F)` | calibration declaration | Convergence ratio `r = b'/(b_eff·D)` |

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

`admissible_strategies(+Recurrence, -List)` returns the strategies the selector may pick from. A strategy is admissible iff *all* of the following hold:

1. **KernelKind permits it** — read from a kernel-kind-to-strategies static table. E.g. `bidirectional_ancestor` permits `per_query(bidirectional)` and `per_query(astar)`; `transitive_closure2` permits `per_query(unidirectional)`, `per_query(bidirectional)` (via upgrade), and `fixed_point(semi_naive)`.
2. **Recurrence-convergence permits it** — for any `fixed_point(...)` strategy, the recurrence must have `numeric_contraction_rate(R)` with `R < 1.0`. If `R` is `none` or `R >= 1.0`, all `fixed_point` strategies are *removed* from admissibility regardless of what KernelKind would permit.
3. **Monotonicity permits cross-class selection** — when `monotone(false)`, cross-class auto-selection is disabled. The selector restricts to whichever strategy class is explicitly requested by intent; if no class is requested, it defaults to the kernel's primary strategy.

If the resulting list is empty, `select_evaluation_strategy/3` throws `error(no_admissible_strategy(Recurrence), _)`.

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

The `Strategy` term represents only the *destination* — what the target will emit code for. Any pre-strategy adjustments (e.g. building CSR at compile time, building a calibration cache) are tracked separately in the `Trace` as `adjustment(...)` step entries, not folded into the strategy term.

This split is deliberate: it keeps the `Strategy` type clean and machine-checkable, and it lets the trace fully describe the side-effects the target must perform before the strategy code runs.

### Decision trace

```prolog
trace(Steps)
```

`Steps` is a list of `step/3` terms. Each step has `step(Name, Outcome, Details)` where `Name` is one of the semantic step names, `Outcome` records what happened, and `Details` is a list of supporting facts:

```prolog
step(classify_signals, classified(Intent, Declared, Inferred), [])
step(cost_model_choice, chosen(Strategy, Score, DecidingRule), [signal(...), reason(...)])
step(no_intent, applied, [no_intent_signals])                     % only fires if Intent = []
step(intent_matches, applied, [matched_class(...)])               % only fires if intent matches CM
step(third_option, found(Alt) | not_found, [signals_considered(...)])
step(scope_disambiguation, resolved(Strategy, By(Scope)) | no_scope_overlap, [intent_a(...), intent_b(...)])
step(satisfiability, satisfiable | adjusted(Action) | degraded(Action), [unmet_intent(...)])
step(caller_wins, applied, [caller_intent(...), overridden_manifest(...)])
step(adjustment, build_csr_at_compile_time | other_adjustment, [reason(...)])
step(final_decision, Strategy, [decided_by(Name), overridden(_), trace_summary(...)])
```

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

| Rule | Priority | When | Output | Score |
|------|----------|------|--------|-------|
| `prefer_bidirectional_csr_present` | 100 | declared: `csr_available(true)`, `query_pattern(single_pair)`, `cardinality(large)` | `per_query(bidirectional)` | +3 |
| `prefer_bidirectional_csr_buildable` | 90 | inferred: `csr_buildable(true)`, declared: `cardinality(large)`, `query_frequency(high)` | `per_query(bidirectional)` + `adjustment(build_csr_at_compile_time)` | +2 |
| `prefer_unidirectional_no_csr` | 80 | declared: `csr_available(false)` AND inferred: `csr_buildable(false)` | `per_query(unidirectional)` | +2 |
| `prefer_unidirectional_small` | 70 | declared: `cardinality(small)`, no contradicting signal | `per_query(unidirectional)` | +1 |
| `prefer_astar_heuristic_available` | 60 | recurrence admits `per_query(astar)`, heuristic predicate available | `per_query(astar)` | +1 |
| `default_fallback` | 1 | always | `per_query(unidirectional)` | +0 |

#### Confidence weighting for inferred signals

A rule that depends only on inferred data takes a confidence multiplier: its raw score is multiplied by 0.8 before being summed. A rule that depends on a mix of declared and inferred data gets a per-signal weighting averaged (declared = 1.0, inferred = 0.8). This is a soft mechanism; future tuning may adjust the constants.

#### Scoring example

A worked example to anchor future rule additions. Suppose the workload is:

- Declared: `cardinality(large)`, `query_pattern(single_pair)`
- Inferred: `csr_buildable(true)`, `csr_available(false)`
- (No intent signals)

Rules that fire:

- `prefer_bidirectional_csr_buildable` — all preconditions match. Raw score +2. Confidence: `csr_buildable` is inferred (0.8), the other two are declared (1.0). Weighted score: 2 × (0.8 + 1.0 + 1.0) / 3 ≈ 1.87.
- `prefer_unidirectional_no_csr` — declared `csr_available(false)` matches; inferred `csr_buildable(false)` does *not* match (we have `csr_buildable(true)`). Rule does not fire.
- `default_fallback` — fires. Raw score +0.

Winner: `per_query(bidirectional)` + `adjustment(build_csr_at_compile_time)`, weighted score ≈ 1.87, deciding rule `prefer_bidirectional_csr_buildable`. The adjustment becomes a `step(adjustment, build_csr_at_compile_time, ...)` entry in the trace; the strategy is `per_query(bidirectional)` (the adjustment is *not* part of the strategy term).

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

#### `step_scope_disambiguation` — narrower strategy term subsumes broader

If two intent signals come from different scopes (manifest vs caller) and the narrower-strategy intent is *subsumed by* the broader-strategy intent (i.e. the narrower picks a specific point within the strategy class the broader names), the narrower wins without it being treated as conflict. This is normal scoping rules, not an override.

Formal definition: intent A *subsumes* intent B if every strategy satisfying B also satisfies A (B's strategy-set is a subset of A's strategy-set). Example: manifest says `strategy(per_query(_))` (broad; matches any per_query algorithm), caller says `kernel_mode(bidirectional)` (narrow; matches only `per_query(bidirectional)`). Caller subsumes manifest in the strategy-space sense; caller wins by specificity.

Trace step: `step(scope_disambiguation, resolved(Strategy, by(narrower_subsumes_broader)), [narrower(...), broader(...)])`.

#### `step_satisfiability` — adjust the unsatisfiable

If the intent is structurally unmet (e.g. caller wants `bidirectional` but declared `csr_available(false)` AND inferred `csr_buildable(false)`):

- Determine if an adjustment makes it satisfiable. Concrete adjustments in priority order:
  - `build_csr_at_compile_time` — if `csr_buildable(true)` (inferred or declared)
  - `degrade_to_compatible` — fall back to a strategy that satisfies the intent's broader class (e.g. `bidirectional` → `astar` if astar admissible) with a warning recorded in the trace
  - `degrade_with_warning` — fall back to a strategy that contradicts the intent but warns loud

- Selector picks the first adjustment whose precondition holds.

Trace steps: `step(satisfiability, adjusted(BuildCsrAtCompileTime), ...)` for the adjustment proper, plus `step(adjustment, build_csr_at_compile_time, [reason(unmet_intent_kernel_mode_bidirectional)])` recording the side-effect the target must perform. The chosen strategy is the post-adjustment one (e.g. `per_query(bidirectional)`); the adjustment is *not* folded into the strategy term.

#### `step_caller_wins` — fallback

Genuine conflict that none of the previous steps resolved. Caller's intent wins; manifest's intent is overridden. Trace step: `step(caller_wins, applied, [caller_intent(kernel_mode(bidirectional)), overridden_manifest(strategy(unidirectional)), reason(unknown_consider_reconciling)])`. The warning is not emitted directly by the selector; it appears in the trace and the renderer surfaces it.

### Intent-compatibility matrix

The matrix used by `step_intent_matches` and `step_third_option` to decide whether a given intent signal is satisfied by a candidate strategy. Indexed by intent type:

| Intent signal | Satisfied by strategy |
|---------------|----------------------|
| `kernel_mode(bidirectional)` | `per_query(bidirectional)`, `per_query(astar)` (astar is bidirectional-flavour) |
| `kernel_mode(unidirectional)` | `per_query(unidirectional)`, `per_query(bfs)`, `per_query(dfs)` |
| `kernel_mode(astar)` | `per_query(astar)` |
| `kernel_mode(dijkstra)` | `per_query(dijkstra)` |
| `strategy(per_query(X))` | `per_query(X)` if X is ground; any `per_query(_)` if X is unbound |
| `strategy(fixed_point(X))` | `fixed_point(X)` if X is ground; any `fixed_point(_)` if X is unbound |
| `strategy(cached)` | `cached` |
| `strategy(hybrid(_))` | any `hybrid(_)` |
| `force_search_algorithm(A)` | any strategy whose inner algorithm matches A |

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

Conflict example:

```
[evaluation-strategy] selecting for category_ancestor/4
[evaluation-strategy]   classify_signals: intent=[kernel_mode(bidirectional), strategy(per_query(unidirectional))], declared=[csr_available(true)], inferred=[]
[evaluation-strategy]   cost_model_choice: per_query(bidirectional), score=3.0, deciding-rule=prefer_bidirectional_csr_present
[evaluation-strategy]   third_option: not_found (no admissible strategy satisfies both intent signals)
[evaluation-strategy]   scope_disambiguation: resolved(per_query(bidirectional), by(narrower_subsumes_broader))
[evaluation-strategy]   final_decision: per_query(bidirectional), decided_by=scope_disambiguation, overridden=[]
```

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
- `ConfigOps` → fed into `Recurrence` properties

The selector does not call the detector; the caller (target adapter) is responsible for invoking detection first and passing the result.

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

If `kernel_mode(quantum)` is passed, the selector adds a `step(classify_signals, ..., [unknown_intent(kernel_mode(quantum))])` warning to the trace and treats it as if no kernel_mode signal was present (i.e. skips it). Forward-compat principle: unknown intent should not break compilation, only be ignored with a warning in the trace.

### Multiple cost-model rules tie at the same score

Resolved by the explicit `Priority` argument of `cost_model_rule/6` (higher wins). If priorities also tie, the rule with the lexicographically earliest `RuleName` wins. Trace records the tie and the tiebreak: `step(cost_model_choice, chosen(...), [tied_with([...]), tiebreak_by(priority)])`.

### Recurrence properties inconsistent

If `numeric_contraction_rate(R)` with `R >= 1.0`, the selector removes all `fixed_point(...)` strategies from admissibility (as specified above) and proceeds with the remaining admissible strategies. If no strategies remain admissible after this filtering, the previous case ("no admissible strategy") applies.

If `monotone(false)` is combined with intent that requests cross-class auto-selection, the selector honours the intent (caller-wins) but records a warning in the trace that auto-selection across classes is disabled and the chosen strategy may not have its convergence properties verified.

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
