# Recurrence Evaluation Strategy — Specification

This document specifies the API, data structures, decision logic, and outputs of the recurrence-evaluation-strategy selector. For the why, see [`RECURRENCE_EVALUATION_STRATEGY_PHILOSOPHY.md`](RECURRENCE_EVALUATION_STRATEGY_PHILOSOPHY.md). For the work plan, see [`RECURRENCE_EVALUATION_STRATEGY_IMPLEMENTATION_PLAN.md`](RECURRENCE_EVALUATION_STRATEGY_IMPLEMENTATION_PLAN.md).

## Module

`src/unifyweaver/core/recurrence_evaluation_strategy.pl`

```prolog
:- module(recurrence_evaluation_strategy, [
    select_evaluation_strategy/3,        % +Recurrence, +Workload, -StrategyAndTrace

    %% Helper exports for target consumers and tests
    classify_signals/3,                  % +Workload, -IntentSignals, -DataSignals
    apply_cost_model/3,                  % +Recurrence, +DataSignals, -CostModelChoice
    resolve_against_intent/5,            % +IntentSignals, +CostModelChoice, +Recurrence,
                                         %   -Strategy, -Trace
    emit_reasoning_trace/1,              % +Trace — to stderr + return for codegen
    format_trace_for_comment/2,          % +Trace, -CommentString — for generated code

    %% Introspection
    admissible_strategies/2,             % +Recurrence, -List
    strategy_pretty/2                    % +Strategy, -String
]).
```

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
| `numeric_contraction_rate(R)` | maybe Float | An upper bound on the iteration contraction rate. `none` if unknown or non-numeric; a float `< 1` if known |
| `monotone(B)` | boolean | `true` if the recurrence operator is monotone over its solution lattice (default `true` for Datalog-shaped recurrences) |
| `directions_admissible(Dirs)` | list | Subset of `[forward, backward, bidirectional]` — which traversal directions are valid for this kernel |
| `result_layout(L)` | term | The output-tuple layout from `recursive_kernel_detection:kernel_result_layout/2` |
| `result_mode(M)` | atom | `stream` / `deterministic` / `deterministic_collection` — from `kernel_result_mode/2` |
| `expected_cardinality(C)` | atom | `small` / `medium` / `large` / `unknown` — from relation-policy if declared, else `unknown` |

The `Recurrence` term is constructed once per `select_evaluation_strategy/3` call. The caller's responsibility is to populate it from the detected kernel and clause analysis. The module does not re-detect; it consumes.

### Workload signals

A list of signal terms. The classifier (`classify_signals/3`) sorts them into intent vs data.

#### Intent signals

| Signal | Source | Meaning |
|--------|--------|---------|
| `kernel_mode(Mode)` | caller option | Mode: `bidirectional`, `unidirectional`, `astar`, `dijkstra` |
| `strategy(S)` | manifest `decl_algorithm_optimization` | S: `per_query(_)`, `fixed_point(_)`, `cached`, `hybrid(_)` |
| `force_search_algorithm(A)` | caller option | A: `bfs`, `dfs`, `bidirectional`, `astar`, `dijkstra` |

Intent signals express the user's required outcome. They bypass the cost model when they are unambiguous.

#### Data signals

| Signal | Source | Meaning |
|--------|--------|---------|
| `csr_path(_)` | caller option | CSR (child-direction index) is available at this path |
| `csr_available(B)` | caller option / inferred | `true` if CSR is available (any path) |
| `csr_buildable(B)` | inferred | `true` if CSR can be built from the existing parent-direction edge predicate in O(\|E\|) |
| `cardinality(C)` | `relation_policy` declaration | `small` / `medium` / `large` / `unknown` for the edge predicate |
| `determinism(D)` | `relation_policy` | `det` / `semidet` / `nondet` / `multi` |
| `unique(B)` | `relation_policy` | `true` / `false` for the relation |
| `query_pattern(P)` | manifest / caller / inferred | `single_pair` / `all_pairs` / `all_from_source` / `sample` |
| `query_frequency(F)` | manifest / caller | `low` / `high` / `sustained` |
| `graph_mutability(M)` | manifest | `static` / `append_only` / `mutable` |
| `graph_stats(b_eff(_), D(_), r(_))` | calibration | Calibration constants from tree-likeness theory; `r` is the contraction rate |

Data signals feed the cost model. They do not constitute intent.

### Strategy

The output of strategy selection:

```prolog
strategy(Mode)

where Mode is one of:
    per_query(SearchAlgo)
    fixed_point(IterAlgo)        % stubbed; not currently auto-selectable
    cached                       % stubbed
    hybrid(Components)           % stubbed

SearchAlgo: unidirectional | bidirectional | astar | dijkstra
IterAlgo:   semi_naive | naive | top_down | bottom_up
```

Within `per_query`, the choice of `SearchAlgo` is what gets selected. Within `fixed_point` (when populated), the choice of `IterAlgo` is what gets selected.

### Decision trace

```prolog
trace(Steps)

Steps is a list of step/N terms:

  step(classify_signals, [intent(_), data(_)], ...)
  step(cost_model_choice, Choice, [signal(...), reason(...)])
  step(conflict_detection, Conflict | none, ...)
  step(third_option_search, found(Alt) | not_found, ...)
  step(scope_disambiguation, resolved(_) | no_scope_overlap, ...)
  step(satisfiability_check, satisfiable | adjusted(Action) | unsatisfiable_warn, ...)
  step(final_decision, Strategy, [decided_by(IntentOrCostModel), overridden(_)])
```

The trace is consumed in two places:

1. **Compile-time stderr.** `emit_reasoning_trace/1` renders the trace as one-line entries with the deciding signal named.
2. **Generated-code comment.** `format_trace_for_comment/2` produces a multi-line comment header that the target inserts at the kernel call site.

Both renderings are derived from the same structured trace, so they are guaranteed consistent.

## Decision logic

### Phase A — classify signals

```
classify_signals(Workload, IntentSignals, DataSignals)
```

Each workload term is matched against the intent/data lookup table. The result is two lists. Anything unrecognised goes into `DataSignals` with a warning logged to stderr (forward-compat: new signal types should be acknowledged, not silently dropped).

### Phase B — cost model evaluation

```
apply_cost_model(Recurrence, DataSignals, CostModelChoice)
```

The cost model is a set of rules that, given recurrence properties and data signals, produces a default strategy choice. Each rule produces a *partial preference* — a strategy plus the data signals that motivated it. Rules compose by preference scoring (the rule with the highest aggregate score wins).

Initial rule set (Phase 1 of the implementation plan):

| Rule | When | Output | Score |
|------|------|--------|-------|
| `prefer_bidirectional_csr_present` | `csr_available(true)`, `query_pattern(single_pair)`, `cardinality(large)` | `per_query(bidirectional)` | +3 |
| `prefer_bidirectional_csr_buildable` | `csr_buildable(true)`, `cardinality(large)`, `query_frequency(high)` | `per_query(bidirectional) with build_csr_step` | +2 |
| `prefer_unidirectional_no_csr` | `csr_available(false)`, `csr_buildable(false)` | `per_query(unidirectional)` | +2 |
| `prefer_unidirectional_small` | `cardinality(small)`, no contradicting signal | `per_query(unidirectional)` | +1 |
| `prefer_astar_heuristic_available` | `directions_admissible([..., astar])`, heuristic predicate available | `per_query(astar)` | +1 |
| `default_fallback` | always | `per_query(unidirectional)` | +0 |

Each rule has access to the recurrence properties (especially `numeric_contraction_rate`, which becomes relevant when a fixed-point rule is added). The scoring is intentionally simple — sum of weights, highest wins, ties broken by deterministic rule order. The full rule-language is *not* an expression evaluator; it is a flat scored list. Adding rules means adding clauses.

This is deliberately simpler than the more sophisticated cost-model designs in [`CACHE_COST_MODEL_PHILOSOPHY.md`](CACHE_COST_MODEL_PHILOSOPHY.md) and [`SCAN_STRATEGY_SPECIFICATION.md`](SCAN_STRATEGY_SPECIFICATION.md), because those models reason about per-operation cost in detail; this one reasons about strategy-class fit at a coarser grain. The two compose: `apply_cost_model/3` can *call* the cache-cost or scan-strategy models as helpers when their inputs are available.

### Phase C — conflict resolution

```
resolve_against_intent(IntentSignals, CostModelChoice, Recurrence, Strategy, Trace)
```

The six-step hierarchy from the philosophy doc, made operational:

#### Step 1 — no intent → cost-model wins

If `IntentSignals = []`, return `CostModelChoice` immediately. Trace records: "no intent signals; cost-model preference applies".

#### Step 2 — intent matches cost-model → no conflict

If the single intent signal points at the same strategy class as `CostModelChoice`, return that strategy. Trace records: "intent and cost-model agree".

#### Step 3 — intent and cost-model disagree → check for compatible third option

The selector iterates over `admissible_strategies(Recurrence, Admissible)` looking for any strategy that satisfies *all* intent signals. If found, returns it. Trace records: "third-option satisfaction".

The compatibility check uses small per-intent matchers — e.g. `kernel_mode(bidirectional)` matches `per_query(bidirectional)` and `per_query(astar)` (A* is bidirectional-flavour); `strategy(per_query(_))` matches any `per_query(...)`.

#### Step 4 — scope disambiguation

If two intent signals come from different scopes (manifest vs caller), and one is a subset of the other in strategy space, the more specific scope wins without raising a conflict. Trace records: "scope subset: manifest broader, caller wins by specificity".

This is *not* the same as caller-wins-fallback. Scope subset is when the broader scope's intent includes the narrower scope's intent (e.g. manifest says `strategy(per_query(_))`, caller says `kernel_mode(bidirectional)` — bidirectional is per_query, no real conflict).

#### Step 5 — satisfiability check

If the intent is structurally unmet (e.g. caller wants `bidirectional` but `csr_available(false), csr_buildable(false)`):

- Determine if an adjustment makes it satisfiable. Concrete adjustments:
  - `build_csr_at_compile_time` — if CSR is buildable
  - `degrade_to_compatible` — fall back to a strategy that satisfies the intent's broader class (e.g. `bidirectional` → `astar` if astar admissible) with a warning
  - `degrade_with_warning` — fall back to a strategy that contradicts the intent but warns loud

- The selector picks the adjustment by preference order: `build_csr_at_compile_time` > `degrade_to_compatible` > `degrade_with_warning`.

Trace records the adjustment chosen: "csr unavailable, building at compile time" or "degraded to unidirectional, warn".

#### Step 6 — caller wins, loud warning

Genuine conflict that none of steps 1–5 resolved. Caller's intent wins; manifest's intent is overridden; the trace records: "caller's `kernel_mode(bidirectional)` overrode manifest's `strategy(unidirectional)`; reason for override unknown; consider reconciling".

Warning emitted to stderr regardless of trace consumption.

### Phase D — emit trace

```
emit_reasoning_trace(Trace)
```

Renders the trace to stderr as multi-line output. One line per step, indented for readability. Critical signals named explicitly.

Example output:

```
[evaluation-strategy] selecting for category_ancestor/4
[evaluation-strategy]   classify_signals: intent=[kernel_mode(bidirectional)], data=[csr_available(true), cardinality(large), query_pattern(single_pair)]
[evaluation-strategy]   cost_model_choice: per_query(bidirectional), score=3, deciding-rule=prefer_bidirectional_csr_present
[evaluation-strategy]   conflict_detection: none (intent and cost-model agree)
[evaluation-strategy]   final_decision: per_query(bidirectional), decided_by=both (intent + cost-model)
```

Another example with conflict:

```
[evaluation-strategy] selecting for category_ancestor/4
[evaluation-strategy]   classify_signals: intent=[kernel_mode(bidirectional), strategy(per_query(unidirectional))], data=[csr_available(true)]
[evaluation-strategy]   cost_model_choice: per_query(bidirectional), score=3, deciding-rule=prefer_bidirectional_csr_present
[evaluation-strategy]   conflict_detection: caller's kernel_mode(bidirectional) vs manifest's strategy(per_query(unidirectional))
[evaluation-strategy]   third_option_search: not_found (no admissible strategy satisfies both)
[evaluation-strategy]   scope_disambiguation: caller more specific than manifest; using caller scope
[evaluation-strategy]   final_decision: per_query(bidirectional), decided_by=caller_scope_priority, overridden=[manifest:strategy(per_query(unidirectional))]
```

### Phase E — emit comment for generated code

```
format_trace_for_comment(Trace, CommentString)
```

Produces a compact multi-line comment string the target inserts at the kernel call site. Example:

```fsharp
// =========================================================================
// Evaluation strategy: per_query(bidirectional)
// Decided by: cost model (preferred) + caller (override-compatible)
// Deciding signal: csr_available(true) + cardinality(large) + query_pattern(single_pair)
// Alternatives considered: per_query(unidirectional) [score 2], per_query(astar) [score 1]
// Trace: src/unifyweaver/core/recurrence_evaluation_strategy.pl + this comment
// =========================================================================
```

The comment header is target-agnostic; targets adapt the comment-syntax wrapping (`//` for F#/C, `--` for Haskell, `%` for Prolog).

## Integration with existing modules

### Upstream: `recursive_kernel_detection.pl`

The detector produces `recursive_kernel(KernelKind, Pred/Arity, ConfigOps)`. The strategy selector reads:

- `KernelKind` → maps to admissible strategies (e.g. `bidirectional_ancestor` admits `per_query(bidirectional)` and `per_query(astar)`; `category_ancestor` admits `per_query(unidirectional)` and `per_query(bidirectional)` via upgrade)
- `Pred/Arity` → passes through to the Recurrence term
- `ConfigOps` → fed into `Recurrence` properties

The selector does not call the detector; the caller (target adapter) is responsible for invoking detection first and passing the result.

### Upstream: `algorithm_manifest.pl`

The selector reads `manifest_optimization_options(Name, Opts)` to populate the intent signals. The `decl_algorithm` declaration's `kernel(Pred/Arity)` field tells the caller which predicate to query for.

### Upstream: `relation_policy.pl`

The selector reads `get_relation_policy(EdgePred/2, cardinality, Card, unknown)` and similar accessors to populate data signals. The relation-policy module is the canonical source for `cardinality`, `determinism`, `unique`, `on_duplicate`.

### Adjacent: `cost_model.pl` and `cost_function.pl`

The selector uses these as helpers when its rules need per-operation cost estimates. For Phase 1, the rules above are coarse enough that they don't need detailed cost estimation; later rules may.

### Downstream: `wam_fsharp_target.pl`

The F# WAM target is the first consumer. Integration shape:

1. Target runs `recursive_kernel_detection:detect_recursive_kernel(...)` as before.
2. **NEW**: target builds the `Recurrence` term from the detected kernel + manifest + relation-policy lookups.
3. **NEW**: target calls `select_evaluation_strategy(Recurrence, Workload, strategy_choice(Strategy, Trace))`.
4. **NEW**: target reads the `Strategy` term and dispatches accordingly (existing `maybe_upgrade_bidirectional/2` becomes one of several upgrade paths driven by the strategy).
5. **NEW**: target inserts `format_trace_for_comment(Trace, Comment)` into the generated F# file as a comment header at the kernel call site.

The existing explicit-option-driven path (`kernel_mode(bidirectional)` from caller) continues to work — the explicit option becomes an intent signal that the selector honours (after running the full hierarchy, which it will of course resolve in favour of the caller).

### Downstream: future WAM targets

`wam_haskell_target.pl` and `wam_c_target.pl` consume the same `select_evaluation_strategy/3` predicate. Their integration follows the same five-step shape. Comment rendering uses each target's comment syntax.

## Edge cases and error handling

### Recurrence with no admissible strategy

If `admissible_strategies(Recurrence, [])`, the selector throws `error(no_admissible_strategy(Recurrence), context(select_evaluation_strategy/3, ...))`. The target adapter catches and emits a compile-time error pointing at the recurrence shape. (This case shouldn't arise in normal use; it's a structural failure.)

### Intent signals reference an unknown strategy

If `kernel_mode(quantum)` is passed, the selector logs to stderr and treats it as if no kernel_mode signal was present (i.e. skips it). Forward-compat principle: unknown intent should not break compilation, only be ignored with a warning.

### Multiple cost-model rules tie at the same score

Resolved by clause order in the cost-model rule list. Trace records the tie and the tiebreak: "tied at score 3: rules [prefer_bidirectional_csr_present, prefer_astar_heuristic_available]; picked first by declaration order".

### Recurrence properties inconsistent

If `directions_admissible([])` (no admissible direction), or `numeric_contraction_rate(R)` with `R >= 1.0` (no convergence guarantee), the selector emits a warning and proceeds with what is admissible. Hard failure is reserved for the "no admissible strategy" case.

## Versioning and compatibility

This module is new; there is no prior version to be compatible with. The API contract is:

- `select_evaluation_strategy/3` is the stable entry point.
- The `Strategy` term shape is stable; new branches (e.g. populating `fixed_point` for F# WAM) may be added; existing branches do not change.
- The `Trace` term shape is stable in its top-level structure; new step types may be added; existing step types do not change semantics.
- Cost-model rules can be added freely; the scoring system is the abstraction.
- Workload signal types can be added; classification falls back to `data` with a warning for unknowns.

## What is *not* in the API

- Re-running kernel detection. The caller already detected; the selector consumes.
- Emitting target-language code. The selector picks a strategy; the target emits.
- Decoding the trace structure into JSON or SARIF. The trace is Prolog terms; renderers exist for stderr and code comments. Other renderings are future work.
- Persisting decisions across compilation runs (caching, learning). Each call is independent.

## See also

- [`RECURRENCE_EVALUATION_STRATEGY_PHILOSOPHY.md`](RECURRENCE_EVALUATION_STRATEGY_PHILOSOPHY.md) — the why.
- [`RECURRENCE_EVALUATION_STRATEGY_IMPLEMENTATION_PLAN.md`](RECURRENCE_EVALUATION_STRATEGY_IMPLEMENTATION_PLAN.md) — the work plan.
- [`KERNEL_SHAPE_RECOGNITION.md`](KERNEL_SHAPE_RECOGNITION.md) — the upstream detector layer.
- [`ALGORITHM_MANIFEST_SPECIFICATION.md`](ALGORITHM_MANIFEST_SPECIFICATION.md) — the manifest spec.
- [`RELATION_POLICY_DECLARATIONS.md`](RELATION_POLICY_DECLARATIONS.md) — relation-policy hints.
- [`SCAN_STRATEGY_SPECIFICATION.md`](SCAN_STRATEGY_SPECIFICATION.md) — closest-sibling spec in shape.
- [`COST_FUNCTION_PHILOSOPHY.md`](COST_FUNCTION_PHILOSOPHY.md) and [`CACHE_COST_MODEL_PHILOSOPHY.md`](CACHE_COST_MODEL_PHILOSOPHY.md) — adjacent cost-model layers.
