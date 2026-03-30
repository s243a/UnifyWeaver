# Implementation Plan: Demand-Driven Prolog Target Optimization

## Related Documents

- **Proposal**: `PROLOG_TARGET_DEMAND_ANALYSIS.md` — problem statement,
  solution overview, O(1) opportunities, relationship to C# query engine
- **Specification**: `PROLOG_TARGET_DEMAND_ANALYSIS_SPEC.md` — algorithm,
  guard insertion rules, shared module interface, correctness criteria
- **Prolog optimization strategies**: `docs/design/PROLOG_TARGET_OPTIMIZATION.md`
- **Per-path visited design**: `docs/design/PER_PATH_VISITED_RECURSION.md`

## Key Files to Study

| File | What to Look For |
|------|-----------------|
| `src/unifyweaver/targets/csharp_target.pl` | `build_pipeline_seeded`, demand closure, `ParamSeedNode` construction |
| `src/unifyweaver/core/advanced/pattern_matchers.pl` | `is_per_path_visited_pattern/4` — how recursive patterns are detected |
| `src/unifyweaver/core/clause_body_analysis.pl` | `classify_goal_sequence` — how body goals are analyzed |
| `examples/benchmark/effective_distance.pl` | The benchmark Prolog program to optimize |
| `examples/benchmark/generate_pipeline.py` | Pipeline generator (see how other targets handle this) |

## Key Commits

| Commit | Description |
|--------|-------------|
| `dce3945b` | `feat(csharp_query): demand-closure for parameterized mutual SCCs` |
| `88346fe9` | `feat(csharp_query): allow parameterized mutual recursion` |
| `e738fee3` | `perf(csharp-query): seed transitive closure for parameters` |
| `da9eed05` | `feat(query): harden parameterized recursion eligibility` |
| `50f02ba9` | `feat(csharp_query): broaden is/2 and arithmetic comparisons` |
| `40f0cf11` | `feat(core): add per-path visited recursion pattern detection` |

## Phase 1: Extract Demand Analysis (shared core)

### Step 1.1: Study C# Demand Closure

Read the C# target's demand closure code in `csharp_target.pl`. Find:
- How it identifies the "seed" (fixed input parameter)
- How it computes backward reachability from the seed
- How it generates the demand-aware plan nodes

The demand closure for the C# target was added in commits `dce3945b`
and `88346fe9`. The `build_pipeline_seeded` predicate (around line 1472)
is the entry point.

### Step 1.2: Create Shared Demand Analysis Module

Create `src/unifyweaver/core/demand_analysis.pl`:

```prolog
:- module(demand_analysis, [
    compute_demand_set/4,
    insert_demand_guards/4,
    generate_demand_init/3
]).

%% compute_demand_set(+StepRelation, +TargetValues, +Direction, -DemandSet)
%% Backward reachability: find all nodes that can reach any TargetValue
%% via the StepRelation.
compute_demand_set(StepRel, Targets, backward, DemandSet) :-
    findall(T, member(T, Targets), Initial),
    fixpoint_expand(StepRel, Initial, [], DemandSet).

fixpoint_expand(StepRel, [], Visited, Visited).
fixpoint_expand(StepRel, [Node|Queue], Visited, Result) :-
    (   member(Node, Visited)
    ->  fixpoint_expand(StepRel, Queue, Visited, Result)
    ;   findall(Child,
            (call(StepRel, Child, Node), \+ member(Child, Visited)),
            NewNodes),
        append(Queue, NewNodes, NewQueue),
        fixpoint_expand(StepRel, NewQueue, [Node|Visited], Result)
    ).
```

### Step 1.3: Test Independently

Test demand analysis on the benchmark facts:
```prolog
compute_demand_set(category_parent, ['Physics'], backward, D),
length(D, N),
format('Demand set size: ~w~n', [N]).
```

Expected: ~300 categories out of 1593 can reach Physics.

## Phase 2: Prolog Target Code Generation

### Step 2.1: Generate Demand Facts

```prolog
generate_demand_init(DemandSet, can_reach_root, InitCode) :-
    findall(Line,
        (member(Node, DemandSet),
         format(atom(Line), 'can_reach_root(~q).', [Node])),
        Lines),
    atomic_list_concat(Lines, '\n', InitCode).
```

### Step 2.2: Insert Guards into Clauses

For each recursive clause, find the step relation call and insert
the demand guard after it:

```prolog
insert_demand_guards(Clauses, can_reach_root, StepRel, OptClauses) :-
    maplist(insert_guard_in_clause(can_reach_root, StepRel), Clauses, OptClauses).

insert_guard_in_clause(DemandPred, StepRel, (Head, Body), (Head, OptBody)) :-
    insert_guard_after_step(Body, DemandPred, StepRel, OptBody).
```

### Step 2.3: Emit Optimized Prolog File

Combine: demand facts + optimized clauses + original non-recursive
predicates.

## Phase 3: Apply to Other Targets

Once the shared demand analysis works for Prolog, apply it to:
- **Go**: Add `reachable` map precomputed at startup, guard DFS
- **Python**: Add `reachable` set, guard recursive calls
- **Rust**: Add `HashSet<String>` demand set
- **AWK**: Add `can_reach[node]` array, guard DFS loop

This is lower priority — the compiled targets are already fast. But
at 50K+ scale the pruning would help them too.

## Phase 4: Wire into Compilation Pipeline

Add demand analysis as an optional optimization pass in the target
compilation pipeline:

```prolog
compile_predicate_to_X(Pred/Arity, Options, Code) :-
    option(demand_analysis(true), Options, false),
    detect_fixed_target(Pred/Arity, TargetValues),
    compute_demand_set(StepRelation, TargetValues, backward, DemandSet),
    insert_demand_guards(Clauses, DemandPred, StepRelation, OptClauses),
    compile_optimized(OptClauses, DemandSet, Options, Code).
```

## Estimated Impact

| Scale | Without Demand | With Demand (est.) | Speedup |
|-------|---------------|-------------------|---------|
| 19 articles (dev) | 0.04s | ~0.03s | ~1.3x |
| 300 articles | 1.26s | ~0.25s | ~5x |
| 1K articles | ~5s | ~1s | ~5x |
| 50K articles | unknown | much better | >>5x |

The benefit grows with scale because larger graphs have more dead-end
branches to prune.

## Notes for Implementing Agent

- **Start by reading the C# demand closure code** — commits `dce3945b`
  and `88346fe9` in `csharp_target.pl`
- The demand analysis is essentially **backward BFS from the target** —
  simple algorithm, the complexity is in wiring it into the compilation
  pipeline
- **Test on the benchmark**: the `effective_distance.pl` program with
  `data/benchmark/300/facts.pl` is the reference workload
- **Correctness first**: verify optimized output matches unoptimized
  before measuring speedup
- This optimization applies to **any target** where a fixed query target
  is known — it's not Prolog-specific
