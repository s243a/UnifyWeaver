# Computed Recursive Increments: Implementation Plan

## Overview

This plan adds support for recursive predicates where the accumulator
increment is computed from auxiliary relation lookups, not a constant.
The work generalizes UnifyWeaver's native lowering beyond constant-
increment transitive closure.

## Phase 0: Pattern Detection (shared core)

**Goal**: Extend `classify_goal_sequence` to recognize computed
increments in recursive clause bodies.

**Location**: `src/unifyweaver/core/advanced/pattern_matchers.pl`

### Step 0.1: Auxiliary goal detection

Add `is_auxiliary_join/4` predicate that identifies goals in a
recursive clause body that:
- Are not the edge relation
- Are not the recursive call
- Bind variables used in the `is/2` expression

```prolog
is_auxiliary_join(Goal, EdgePred, RecPred, AuxInfo) :-
    Goal \= (_is_),
    functor(Goal, Name, Arity),
    Name/Arity \= EdgePred,
    Name/Arity \= RecPred,
    % ... extract bound variables and match against is/2 usage
```

### Step 0.2: Computed increment extraction

Add `extract_computed_increment/5` that analyzes the `is/2` goal:
- Identifies the recursive accumulator variable
- Identifies auxiliary variables
- Extracts the expression structure (operator, operands)
- Validates the expression is supported (linear accumulation)

```prolog
extract_computed_increment(IsGoal, RecAccVar, AuxVars, Op, Expression) :-
    IsGoal = (Result is Expr),
    decompose_arithmetic(Expr, RecAccVar, Op, IncrementExpr),
    term_variables(IncrementExpr, ExprVars),
    intersection(ExprVars, AuxVars, Matched),
    Matched \= [].
```

### Step 0.3: Unified pattern result

Produce a classification that downstream compilers consume:

```prolog
computed_recursive_pattern{
    edge: predicate{name:EdgeName, arity:2},
    head: predicate{name:HeadName, arity:HeadArity},
    auxiliaries: [aux{predicate:AuxPred, bind_vars:Binds, result_vars:Results}],
    accumulator: acc{position:AccPos, base_expr:BaseExpr, rec_expr:RecExpr, op:Op},
    visited: VisitedInfo  % from existing per-path visited detection
}
```

**Depends on**: existing `is_per_path_visited_pattern/4`

## Phase 1: C# Query Engine (approach 2)

**Goal**: The plan compiler generates a `FixpointNode` with joined
auxiliary relations and computed arithmetic for the accumulator.

**Location**: `src/unifyweaver/targets/csharp_target.pl`

### Step 1.1: Plan builder for computed increments

Add `maybe_computed_increment_plan/7` alongside
`maybe_path_aware_transitive_closure_plan`:

```prolog
maybe_computed_increment_plan(HeadSpec, GroupSpecs, BaseClauses, RecClauses, Modes, Root, Relations) :-
    % Detect computed increment pattern from Phase 0
    % Build FixpointNode with:
    %   base plan: Join(edge, aux) -> Arithmetic -> Project
    %   recursive plan: Join(edge, aux, RecRef) -> Arithmetic -> Project
```

This uses existing node types — no runtime changes needed.

### Step 1.2: Auxiliary relation wiring

The plan must include the auxiliary relation in the provider's fact
store. Two cases:

- **Explicit auxiliary**: The user provides `node_degree/2` as facts.
  The compiler adds it to the relation list.
- **Derived auxiliary**: The compiler recognizes that `node_degree(X, D)`
  can be computed as `aggregate_all(count, edge(X, _), D)` and generates
  the derived relation as a preprocessing step in the plan.

Derived auxiliaries require a new plan node or a preprocessing step
before fixpoint evaluation.

### Step 1.3: Test and validate

Test against the degree-corrected semantic distance on the dev dataset.
Compare with Prolog reference output.

**Depends on**: Phase 0

## Phase 2: Go Target (approach 1)

**Goal**: Extend `compile_general_recursive_to_go` to handle computed
increments.

**Location**: `src/unifyweaver/targets/go_target.pl`

### Step 2.1: Recognize computed increment pattern

In `is_general_recursive_pattern_go`, check for the
`computed_recursive_pattern` from Phase 0. If found, dispatch to a
new code generator instead of the constant-increment path.

### Step 2.2: Generate auxiliary lookup code

Emit Go code to:
1. Build auxiliary lookup maps (e.g., degree map from adjacency)
2. Look up auxiliary values in the DFS loop
3. Compute the increment from the expression

```prolog
emit_go_auxiliary_lookup(AuxInfo, Code) :-
    % Generate: degreeMap := make(map[string]int)
    % Generate: for k, v := range adj { degreeMap[k] = len(v) }
```

### Step 2.3: Generate computed DFS loop

Modify the DFS template to use the computed increment:

```prolog
emit_go_computed_dfs_loop(Pattern, Code) :-
    % Generate: step := math.Log(float64(auxVal)) / math.Log(float64(n))
    % Generate: newAcc := acc + step
    % instead of: newAcc := acc + 1
```

**Depends on**: Phase 0

## Phase 3: Rust Target (approach 1)

**Goal**: Same as Go but for `compile_general_recursive_to_rust`.

**Location**: `src/unifyweaver/targets/rust_target.pl`

Mirrors Phase 2 structure:
- Recognize pattern
- Generate auxiliary `HashMap` construction
- Generate computed DFS loop with `f64` arithmetic

**Depends on**: Phase 0

## Phase 4: Python Target (approach 1)

**Goal**: Same for `generate_ternary_worker` and Codon variant.

**Location**: `src/unifyweaver/targets/python_target.pl`

Note: Python and Codon share the generator but Codon has type
annotation differences. The computed increment may require explicit
`float()` casts for Codon compatibility.

**Depends on**: Phase 0

## Phase 5: AWK Target (approach 1)

**Goal**: Same for `compile_general_recursive_to_awk`.

**Location**: `src/unifyweaver/targets/awk_target.pl`

AWK has no native `log()` function in all implementations. Options:
- Use `gawk` which has `log()`
- Precompute the log values and embed as a lookup array
- Skip AWK for floating-point computed increments (document limitation)

**Depends on**: Phase 0

## Phase 6: Evaluation

### 6.1: Correctness

Run degree-corrected semantic distance on dev dataset across all
targets. Compare outputs (within floating-point tolerance).

### 6.2: Generality test

Verify the same machinery handles other computed increment patterns:
- Weighted shortest path (edge cost from a `weight/2` relation)
- Accumulated probability (transition probability from a `prob/2` relation)

These should compile without additional code — the pattern detection
and code generation are generic.

### 6.3: Performance

Benchmark at 300/1K/5K/10K scales. The computed increment adds one
hash lookup per hop — measure the overhead vs constant increment.

## Priority and Dependencies

```
Phase 0 (pattern detection)
  ├── Phase 1 (C# query engine)
  ├── Phase 2 (Go)
  ├── Phase 3 (Rust)
  ├── Phase 4 (Python/Codon)
  └── Phase 5 (AWK)
Phase 6 (evaluation) — after any target completes
```

Phase 0 is the critical path. Once pattern detection works, each
target can be implemented independently. Phase 1 (C# query engine)
and Phase 2 (Go) are highest priority as they represent the two
compilation strategies.

## Effort Estimates

| Phase | Effort | Notes |
|-------|--------|-------|
| 0 | Medium | Core pattern detection, must be general |
| 1 | Medium | Plan composition, no new runtime nodes |
| 2 | Low-Medium | Extends existing Go DFS template |
| 3 | Low-Medium | Mirrors Go work |
| 4 | Low | Python generator is simpler |
| 5 | Low | May skip for float expressions |
| 6 | Low | Reuses existing benchmark infrastructure |

## Related Work

| Document | Relevance |
|----------|-----------|
| `docs/proposals/COMPUTED_RECURSIVE_INCREMENT_PHILOSOPHY.md` | Why this matters, eval-not-goal principle |
| `docs/proposals/COMPUTED_RECURSIVE_INCREMENT_SPEC.md` | Pattern definition, expression support, code gen templates |
| `docs/proposals/CROSS_TARGET_EFFECTIVE_DISTANCE_THEORY.md` | Degree-corrected semantic distance theory |
| `docs/design/PER_PATH_VISITED_RECURSION.md` | Per-path visited pattern (reused in Phase 0) |
| `src/unifyweaver/core/advanced/pattern_matchers.pl` | `is_per_path_visited_pattern/4` (extended in Phase 0) |
