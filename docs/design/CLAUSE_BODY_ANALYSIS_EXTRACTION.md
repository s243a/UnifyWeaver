# Clause Body Analysis — Status and Next Steps

## Discovery

The shared `clause_body_analysis.pl` module **already exists** (572
lines) and is **already imported by 24 targets**. It provides:

- Goal normalization (`normalize_goals/2`)
- Pattern detectors (`if_then_else_goal/4`, `if_then_goal/3`,
  `disjunction_alternatives/2`)
- Goal classification (`classify_goal/3`, `is_guard_goal/2`,
  `is_output_goal/2`)
- Shared output variable detection
- VarMap management (`build_head_varmap/3`, `ensure_var/5`, etc.)
- Expression analysis (`translate_expr/3`, `expr_op/2`)
- Clause structure analysis (`analyze_clauses/2`,
  `clause_guard_output_split/4`)

Multiple targets already use it — C, C++, Go, Rust, Clojure, AWK,
F#, Haskell, Java, and others call `clause_guard_output_split` for
multi-clause predicate compilation.

**TypR does NOT use it.** TypR still has its own duplicated copies:
`normalize_typr_goals`, `typr_if_then_else_goal`,
`typr_if_then_goal`, `typr_disjunction_alternatives`,
`build_head_varmap`. These are functionally identical to the shared
versions.

## Revised Goal

Wire TypR to use the shared module, eliminating the duplicate code
and ensuring improvements to the shared module benefit TypR too.
Then identify where the shared module's analysis capabilities aren't
being fully utilized by non-TypR targets.

## What Gets Extracted

### 1. Pattern Detectors (fully target-independent)

These operate on Prolog AST and detect structural patterns:

```prolog
%% extract_if_then_else(+Goal, -If, -Then, -Else)
extract_if_then_else((If -> Then ; Else), If, Then, Else).

%% extract_if_then(+Goal, -If, -Then)
extract_if_then((If -> Then), If, Then).

%% extract_disjunction(+Goal, -Alternatives)
extract_disjunction((A ; B), Alts) :-
    extract_disjunction(A, AltsA),
    extract_disjunction(B, AltsB),
    append(AltsA, AltsB, Alts).
extract_disjunction(Goal, [Goal]) :- Goal \= (_ ; _).
```

Currently in TypR as `typr_if_then_else_goal`, `typr_if_then_goal`,
`typr_disjunction_alternatives`. Same logic, TypR-prefixed names.

### 2. Goal Flattening (fully target-independent)

Normalize conjunction chains into flat goal lists:

```prolog
%% flatten_goals(+Body, -GoalList)
flatten_goals((A, B), Goals) :-
    flatten_goals(A, GA), flatten_goals(B, GB),
    append(GA, GB, Goals).
flatten_goals(true, []).
flatten_goals(Goal, [Goal]) :- Goal \= (_, _), Goal \= true.
```

Currently in TypR as `normalize_typr_goals`.

### 3. Head Variable Mapping (fully target-independent)

Build a mapping from Prolog variables to positional argument names:

```prolog
%% build_head_varmap(+HeadArgs, +StartIndex, -VarMap)
build_head_varmap([], _, []).
build_head_varmap([Arg|Rest], I, [Arg-ArgName|Map]) :-
    var(Arg),
    format(atom(ArgName), 'arg~w', [I]),
    I1 is I + 1,
    build_head_varmap(Rest, I1, Map).
```

Currently in TypR as `build_head_varmap`.

### 4. Shared Output Variable Detection (fully target-independent)

Detect when multiple branches bind the same output variable:

```prolog
%% shared_output_vars(+Branches, -SharedVars)
%  Find variables that appear as outputs in ALL branches.
```

Currently embedded in TypR's multi-result output handling.

### 5. Goal Classification Interface (target-parameterized)

The classifier needs a target-specific binding registry to decide
what's a guard vs output vs control flow:

```prolog
%% classify_goal(+Goal, +VarMap, +Target, -Classification)
%  Classification = guard(Expr)
%               | output(Var, Expr)
%               | control(if_then_else(If, Then, Else))
%               | control(if_then(If, Then))
%               | control(disjunction(Alts))
%               | multi_result_output(Vars, Branches)
%               | unknown

%% Default classification uses pattern detectors:
classify_goal(Goal, _VM, _Target, control(if_then_else(If, Then, Else))) :-
    extract_if_then_else(Goal, If, Then, Else), !.
classify_goal(Goal, _VM, _Target, control(if_then(If, Then))) :-
    extract_if_then(Goal, If, Then), !.
classify_goal(Goal, _VM, _Target, control(disjunction(Alts))) :-
    extract_disjunction(Goal, Alts), Alts = [_,_|_], !.

%% Target-specific: is this a guard? (needs binding registry)
classify_goal(Goal, VarMap, Target, guard(Expr)) :-
    target_guard_goal(Target, Goal, VarMap, Expr), !.

%% Target-specific: is this an output binding?
classify_goal(Goal, VarMap, Target, output(Var, Expr)) :-
    target_output_goal(Target, Goal, VarMap, Var, Expr), !.

classify_goal(_, _, _, unknown).
```

### 6. Clause Body Dispatcher (target-parameterized)

```prolog
%% compile_clause_body(+Target, +PredSpec, +Clauses, -Code)
%  Dispatches to target-specific rendering after classification.
compile_clause_body(Target, PredSpec, Clauses, Code) :-
    maplist(classify_clause(Target), Clauses, ClassifiedClauses),
    multi_clause_strategy(Target, Strategy),
    render_clause_body(Target, Strategy, PredSpec, ClassifiedClauses, Code).

%% multi_clause_strategy(+Target, -Strategy)
multi_clause_strategy(rust, match_arms).
multi_clause_strategy(haskell, pattern_heads).
multi_clause_strategy(elixir, pattern_heads).
multi_clause_strategy(go, switch_cases).
multi_clause_strategy(_, if_else_chain).  % default for most targets
```

## What Stays in TypR

- `@{ }@` wrapping logic
- R operator mapping (`r_expr_op_map`)
- R binding registry lookups
- DataFrame operation fallbacks
- Wrapped R body expression (IIFE generation)
- Type annotation computation

These implement `target_guard_goal(typr, ...)` and
`target_output_goal(typr, ...)` — the target-specific hooks
that the shared classifier calls.

## Where TypR Benefits From Templates

Large format strings in TypR that generate multi-line code blocks
are effectively inline templates. Extracting them to `.mustache`
files makes them readable, auditable, and modifiable without
touching Prolog:

| Current (format string) | Proposed template |
|------------------------|-------------------|
| Tree recursion body (~40 lines) | `structural_tree_recursive.mustache` |
| Mutual recursion wrapper (~12 lines) | `mutual_recursive_wrapper.mustache` |
| Linear recursion loops (~20 lines) | `linear_recursive_loop.mustache` |
| Generic R fallback IIFE | `generic_r_wrapper.mustache` |
| TC template (monolithic) | Split like other targets |

## Revised Implementation Order

### Step 1: Wire TypR to the shared module

Replace TypR's duplicated predicates with imports from
`clause_body_analysis`:
- `normalize_typr_goals` → `normalize_goals`
- `typr_if_then_else_goal` → `if_then_else_goal`
- `typr_if_then_goal` → `if_then_goal`
- `typr_disjunction_alternatives` → `disjunction_alternatives`
- `build_head_varmap` (TypR) → `build_head_varmap` (shared)

This is a mechanical refactor. The shared versions are
functionally identical. Verify TypR tests still pass after.

### Step 2: Audit target usage depth

The shared module has `classify_goal/3`, `analyze_clauses/2`,
`translate_expr/3` — but most targets only use
`clause_guard_output_split`. Audit which targets could use
more of the shared analysis to handle more predicate structures.

### Step 3: TypR template extraction

Move TypR's large format strings (tree recursion body, mutual
recursion wrapper, linear recursion loops) into `.mustache` files.
Independent of the shared module work.

### Step 4: Identify shared module gaps

Compare what TypR's native lowering can handle vs what the
shared module provides. The gap is TypR's more sophisticated
multi-result output handling, container patterns, and guarded
tail sequences. These may be worth adding to the shared module.

## Fallback Chains (Possible Future Work)

When native lowering fails for a target, embedded fallbacks could
provide a safety net. The chain should be configurable and ordered:
try the simpler option first, escalate if it fails.

Example: GNU Prolog first (handles ISO-compatible predicates),
then WAM if GNU Prolog fails (handles SWI-specific syntax that
GNU Prolog rejects):

```prolog
%% Configurable fallback order:
:- set_fallback_chain(rust, [gnu_prolog, wam]).

%% Try GNU Prolog first (ISO Prolog subset)
%% If that fails (SWI-specific features) → try WAM
%% WAM fans out to WAT/Jamaica/Krakatau
```

This is speculative and should only be investigated after the core
clause body analysis is working across multiple targets. The value
proposition of WAM as a fallback hub needs validation — it may or
may not be worth the implementation cost.
