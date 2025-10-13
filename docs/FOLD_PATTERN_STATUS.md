# Fold Pattern Implementation - Current Status

**Branch**: `feat/fold-helper-pattern`
**Status**: Partially Complete - Detection works, Generation needs variable binding fix
**Last Updated**: 2025-01-13

## Table of Contents

1. [Overview](#overview)
2. [What Works](#what-works)
3. [The Variable Binding Problem](#the-variable-binding-problem)
4. [Code Structure](#code-structure)
5. [Testing Status](#testing-status)
6. [Next Steps](#next-steps)
7. [Merge Safety Assessment](#merge-safety-assessment)

## Overview

The fold pattern implementation enables two-phase execution for tree-recursive predicates:

1. **Phase 1 - Build Graph**: Construct dependency tree showing recursive structure
2. **Phase 2 - Fold Graph**: Traverse tree to compute final result

**Benefits**:
- Visualize recursion structure (helpful for debugging)
- Separate structure from computation (enables caching)
- Makes parallelization explicit
- Educational value (shows recursion tree)

**User-Facing API**:
```prolog
% Mark predicate to use fold pattern
:- forbid_linear_recursion(fib/2).

% Original definition
fib(0, 0).
fib(1, 1).
fib(N, F) :- N > 1, N1 is N-1, N2 is N-2, fib(N1,F1), fib(N2,F2), F is F1+F2.

% Manual fold implementation (current requirement)
fib_graph(0, leaf(0)).
fib_graph(1, leaf(1)).
fib_graph(N, node(N, [L, R])) :- N > 1, N1 is N-1, N2 is N-2, fib_graph(N1,L), fib_graph(N2,R).

fold_fib(leaf(V), V).
fold_fib(node(_, [L, R]), V) :- fold_fib(L, VL), fold_fib(R, VR), V is VL+VR.

fib_fold(N, F) :- fib_graph(N, Graph), fold_fib(Graph, F).

% Query: inspect structure or compute value
?- fib_graph(3, G).
G = node(3, [node(2, [leaf(1), leaf(0)]), leaf(1)]).

?- fib_fold(5, F).
F = 5.
```

## What Works

### 1. Pattern Detection ✅

**File**: `src/unifyweaver/core/advanced/pattern_matchers.pl`

**Predicate**: `is_tree_fold_pattern/1`

**What it does**: Detects when a predicate should use the fold helper pattern.

**Detection Criteria**:
```prolog
is_tree_fold_pattern(Pred/Arity) :-
    % 1. Must be forbidden from linear recursion
    is_forbidden_linear_recursion(Pred/Arity),

    % 2. Arity must be 2 (input, output)
    Arity =:= 2,

    % 3. Must have 2+ recursive calls (tree pattern)
    functor(Head, Pred, Arity),
    user:clause(Head, Body),
    contains_call_to(Body, Pred),
    count_recursive_calls(Body, Pred, Count),
    Count >= 2,

    % 4. Recursive calls must be in conjunction (AND), not disjunction (OR)
    recursive_calls_independent(Body, Pred).
```

**Test Results**: ✅ All tests passing
```prolog
?- forbid_linear_recursion(test_fib/2).
?- is_tree_fold_pattern(test_fib/2).
true.  % ✓ Correctly detects fibonacci matches fold pattern
```

**Location**: Lines 430-484 in `pattern_matchers.pl`

### 2. Manual Examples ✅

**Files**:
- `examples/fibonacci_fold.pl` - Fibonacci with fold pattern
- `examples/binomial_fold.pl` - Pascal's triangle with fold pattern

**What they demonstrate**: Complete working implementations of fold pattern, manually written.

**Example from fibonacci_fold.pl**:
```prolog
% Original fibonacci (will use tree recursion due to forbid)
:- forbid_linear_recursion(fib/2).
fib(0, 0).
fib(1, 1).
fib(N, F) :- N > 1, N1 is N-1, N2 is N-2, fib(N1,F1), fib(N2,F2), F is F1+F2.

% Manually implemented fold helpers
fib_graph(0, leaf(0)).
fib_graph(1, leaf(1)).
fib_graph(N, node(N, [L, R])) :-
    N > 1,
    N1 is N-1,
    N2 is N-2,
    fib_graph(N1, L),
    fib_graph(N2, R).

fold_fib(leaf(V), V).
fold_fib(node(_, [L, R]), V) :-
    fold_fib(L, VL),
    fold_fib(R, VR),
    V is VL+VR.

fib_fold(N, F) :- fib_graph(N, Graph), fold_fib(Graph, F).
```

**Test Results**: ✅ Both examples tested and working
```bash
$ swipl examples/fibonacci_fold.pl
?- fib_fold(5, F).
F = 5.

?- fib_graph(3, G).
G = node(3, [node(2, [leaf(1), leaf(0)]), leaf(1)]).
```

### 3. Design Documentation ✅

**File**: `docs/FOLD_GENERATOR_DESIGN.md`

**What it contains**:
- Complete explanation of two-phase fold pattern
- General term transformation approach (v0.1) - complex, has bugs
- Template-based approach (v0.2) - simpler, partially working
- Trade-offs analysis
- Future work: mustache templates, hybrid approach
- References to Prolog variable substitution techniques

**Key Insight from Document**:
> The general transformation approach is **more powerful** but **harder to implement correctly** due to Prolog's immutable variables. The template approach is **simpler** and **handles 80% of cases**, making it better for initial implementation.

### 4. Template Framework ✅ (Partial)

**File**: `src/unifyweaver/core/advanced/fold_helper_generator.pl`

**What works**:
- `generate_fold_helpers/2` - Main entry point
- `extract_template_params/3` - Extracts parameters from original predicate
- `generate_from_template/4` - Binary tree template instantiation
- `build_conjunction/2` - Helper for building goal conjunctions

**Generated Output Example**:
```prolog
?- generate_fold_helpers(test_fib2/2, Clauses).
Clauses = [
    clause(test_fib2_graph(0, leaf(0)), true),
    clause(test_fib2_graph(1, leaf(1)), true),
    clause(test_fib2_graph(_N, node(_N, [_L, _R])),
           (_N>1, (_N1 is _N-1, _N2 is _N-2), test_fib2_graph(_N1,_L), test_fib2_graph(_N2,_R))),
    clause(fold_test_fib2(leaf(_V), _V), true),
    clause(fold_test_fib2(node(_, [_FL, _FR]), _Out),
           (fold_test_fib2(_FL,_VL), fold_test_fib2(_FR,_VR), _Out is _VL+_VR)),
    clause(test_fib2_fold(_In, _Out),
           (test_fib2_graph(_In,_G), fold_test_fib2(_G,_Out)))
].
```

**Structure is correct**, but there's a variable binding issue...

## The Variable Binding Problem

### The Issue

When we extract guards and computations from the original clause, they contain **variable references from that specific clause instance**. When we try to build a new clause using these goals, the variables are already bound, causing runtime errors.

### Example Walkthrough

**Original Clause**:
```prolog
% When Prolog loads this clause, it creates internal variable instances
fib(N, F) :- N > 1, N1 is N-1, N2 is N-2, fib(N1,F1), fib(N2,F2), F is F1+F2.
%   ^N=Var1234  ^Var1234   ^Var1235=N1    ^Var1236=N2
```

**Extraction in `extract_guards/3`** (line 106-112 in fold_helper_generator.pl):
```prolog
extract_guards(Body, Pred, Guards) :-
    findall(Goal, (
        extract_goal(Body, Goal),  % Extracts goals from body
        \+ functor(Goal, is, 2),   % Skip arithmetic
        \+ functor(Goal, Pred, _)  % Skip recursive calls
    ), Guards).
```

When this extracts `N > 1`, it gets the actual goal with the bound variable:
```prolog
Guards = [Var1234 > 1]  % Var1234 is the ACTUAL variable from original clause!
```

**Problem in `generate_from_template/4`** (line 182-188):
```prolog
% Try to build new clause
build_conjunction(Guards, GuardConj),  % GuardConj = (Var1234 > 1)
...
GraphRecHead =.. [GraphPred, _InputVar, node(_InputVar, [_L, _R])],
GraphRecBody = (GuardConj, CompConj, GraphRecCall1, GraphRecCall2),
%               ^^^^^^^^^
%               This contains Var1234 from ORIGINAL clause!
%               But new clause head uses _InputVar (different variable)!
```

**Runtime Error**:
```prolog
?- test_fib2_fold(5, F).
ERROR: >/2: Arguments are not sufficiently instantiated
%      Because: Var1234 is unbound in new clause context!
```

### Why This Happens

In Prolog:
1. **Variables are immutable** - once created, their identity is fixed
2. **Clause bodies capture variable instances** - when you extract a goal like `N > 1`, you get the actual variable `N`, not a "pattern" or "template"
3. **Different clause = different variables** - `_InputVar` in new clause is not the same as `N` in original clause

### Visual Diagram

```
Original Clause:
    fib(N, F) :- N > 1, ...
        |         |
        v         v
    Var1234   (Var1234 > 1)  <-- Actual goal with actual variable

Extraction:
    Guards = [(Var1234 > 1)]  <-- We get the goal WITH its variables

New Clause Generation:
    fib_graph(_InputVar, ...) :- (Var1234 > 1), ...
              ^                    ^
              |                    |
              New Var              Old Var from different clause!

    MISMATCH! Var1234 is unbound in this clause scope.
```

### What We Need

**Variable Substitution/Renaming**: Replace variables in extracted goals with fresh variables that match the new clause head.

**Desired Transformation**:
```prolog
% From original clause: N > 1 (where N is input variable)
% To new clause:        _InputVar > 1 (where _InputVar is new input variable)
```

### Related Code Locations

**Where extraction happens**:
- `extract_guards/3` - Line 106 in fold_helper_generator.pl
- `extract_computations/3` - Line 122 in fold_helper_generator.pl
- `extract_operator/3` - Line 144 in fold_helper_generator.pl

**Where new clause is built**:
- `generate_from_template/4` - Line 158 in fold_helper_generator.pl
- Specifically lines 182-188 (graph recursive clause generation)

**Helper predicates involved**:
- `build_conjunction/2` - Line 218 in fold_helper_generator.pl
- `extract_goal/2` - From pattern_matchers.pl (exported helper)

## Code Structure

### Module Dependencies

```
fold_helper_generator.pl
    ├── uses: pattern_matchers.pl
    │   ├── is_tree_fold_pattern/1 (detection)
    │   ├── extract_goal/2 (goal extraction)
    │   ├── contains_call_to/2 (recursive call detection)
    │   └── count_recursive_calls/2 (call counting)
    │
    └── exports:
        ├── generate_fold_helpers/2 (main API)
        ├── generate_graph_builder/3 (compatibility)
        ├── generate_fold_computer/3 (compatibility)
        ├── generate_wrapper/2 (compatibility)
        └── install_fold_helpers/1 (install to user module)
```

### Key Predicates

**1. Main Entry Point**:
```prolog
% File: fold_helper_generator.pl, Line 53
generate_fold_helpers(Pred/Arity, AllClauses) :-
    % Get original clauses from user module
    functor(Head, Pred, Arity),
    findall(clause(Head, Body), user:clause(Head, Body), OrigClauses),

    % Extract parameters (guards, computations, operator)
    extract_template_params(Pred/Arity, OrigClauses, Params),

    % Generate using binary tree template
    generate_from_template(binary_tree, Pred/Arity, Params, AllClauses).
```

**2. Parameter Extraction**:
```prolog
% File: fold_helper_generator.pl, Line 69
extract_template_params(Pred/Arity, OrigClauses, Params) :-
    % Separate base vs recursive clauses
    partition_clauses(Pred, OrigClauses, BaseClauses, RecClauses),

    % Extract from recursive clause
    RecClauses = [clause(RecHead, RecBody)|_],

    % Extract guards: N > 1
    extract_guards(RecBody, Pred, Guards),

    % Extract computations: N1 is N-1, N2 is N-2
    extract_computations(RecBody, Pred, Computations, RecArgs),

    % Extract operator: + (from F is F1+F2)
    extract_operator(RecBody, _, Operator),

    Params = params(
        base_clauses(BaseClauses),
        guards(Guards),
        computations(Computations),
        rec_args(RecArgs),
        operator(Operator)
    ).
```

**3. Template Instantiation**:
```prolog
% File: fold_helper_generator.pl, Line 158
generate_from_template(binary_tree, Pred/_Arity, Params, AllClauses) :-
    % Generate predicate names
    atom_concat(Pred, '_graph', GraphPred),
    atom_concat('fold_', Pred, FoldPred),
    atom_concat(Pred, '_fold', WrapperPred),

    % Extract parameters
    Params = params(
        base_clauses(BaseClauses),
        guards(Guards),
        computations(Computations),
        rec_args([Arg1, Arg2]),
        operator(Operator)
    ),

    % Generate 3 sets of clauses:
    % 1. Graph builder (base + recursive)
    % 2. Fold computer (leaf + node)
    % 3. Wrapper (calls graph then fold)
    ...
```

### File Organization

```
UnifyWeaver/
├── src/unifyweaver/core/advanced/
│   ├── pattern_matchers.pl          % Detection logic ✅
│   │   ├── is_tree_fold_pattern/1   % Lines 430-484
│   │   ├── extract_goal/2           % Lines 64-74 (exported)
│   │   └── Tests                    % Lines 490-605 (Test 7)
│   │
│   └── fold_helper_generator.pl     % Generation logic ⚠️
│       ├── Template-based (v0.2)    % Lines 48-221 (CURRENT)
│       └── Old implementation       % Lines 262-489 (REFERENCE)
│
├── examples/
│   ├── fibonacci_fold.pl            % Working example ✅
│   └── binomial_fold.pl             % Working example ✅
│
└── docs/
    ├── FOLD_HELPER_PATTERN.md       % User documentation ✅
    ├── FOLD_GENERATOR_DESIGN.md     % Design rationale ✅
    └── FOLD_PATTERN_STATUS.md       % This file
```

## Testing Status

### Unit Tests

**Pattern Detection** - ✅ PASSING
```bash
$ swipl -g "use_module('src/unifyweaver/core/advanced/pattern_matchers'), test_pattern_matchers, halt."
Test 7: Tree fold pattern (fibonacci with forbid)
  ✓ PASS - test_fib without forbid does NOT match fold pattern
  ✓ PASS - forbidden test_fib matches tree fold pattern
  ✓ PASS - should_use_fold_helper alias works
```

**Code Generation** - ⚠️ PARTIAL
```bash
$ swipl -g "use_module('src/unifyweaver/core/advanced/fold_helper_generator'), test_fold_helper_generator, halt."
Test 1: Generate fold helpers for fibonacci
  ✓ PASS - generated 6 clauses
  Generated clauses:
    test_fib2_graph(0,leaf(0)) :- true
    test_fib2_graph(1,leaf(1)) :- true
    test_fib2_graph(_N,node(_N,[_L,_R])) :- _N>1,(_N1 is _N-1,_N2 is _N-2),test_fib2_graph(_N1,_L),test_fib2_graph(_N2,_R)
    fold_test_fib2(leaf(_V),_V) :- true
    fold_test_fib2(node(_,[_FL,_FR]),_Out) :- fold_test_fib2(_FL,_VL),fold_test_fib2(_FR,_VR),_Out is _VL+_VR
    test_fib2_fold(_In,_Out) :- test_fib2_graph(_In,_G),fold_test_fib2(_G,_Out)
```
**Note**: Clauses are generated, but variable binding issue prevents execution.

**Manual Examples** - ✅ PASSING
```bash
$ swipl examples/fibonacci_fold.pl
?- fib_fold(5, F).
F = 5.

?- fib_fold(10, F).
F = 55.

?- fib_graph(3, G).
G = node(3, [node(2, [leaf(1), leaf(0)]), leaf(1)]).
```

### Regression Tests

**Original Tests** - ✅ PASSING

All existing UnifyWeaver tests still pass. The fold pattern work:
- Adds new functionality (doesn't break existing)
- Is opt-in (requires `forbid_linear_recursion/1`)
- Doesn't affect existing compilation paths

```bash
$ swipl -g "use_module('src/unifyweaver/core/advanced/test_advanced'), test_advanced, halt."
=== ADVANCED TESTS ===
... (all existing tests pass)
```

## Next Steps

### Immediate: Fix Variable Binding

**Goal**: Make generated code executable, not just syntactically correct.

**Approach**: Implement variable renaming in goal extraction.

**Option 1 - Variable Mapping** (Recommended):
```prolog
% Build mapping from original variables to new variables
extract_guards_with_mapping(RecHead, RecBody, Pred, Guards) :-
    % Get variable positions from head
    RecHead =.. [Pred, InputVar, OutputVar],

    % Create fresh variables for new clause
    VarMap = [InputVar-_NewInput, OutputVar-_NewOutput],

    % Extract and rename guards
    findall(RenamedGoal, (
        extract_goal(RecBody, Goal),
        \+ functor(Goal, is, 2),
        \+ functor(Goal, Pred, _),
        rename_goal_vars(Goal, VarMap, RenamedGoal)
    ), Guards).

% Rename variables in a goal using mapping
rename_goal_vars(Goal, VarMap, RenamedGoal) :-
    copy_term(Goal, RenamedGoal),
    rename_vars_in_term(RenamedGoal, VarMap).

rename_vars_in_term(Term, VarMap) :-
    ( var(Term) ->
        % If variable, look up in map and unify
        ( member(OldVar-NewVar, VarMap), Term == OldVar ->
            Term = NewVar
        ; true )
    ; compound(Term) ->
        % Recursively process arguments
        Term =.. [Functor|Args],
        maplist(rename_vars_in_term_with_map(VarMap), Args)
    ; true  % Atom or number - nothing to rename
    ).
```

**Option 2 - Term Rewriting**:
```prolog
% Use term_variables/2 to find all variables, then create substitution
extract_guards_with_substitution(RecHead, RecBody, Pred, Guards) :-
    RecHead =.. [Pred, InputVar, OutputVar],
    term_variables(RecBody, AllVars),

    % Create fresh variables
    length(AllVars, N),
    length(FreshVars, N),

    % Build substitution list
    pairs_keys_values(Subst, AllVars, FreshVars),

    % Apply substitution to extracted goals
    findall(SubstGoal, (
        extract_goal(RecBody, Goal),
        \+ functor(Goal, is, 2),
        \+ functor(Goal, Pred, _),
        apply_subst(Goal, Subst, SubstGoal)
    ), Guards).
```

**Files to Modify**:
- `src/unifyweaver/core/advanced/fold_helper_generator.pl`
  - Lines 106-117: `extract_guards/3`
  - Lines 122-141: `extract_computations/3`
  - Lines 144-153: `extract_operator/3`

**Testing**:
1. Update `/tmp/test_fold_generation.pl` with working generation
2. Test `fib_fold(5, F)` executes successfully
3. Test `fib_graph(3, G)` returns correct structure
4. Add automated test to `test_fold_helper_generator/0`

### Short-term: More Templates

**Goal**: Support ternary trees, n-ary trees, different operators.

**Templates to Add**:
1. Ternary tree (3 recursive calls)
2. N-ary tree (variable number of calls)
3. Different operators (max, min, *, custom)

**Approach**: Extend `generate_from_template/4` with pattern matching:
```prolog
generate_from_template(binary_tree, ...) :- ... .  % Existing
generate_from_template(ternary_tree, ...) :- ... . % New
generate_from_template(nary_tree, ...) :- ... .    % New
```

### Medium-term: Bash Compilation

**Goal**: Generate bash code for two-phase fold pattern.

**Approach**:
1. Graph builder → bash function that builds array representation
2. Fold computer → bash function that folds over array
3. Wrapper → bash function that calls both

**Integration Point**: `src/unifyweaver/core/advanced/tree_recursion.pl`

### Long-term: Return to General Approach

**Goal**: Handle arbitrary predicates, not just templates.

**See**: `docs/FOLD_GENERATOR_DESIGN.md` for full design.

**Key Challenges**:
- Variable substitution in arbitrary expressions
- Preserving complex guard logic
- Handling non-uniform recursion patterns

**Hybrid Approach**:
- Use templates for common cases (80%)
- Fall back to general transformation for edge cases (20%)
- Provide escape hatch for user-defined templates

## Merge Safety Assessment

### Can We Merge Now?

**YES** ✅ - Safe to merge with caveats

### Why It's Safe

1. **No Breaking Changes**
   - All existing tests pass
   - New functionality is opt-in (requires `forbid_linear_recursion/1`)
   - Doesn't affect existing compilation paths

2. **Detection Logic is Solid**
   - `is_tree_fold_pattern/1` works correctly
   - Properly integrated with `pattern_matchers.pl`
   - All detection tests passing

3. **Manual Examples Work**
   - `examples/fibonacci_fold.pl` - complete working implementation
   - `examples/binomial_fold.pl` - complete working implementation
   - Users can manually implement fold pattern following examples

4. **Good Documentation**
   - User-facing: `docs/FOLD_HELPER_PATTERN.md`
   - Design rationale: `docs/FOLD_GENERATOR_DESIGN.md`
   - Status/TODO: `docs/FOLD_PATTERN_STATUS.md` (this file)

5. **Clear Path Forward**
   - Variable binding issue is well-documented
   - Solution approaches outlined
   - Can continue work on branch after merge

### What's Not Ready

1. **Auto-generation** - Generates syntactically correct but not executable code
2. **`install_fold_helpers/1`** - Won't work until variable binding fixed
3. **Bash compilation** - Not yet implemented

### Merge Strategy

**Recommended**: Merge now, continue work later

**Reasoning**:
- Core infrastructure is solid
- Manual pattern works and is documented
- Future work is additive, not corrective
- Delaying merge increases merge conflict risk

**After Merge**:
1. Can continue work on `feat/fold-helper-pattern` branch
2. Or create new branch `feat/fold-auto-generation` for completion
3. Users can use manual pattern while auto-generation is perfected

### Alternative: Wait to Merge

**If you prefer to wait**, criteria for "ready to merge":
1. ✅ Variable binding fix implemented
2. ✅ Auto-generation tested with fibonacci
3. ✅ Auto-generation tested with binomial
4. ✅ At least 2 templates working (binary + one more)
5. ⚠️ Bash compilation (optional - could be separate PR)

**Estimated Time**: 2-4 hours of focused work

## References

### Documentation Files

- `docs/FOLD_HELPER_PATTERN.md` - User guide and examples
- `docs/FOLD_GENERATOR_DESIGN.md` - Design rationale and approaches
- `docs/FOLD_PATTERN_STATUS.md` - This file (status and next steps)

### Source Files

- `src/unifyweaver/core/advanced/pattern_matchers.pl` - Detection (lines 430-484)
- `src/unifyweaver/core/advanced/fold_helper_generator.pl` - Generation
- `examples/fibonacci_fold.pl` - Working manual example
- `examples/binomial_fold.pl` - Working manual example

### Test Files

- Pattern detection tests - `pattern_matchers.pl` lines 578-604
- Generation tests - `fold_helper_generator.pl` lines 505-528
- Manual example tests - Run examples directly with `swipl`

### Related Docs

- `src/unifyweaver/core/advanced/tree_recursion.pl` - Tree recursion compiler
- `src/unifyweaver/core/advanced/linear_recursion.pl` - Linear recursion compiler
- `README.md` - Project overview

## Continuation After Merge

### If Merged Now

**Branch**: Keep `feat/fold-helper-pattern` for continued work

**OR**: Create new branch `feat/fold-auto-generation`

**Next Commits**:
1. "fix: Variable binding in fold helper generation"
2. "test: End-to-end fold generation tests"
3. "feat: Add ternary tree template"
4. "feat: Bash code generation for fold pattern"

### If Waiting

**Complete on current branch**, then merge when fully working.

**Final merge commit**: "feat: Complete fold pattern with auto-generation"

## Questions for Review

1. **Merge Now or Wait?**
   - Merge now: Get detection + manual pattern into main, continue work later
   - Wait: Complete auto-generation first, merge when fully working

2. **Branch Strategy After Merge?**
   - Continue on `feat/fold-helper-pattern`
   - New branch `feat/fold-auto-generation`
   - Fold into `main` development

3. **Priority for Auto-Generation?**
   - High: Fix variable binding next
   - Medium: Manual pattern is good enough for now
   - Low: Focus on other features first

---

**Last Updated**: 2025-01-13
**Review By**: John William Creighton (s243a)
**Status**: Awaiting review decision
