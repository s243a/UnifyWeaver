# Fold Auto-Generation - Implementation Plan

**Goal**: Fix variable binding issue and complete auto-generation of fold helpers
**Branch**: `feat/fold-helper-pattern` (continue on existing branch)
**Estimated Time**: 2-4 hours
**Difficulty**: Medium (requires careful variable handling, but approach is clear)

## Problem Summary

Currently, `generate_fold_helpers/2` produces syntactically correct but non-executable code because extracted goals contain variable references from the original clause.

**Root Cause**: When we extract `N > 1` from original clause, we get the actual variable `N`, not a pattern. When building new clause with different variable `_InputVar`, they don't match.

**Solution**: Implement variable renaming during goal extraction.

## Implementation Strategy

### Phase 1: Variable Mapping Approach (Recommended)

Build explicit mapping from original variables to new variables, then rename all extracted goals.

**Why This Approach**:
- ✅ Clean and explicit
- ✅ Easy to debug
- ✅ Works for all goal types
- ✅ Standard Prolog technique

**Implementation Steps** (1-2 hours):

#### Step 1.1: Create Variable Mapping Helper

**File**: `src/unifyweaver/core/advanced/fold_helper_generator.pl`
**Location**: Add after line 153 (after `extract_operator/3`)

```prolog
%% build_var_mapping(+OrigHead, +NewVars, -VarMap)
%  Build mapping from original clause head variables to new variables
%
%  Example:
%    OrigHead = fib(N, F)  % N and F are actual variables from clause
%    NewVars = [_InputVar, _OutputVar]  % Fresh variables for new clause
%    VarMap = [N-_InputVar, F-_OutputVar]  % Mapping pairs
%
build_var_mapping(OrigHead, NewVars, VarMap) :-
    OrigHead =.. [_Pred|OrigArgs],
    pairs_keys_values(VarMap, OrigArgs, NewVars).

%% rename_goal_vars(+Goal, +VarMap, -RenamedGoal)
%  Rename all variables in a goal using the mapping
%
%  Example:
%    Goal = (N > 1)
%    VarMap = [N-_InputVar, F-_OutputVar]
%    RenamedGoal = (_InputVar > 1)  % N replaced with _InputVar
%
rename_goal_vars(Goal, VarMap, RenamedGoal) :-
    % Copy term to get fresh structure
    copy_term(Goal, RenamedGoal),
    % Apply variable substitutions
    apply_var_mapping(RenamedGoal, VarMap).

%% apply_var_mapping(+Term, +VarMap)
%  Recursively apply variable mapping to term (modifies in place via unification)
%
apply_var_mapping(Term, VarMap) :-
    ( var(Term) ->
        % If it's a variable, look it up in mapping
        ( member(OldVar-NewVar, VarMap), Term == OldVar ->
            Term = NewVar  % Unify with new variable
        ; true )  % Not in mapping, leave as is
    ; compound(Term) ->
        % Recursively process all arguments
        Term =.. [_Functor|Args],
        maplist(apply_var_mapping_with_map(VarMap), Args)
    ; true ).  % Atom or number, nothing to do

apply_var_mapping_with_map(VarMap, Arg) :-
    apply_var_mapping(Arg, VarMap).
```

**Test**:
```prolog
?- OrigHead = fib(N, F),
   build_var_mapping(OrigHead, [_I, _O], Map),
   Goal = (N > 1),
   rename_goal_vars(Goal, Map, Renamed).
% Should produce: Renamed = (_I > 1)
```

#### Step 1.2: Update `extract_guards/3`

**File**: `src/unifyweaver/core/advanced/fold_helper_generator.pl`
**Location**: Replace lines 106-117

```prolog
%% extract_guards(+RecHead, +RecBody, +Pred, +NewVars, -Guards)
%  Extract guard conditions with variable renaming
%
extract_guards(RecHead, RecBody, Pred, NewVars, Guards) :-
    % Build variable mapping
    build_var_mapping(RecHead, NewVars, VarMap),

    % Extract and rename guards
    findall(RenamedGoal, (
        extract_goal(RecBody, Goal),
        \+ functor(Goal, is, 2),  % Skip arithmetic
        \+ functor(Goal, Pred, _),  % Skip recursive calls
        rename_goal_vars(Goal, VarMap, RenamedGoal)
    ), Guards).
```

#### Step 1.3: Update `extract_computations/3`

**File**: `src/unifyweaver/core/advanced/fold_helper_generator.pl`
**Location**: Replace lines 122-141

```prolog
%% extract_computations(+RecHead, +RecBody, +Pred, +NewVars, -Computations, -RecArgs)
%  Extract computations with variable renaming
%
extract_computations(RecHead, RecBody, Pred, NewVars, Computations, RecArgs) :-
    % Build variable mapping
    build_var_mapping(RecHead, NewVars, VarMap),

    % Extract computations that feed into recursive calls
    findall(RenamedComp, (
        extract_goal(RecBody, Comp),
        Comp =.. [is, Var, _],
        % Check if this variable is used in recursive call
        extract_goal(RecBody, RecCall),
        functor(RecCall, Pred, _),
        RecCall =.. [Pred, Arg, _],
        Arg == Var,
        % Rename the computation
        rename_goal_vars(Comp, VarMap, RenamedComp)
    ), ComputationsWithDups),
    sort(ComputationsWithDups, Computations),

    % Extract renamed recursive arguments
    findall(RenamedArg, (
        extract_goal(RecBody, RecCall),
        functor(RecCall, Pred, _),
        RecCall =.. [Pred, Arg, _],
        rename_goal_vars(Arg, VarMap, RenamedArg)
    ), RecArgs).
```

#### Step 1.4: Update `extract_template_params/3`

**File**: `src/unifyweaver/core/advanced/fold_helper_generator.pl`
**Location**: Modify lines 69-92

```prolog
extract_template_params(Pred/Arity, OrigClauses, Params) :-
    % Separate base and recursive clauses
    partition_clauses(Pred, OrigClauses, BaseClauses, RecClauses),

    % Extract from recursive clause
    RecClauses = [clause(RecHead, RecBody)|_],

    % Create fresh variables for new clause (these will be used throughout)
    length([_InputVar, _OutputVar], Arity),
    NewVars = [_InputVar, _OutputVar],

    % Extract guards with renaming
    extract_guards(RecHead, RecBody, Pred, NewVars, Guards),

    % Extract computations with renaming
    extract_computations(RecHead, RecBody, Pred, NewVars, Computations, RecArgs),

    % Extract operator (doesn't need renaming, just pattern)
    extract_operator(RecBody, _, Operator),

    Params = params(
        base_clauses(BaseClauses),
        new_vars(NewVars),  % NEW: pass fresh variables
        guards(Guards),
        computations(Computations),
        rec_args(RecArgs),
        operator(Operator)
    ).
```

#### Step 1.5: Update `generate_from_template/4`

**File**: `src/unifyweaver/core/advanced/fold_helper_generator.pl`
**Location**: Modify lines 163-170

```prolog
generate_from_template(binary_tree, Pred/_Arity, Params, AllClauses) :-
    atom_concat(Pred, '_graph', GraphPred),
    atom_concat('fold_', Pred, FoldPred),
    atom_concat(Pred, '_fold', WrapperPred),

    % Extract parameters
    Params = params(
        base_clauses(BaseClauses),
        new_vars([InputVar, OutputVar]),  % USE these instead of creating fresh
        guards(Guards),
        computations(Computations),
        rec_args([Arg1, Arg2]),
        operator(Operator)
    ),

    % ... rest of template generation uses InputVar and OutputVar
    % which now match the variables in Guards and Computations!
```

**Key Change**: Instead of creating `_InputVar` and `_OutputVar` in template, use the ones from Params which were used during extraction.

### Phase 2: Testing (30 minutes)

#### Test 2.1: Unit Test Variable Mapping

Create test file `/tmp/test_var_mapping.pl`:
```prolog
:- use_module('/path/to/fold_helper_generator').

test_var_mapping :-
    % Test variable mapping creation
    Head = fib(N, F),
    build_var_mapping(Head, [_I, _O], Map),
    format('Mapping: ~w~n', [Map]),

    % Test goal renaming
    Goal1 = (N > 1),
    rename_goal_vars(Goal1, Map, Renamed1),
    format('~w => ~w~n', [Goal1, Renamed1]),

    Goal2 = (N1 is N - 1),
    rename_goal_vars(Goal2, Map, Renamed2),
    format('~w => ~w~n', [Goal2, Renamed2]),

    writeln('Variable mapping tests passed!').
```

Run: `swipl -g "['/ tmp/test_var_mapping'], test_var_mapping, halt."`

Expected output:
```
Mapping: [N-_123, F-_456]
N>1 => _123>1
N1 is N-1 => _789 is _123-1
```

#### Test 2.2: End-to-End Generation Test

Update `/tmp/test_fold_generation.pl`:
```prolog
:- use_module('/path/to/fold_helper_generator').

:- dynamic test_fib3/2.
test_fib3(0, 0).
test_fib3(1, 1).
test_fib3(N, F) :- N > 1, N1 is N - 1, N2 is N - 2, test_fib3(N1, F1), test_fib3(N2, F2), F is F1 + F2.

:- install_fold_helpers(test_fib3/2).

test_all :-
    writeln('Testing generated fold helpers:'),

    % Test execution (should work now!)
    test_fib3_fold(5, F5),
    format('  fib_fold(5) = ~w (expected: 5)~n', [F5]),
    ( F5 = 5 -> writeln('  ✓ PASS') ; writeln('  ✗ FAIL') ),

    test_fib3_fold(10, F10),
    format('  fib_fold(10) = ~w (expected: 55)~n', [F10]),
    ( F10 = 55 -> writeln('  ✓ PASS') ; writeln('  ✗ FAIL') ),

    % Test structure
    test_fib3_graph(3, G3),
    format('  fib_graph(3) = ~w~n', [G3]),
    ( G3 = node(3, [node(2, [leaf(1), leaf(0)]), leaf(1)]) ->
        writeln('  ✓ PASS')
    ;   writeln('  ✗ FAIL') ),

    writeln('All tests passed!').
```

Run: `swipl -g "test_all, halt."`

**Success Criteria**:
- ✅ No "Arguments are not sufficiently instantiated" error
- ✅ `fib_fold(5, F)` returns `F = 5`
- ✅ `fib_fold(10, F)` returns `F = 55`
- ✅ `fib_graph(3, G)` returns correct structure

#### Test 2.3: Binomial Test

```prolog
:- dynamic binom_test/3.
binom_test(_, 0, 1).
binom_test(N, N, 1).
binom_test(N, K, C) :-
    N > 0, K > 0, K < N,
    N1 is N - 1,
    K1 is K - 1,
    binom_test(N1, K1, C1),
    binom_test(N1, K, C2),
    C is C1 + C2.

:- install_fold_helpers(binom_test/3).

?- binom_test_fold(5, 2, C).
C = 10.  % ✓ Should work!
```

### Phase 3: Integration Test (30 minutes)

#### Update `test_fold_helper_generator/0`

**File**: `src/unifyweaver/core/advanced/fold_helper_generator.pl`
**Location**: Lines 505-528 (test section)

Add comprehensive test:
```prolog
test_fold_helper_generator :-
    writeln('=== FOLD HELPER GENERATOR TESTS ==='),

    % Test 1: Generate helpers
    writeln('Test 1: Generate fold helpers for fibonacci'),
    catch(abolish(user:test_fib2/2), _, true),
    assertz(user:(test_fib2(0, 0))),
    assertz(user:(test_fib2(1, 1))),
    assertz(user:(test_fib2(N, F) :- N > 1, N1 is N - 1, N2 is N - 2, test_fib2(N1, F1), test_fib2(N2, F2), F is F1 + F2)),

    ( generate_fold_helpers(test_fib2/2, Clauses) ->
        length(Clauses, NumClauses),
        format('  ✓ Generated ~w clauses~n', [NumClauses])
    ;   writeln('  ✗ FAIL - could not generate helpers')
    ),

    % Test 2: Install and execute (NEW)
    writeln('Test 2: Install and execute generated code'),
    catch(abolish(user:test_fib2_graph/2), _, true),
    catch(abolish(user:fold_test_fib2/2), _, true),
    catch(abolish(user:test_fib2_fold/2), _, true),

    install_fold_helpers(test_fib2/2),

    % Test execution
    ( test_fib2_fold(5, F), F = 5 ->
        writeln('  ✓ PASS - fib_fold(5) = 5')
    ;   writeln('  ✗ FAIL - execution failed')
    ),

    ( test_fib2_fold(10, F10), F10 = 55 ->
        writeln('  ✓ PASS - fib_fold(10) = 55')
    ;   writeln('  ✗ FAIL - fib_fold(10) incorrect')
    ),

    ( test_fib2_graph(3, G), G = node(3, [node(2, [leaf(1), leaf(0)]), leaf(1)]) ->
        writeln('  ✓ PASS - graph structure correct')
    ;   writeln('  ✗ FAIL - graph structure incorrect')
    ),

    writeln('=== FOLD HELPER GENERATOR TESTS COMPLETE ===').
```

Run full test suite:
```bash
$ swipl -g "use_module('src/unifyweaver/core/advanced/fold_helper_generator'), test_fold_helper_generator, halt."
```

### Phase 4: Documentation Update (30 minutes)

#### Update `docs/FOLD_PATTERN_STATUS.md`

Add section after "The Variable Binding Problem":

```markdown
## Solution Implemented

### Variable Mapping Approach

We solved the variable binding problem using explicit variable mapping:

1. **Extract Variables**: Get original variables from clause head
2. **Create Fresh Variables**: Generate new variables for new clause
3. **Build Mapping**: Create pairs `[OrigVar-NewVar, ...]`
4. **Rename Goals**: Apply mapping to all extracted goals

**Implementation**:
- `build_var_mapping/3` - Creates mapping from original to new variables
- `rename_goal_vars/3` - Renames variables in a goal using mapping
- `apply_var_mapping/2` - Recursively applies mapping to compound terms

**Result**: Guards, computations, and operators all use consistent variables that match the new clause head.

### Testing

All tests now pass:
- ✅ Variable mapping unit tests
- ✅ fibonacci end-to-end generation
- ✅ binomial end-to-end generation
- ✅ Integrated test suite

**Status**: Auto-generation COMPLETE ✅
```

#### Update `docs/REVIEW_SUMMARY.md`

Change status from ⚠️ to ✅:
```markdown
### Auto-Generation ✅ COMPLETE

**Problem**: Variable binding issue - FIXED
**Solution**: Variable mapping approach implemented

All generated code now executes correctly.
```

## Validation Checklist

Before considering complete:

- [ ] Variable mapping helpers implemented and tested
- [ ] `extract_guards/3` updated with renaming
- [ ] `extract_computations/3` updated with renaming
- [ ] `extract_template_params/3` passes new variables
- [ ] `generate_from_template/4` uses provided variables
- [ ] Unit test for variable mapping passes
- [ ] fibonacci end-to-end test passes
- [ ] binomial end-to-end test passes
- [ ] Integrated test suite updated and passes
- [ ] Documentation updated (STATUS.md, REVIEW_SUMMARY.md)
- [ ] All regression tests still pass

## Estimated Timeline

| Phase | Task | Time | Cumulative |
|:------|:-----|:-----|:-----------|
| 1.1 | Variable mapping helpers | 30 min | 0.5h |
| 1.2 | Update extract_guards | 15 min | 0.75h |
| 1.3 | Update extract_computations | 20 min | 1h |
| 1.4 | Update extract_template_params | 15 min | 1.25h |
| 1.5 | Update generate_from_template | 10 min | 1.5h |
| 2.1 | Unit tests | 15 min | 1.75h |
| 2.2 | End-to-end tests | 20 min | 2h |
| 2.3 | Binomial test | 10 min | 2.25h |
| 3 | Integration test update | 30 min | 2.75h |
| 4 | Documentation update | 30 min | 3.25h |

**Total**: ~3.25 hours

**Buffer for debugging**: +1 hour = **4.25 hours total**

## Confidence Level

**High (85%)**

**Reasoning**:
- ✅ Clear solution approach (variable mapping is standard technique)
- ✅ Isolated problem (only affects extraction, not template)
- ✅ Good test coverage planned
- ✅ Similar to existing Prolog patterns (copy_term + unification)

**Risks**:
- ⚠️ Compound terms with nested variables (low risk, handled by recursion)
- ⚠️ Edge cases in variable identity checking (low risk, well-tested pattern)

## Alternative Approach (If Needed)

If variable mapping proves difficult, fallback to **string-based templates**:

1. Convert goals to strings
2. Use regex to replace variable names
3. Parse strings back to terms

**Pros**: Simpler logic
**Cons**: Less robust, harder to maintain

**Estimated Fallback Time**: +2 hours

## Next Commits (After Completion)

1. `fix: Implement variable mapping for fold helper generation`
2. `test: Add comprehensive tests for fold auto-generation`
3. `docs: Update status - fold auto-generation complete`

## Questions Before Starting?

- Approach clear?
- Any concerns about variable mapping technique?
- Want to see proof-of-concept for variable mapping first?
- Should we proceed or wait for your review?

---

**Prepared by**: Claude (UnifyWeaver Assistant)
**Confidence**: High (85%)
**Ready to Start**: Yes
