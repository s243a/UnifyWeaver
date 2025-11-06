# Recursion Pattern Theory: Tree vs Linear with Aggregation

## Problem Statement

When a predicate has multiple recursive calls (2+), how do we distinguish between:
1. **True tree recursion** - requires tree-specific compilation (structural decomposition)
2. **Linear recursion with aggregation** - can use linear compilation with memoization

This distinction is critical for correct code generation in UnifyWeaver.

## The Core Insight

**The key is NOT the number of recursive calls, but TWO factors:**
1. **WHAT is passed to those calls** (structural vs scalar)
2. **Whether calls are INDEPENDENT** (can be memoized separately)

### Tree Recursion
Recursive calls receive **structural arguments** - parts of decomposed data structures:
```prolog
tree_sum([V, L, R], Sum) :-
    tree_sum(L, LS),      % L is a tree structure (left subtree)
    tree_sum(R, RS),      % R is a tree structure (right subtree)
    Sum is V + LS + RS.
```

**Characteristics:**
- Input is a structure (tree, list, compound term)
- Structure is **decomposed** into parts (pattern matching `[V,L,R]`)
- Recursive calls receive **those structural parts** (L and R)
- No further recursion needed to aggregate - just combine values

### Linear Recursion with Aggregation
Recursive calls receive **computed scalar values** - not structure parts:
```prolog
fib(N, F) :-
    N > 1,
    N1 is N - 1,         % N1 is a COMPUTED value
    N2 is N - 2,         % N2 is a COMPUTED value
    fib(N1, F1),         % Passing computed scalars
    fib(N2, F2),         % Passing computed scalars
    F is F1 + F2.
```

**Characteristics:**
- Input is a scalar (number, atom)
- Values are **computed** (arithmetic: N-1, N-2)
- Recursive calls receive **computed scalars** (N1 and N2)
- **Recursive calls are INDEPENDENT** - can be computed in any order
- Results are **aggregated/folded** into single value (F1 + F2)
- No structural decomposition
- **Purity requirement**: No side effects, calls don't depend on each other

## Why This Matters for Compilation

### Tree Recursion Compilation Requirements
```bash
tree_sum() {
    local tree="$1"

    # MUST parse tree structure
    parse_tree "$tree" value left right

    # Recurse on structure parts
    local left_result=$(tree_sum "$left")
    local right_result=$(tree_sum "$right")

    # Combine
    echo $((value + left_result + right_result))
}
```

**Requires:**
- Tree structure parser (handle `[V,[L],[R]]` notation)
- Structural pattern matching
- Tree-aware base cases (`[] = empty`)

### Linear with Aggregation Compilation Requirements
```bash
fib() {
    local n="$1"

    # Check memo table
    if [[ -n "${MEMO[$n]}" ]]; then
        echo "${MEMO[$n]}"
        return
    fi

    # Base cases (scalars)
    [[ $n -eq 0 ]] && echo 0 && return
    [[ $n -eq 1 ]] && echo 1 && return

    # Compute arguments (NOT structure decomposition)
    local n1=$((n - 1))
    local n2=$((n - 2))

    # Recurse and aggregate
    local f1=$(fib "$n1")
    local f2=$(fib "$n2")
    local result=$((f1 + f2))

    # Memoize
    MEMO[$n]=$result
    echo $result
}
```

**Requires:**
- Memoization (critical for performance with multiple calls)
- Scalar arithmetic
- Simple base case checks
- NO structure parsing

## Detection Algorithm

### Current (Incorrect) Detection
```prolog
is_tree_recursive(Pred/Arity) :-
    % Has 2+ recursive calls
    count_recursive_calls >= 2.  % ← TOO SIMPLE!
```

**Problem:** Fibonacci has 2 recursive calls, so it's detected as tree recursion, but it's NOT structural recursion.

### Proposed (Correct) Detection
```prolog
is_tree_recursive(Pred/Arity) :-
    % Has 2+ recursive calls
    count_recursive_calls >= 2,

    % AND recursive calls receive structural arguments
    has_structural_recursive_calls(Pred/Arity).
```

**Check for structural arguments:**
1. Extract recursive call arguments from body
2. Determine if arguments are:
   - **Variables bound to structure parts** (from pattern matching head) → Tree recursion
   - **Computed expressions** (from `is` goals) → Linear with aggregation

## The Independence/Purity Requirement

### Why Linear Recursion Works for Multiple Calls

Linear recursion with memoization can handle patterns with multiple recursive calls **IF AND ONLY IF** the calls are independent:

```prolog
% INDEPENDENT calls - works with linear + memo
fib(N, F) :-
    N1 is N - 1,
    N2 is N - 2,
    fib(N1, F1),    % Independent - just a lookup
    fib(N2, F2),    % Independent - just a lookup
    F is F1 + F2.   % Combine after both computed
```

**Why this works:**
- `fib(N1, F1)` and `fib(N2, F2)` can be memoized separately
- N1 and N2 are determined BEFORE any recursive calls
- F1 and F2 are just looked up values
- No call depends on another call's result
- Order doesn't matter: could compute F2 before F1
- **Pure computation**: No side effects, deterministic

### When Independence Breaks

```prolog
% DEPENDENT calls - would NOT work with simple linear
dependent_process(N, Result) :-
    process(N, R1),
    UseR1 is R1 + 1,        % Compute based on R1
    process(UseR1, R2),     % R2's INPUT depends on R1's OUTPUT!
    Result is R1 + R2.
```

**Why this fails independence:**
- Second call's argument (UseR1) depends on first call's result (R1)
- Calls must execute sequentially: R1 THEN R2
- Cannot memoize independently - R2's key depends on R1's value
- **Sequential dependency** breaks the independence assumption

### Purity Levels in UnifyWeaver

Different compilation strategies have different purity requirements:

**1. Stream Compilation (Strictest Purity)**
- No recursion at all
- Pure data transformation
- Pipeable operations
- Example: `parent(X,Y)` facts

**2. Linear Recursion with Aggregation (Moderate Purity)**
- Multiple recursive calls allowed
- Calls must be **independent** (parallel-safe)
- No side effects
- Pure functions with memoization
- Example: `fib(N, F)` - all calls independent

**3. Tree Recursion (Structural Purity)**
- Operates on structures, not necessarily pure
- Calls naturally independent (different subtrees)
- No requirement for computational purity
- Example: `tree_sum([V,L,R], Sum)` - L and R are independent subtrees

### Testing for Independence

**Independent recursive calls satisfy:**
1. **Arguments determined before calls** - all recursive call arguments computed/known before any call
2. **No data flow between calls** - result of one call not used as input to another
3. **Commutative computation** - could compute in any order, result same
4. **Pure aggregation** - results only combined after all calls complete

```prolog
% TEST: Are recursive calls independent?

% Case 1: INDEPENDENT ✓
fib(N, F) :-
    N1 is N - 1,      % Arguments determined BEFORE calls
    N2 is N - 2,
    fib(N1, F1),      % Call 1
    fib(N2, F2),      % Call 2 - doesn't use F1
    F is F1 + F2.     % Aggregation AFTER both calls

% Case 2: DEPENDENT ✗
bad_example(N, R) :-
    bad_example(N-1, R1),
    X is R1 + 1,            % Compute from R1
    bad_example(X, R2),     % R2 input DEPENDS on R1 output
    R is R1 + R2.

% Case 3: INDEPENDENT (3 calls) ✓
trib(N, T) :-
    N1 is N - 1,      % All args determined BEFORE calls
    N2 is N - 2,
    N3 is N - 3,
    trib(N1, T1),     % Independent
    trib(N2, T2),     % Independent
    trib(N3, T3),     % Independent
    T is T1 + T2 + T3. % Aggregation AFTER
```

## Pattern Recognition

### Structural Argument Indicators
Arguments to recursive calls are tree recursion if they are:
- **Pattern-matched from head**: `tree_sum([V,L,R], Sum) :- tree_sum(L, ...)`
  - L comes from destructuring `[V,L,R]`
- **List/compound term patterns**: `[_|Rest]`, `node(Left, Right)`
- **Directly from input structure decomposition**

### Scalar Argument Indicators
Arguments are linear aggregation if they are:
- **Computed via arithmetic**: `N1 is N - 1`, then `fib(N1, ...)`
- **Simple variables from scalar inputs**: Input is number/atom, not structure
- **Results of `is` expressions**: `X is Y + 1`

## Examples Analysis

### Example 1: Binary Tree Sum (TRUE Tree Recursion)
```prolog
tree_sum([], 0).
tree_sum([V, L, R], Sum) :-
    tree_sum(L, LS),           % L from pattern match [V,L,R]
    tree_sum(R, RS),           % R from pattern match [V,L,R]
    Sum is V + LS + RS.
```

**Analysis:**
- Head pattern: `[V, L, R]` - destructures tree structure
- Recursive calls: `tree_sum(L, LS)` and `tree_sum(R, RS)`
- Arguments L and R: **Bound from head pattern matching** ✓
- **Verdict: TRUE tree recursion**

### Example 2: Fibonacci (LINEAR with Aggregation)
```prolog
fib(0, 0).
fib(1, 1).
fib(N, F) :-
    N > 1,
    N1 is N - 1,              % N1 COMPUTED
    N2 is N - 2,              % N2 COMPUTED
    fib(N1, F1),              % Passing computed scalars
    fib(N2, F2),              % Passing computed scalars
    F is F1 + F2.             % Aggregation/folding
```

**Analysis:**
- Head pattern: `fib(N, F)` - N is a scalar (number)
- Recursive calls: `fib(N1, F1)` and `fib(N2, F2)`
- Arguments N1 and N2: **Computed via arithmetic** (`is` expressions) ✗
- **Verdict: NOT tree recursion - this is linear with aggregation**

### Example 3: List Processing (TRUE Tree Recursion)
```prolog
process_list([], []).
process_list([H|T], [H2|T2]) :-
    process_item(H, H2),
    process_list(T, T2).
```

**Analysis:**
- Head pattern: `[H|T]` - destructures list structure
- Recursive call: `process_list(T, T2)`
- Argument T: **Bound from head pattern matching** ✓
- Only 1 recursive call, so not tree recursion by count, but IS structural
- **Verdict: This is actually LINEAR structural recursion (tail recursion)**

### Example 4: Tribonacci (LINEAR with Aggregation)
```prolog
trib(0, 0).
trib(1, 1).
trib(2, 1).
trib(N, T) :-
    N > 2,
    N1 is N - 1,
    N2 is N - 2,
    N3 is N - 3,
    trib(N1, T1),
    trib(N2, T2),
    trib(N3, T3),
    T is T1 + T2 + T3.
```

**Analysis:**
- 3 recursive calls (more than fibonacci!)
- Arguments N1, N2, N3: **All computed via arithmetic** ✗
- Results aggregated: `T is T1 + T2 + T3`
- **Verdict: NOT tree recursion - linear with aggregation (3 lookups)**

## The Distinguishing Test

**Question to ask:** Where do the arguments to recursive calls come from?

1. **From pattern matching the input structure** (head unification with compound terms)
   → **TREE RECURSION** - needs structural compilation

2. **From arithmetic/computation on scalar inputs** (`is` expressions)
   → **LINEAR WITH AGGREGATION** - needs memoization, not structure parsing

## Implementation Strategy

### Pattern Detection Predicate
```prolog
has_structural_recursive_calls(Pred/Arity) :-
    functor(Head, Pred, Arity),
    clause(Head, Body),

    % Get recursive calls from body
    findall(RecCall, (
        extract_goal_from_body(Body, RecCall),
        functor(RecCall, Pred, Arity)
    ), RecCalls),

    % Check if arguments come from head pattern matching
    % (not from 'is' expressions)
    forall(member(RecCall, RecCalls),
        args_from_pattern_match(RecCall, Head, Body)
    ).

args_from_pattern_match(RecCall, Head, Body) :-
    % Get arguments of recursive call
    RecCall =.. [_|RecArgs],

    % Check each argument
    forall(member(Arg, RecArgs), (
        % Arg should be bound from Head pattern, not from 'is'
        (   var(Arg) -> fail  % Free variable means computed
        ;   bound_in_head(Arg, Head) -> true
        ;   \+ computed_in_body(Arg, Body)
        )
    )).

bound_in_head(Arg, Head) :-
    % Check if Arg appears in a structured position in Head
    Head =.. [_|HeadArgs],
    member(HeadArg, HeadArgs),
    (   HeadArg == Arg
    ;   compound(HeadArg), sub_term(Arg, HeadArg)
    ).

computed_in_body(Arg, Body) :-
    % Check if Arg is result of 'is' expression
    extract_goal_from_body(Body, (Arg is _)).
```

## Compilation Routing

### Decision Tree
```
Has multiple recursive calls (2+)?
├─ NO → Linear or Tail recursion
└─ YES → Check argument types
    ├─ Structural (from pattern match) → TREE RECURSION
    │   ├─ Generate parse_tree helper
    │   ├─ Structure-aware base cases
    │   └─ Structural decomposition code
    │
    └─ Scalar (from computation) → LINEAR WITH AGGREGATION
        ├─ Generate memoization table
        ├─ Scalar base cases
        └─ Arithmetic computation code
```

## Test Cases

### Should Detect as Tree Recursion
- ✓ `tree_sum([V,L,R], Sum)` - binary tree operations
- ✓ `list_append([H|T], L2, [H|R])` - list structural recursion
- ✓ `flatten_tree(node(L,R), Flat)` - compound term decomposition

### Should NOT Detect as Tree Recursion
- ✗ `fib(N, F)` with `N1 is N-1, N2 is N-2` - computed scalars
- ✗ `ackermann(M, N, A)` with arithmetic - scalar computation
- ✗ `tribonacci(N, T)` - even with 3 calls, still scalar

## Conclusion

The classification of recursion patterns depends on THREE key factors:

### 1. Structural vs Scalar Arguments
- **Tree recursion**: Recursive calls receive **structural parts** from pattern matching
  - Example: `tree_sum(L, LS)` where L is from `[V,L,R]`
- **Linear with aggregation**: Recursive calls receive **computed scalar values**
  - Example: `fib(N1, F1)` where N1 is from `N-1`

### 2. Independence of Recursive Calls
- **Independent calls** can use linear recursion with memoization
  - All arguments determined BEFORE any recursive calls
  - No data flow between calls (result of one not used as input to another)
  - Can compute in any order (commutative)
  - Pure aggregation after all calls complete

- **Dependent calls** require different handling
  - One call's result feeds into another call's argument
  - Sequential execution required
  - Cannot parallelize or memoize independently

### 3. Purity Requirements

**Stream (strictest)** → No recursion, pure transformation
**Linear with aggregation** → Independent calls, no side effects, memoizable
**Tree recursion** → Structural independence (subtrees naturally independent)

### Decision Criteria

For predicates with 2+ recursive calls:

```
IF recursive calls pass STRUCTURES (from pattern matching)
  → TREE RECURSION
  → Generate: structure parser, decomposition code

ELSE IF recursive calls pass SCALARS (from computation)
  AND calls are INDEPENDENT
  → LINEAR RECURSION WITH AGGREGATION
  → Generate: memoization table, scalar arithmetic

ELSE IF calls are DEPENDENT
  → (Not yet supported in v1.0)
  → Fall back to basic recursion or error
```

### Key Insight

**Fibonacci is NOT tree recursion because:**
1. Arguments are computed scalars (N-1, N-2), not structural parts ✗
2. Calls ARE independent (N1 and N2 determined before calls) ✓
3. Pure aggregation (F1 + F2 after both calls) ✓

**Therefore:** Fibonacci should compile as **linear recursion with memoization**, not tree recursion.

The current issue is that pattern detection only checks for "2+ calls" but doesn't verify structural arguments. This causes fibonacci to be misclassified as tree recursion, generating incorrect code with structure parsers instead of memoization tables.

## Open Questions for Implementation

### Q1: How to reliably detect structural vs scalar arguments?

**Current approach idea:**
```prolog
% Check if argument comes from head pattern matching
args_from_pattern_match(RecCall, Head, Body) :-
    RecCall =.. [_|RecArgs],
    forall(member(Arg, RecArgs), (
        bound_in_head(Arg, Head)  % From pattern match
        ; \+ computed_in_body(Arg, Body)  % Not from 'is'
    )).
```

**Concerns:**
- What about variables that are neither in head nor computed? (intermediate variables)
- What if a structural argument is assigned to a variable first?
  ```prolog
  tree_sum([V,L,R], Sum) :-
      Left = L,              % Intermediate variable
      tree_sum(Left, LS),    % Still structural!
      ...
  ```
- Should we trace variable bindings through the body?

**Question:** Is it sufficient to check:
1. Variable appears in compound term in head (structural) ✓
2. Variable is result of `is` expression (scalar) ✓
3. Everything else → assume scalar (safe fallback)?

### Q2: What about mixed patterns?

What if a predicate has BOTH structural and scalar recursive calls?

```prolog
mixed([], N, Result) :- Result is N.
mixed([H|T], N, Result) :-
    N1 is N + 1,
    mixed(T, N1, R1),    % Scalar argument N1
    mixed(T, N, R2),     % Structural argument T
    Result is R1 + R2.
```

**Question:** How should this be classified?
- Tree recursion (has structural arg)?
- Linear recursion (has scalar arg)?
- Hybrid (not supported)?
- Error (inconsistent pattern)?

### Q3: Independence detection - how deep should we check?

For independence, we need to verify that recursive call arguments are computed BEFORE any recursive calls.

**Simple case (easy to detect):**
```prolog
fib(N, F) :-
    N1 is N - 1,    % Computed BEFORE calls
    N2 is N - 2,    % Computed BEFORE calls
    fib(N1, F1),    % Calls use pre-computed values
    fib(N2, F2),
    F is F1 + F2.
```

**Complex case (harder to detect):**
```prolog
complex(N, R) :-
    helper(N, X),        % Some helper computation
    N1 is X - 1,        % N1 depends on X
    complex(N1, R1),    % Is N1 "pre-computed"?
    ...
```

**Question:** Should we:
1. Only check direct `is` expressions before recursive calls?
2. Track all variable bindings through arbitrary goals?
3. Be conservative - require `is` expressions to be immediately before recursive calls?

### Q4: What about the linear recursion compiler - does it handle multiple calls?

Looking at `linear_recursion.pl`, does it currently support multiple recursive calls with memoization?

**Current check in pattern_matchers.pl:**
```prolog
is_linear_recursive_streamable(Pred/Arity) :-
    ...
    count_recursive_calls(Body, Pred, Count),
    Count =:= 1  % Exactly 1 call!
```

**This means:** Linear recursion currently ONLY accepts 1 recursive call!

**Question:** Do we need to:
1. Extend linear recursion compiler to handle multiple calls with memoization?
2. Create a new "aggregation" pattern compiler for independent multi-call patterns?
3. Keep linear as-is and route fibonacci elsewhere?

### Q5: Memoization strategy for multiple calls

If we extend linear recursion for multiple calls, how should memoization work?

```bash
fib() {
    local n="$1"

    # Check memo ONCE per call
    if [[ -n "${MEMO[$n]}" ]]; then
        echo "${MEMO[$n]}"
        return
    fi

    # Multiple recursive calls
    local f1=$(fib "$n1")  # Will check MEMO[$n1]
    local f2=$(fib "$n2")  # Will check MEMO[$n2]

    # Aggregate and memoize THIS call's result
    local result=$((f1 + f2))
    MEMO[$n]=$result
    echo $result
}
```

This should work! Each call memoizes independently.

**Question:** Is this the correct approach, or do we need special handling?

### Q6: What's the priority order for pattern matching?

Current order: Tail → Linear → Tree → Mutual

**If we fix tree recursion detection:**
- Fibonacci won't match Tree (scalar args, not structural)
- Will it fall through to Mutual? (has 2+ calls in SCC)
- Or should it match an extended Linear pattern?

**Question:** What should the new priority be?
1. Tail → Linear (1 call) → Linear-Aggregation (2+ independent scalar) → Tree (structural) → Mutual?
2. Or keep order but fix detection logic?

### Q7: Testing strategy

How do we verify the fix works correctly?

**Test cases needed:**
1. Fibonacci → should compile with memoization, NOT parse_tree
2. Tree_sum → should compile with parse_tree, NOT scalar memo
3. Tribonacci (3 calls) → should work like fibonacci
4. Mixed structural/scalar → should error or route correctly

**Question:** Should we add explicit test predicates for each case?

## Implementation Plan (Pending Question Resolution)

**Option A: Minimal Fix (Conservative)**
1. Make tree recursion detection stricter (require structural arguments)
2. Let fibonacci fall through to basic recursion (existing fallback)
3. Accept suboptimal performance for now

**Option B: Extend Linear Recursion**
1. Relax linear recursion to accept 2+ calls if independent
2. Update pattern detection to check independence
3. Ensure memoization works for multiple calls

**Option C: New Aggregation Pattern**
1. Create separate "aggregation_recursion.pl" module
2. Handle independent multi-call patterns explicitly
3. Insert in priority: Tail → Linear → Aggregation → Tree → Mutual

**Which approach should we take?**

## Variable Independence Analysis (Key Insight)

### The Conjunction Pattern

**Linear recursion works when the body is a series of conjunctions (AND) with unique variables:**

```prolog
% WORKS - unique variables in conjunction
fib(N, F) :-
    N1 is N - 1,      % Binds N1 (unique)
    N2 is N - 2,      % Binds N2 (unique, no dependency on N1)
    fib(N1, F1),      % Binds F1 (unique)
    fib(N2, F2),      % Binds F2 (unique, no dependency on F1)
    F is F1 + F2.     % Uses F1, F2 (but doesn't feed into recursive calls)
```

**Key characteristic:** Each goal in the conjunction binds NEW variables or uses already-bound ones, but doesn't create data flow between recursive calls.

### Shared Variables Require Different Handling

**When clauses share variables, we need streaming or tree recursion:**

```prolog
% Streaming (non-recursive, shared variables)
grandparent(X, Z) :-
    parent(X, Y),     % Binds Y
    parent(Y, Z).     % Uses Y (shared variable)
% This creates a data pipeline: X → Y → Z
% Streamable because non-recursive
```

```prolog
% Tree recursion (structural sharing)
tree_sum([V, L, R], Sum) :-
    tree_sum(L, LS),  % LS is from subtree L
    tree_sum(R, RS),  % RS is from subtree R
    Sum is V + LS + RS.  % All three values combined
% The structure [V,L,R] naturally partitions the data
```

### AND/OR Patterns (Disjunction)

**If we have OR clauses (multiple clauses for same predicate), each AND branch must follow the linear pattern:**

```prolog
% Multiple clauses - each must be linear
fib(0, 0).           % Clause 1: base case (linear)
fib(1, 1).           % Clause 2: base case (linear)
fib(N, F) :-         % Clause 3: recursive (check if linear)
    N > 1,
    N1 is N - 1,
    N2 is N - 2,
    fib(N1, F1),
    fib(N2, F2),
    F is F1 + F2.
```

**Can use linear recursion IF:**
- Each clause (OR branch) is independent
- Within each clause, conjunctions have unique variables
- No data flow between recursive calls in same clause

### Order Independence Check

**Linear recursion assumes goals can execute in sequence without order dependency:**

```prolog
% Order independent ✓
fib(N, F) :-
    N1 is N - 1,    % Can compute N1
    N2 is N - 2,    % Can compute N2 (doesn't need N1)
    fib(N1, F1),    % Can call in any order
    fib(N2, F2),    % Can call in any order
    F is F1 + F2.   % Aggregation after both

% Order DEPENDENT ✗
bad(N, R) :-
    bad(N-1, R1),
    X is R1 + 1,    % DEPENDS on R1 (from previous call)
    bad(X, R2),     % This call DEPENDS on previous result
    R is R1 + R2.
```

### Side Effects and Annotations

**Idea:** Use predicates/annotations to explicitly mark non-linear patterns:

```prolog
:- pragma(side_effects, has_side_effects/1).

has_side_effects(X) :-
    write(X),        % Side effect!
    process(X, Y),
    Y > 0.
```

**Or detect side effects:**
- Uses `write`, `writeln`, `format` (I/O)
- Uses `assert`, `retract` (database modification)
- Uses `!` (cut) in ways that affect order

### The Decision Rule

For a predicate with multiple recursive calls to be **linear recursion**:

1. **Body is conjunction** (series of AND goals) ✓
2. **Variables are unique** (each goal binds new vars or uses existing, no inter-call data flow) ✓
3. **Order independent** (recursive call arguments computed before any call) ✓
4. **No side effects** (pure computation) ✓

**Note:** We don't need a separate "non-structural arguments" check because:
- Structural arguments (from pattern matching like [V,L,R]) would create variable sharing
- Variable sharing violates criterion 2 (unique variables)
- So structural patterns automatically fail the linear test and route to tree recursion

If all these hold → **Linear recursion with memoization** (even with 2+ calls!)

If any fail → Route to appropriate pattern:
- Structural arguments (fails variable uniqueness) → Tree recursion
- Side effects → Basic recursion (no optimization)
- Order dependent → Basic recursion or error

### Examples Re-analyzed

**Fibonacci:**
1. Body is conjunction ✓ (`N1 is ..., N2 is ..., fib(...), fib(...), F is ...`)
2. Variables unique ✓ (N1, N2, F1, F2 all distinct)
3. Order independent ✓ (N1, N2 computed before calls)
4. No side effects ✓ (pure arithmetic)
5. Each clause independent ✓ (base cases simple, recursive case clean)

**Verdict: LINEAR RECURSION** ✓

**Tree Sum:**
1. Body is conjunction ✓
2. Variables from structure ⚠️ (L, R from [V,L,R])
3. Order independent ✓ (can compute L, R in any order)
4. No side effects ✓
5. Structural arguments → **TREE RECURSION**

**Verdict: TREE RECURSION** (because of structural decomposition)

**Grandparent:**
1. Body is conjunction ✓
2. Variables shared ⚠️ (Y used in both parent calls)
3. Order dependent ⚠️ (second call needs Y from first)
4. No side effects ✓
5. **Non-recursive** → **STREAMING**

**Verdict: STREAMING** (transitive closure, BFS optimization)

### Implementation Implications

**For linear recursion pattern matcher:**
```prolog
is_linear_recursive_streamable(Pred/Arity) :-
    % Get all clauses
    functor(Head, Pred, Arity),
    findall(clause(Head, Body), clause(Head, Body), Clauses),

    % Separate base and recursive cases
    partition(is_recursive_for_pred(Pred), Clauses, RecClauses, BaseClauses),
    RecClauses \= [],
    BaseClauses \= [],

    % For EACH recursive clause, verify linear pattern
    forall(member(clause(RecHead, RecBody), RecClauses), (
        % 1. Body is conjunction (implicitly checked by following tests)

        % 2. Variables are unique (no inter-call data flow)
        has_unique_variables_in_recursion(RecBody, Pred),

        % 3. Order independent (recursive args pre-computed)
        recursive_args_precomputed(RecBody, Pred),

        % 4. No side effects (for now, assumed - future enhancement)
        true  % Could add: \+ has_side_effects(RecBody)
    )).

% Helper: Check that recursive calls don't share variables in a dependent way
has_unique_variables_in_recursion(Body, Pred) :-
    % Get all recursive calls
    findall(RecCall, (
        extract_goal_from_body(Body, RecCall),
        functor(RecCall, Pred, _)
    ), RecCalls),

    % Check that no recursive call result is used as input to another
    \+ has_inter_call_dependency(Body, RecCalls, Pred).

% Helper: Check if recursive call arguments are computed before any recursive calls
recursive_args_precomputed(Body, Pred) :-
    % Extract goals in order
    extract_goals_in_order(Body, Goals),

    % Find first recursive call position
    nth0(FirstRecPos, Goals, FirstRec),
    functor(FirstRec, Pred, _),

    % All recursive call arguments should be bound before first recursive call
    findall(RecCall, (
        member(RecCall, Goals),
        functor(RecCall, Pred, _)
    ), RecCalls),

    forall(member(RecCall, RecCalls), (
        RecCall =.. [_|Args],
        forall(member(Arg, Args), (
            % Arg should be bound by goals before FirstRecPos
            is_bound_before_position(Arg, Goals, FirstRecPos)
        ))
    )).
```

This extends linear recursion to handle **multiple recursive calls** when they meet the independence criteria!

## Forbidding Linear Recursion Compilation

### Use Case: Graph Recursion with Helper Functions

Sometimes a predicate matches the linear recursion pattern (independent calls, computed scalars) but should be compiled differently for other reasons:

1. **Ordered Constraints**: Predicate has `unordered=false` (results must maintain order)
2. **Graph Recursion**: Need structural traversal with fold helper
3. **Side Effects**: Predicate has I/O or state modification
4. **Testing**: Want to force a specific compilation strategy

### The `forbid_linear_recursion` System

**API:**
```prolog
% Mark predicate as forbidden for linear recursion
forbid_linear_recursion(fibonacci/2).

% Check if predicate is forbidden
is_forbidden_linear_recursion(fibonacci/2).

% Remove forbid (allow linear recursion again)
clear_linear_recursion_forbid(fibonacci/2).
```

**Automatic Forbidding via Constraints:**
```prolog
% Declare constraint
:- constraint(my_pred/2, [unique, ordered]).

% This automatically forbids linear recursion because:
% - ordered=true means unordered=false
% - unordered=false means order matters
% - Linear recursion with memoization may not preserve order
```

**Integration with Pattern Detection:**
```prolog
is_linear_recursive_streamable(Pred/Arity) :-
    % Check if forbidden FIRST (fast fail)
    \+ is_forbidden_linear_recursion(Pred/Arity),

    % Then check pattern criteria...
    ...
```

### Example: Fibonacci with Graph Recursion

**Scenario:** You want fibonacci to use graph recursion (building dependency graph) instead of linear recursion (memoization).

**Approach:**
```prolog
% 1. Define fibonacci normally
fib(0, 0).
fib(1, 1).
fib(N, F) :-
    N > 1,
    N1 is N - 1,
    N2 is N - 2,
    fib(N1, F1),
    fib(N2, F2),
    F is F1 + F2.

% 2. Forbid linear recursion for fibonacci
:- forbid_linear_recursion(fib/2).

% 3. Now fibonacci won't match linear pattern
% It will fall through to tree or mutual recursion

% 4. Create wrapper that folds graph results
fib_folded(N, F) :-
    fib_graph(N, Graph),  % Graph recursion builds structure
    fold_graph(Graph, F).  % Fold helper aggregates to value
```

**Why This Works:**
- Graph recursion builds the dependency structure
- Fold helper traverses structure and aggregates
- Main function provides clean interface
- Forbid ensures correct compilation strategy

### Constraint-Based Forbidding

**Ordered predicates are automatically forbidden:**
```prolog
% This declaration automatically forbids linear recursion
:- constraint(temporal_query/2, [unique, ordered]).

% Because:
is_forbidden_linear_recursion(temporal_query/2) :-
    get_constraints(temporal_query/2, Constraints),
    member(unordered(false), Constraints).  % ordered=true
```

**Rationale:**
- Linear recursion with memoization doesn't guarantee order
- Memoization returns cached results in arbitrary order
- Ordered constraints require sequential processing
- Therefore: ordered → forbid linear

### Testing the Forbid System

**Test case from `pattern_matchers.pl`:**
```prolog
% Fibonacci normally matches linear pattern
?- is_linear_recursive_streamable(fib/2).
true.

% Forbid it
?- forbid_linear_recursion(fib/2).
true.

% Now it doesn't match
?- is_linear_recursive_streamable(fib/2).
false.

% Check forbid status
?- is_forbidden_linear_recursion(fib/2).
true.

% Unforbid
?- clear_linear_recursion_forbid(fib/2).
true.

% Matches again
?- is_linear_recursive_streamable(fib/2).
true.
```

### Use Cases Summary

**1. Graph Recursion Pattern:**
- Forbid linear recursion for predicates that should use graph traversal
- Main predicate builds dependency graph
- Helper predicate folds graph to value
- Clean separation of concerns

**2. Ordered Results:**
- Automatically forbidden via constraint system
- Ensures sequential processing preserves order
- No manual forbid needed

**3. Side Effects:**
- Manual forbid for predicates with I/O or state
- Forces basic recursion (no memoization assumptions)
- Explicit marking makes intent clear

**4. Testing Different Strategies:**
- Temporarily forbid to test alternative compilation
- Compare performance of different strategies
- Easy to toggle for benchmarking

## Final Implementation Plan

**Phase 1: Extend Linear Recursion Pattern Matcher**
1. Modify `is_linear_recursive_streamable/1` in `pattern_matchers.pl`
2. Remove "exactly 1 call" restriction
3. Add checks for variable uniqueness and order independence
4. Keep memoization (already handles multiple calls correctly)

**Phase 2: Fix Tree Recursion Detection**
1. Keep tree recursion for structural decomposition patterns only
2. Tree recursion naturally has independent subtrees (L, R from [V,L,R])
3. No changes needed to tree recursion code generation

**Phase 3: Test**
1. Fibonacci → should compile as linear recursion with memo
2. Tree_sum → should compile as tree recursion with parse_tree
3. Tribonacci → should compile as linear recursion with memo

**Result:** Fibonacci gets memoization (correct), not parse_tree (incorrect)
