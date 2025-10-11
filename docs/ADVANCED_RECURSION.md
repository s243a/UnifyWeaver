<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2025 John William Creighton (@s243a)
-->
# Advanced Recursion Compilation

## Overview

The `advanced/` module extends UnifyWeaver's basic recursion support with sophisticated pattern detection and compilation strategies. It handles tail recursion, linear recursion, and mutual recursion through SCC (Strongly Connected Components) analysis.

## Architecture

### Design Philosophy

**Priority-Based Compilation**: Try simpler patterns before complex ones
1. **Tail recursion** → Compile to iterative loops
2. **Linear recursion** → Compile with memoization (handles 1+ independent calls)
3. **Tree recursion** → Compile structural decomposition patterns
4. **Mutual recursion** → Detect via SCC, compile with shared memo tables

**Separation of Concerns**: Advanced patterns isolated from basic compiler
- Basic recursion (transitive closures) stays in `recursive_compiler.pl`
- Advanced patterns live in `src/unifyweaver/core/advanced/`
- Minimal coupling via optional hook

### Module Structure

```
src/unifyweaver/core/advanced/
├── advanced_recursive_compiler.pl   # Orchestrator (main entry point)
├── call_graph.pl                   # Build predicate dependency graphs
├── scc_detection.pl                # Tarjan's SCC algorithm
├── pattern_matchers.pl             # Pattern detection utilities
├── tail_recursion.pl               # Tail recursion → loop compiler
├── linear_recursion.pl             # Linear recursion → memoized compiler
├── tree_recursion.pl               # Tree recursion → structural compiler
├── mutual_recursion.pl             # Mutual recursion → joint memo compiler
└── test_advanced.pl                # Comprehensive test suite
```

## Pattern Detection

### Tail Recursion

**Pattern**: Recursive call is the last operation in a clause

```prolog
% Accumulator pattern (arity 3)
count([], Acc, Acc).
count([_|T], Acc, N) :-
    Acc1 is Acc + 1,
    count(T, Acc1, N).
```

**Compilation Strategy**: Convert to bash `for` loop with accumulator variable

**Detected by**: `pattern_matchers:is_tail_recursive_accumulator/2`

### Linear Recursion

**Pattern**: One OR MORE recursive calls per clause, with independent calls

**Key Requirements:**
1. Recursive calls are independent (no inter-call data flow)
2. Arguments are pre-computed (order independent)
3. No structural decomposition in head (not `[V,L,R]` patterns)
4. Pure computation (no side effects)

**Examples:**

*Single recursive call (classic linear):*
```prolog
length([], 0).
length([_|T], N) :-
    length(T, N1),
    N is N1 + 1.
```

*Multiple recursive calls (fibonacci-style):*
```prolog
fib(0, 0).
fib(1, 1).
fib(N, F) :-
    N > 1,
    N1 is N - 1,      % Pre-computed
    N2 is N - 2,      % Pre-computed
    fib(N1, F1),      % Independent call
    fib(N2, F2),      % Independent call
    F is F1 + F2.     % Aggregation
```

**Compilation Strategy**: Memoization with associative arrays

**Detected by**: `pattern_matchers:is_linear_recursive_streamable/1`

**See also**: `docs/RECURSION_PATTERN_THEORY.md` for detailed theory on variable independence

### Tree Recursion

**Pattern**: Structural decomposition with recursive calls on structure parts

**Key Requirements:**
1. Head uses structural pattern (e.g., `[V,L,R]` for trees)
2. Recursive calls on decomposed parts (L and R)
3. Natural independence (different subtrees)

**Example:**
```prolog
tree_sum([], 0).
tree_sum([V, L, R], Sum) :-
    tree_sum(L, LS),      % Recurse on left subtree
    tree_sum(R, RS),      % Recurse on right subtree
    Sum is V + LS + RS.   % Combine
```

**Compilation Strategy**: Generate tree parser and recursive structure handling

**Detected by**: `tree_recursion:is_tree_recursive/1`

**Note:** Tree recursion is for STRUCTURAL patterns. Fibonacci-style patterns with multiple calls on SCALARS compile as linear recursion with aggregation (see above).

### Mutual Recursion

**Pattern**: Multiple predicates calling each other in a cycle

```prolog
is_even(0).
is_even(N) :- N > 0, N1 is N - 1, is_odd(N1).

is_odd(1).
is_odd(N) :- N > 1, N1 is N - 1, is_even(N1).
```

**Compilation Strategy**:
1. Build call graph
2. Detect SCCs using Tarjan's algorithm
3. Compile SCC members with shared memo table

**Detected by**: `scc_detection:find_sccs/2`

## API

### Main Entry Points

```prolog
% Compile single predicate with advanced patterns
advanced_recursive_compiler:compile_advanced_recursive(+Pred/Arity, +Options, -BashCode)

% Compile group of predicates (for explicit mutual recursion)
advanced_recursive_compiler:compile_predicate_group(+[Pred/Arity, ...], +Options, -BashCode)
```

### Options and Constraints

All advanced compilers accept an `Options` parameter for configuration:

```prolog
% All compilers support Options
compile_tail_recursion(Pred/Arity, Options, BashCode).
compile_linear_recursion(Pred/Arity, Options, BashCode).
compile_mutual_recursion(Predicates, Options, BashCode).
```

**Options Format:** List of `Key=Value` pairs
- Example: `[unique(true), ordered=false, output_lang=bash]`

**Constraint Integration:**
- Compilers query `constraint_analyzer:get_constraints/2` for predicate constraints.
- Runtime options are merged with these constraints.
- The `tail_recursion` compiler now uses the `unique(true)` constraint to generate more efficient code. When this constraint is present, an `exit 0` is added to the generated bash script, causing it to terminate immediately after producing its single result.
- Other compilers do not yet act on constraints, but the plumbing is in place for future enhancements.

### Integration Hook

`recursive_compiler.pl` automatically tries advanced patterns before falling back to basic memoization:

```prolog
compile_recursive(Pred/Arity, Options, BashCode) :-
    % ... try basic patterns ...
    ;
        catch(
            advanced_recursive_compiler:compile_advanced_recursive(
                Pred/Arity, Options, BashCode
            ),
            error(existence_error(procedure, _), _),
            fail
        ) ->
        true
    ;
        % Fall back to basic memoization
        compile_memoized_recursion(Pred, Arity, Options, BashCode)
    ).
```

## SCC Detection Algorithm

Uses **Tarjan's algorithm** for finding strongly connected components:

1. **Build call graph**: Extract all predicate calls
2. **DFS traversal**: Visit nodes, track index/lowlink
3. **Identify SCCs**: When `index = lowlink`, pop SCC from stack
4. **Topological order**: SCCs ordered by dependencies

**Time Complexity**: O(V + E) where V = predicates, E = calls

**Implementation**: `scc_detection.pl`

## Code Generation

### Template Style

Uses **list-of-strings** for bash templates (better readability):

```prolog
TemplateLines = [
    "#!/bin/bash",
    "# {{pred}} - tail recursive",
    "{{pred}}() {",
    "    echo 'Hello'",
    "}"
],
atomic_list_concat(TemplateLines, '\n', Template),
render_template(Template, [pred=PredStr], BashCode).
```

**Benefits**:
- No escape sequence confusion
- Each line clearly visible
- Better syntax highlighting

### Generated Code Structure

**Tail Recursion Example**:
```bash
#!/bin/bash
# count_items - tail recursive accumulator pattern

count_items() {
    local input="$1"
    local acc="$2"

    # Convert to array, iterate with accumulator
    for item in "${items[@]}"; do
        acc=$((acc + 1))
    done

    echo "$acc"
}
```

**Mutual Recursion Example**:
```bash
#!/bin/bash
# Mutually recursive group: is_even_is_odd
declare -gA is_even_is_odd_memo

is_even() {
    local key="is_even:$*"
    [[ -n "${is_even_is_odd_memo[$key]}" ]] && echo "${is_even_is_odd_memo[$key]}" && return
    # ... base cases, recursive logic ...
    is_even_is_odd_memo["$key"]="$result"
}

is_odd() {
    local key="is_odd:$*"
    [[ -n "${is_even_is_odd_memo[$key]}" ]] && echo "${is_even_is_odd_memo[$key]}" && return
    # ... base cases, recursive logic ...
    is_even_is_odd_memo["$key"]="$result"
}
```

## Real-World Examples

### Example 1: Tail Recursion - Summing a List

**Prolog Source:**
```prolog
sum_list([], Acc, Acc).
sum_list([H|T], Acc, Sum) :-
    Acc1 is Acc + H,
    sum_list(T, Acc1, Sum).
```

**Detection:**
```prolog
?- is_tail_recursive_accumulator(sum_list/3, Info).
Info = acc_pattern([clause(sum_list([], Acc, Acc), true)],
                   [clause(sum_list([H|T], Acc, Sum), (...))],
                   2).  % Accumulator at position 2
```

**Generated Bash Code:**
```bash
#!/bin/bash
# sum_list - tail recursive accumulator pattern
# Compiled to iterative while loop

sum_list() {
    local input="$1"
    local acc="$2"
    local result_var="$3"

    # Convert input to array if it's a list notation
    if [[ "$input" =~ ^\[.*\]$ ]]; then
        input="${input#[}"
        input="${input%]}"
        IFS=',' read -ra items <<< "$input"
    else
        items=()
    fi

    local current_acc="$acc"

    # Iterative loop (tail recursion optimization)
    for item in "${items[@]}"; do
        # Step operation: Acc1 is Acc + H
        current_acc=$((current_acc + item))
    done

    # Return result
    if [[ -n "$result_var" ]]; then
        eval "$result_var=$current_acc"
    else
        echo "$current_acc"
    fi
}

# Helper function for common use case
sum_list_eval() {
    sum_list "$1" 0 result
    echo "$result"
}
```

**Usage:**
```bash
$ source sum_list.sh
$ sum_list "[1,2,3]" 0 result
$ echo $result
6
$ sum_list_eval "[5,10,15]"
30
```

---

### Example 2: Linear Recursion - Factorial

**Prolog Source:**
```prolog
factorial(0, 1).
factorial(N, F) :-
    N > 0,
    N1 is N - 1,
    factorial(N1, F1),
    F is F1 * N.
```

**Detection:**
```prolog
?- is_linear_recursive_streamable(factorial/2).
true.  % Single recursive call (classic linear recursion)
```

**Generated Bash Code:**
```bash
#!/bin/bash
# factorial - linear recursive with memoization
declare -gA factorial_memo

factorial() {
    local n="$1"
    local expected="$2"

    # Memoization check
    local key="${n}"
    if [[ -n "${factorial_memo[$key]}" ]]; then
        local cached="${factorial_memo[$key]}"
        if [[ -n "$expected" ]]; then
            [[ "$cached" == "$expected" ]] && echo "true" || echo "false"
        else
            echo "$cached"
        fi
        return
    fi

    # Base case: factorial(0, 1)
    if [[ "$n" == "0" ]]; then
        factorial_memo["$key"]="1"
        if [[ -n "$expected" ]]; then
            [[ "1" == "$expected" ]] && echo "true" || echo "false"
        else
            echo "1"
        fi
        return
    fi

    # Recursive case
    if (( n > 0 )); then
        local n1=$((n - 1))
        local f1=$(factorial "$n1" "")
        local f=$((f1 * n))

        factorial_memo["$key"]="$f"
        if [[ -n "$expected" ]]; then
            [[ "$f" == "$expected" ]] && echo "true" || echo "false"
        else
            echo "$f"
        fi
    fi
}
```

**Usage:**
```bash
$ source factorial.sh
$ factorial 5
120
$ factorial 10
3628800
# Second call uses memoization - instant!
$ factorial 5
120
```

---

### Example 3: Mutual Recursion - Even/Odd

**Prolog Source:**
```prolog
is_even(0).
is_even(N) :- N > 0, N1 is N - 1, is_odd(N1).

is_odd(1).
is_odd(N) :- N > 1, N1 is N - 1, is_even(N1).
```

**Detection:**
```prolog
?- build_call_graph([is_even/1, is_odd/1], Graph).
Graph = [(is_even/1 -> is_odd/1), (is_odd/1 -> is_even/1)].

?- find_sccs(Graph, SCCs).
SCCs = [[is_even/1, is_odd/1]].  % Single SCC = mutual recursion
```

**Generated Bash Code:**
```bash
#!/bin/bash
# Mutually recursive group: is_even, is_odd

# Shared memoization table for the SCC
declare -gA even_odd_memo

is_even() {
    local n="$1"
    local key="is_even:${n}"

    # Check memo
    if [[ -n "${even_odd_memo[$key]}" ]]; then
        echo "${even_odd_memo[$key]}"
        return
    fi

    # Base case
    if [[ "$n" == "0" ]]; then
        even_odd_memo["$key"]="true"
        echo "true"
        return
    fi

    # Recursive case: call is_odd
    if (( n > 0 )); then
        local n1=$((n - 1))
        local result=$(is_odd "$n1")
        even_odd_memo["$key"]="$result"
        echo "$result"
    else
        even_odd_memo["$key"]="false"
        echo "false"
    fi
}

is_odd() {
    local n="$1"
    local key="is_odd:${n}"

    # Check memo
    if [[ -n "${even_odd_memo[$key]}" ]]; then
        echo "${even_odd_memo[$key]}"
        return
    fi

    # Base case
    if [[ "$n" == "1" ]]; then
        even_odd_memo["$key"]="true"
        echo "true"
        return
    fi

    # Recursive case: call is_even
    if (( n > 1 )); then
        local n1=$((n - 1))
        local result=$(is_even "$n1")
        even_odd_memo["$key"]="$result"
        echo "$result"
    else
        even_odd_memo["$key"]="false"
        echo "false"
    fi
}
```

**Usage:**
```bash
$ source even_odd.sh
$ is_even 4
true
$ is_even 7
false
$ is_odd 3
true
$ is_odd 8
false
```

---

## Module Visibility Pattern

### The `user:clause` Technique

**Problem:** Pattern matchers in modules can't see predicates defined in other modules or user context.

**Solution:** Use `user:clause/2` instead of `clause/2`:

```prolog
% ❌ WRONG - Won't see predicates from other modules
my_pattern_matcher(Pred/Arity) :-
    functor(Head, Pred, Arity),
    clause(Head, Body),  % Only sees current module!
    analyze(Body).

% ✅ CORRECT - Sees predicates in user namespace
my_pattern_matcher(Pred/Arity) :-
    functor(Head, Pred, Arity),
    user:clause(Head, Body),  % Sees user predicates!
    analyze(Body).
```

**When to use:**
- Pattern detection predicates in modules
- Call graph analysis across modules
- SCC detection with external predicates
- Any cross-module predicate introspection

**Example from pattern_matchers.pl:**
```prolog
is_tail_recursive_accumulator(Pred/Arity, AccInfo) :-
    functor(BaseHead, Pred, Arity),
    user:clause(BaseHead, true),  % ← user: prefix

    functor(RecHead, Pred, Arity),
    user:clause(RecHead, RecBody),  % ← user: prefix
    contains_call_to(RecBody, Pred/Arity).
```

**Test predicates must also use user namespace:**
```prolog
% In test code
test_my_predicate :-
    % ❌ WRONG
    assertz(foo(X) :- bar(X)),  % Goes to current module

    % ✅ CORRECT
    assertz(user:(foo(X) :- bar(X))),  % Goes to user namespace
    my_pattern_matcher(foo/1).  % Now can see foo/1!
```

---

## Testing

### Run All Tests

```prolog
?- [test_advanced].
?- test_all_advanced.  % Run all module tests
?- test_all.           % Include integration, performance, regression tests
```

### Individual Module Tests

```prolog
?- test_call_graph.
?- test_scc_detection.
?- test_pattern_matchers.
?- test_tail_recursion.
?- test_linear_recursion.
?- test_mutual_recursion.
?- test_advanced_compiler.
```

### Generated Files

Tests generate bash scripts in `output/advanced/`:
- `count_items.sh` - Tail recursion example
- `list_length.sh` - Linear recursion example
- `even_odd.sh` - Mutual recursion example

## Future Work

### Planned Enhancements

1. **Better accumulator detection**: Analyze data flow to identify accumulator positions
2. **Loop fusion**: Combine multiple tail-recursive predicates
3. **Mutual transitive closures**: Detect and optimize multi-relation closures
4. **Pattern learning**: Suggest optimizations based on usage patterns

### Under Consideration

1. **Parallel execution**: Generate concurrent bash code for independent SCCs
2. **Memory optimization**: Smart memo table pruning
3. **Custom patterns**: User-defined compilation strategies
4. **Static analysis**: Warn about inefficient recursion patterns

## Design Decisions

### Why SCC Detection?

**Problem**: How to identify mutual recursion groups?

**Solution**: SCCs group predicates that call each other cyclically

**Benefit**: Handles arbitrary-sized mutual recursion (not just pairs)

### Why Priority-Based Compilation?

**Problem**: Multiple patterns might match same predicate

**Solution**: Try simplest (most optimized) pattern first

**Benefit**: Ensures best performance for each pattern type

### Why Minimal Integration?

**Problem**: Don't want to modify existing working code

**Solution**: Optional hook with graceful fallback

**Benefit**:
- Existing functionality preserved
- Advanced module can be developed independently
- Easy to disable if issues arise

## Troubleshooting

### Common Issues

#### "Pattern not detected" / Fallback to basic memoization

**Symptom:** Your tail recursive predicate compiles to memoization instead of iterative loop

**Causes:**
1. Accumulator not detected (position ambiguous)
2. Not truly tail recursive (operations after recursive call)
3. Multiple recursive calls

**Solutions:**
- Ensure accumulator pattern: `pred(..., Acc, Result)` with `Acc = Result` in base case
- Make sure recursive call is the LAST operation
- Use `is_tail_recursive_accumulator/2` to debug

**Example:**
```prolog
% ❌ NOT tail recursive - multiplication happens after
length([], 0).
length([_|T], N) :-
    length(T, N1),
    N is N1 + 1.  % ← Not tail recursive!

% ✅ Tail recursive - uses accumulator
length_acc(List, Len) :- length_acc(List, 0, Len).
length_acc([], Acc, Acc).
length_acc([_|T], Acc, Len) :-
    Acc1 is Acc + 1,
    length_acc(T, Acc1, Len).  % ← Last operation!
```

#### "Unknown procedure: user:clause/2"

**Symptom:** Error when loading pattern_matchers.pl

**Solution:** Ensure you're using SWI-Prolog (other Prolog systems may use different module syntax)

#### Generated bash script doesn't work

**Symptom:** Syntax errors or incorrect output from generated scripts

**Debugging:**
1. Check bash syntax: `bash -n output/advanced/yourscript.sh`
2. Enable bash debugging: `bash -x output/advanced/yourscript.sh`
3. Verify Prolog predicate is correct first
4. Check output/advanced/test_runner.sh for working examples

#### Singleton variable warnings

**Symptom:** `Warning: Singleton variables: [X]`

**Cause:** Variable appears only once in a clause

**Fix:** Prefix unused variables with underscore:
```prolog
% ❌ Warning
expr_to_bash(Acc + Const, Expr) :-

% ✅ No warning
expr_to_bash(_Acc + Const, Expr) :-
```

#### Mutual recursion not detected

**Symptom:** Even/odd style predicates compile separately instead of as a group

**Debugging:**
```prolog
?- build_call_graph([is_even/1, is_odd/1], Graph).
% Should show: [(is_even/1->is_odd/1), (is_odd/1->is_even/1)]

?- find_sccs(Graph, SCCs).
% Should show: [[is_even/1, is_odd/1]]
```

**Common cause:** Predicates not asserted to user namespace:
```prolog
% ✅ Correct
assertz(user:(is_even(0))).
assertz(user:(is_even(N) :- ...)).
```

---

## Performance Tips

### Tail Recursion vs Linear Recursion

**Tail recursion** (iterative loop):
- ✅ O(1) space (no call stack)
- ✅ Fastest for list processing
- ❌ Requires accumulator pattern

**Linear recursion** (memoization):
- ✅ Works for 1+ recursive calls with independence
- ✅ Caches results for reuse
- ✅ Excellent for fibonacci-style patterns with multiple calls
- ❌ O(n) space for memo table

**When to use which:**
- List processing → Tail recursion
- Scalar computations (fibonacci, etc.) → Linear recursion (memo helps!)
- Tree/structural recursion → Tree recursion compiler

### Memoization Trade-offs

**Benefits:**
- Avoids recomputation
- Essential for overlapping subproblems
- Shared tables for mutual recursion

**Costs:**
- Memory usage grows with unique inputs
- Lookup overhead for simple predicates

**Best for:**
- Expensive computations
- Repeated queries with same inputs
- Mutual recursion groups

**Not worth it for:**
- One-time queries
- Monotonically increasing inputs
- Very cheap predicates

---

## References

- **Tarjan's SCC Algorithm**: R. Tarjan (1972), "Depth-first search and linear graph algorithms"
- **Call Graphs**: Aho, Sethi, Ullman (1986), "Compilers: Principles, Techniques, and Tools"
- **Tail Call Optimization**: Steele (1977), "Debunking the 'Expensive Procedure Call' Myth"

---

*Last updated: 2025-10-11*