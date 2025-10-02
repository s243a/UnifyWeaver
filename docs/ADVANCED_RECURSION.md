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
2. **Linear recursion** → Compile with memoization
3. **Mutual recursion** → Detect via SCC, compile with shared memo tables

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

**Pattern**: Exactly one recursive call per clause

```prolog
length([], 0).
length([_|T], N) :-
    length(T, N1),
    N is N1 + 1.
```

**Compilation Strategy**: Memoization with associative arrays

**Detected by**: `pattern_matchers:is_linear_recursive_streamable/1`

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

### Integration Hook

`recursive_compiler.pl` automatically tries advanced patterns before falling back to basic memoization:

```prolog
compile_recursive(Pred/Arity, Options, BashCode) :-
    % ... try basic patterns ...
    ;   % Try advanced patterns
        catch(
            advanced_recursive_compiler:compile_advanced_recursive(
                Pred/Arity, Options, BashCode
            ),
            error(existence_error(procedure, _), _),
            fail
        ) ->
        true
    ;   % Fall back to basic memoization
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

## References

- **Tarjan's SCC Algorithm**: R. Tarjan (1972), "Depth-first search and linear graph algorithms"
- **Call Graphs**: Aho, Sethi, Ullman (1986), "Compilers: Principles, Techniques, and Tools"
- **Tail Call Optimization**: Steele (1977), "Debunking the 'Expensive Procedure Call' Myth"
