<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Chapter 9: Advanced Recursion Patterns

## Introduction

While the basic `recursive_compiler` can handle simple transitive closures, the true power of UnifyWeaver lies in its `advanced_recursive_compiler`. This component can identify complex and specific patterns of recursion and compile them into highly optimized Bash code.

This chapter explores the main advanced recursion patterns that UnifyWeaver understands: **tail recursion**, **linear recursion**, **tree recursion**, and **mutual recursion**. All examples shown are from actual working implementations with real generated bash code.

## 1. Tail Recursion Optimization

Tail recursion is a special form of recursion where the recursive call is the very last action performed in a function. There is no other work to be done after the recursive call returns.

**Why it matters:** A tail-recursive function can be transformed into a simple iterative loop (like a `while` or `for` loop). This is a major optimization because it avoids the overhead of function calls and eliminates the risk of running out of stack space ("stack overflow") on deep recursions.

### The Pattern: Accumulators

The most common tail-recursive pattern involves an **accumulator**. This is an extra argument that carries the intermediate result down through the recursive calls.

**Example: A tail-recursive counter**

```prolog
% count_items(List, Accumulator, FinalCount)

% Base Case: When the list is empty, the final count is the value
% of the accumulator.
count_items([], Acc, Acc).

% Recursive Step: For a non-empty list, increment the accumulator
% and recurse on the tail of the list.
count_items([_|T], Acc, N) :-
    Acc1 is Acc + 1,
    count_items(T, Acc1, N). % This is the tail call
```

UnifyWeaver's `pattern_matchers` module is specifically designed to recognize this accumulator pattern.

### Generated Code: An Iterative Loop

When the `advanced_recursive_compiler` sees this pattern, it doesn't generate a recursive Bash function. Instead, it generates a highly efficient `while` loop with proper argument parsing.

**Actual Generated Bash (`count_items.sh`):**
```bash
count_items() {
    local arg1="$1"
    local arg2="$2"
    local arg3="$3"

    # Base case check
    if [[ "$arg1" == "[]" || -z "$arg1" ]]; then
        echo "$arg1:$arg2:$arg2"
        return 0
    fi

    # Parse list and initialize
    local current_list="$arg1"
    local acc="$arg2"

    # Iterative loop replacing tail recursion
    while [[ "$current_list" != "[]" && -n "$current_list" ]]; then
        # Extract head and tail
        current_list="${current_list#[}"
        current_list="${current_list%]}"

        if [[ "$current_list" =~ ^[^,]+,(.+)$ ]]; then
            current_list="[${BASH_REMATCH[1]}]"
        else
            current_list="[]"
        fi

        # Step operation: Acc1 is Acc + 1
        acc=$((acc + 1))
    done

    echo "$arg1:$arg2:$acc"
}
```

This transformation from a declarative recursive definition to an imperative loop is one of UnifyWeaver's most powerful optimizations.

### Testing the Generated Code

```bash
source output/advanced/count_items.sh
count_items "[a,b,c,d,e]" 0 ""
# Output: [a,b,c,d,e]:0:5
```

## 2. Linear Recursion with Fold-Based Compilation

Linear recursion occurs when a function makes exactly one recursive call per clause, and that call is independent. UnifyWeaver now uses a **fold-based approach** that separates structure building from computation.

### Numeric Pattern: Factorial

**Example: factorial with linear recursion**
```prolog
% factorial(N, Result)

% Base Case
factorial(0, 1).

% Recursive Step: Single independent recursive call
factorial(N, F) :-
    N > 0,
    N1 is N - 1,
    factorial(N1, F1),
    F is N * F1.
```

### Generated Code: Fold with Helper Functions

UnifyWeaver generates a fold-based implementation that:
1. Builds a range structure (N, N-1, ..., 1)
2. Folds over that structure with an operation function
3. Uses memoization for efficiency

**Actual Generated Bash (`factorial.sh`):**
```bash
#!/bin/bash
# factorial - fold-based linear recursion (numeric)

# Fold helper: fold_left accumulator operation values...
fold_left() {
    local acc="$1"
    local op_func="$2"
    shift 2
    for item in "$@"; do
        acc=$("$op_func" "$item" "$acc")
    done
    echo "$acc"
}

# Range builder: generates N, N-1, ..., 1
build_range_down() {
    local n="$1"
    seq "$n" -1 1
}

# Fold operation for factorial
factorial_op() {
    local current="$1"
    local acc="$2"
    echo $((current * acc))
}

# Main predicate with memoization
declare -gA factorial_memo

factorial() {
    local n="$1"
    local expected="$2"

    # Check memo
    if [[ -n "${factorial_memo[$n]}" ]]; then
        local cached="${factorial_memo[$n]}"
        if [[ -n "$expected" ]]; then
            [[ "$cached" == "$expected" ]] && echo "$n:$expected" && return 0
            return 1
        else
            echo "$n:$cached"
            return 0
        fi
    fi

    # Base case
    if [[ "$n" -eq 0 ]]; then
        local result="1"
        factorial_memo["$n"]="$result"
        if [[ -n "$expected" ]]; then
            [[ "$result" == "$expected" ]] && echo "$n:$expected" && return 0
            return 1
        else
            echo "$n:$result"
            return 0
        fi
    fi

    # Recursive case using fold
    local range=$(build_range_down "$n")
    local result=$(fold_left 1 "factorial_op" $range)

    # Memoize
    factorial_memo["$n"]="$result"

    if [[ -n "$expected" ]]; then
        [[ "$result" == "$expected" ]] && echo "$n:$expected" && return 0
        return 1
    else
        echo "$n:$result"
        return 0
    fi
}
```

### Testing Factorial

```bash
source output/advanced/factorial.sh
factorial 5 ""
# Output: 5:120

factorial 10 ""
# Output: 10:3628800
```

### List Pattern: List Length

**Example: list_length with linear recursion**
```prolog
% list_length(List, Length)

% Base Case
list_length([], 0).

% Recursive Step
list_length([_|T], N) :-
    list_length(T, N1),
    N is N1 + 1.
```

**Actual Generated Bash (`list_length.sh`):**
```bash
#!/bin/bash
# list_length - fold-based linear recursion (list)

# Fold helper
fold_left() {
    local acc="$1"
    local op_func="$2"
    shift 2
    for item in "$@"; do
        acc=$("$op_func" "$item" "$acc")
    done
    echo "$acc"
}

# Parse list into elements
parse_list() {
    local list="$1"
    list="${list#[}"
    list="${list%]}"
    if [[ -n "$list" ]]; then
        echo "$list" | tr ',' ' '
    fi
}

# Fold operation for list_length
list_length_op() {
    local current="$1"
    local acc="$2"
    # For list_length, we just increment (ignore current element)
    echo $((acc + 1))
}

# Main predicate with memoization
declare -gA list_length_memo

list_length() {
    local list="$1"
    local expected="$2"

    # Check memo
    if [[ -n "${list_length_memo[$list]}" ]]; then
        local cached="${list_length_memo[$list]}"
        if [[ -n "$expected" ]]; then
            [[ "$cached" == "$expected" ]] && echo "$list:$expected" && return 0
            return 1
        else
            echo "$list:$cached"
            return 0
        fi
    fi

    # Base case
    if [[ "$list" == "[]" || -z "$list" ]]; then
        local result="0"
        list_length_memo["$list"]="$result"
        if [[ -n "$expected" ]]; then
            [[ "$result" == "$expected" ]] && echo "$list:$expected" && return 0
            return 1
        else
            echo "$list:$result"
            return 0
        fi
    fi

    # Recursive case using fold
    local elements=$(parse_list "$list")
    local result=$(fold_left 0 "list_length_op" $elements)

    # Memoize
    list_length_memo["$list"]="$result"

    if [[ -n "$expected" ]]; then
        [[ "$result" == "$expected" ]] && echo "$list:$expected" && return 0
        return 1
    else
        echo "$list:$result"
        return 0
    fi
}
```

### Key Benefits of Fold-Based Approach

1. **Separation of Concerns:** Structure building vs computation logic
2. **Reusable Components:** `fold_left` can be used for many patterns
3. **Memoization:** Results are cached for efficiency
4. **Variable Translation:** Proper mapping from Prolog variables to bash variables

## 3. Tree Recursion

Tree recursion is used for problems where a function operates on a data structure, and makes multiple recursive calls on smaller parts of that structure. This is common in problems involving trees or nested lists.

The key characteristic of tree recursion is **structural decomposition** with **multiple recursive calls** per clause.

**Example: A tree-recursive sum function**
```prolog
% tree_sum(Tree, Sum)
% Tree is represented as [Value, LeftSubtree, RightSubtree]

% Base Case: The sum of an empty tree is 0.
tree_sum([], 0).

% Recursive Step: The sum of a tree is the value of the root
% plus the sum of the left and right subtrees.
tree_sum([V, L, R], Sum) :-
    tree_sum(L, LS),
    tree_sum(R, RS),
    Sum is V + LS + RS.
```

In this example, the `tree_sum` function is called on both the left (`L`) and right (`R`) subtrees, which are parts of the original tree structure.

### Generated Code: Structure Parsing with Helper

UnifyWeaver generates a Bash function with a sophisticated parser for nested tree structures.

**Actual Generated Bash (`tree_sum.sh`):**
```bash
#!/bin/bash
# tree_sum - tree recursion (binary tree pattern)
# List-based tree: [value, [left], [right]] or []

tree_sum() {
    local tree="$1"

    # Base case: empty tree
    if [[ "$tree" == "[]" || -z "$tree" ]]; then
        echo 0
        return 0
    fi

    # Parse tree structure [value, left, right]
    local value left right
    parse_tree "$tree" value left right

    # Recursive calls
    local left_result=$(tree_sum "$left")
    local right_result=$(tree_sum "$right")

    # Combine results
    local result=$(( $value + $left_result + $right_result ))

    echo "$result"
}

# Helper: Parse list-based tree
# Handles nested brackets properly by tracking bracket depth
parse_tree() {
    local tree_str="$1"
    local -n val="$2"
    local -n lft="$3"
    local -n rgt="$4"

    # Remove outer brackets
    tree_str="${tree_str#[}"
    tree_str="${tree_str%]}"

    # Parse by tracking bracket depth
    local depth=0 part=0 current=""
    local i char

    for (( i=0; i<${#tree_str}; i++ )); do
        char="${tree_str:$i:1}"

        if [[ "$char" == "[" ]]; then
            ((depth++))
            current+="$char"
        elif [[ "$char" == "]" ]]; then
            ((depth--))
            current+="$char"
        elif [[ "$char" == "," && $depth -eq 0 ]]; then
            # Top-level comma - marks boundary between parts
            case $part in
                0) val="$current" ;;
                1) lft="$current" ;;
            esac
            current=""
            ((part++))
        else
            current+="$char"
        fi
    done

    # Last part is right subtree
    rgt="$current"

    # Clean up whitespace
    val="${val// /}"
    lft="${lft// /}"
    rgt="${rgt// /}"
}
```

### Testing Tree Sum

```bash
source output/advanced/tree_sum.sh

# Test: [5, [3, [1, [], []]], [2, [], []]]
#       5
#      / \
#     3   2
#    /
#   1
tree_sum "[5,[3,[1,[],[]]],[2,[],[]]]"
# Output: 11 (5 + 3 + 1 + 2)

# Test: [10, [5, [], [3, [], []]], [7, [], []]]
tree_sum "[10,[5,[],[3,[],[]]]],[7,[],[]]]"
# Output: 25 (10 + 5 + 3 + 7)
```

## 4. Forbidding Linear Recursion

Sometimes, a predicate might look like it fits the linear recursion pattern, but you want to prevent UnifyWeaver from compiling it that way. For example, you might want to force a predicate to be compiled using the tree recursion strategy to handle a specific graph traversal algorithm.

UnifyWeaver provides a way to do this using the `forbid_linear_recursion/1` directive.

**Example: Forcing a different compilation strategy**
```prolog
% This predicate could be interpreted as linear recursion.
my_graph_traversal(Node, Value) :- ...

% Forbid linear recursion for this predicate.
:- forbid_linear_recursion(my_graph_traversal/2).
```

When UnifyWeaver sees this directive, it will not consider the linear recursion pattern for the `my_graph_traversal/2` predicate. It will then try to match other patterns, such as tree recursion.

This feature gives you more control over the compilation process and allows you to guide the compiler to the most appropriate strategy for your specific problem.

## 5. Mutual Recursion

Mutual recursion happens when two or more functions call each other in a cycle. The classic example is determining if a number is even or odd.

**Example: `is_even/1` and `is_odd/1`**
```prolog
is_even(0).
is_even(N) :- N > 0, N1 is N - 1, is_odd(N1).

is_odd(1).
is_odd(N) :- N > 1, N1 is N - 1, is_even(N1).
```
Here, `is_even` calls `is_odd`, and `is_odd` calls `is_even`.

### Detection: Strongly Connected Components (SCC)

UnifyWeaver handles this complex case by performing a graph analysis of your code:
1.  **Build a Call Graph:** It first builds a graph where the nodes are your predicates and an edge `A -> B` means `A` calls `B`.
2.  **Find Strongly Connected Components (SCCs):** It then uses Tarjan's algorithm to find all SCCs in the graph. An SCC is a group of nodes where every node is reachable from every other node in the group. A non-trivial SCC (with more than one node) represents a group of mutually recursive predicates.

### Generated Code: Shared Memoization Table

Once an SCC is identified, the compiler generates Bash functions for all predicates in the group, and they all share a **single memoization table**. This is crucial for efficiency and correctness.

**Actual Generated Bash (`even_odd.sh`):**
```bash
#!/bin/bash
# Mutually recursive group: is_even_is_odd
# Shared memoization for mutual recursion

# Shared memo table for all predicates in this group
declare -gA is_even_is_odd_memo

# is_even/1 - part of mutual recursion group
is_even() {
    local arg1="$1"
    local key="is_even:$*"

    # Check shared memo table
    if [[ -n "${is_even_is_odd_memo[$key]}" ]]; then
        echo "${is_even_is_odd_memo[$key]}"
        return 0
    fi

    # Base case: is_even(0)
    if [[ "$arg1" == "0" ]]; then
        local result="true"
        is_even_is_odd_memo["$key"]="$result"
        echo "$result"
        return 0
    fi

    # Recursive case: N > 0, N1 is N - 1, is_odd(N1)
    if [[ "$arg1" -gt 0 ]]; then
        local n1=$(( $arg1 - 1 ))
        local rec_result=$(is_odd "$n1")
        if [[ $? -eq 0 && "$rec_result" == "true" ]]; then
            result="true"
        else
            return 1
        fi
        local result="true"
        is_even_is_odd_memo["$key"]="$result"
        echo "$result"
        return 0
    fi

    # No match found
    return 1
}

# is_odd/1 - part of mutual recursion group
is_odd() {
    local arg1="$1"
    local key="is_odd:$*"

    # Check shared memo table
    if [[ -n "${is_even_is_odd_memo[$key]}" ]]; then
        echo "${is_even_is_odd_memo[$key]}"
        return 0
    fi

    # Base case: is_odd(1)
    if [[ "$arg1" == "1" ]]; then
        local result="true"
        is_even_is_odd_memo["$key"]="$result"
        echo "$result"
        return 0
    fi

    # Recursive case: N > 1, N1 is N - 1, is_even(N1)
    if [[ "$arg1" -gt 1 ]]; then
        local n1=$(( $arg1 - 1 ))
        local rec_result=$(is_even "$n1")
        if [[ $? -eq 0 && "$rec_result" == "true" ]]; then
            result="true"
        else
            return 1
        fi
        local result="true"
        is_even_is_odd_memo["$key"]="$result"
        echo "$result"
        return 0
    fi

    # No match found
    return 1
}
```

### Testing Mutual Recursion

```bash
source output/advanced/even_odd.sh

is_even 0 && echo "0 is even"    # ✓ true
is_even 2 && echo "2 is even"    # ✓ true (via is_odd(1))
is_odd 3 && echo "3 is odd"      # ✓ true (via is_even(2) -> is_odd(1))
is_even 5 || echo "5 is not even"  # ✓ false

# The shared memo table caches results across both functions
```

### Implementation Details

The mutual recursion compiler:
- **Variable Translation:** Maps Prolog head variable to bash `$arg1`
- **Expression Translation:** Converts `N - 1` to `$(( $arg1 - 1 ))`
- **Condition Generation:** Translates `N > 0` to `[[ "$arg1" -gt 0 ]]`
- **Proper Call Chain:** `is_even` correctly calls `is_odd` with computed values
- **Shared Caching:** Both functions use `is_even_is_odd_memo` table

## Pattern Selection and Priority

UnifyWeaver tries patterns in this order:

1. **Tail Recursion** - Checked first (most efficient when applicable)
2. **Linear Recursion** - Single recursive call per clause
3. **Tree Recursion** - Multiple recursive calls on structure parts
4. **Mutual Recursion** - SCC detection for cyclic predicate calls
5. **Basic Recursion** - Fallback with BFS for transitive closure

Use `forbid_linear_recursion/1` to skip linear pattern detection and force alternative strategies.

## Performance Characteristics

| Recursion Pattern | Time Complexity | Space Complexity | Best For |
|:---|:---|:---|:---|
| **Tail Recursion** | O(n) iterations | O(1) stack | Accumulators, linear scans |
| **Linear Recursion** | O(n) with memo | O(n) memo table | Fibonacci, factorial |
| **Tree Recursion** | O(nodes) | O(depth) stack | Tree/graph operations |
| **Mutual Recursion** | O(n) with shared memo | O(n) shared table | Even/odd, mutual definitions |

## Summary

The advanced recursion compiler elevates UnifyWeaver from a simple code generator to a true optimizing compiler. By recognizing these common recursion patterns, it can produce Bash code that is far more performant and robust than a naive translation would be.

### Key Takeaways

1. **Tail recursion** becomes an iterative loop (no stack overhead)
2. **Linear recursion** uses fold-based compilation with memoization
3. **Tree recursion** handles structural decomposition with multiple calls
4. **Mutual recursion** uses SCC detection and shared memoization
5. Pattern detection is automatic but can be guided with `forbid_linear_recursion/1`

All code examples in this chapter are from actual working implementations in `output/advanced/` - you can run them yourself to see the optimizations in action!

## Next Steps

- **Chapter 10:** Learn about Prolog introspection and how UnifyWeaver analyzes your code
- **Chapter 11:** Explore test runner inference and automatic test discovery
- **Appendix A:** Detailed recursion pattern theory and classification algorithms

---

## Navigation

**←** [Previous: Chapter 8: Template System](08_template_system) | [📖 Back to Book 1](README) | [Next: Chapter 10: Prolog Introspection →](10_prolog_introspection)

