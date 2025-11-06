<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Chapter 6: Advanced Topic: The Constraint System

By default, UnifyWeaver makes certain assumptions about the desired output of your compiled predicates. Specifically, it assumes you want unique results in no particular order. While this is often true, it's not always the case. 

This is where the **constraint system** comes in. It provides a way for you to give hints to the compiler, allowing you to control the behavior of the generated script to match your specific needs.

## Why Constraints are Necessary

Prolog's execution model can sometimes produce the same result multiple times through different logical paths. For most queries (like "who are the ancestors of Jacob?"), these duplicates are not useful. Furthermore, the order in which Prolog finds solutions is an implementation detail and is often not guaranteed.

To provide predictable and clean output, UnifyWeaver's default behavior is to:
1.  **Eliminate duplicates**.
2.  **Not preserve any specific order**.

It achieves this by piping the final output of a stream through `sort -u`, which sorts the results and removes duplicates. This is efficient and effective for many use cases.

However, consider a predicate that processes a log file where the order of events is critical, and duplicate events might be significant. In this scenario, the default behavior would be incorrect. The constraint system lets you override these defaults.

## The Two Constraint Dimensions

The system is built on two independent (orthogonal) constraints:

1.  **`unique`**: Controls whether to eliminate duplicate results.
    *   `unique(true)` or `unique`: Eliminates duplicates (the default).
    *   `unique(false)`: Allows duplicates to appear in the output.

2.  **`unordered`**: Controls whether the original result order matters.
    *   `unordered(true)` or `unordered`: Result order does not matter (the default). This allows the compiler to use efficient sorting-based deduplication.
    *   `unordered(false)` or `ordered`: Result order must be preserved.

## Declaring Constraints

The recommended way to apply a constraint to a predicate is by using a **pragma**, which is a special directive to the compiler written at the top of your Prolog file.

The syntax is: `:- constraint(Predicate/Arity, [ListOfConstraints]).`

### Example Declarations

*   **Default behavior (explicitly stated):**
    ```prolog
    :- constraint(grandparent/2, [unique, unordered]).
    ```

*   **Preserve order, but ensure uniqueness:** (e.g., for a timeline of unique events)
    ```prolog
    :- constraint(event_log/3, [unique, ordered]).
    ```

*   **Allow duplicates, order doesn't matter:** (e.g., collecting all instances of something)
    ```prolog
    :- constraint(all_mentions/2, [unique(false), unordered]).
    ```

*   **Preserve everything, including order and duplicates:** (e.g., for raw log processing)
    ```prolog
    :- constraint(raw_log/3, [unique(false), ordered]).
    ```

## How Constraints Change the Generated Code

UnifyWeaver selects a different **deduplication strategy** based on the constraints you provide.

| `unique` | `unordered` | Strategy | Bash Implementation | 
|:---|:---|:---|:---|
| `true` | `true` | **Sort & Unique** | The output is piped through `... | sort -u`. This is the efficient default. |
| `true` | `false` | **Hash Deduplication** | The script uses a Bash associative array (`declare -A seen`) to keep track of results it has already printed. This preserves the original order but is slightly less performant than `sort -u`. |
| `false` | `*` | **No Deduplication** | The script simply outputs results directly from the pipeline with no deduplication mechanism. The `unordered` constraint has no effect in this case. |

### Example: An Ordered Log

Imagine a predicate `user_action(Timestamp, User, Action).` that represents a sequence of user actions. The order is critical.

**Prolog Code:**
```prolog
:- constraint(user_action/3, [unique, ordered]).

% Facts are assumed to be in chronological order
user_action(100, 'alice', 'login').
user_action(101, 'bob', 'login').
user_action(102, 'alice', 'read_file').
user_action(103, 'alice', 'read_file'). % A duplicate action
```

When you compile this, the `[unique, ordered]` constraint tells the compiler not to use `sort -u`, which would reorder the results by timestamp (and in this case, also remove the duplicate `read_file` action). Instead, it will generate a script that uses an associative array to print only the *first* occurrence of each unique line, preserving the original order.

**Generated Code (Conceptual):**
```bash
user_action_stream() {
    declare -A seen
    # Stream the raw data from the facts array
    for key in "${user_action_data[@]}"; do
        if [[ -z "${seen[$key]}" ]]; then
            seen[$key]=1
            echo "$key"
        fi
    done
}
```
If you had used `[unique(false), ordered]`, the generated code would be even simpler, omitting the `seen` array and the check entirely, printing every single action as it appears.

### Constraints in Action: The `unique` Optimization

As of the latest updates, the constraint system is now beginning to be used by the **advanced compilers** for optimization, not just for deduplication in the stream compiler.

The `tail_recursion` compiler now understands the `unique(true)` constraint.

Consider a tail-recursive predicate to count list items:
```prolog
:- constraint(count_items/3, [unique(true)]).

count_items([], Acc, Acc).
count_items([_|T], Acc, N) :-
    Acc1 is Acc + 1,
    count_items(T, Acc1, N).
```
Because this predicate is declared as `unique`, the compiler knows it will only ever produce one final count. It can use this information to generate more efficient code.

**Generated Code WITHOUT `unique(true)` (Conceptual):**
```bash
count_items() {
    # ... loop logic ...
    echo "$final_accumulator_value"
}
```

**Generated Code WITH `unique(true)` (Actual):**
```bash
count_items() {
    # ... loop logic ...
    echo "$final_accumulator_value"
    exit 0
}
```
The addition of `exit 0` is a subtle but powerful optimization. It tells the script to terminate immediately after producing its one and only result. In a complex pipeline, this prevents the shell from doing any further work and can significantly improve performance by allowing downstream processes to finish sooner.

This is the first of many planned optimizations that will leverage the constraint system.

## Runtime Overrides

It is also possible to override declared constraints at compile time by passing them in the options list to `compile_predicate/3`. This can be useful for debugging or for generating a special version of a script without changing the source file.

```prolog
% Override the declared constraint to get duplicates
?- compile_predicate(user_action/3, [unique(false)], BashCode).
```

## Next Steps

This concludes the main educational series on UnifyWeaver. You have learned about:
*   The core concepts of Prolog.
*   The architecture of the UnifyWeaver compiler.
*   How to write, compile, and run both recursive and non-recursive predicates.
*   How to use the constraint system to fine-tune the output.

From here, you are well-equipped to start writing your own declarative programs and compiling them into powerful Bash scripts. I encourage you to explore the `examples/` directory in the project to see more complex use cases and to experiment with different kinds of predicates and constraints.
