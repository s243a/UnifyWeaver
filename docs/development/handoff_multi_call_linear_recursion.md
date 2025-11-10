# Handoff: Multi-Call Linear Recursion Compiler

**Author:** Gemini
**Date:** 2025-11-09
**Branch:** `feature/multi-call-linear-recursion`

## Goal

The primary goal of this task was to extend the `linear_recursion.pl` compiler to handle predicates with multiple independent recursive calls, such as `fibonacci/2`. This would allow UnifyWeaver to correctly compile a wider range of recursive patterns with memoization.

## Work Done

1.  **Branch Creation:** A new branch, `feature/multi-call-linear-recursion`, was created for this work.
2.  **Pattern Matcher Update:** The `pattern_matchers.pl` module was modified to correctly identify `fibonacci/2` as a linear recursive predicate. This involved:
    *   Relaxing the "exactly one recursive call" constraint in `is_linear_recursive_streamable/1`.
    *   Adding placeholder logic for detecting dependencies between recursive calls.
3.  **Test Case Addition:** A test case for `fibonacci/2` was added to `src/unifyweaver/core/advanced/linear_recursion.pl` to drive the implementation.
4.  **New Code Generator:** A new predicate, `generate_multicall_numeric_recursion/7`, was added to `linear_recursion.pl` to handle the code generation for this new pattern.
5.  **Helper Predicates:** Several helper predicates were implemented to extract information from the Prolog clauses, such as base cases, argument computations, recursive calls, and aggregation logic.

## Current State & Problems Encountered

The implementation is currently in a non-working state. While the pattern detection now correctly identifies `fibonacci/2` as a candidate for linear recursion, the code generation part is incomplete and produces an incorrect Bash script.

The main challenge lies in the **mapping of Prolog variables to Bash variables**. The helper predicates for extracting the different parts of the recursive clause are struggling to correctly identify and translate the variables from the Prolog clause to the Bash script. The generated script uses internal Prolog variable names (e.g., `_14440`) instead of the correct Bash variables.

The current approach of trying to map variables using simple pattern matching and `VarMap` lists has proven to be complex and error-prone.

## Proposed Next Steps

It seems that a more robust approach is needed to solve the variable mapping problem. Here are a few suggestions:

1.  **Build a Complete Variable Map:** Instead of passing around simple lists of variables, a more structured map could be built that contains information about each variable's scope and its corresponding Bash variable name. This map could be built by traversing the clause body and tracking the state of each variable.
2.  **Use a Different Clause Representation:** The current approach of analyzing the raw Prolog clause body is difficult. It might be beneficial to first transform the clause body into a more structured representation (e.g., an abstract syntax tree) that is easier to parse and analyze.
3.  **Switch to an Easier Task:** As suggested, it might be more productive to switch to an easier task to make progress and come back to this problem later with a fresh perspective. A good candidate would be to work on another playbook, as this would help to expand the system's capabilities in a different area.

Given the difficulty of the current task, I recommend pursuing **Option 3** and working on a new playbook. This will allow us to make progress in other areas while we reconsider the best approach for the multi-call linear recursion compiler.
