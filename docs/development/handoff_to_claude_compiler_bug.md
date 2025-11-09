# Handoff to Claude Code: Compiler Arithmetic Transpilation Bug

**To:** Claude Code
**From:** Gemini CLI
**Date:** 2025-10-26
**Subject:** Assistance Needed: UnifyWeaver Compiler Bug with Arithmetic Predicates

## 1. Overview

We are encountering a persistent bug in the UnifyWeaver compiler when attempting to transpile Prolog predicates that contain arithmetic operations (e.g., `is/2`, `>/2`, `=</2`) and negation (`\+/1`) to Bash. Despite multiple attempts to fix it, the compilation continues to fail.

We require your expertise to diagnose and resolve this issue.

## 2. Original Problem Context

The bug was discovered during an end-to-end test of our new LLM-driven workflow feature. The goal was to compile a simple Prolog predicate `choose_strategy/3` (defined in `temp_strategy.pl`) into a Bash script using `unifyweaver.compile` (which internally calls `compiler_driver.pl`).

**`temp_strategy.pl` content:**
```prolog
% choose_strategy/3 - Determines the best context-gathering strategy.
% Signature: choose_strategy(+FileCount, +Budget, -Strategy)

choose_strategy(1, _, single_file_precision).

choose_strategy(FileCount, Budget, balanced_deep_dive) :-
    FileCount > 1,
    EstimatedCost is 0.002 * FileCount,
    EstimatedCost =< Budget.

choose_strategy(FileCount, Budget, quick_triage) :-
    FileCount > 1,
    EstimatedCost is 0.002 * FileCount,
    EstimatedCost > Budget.
```

## 3. Initial Error

The first compilation attempt failed with:
`ERROR: ... clause/2: No permission to access private_procedure '(>)/2'`

This indicated that the `dependency_analyzer.pl` (called by `compiler_driver.pl`) was trying to introspect built-in predicates like `>/2`, which are private.

## 4. Attempts to Fix and Current Status

### Attempt 1: Filter Built-ins in `dependency_analyzer.pl` (Incorrect)

*   **Idea:** Modify `dependency_analyzer.pl` to ignore built-in predicates.
*   **Outcome:** This was incorrect. As discussed, built-in predicates *are* dependencies that need to be translated by the backend compilers. This fix was reverted.

### Attempt 2: Filter Built-ins in `compiler_driver.pl` (Corrected Approach)

*   **Idea:** Modify `compiler_driver.pl` to filter out built-in predicates from the dependency list *before* attempting to compile them. This ensures only user-defined predicates are passed to the backend compilers.
*   **Outcome:** This fix was applied and is currently present in `compiler_driver.pl`. It successfully prevented the `clause/2: No permission` error for the `compiler_driver` itself.

### Attempt 3: Implement Translation in `stream_compiler.pl` (Current State)

*   **Idea:** The `stream_compiler.pl` needs to know how to translate arithmetic and logical built-ins into Bash.
*   **Actions:**
    *   Refactored `stream_compiler.pl` to introduce `translate_body_to_bash/3` and `translate_expr/3` predicates.
    *   These predicates are designed to map Prolog variables to Bash positional parameters and translate `is/2`, `>/2`, `=</2`, `\+/1`, etc., into Bash equivalents (using `bc` for arithmetic).
    *   Restored `compile_multiple_rules/4` to handle predicates with multiple clauses.
*   **Current Errors:**
    1.  `Syntax error: End of file in quoted string` in `stream_compiler.pl` at line 124. This points to an issue in the `compile_facts` predicate, likely due to missing helper predicates (`compile_facts_no_dedup`, `format_fact_entry`) that were removed during refactoring.
    2.  `functor/3: Arguments are not sufficiently instantiated`. This indicates a variable instantiation issue, possibly related to `Options` not being correctly passed or used in `generate_dedup_wrapper`.
    3.  `Warning: Local definition of stream_compiler:get_dedup_strategy/2 overrides weak import from constraint_analyzer`. This is a minor warning but indicates a potential conflict.

## 5. Request for Assistance

We are currently on the `fix/arithmetic-transpilation` branch.

Claude, we need your help to:

1.  **Diagnose the current errors:** Specifically, the `Syntax error: End of file in quoted string` and `functor/3: Arguments are not sufficiently instantiated` errors in `stream_compiler.pl`.
2.  **Complete the `stream_compiler.pl` fix:** Ensure that `compile_facts`, `compile_single_rule`, and `compile_multiple_rules` correctly translate all necessary Prolog constructs (including arithmetic and logical built-ins) into functional Bash code.
3.  **Validate the overall approach:** Confirm that the current strategy of filtering built-ins in `compiler_driver.pl` and translating them in `stream_compiler.pl` is the correct and most robust way to handle this.

The goal is to successfully compile `choose_strategy/3` from `temp_strategy.pl` into a working Bash script.

## 6. Relevant Files

*   `src/unifyweaver/core/compiler_driver.pl` (Contains the fix for filtering built-ins)
*   `src/unifyweaver/core/dependency_analyzer.pl` (Should be in its original state)
*   `src/unifyweaver/core/stream_compiler.pl` (Contains the latest, but still buggy, translation logic)
*   `temp_strategy.pl` (The Prolog code we are trying to compile)

Thank you for your assistance.
