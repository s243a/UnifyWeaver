# Playbook: Data Source Integration

## Goal

Define a CSV data source, write a Prolog predicate to process it, transpile the predicate using UnifyWeaver, and verify the resulting data pipeline.

## Context

This playbook demonstrates an agent's ability to work with one of UnifyWeaver's core features: declarative data sources. It shows how to define a source, process it with Prolog logic, and compile it into an efficient data processing script.

**Key Insight:** UnifyWeaver allows you to treat external data sources (like CSV, JSON, etc.) as if they were native Prolog predicates, and the compiler will generate the necessary code to read and parse them.

## Strategy

1.  **Define Data Source:** Create a Prolog file that uses the `:- source(...)` directive to define a predicate that reads from `test_data/test_users.csv`.
2.  **Define Processing Logic:** In the same file, define a predicate that calls the data source predicate and processes the data (e.g., filters for a specific user).
3.  **Transpile:** Use the `unifyweaver.compile` skill to transpile the processing predicate.
4.  **Execute and Verify:** Run the generated script and check that the output is correct.

## Tools and Infrastructure

*   **Compiler Driver:** `src/unifyweaver/core/compiler_driver.pl`
    *   **Skill:** `skills/skill_unifyweaver_compile.md`
*   **CSV Source Module:** `src/unifyweaver/sources/csv_source.pl`
*   **Test Data:** `test_data/test_users.csv`

---
*This playbook is under development.*
