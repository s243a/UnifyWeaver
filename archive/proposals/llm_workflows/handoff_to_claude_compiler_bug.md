# Handoff to Claude: Compiler Bug with Dynamic Data Sources

**To:** Claude AI Agent
**From:** Gemini CLI
**Date:** 2025-11-11
**Subject:** Analysis of a Compiler Bug and a Flawed Playbook When Using Dynamic Data Sources

## 1. Context

I was performing an end-to-end test of the `examples/csv_data_source_playbook.md`. The goal was to act as a fresh agent, follow the playbook's strategy, and create a data processing pipeline that reads from a CSV file.

The test failed during the compilation step. This document outlines my actions, the root cause of the failure, and suggestions for fixing both the compiler and the playbook.

## 2. Summary of Actions and Failure

Following the playbook's strategy, I performed these steps:

1.  **Generated Prolog Code:** I created a file, `tmp/csv_pipeline.pl`, containing the necessary logic:
    ```prolog
    :- module(csv_pipeline, [get_user_age/2]).
    :- use_module('src/unifyweaver/sources').
    :- use_module('src/unifyweaver/sources/csv_source').

    % 1. Define the data source from the CSV file
    :- source(csv, users/3, [file('test_data/test_users.csv'), has_header(true)]).

    % 2. Define the processing logic
    get_user_age(Name, Age) :-
        users(_, Name, Age).
    ```
2.  **Attempted Compilation:** I used the `unifyweaver.compile` skill to transpile the processing predicate, `get_user_age/2`. This invoked the core `compiler_driver.pl`.
3.  **Failure:** The compilation failed with the error: `ERROR: No clauses found for users/3`.

## 3. Root Cause Analysis

The error occurs because of a limitation in the `compiler_driver.pl`.

My investigation of the driver's source code revealed that it correctly identifies `users/3` as a dependency of `get_user_age/2`. However, it then tries to find the static clauses for `users/3` to compile it. It has **no awareness of the dynamic source registry**.

When it finds no clauses for `users/3` (because it's a dynamic source, not a regular predicate), it fails. The `compiler_driver` is built for transpiling static Prolog code, and it does not know how to defer to the `csv_source.pl` plugin to generate the code for `users/3`.

The playbook's strategy is therefore fundamentally flawed, as it directs the agent to use a tool (`compiler_driver`) in a way it doesn't support.

## 4. Suggested Improvements

There are two areas to fix: the compiler (the ideal solution) and the playbook (a temporary workaround).

### Suggestion 1: Improve the Compiler (The Correct Fix)

The `compiler_driver.pl` should be made aware of dynamic sources.

**Proposed Logic:**
1.  In `compiler_driver.pl`, after the `find_dependencies/2` call, the driver gets a list of predicates to compile.
2.  Before attempting to compile a dependency, it should first check if that predicate is a registered dynamic source. A new predicate, perhaps `dynamic_source_compiler:is_dynamic_source(Predicate)`, could be created for this.
3.  **If it is a dynamic source**, the driver should **not** proceed with its normal static compilation workflow. Instead, it should call `dynamic_source_compiler:compile_dynamic_source(Predicate, Options, BashCode)`.
4.  This will correctly invoke the appropriate plugin (e.g., `csv_source.pl`) to generate the necessary data-access script (`users.sh`).
5.  The driver can then proceed to compile the original predicate (`get_user_age/2`), knowing that its dependency will be available as a shell script.

This fix would make the compiler behave exactly as the playbook originally intended, making data source integration seamless.

### Suggestion 2: Improve the Playbook (The Workaround)

Until the compiler is fixed, the `csv_data_source_playbook.md` is unusable. It should be updated to provide a working strategy.

**Proposed New Strategy for the Playbook:**
1.  **Compile the Source, Not the Logic:** Instruct the agent to compile the data source predicate (`users/3`) directly, not the processing predicate. This will likely require using a more specific tool than the generic `compiler_driver`.
2.  **Use Unix Pipelines:** The result of compiling the source will be a `users.sh` script that dumps the entire CSV content. The playbook should then instruct the agent to pipe the output of this script to standard tools like `grep` and `awk`, `cut` to perform the filtering.

This is less elegant but would provide a working example with the current limitations.

## 5. Your Task

Please review this analysis. I recommend focusing on **Suggestion 1: Improving the Compiler**. This is the correct, long-term solution that will make the entire UnifyWeaver system more powerful and intuitive.

Your task is to modify the `compiler_driver.pl` to make it aware of dynamic data sources and delegate their compilation to the appropriate source plugins.

Thank you.
