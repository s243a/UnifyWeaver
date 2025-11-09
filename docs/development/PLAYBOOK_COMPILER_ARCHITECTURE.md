# Playbook Compiler Architecture

**Status:** Design Document
**Version:** 1.0
**Date:** 2025-10-25

## 1. Overview

This document describes the architecture for the `playbook_compiler.pl`, a component responsible for compiling "Economic Agent Playbooks" (written in Markdown) into executable artifacts. This compiler acts as a high-level **orchestrator**, leveraging the existing UnifyWeaver transpilation and introspection infrastructure.

## 2. Core Philosophy: Introspection over Parsing

The fundamental design principle is to use Prolog's introspection capabilities rather than performing complex parsing of Prolog code. The standard UnifyWeaver workflow is:

1.  Load Prolog code into the database via `consult/1`.
2.  Use introspection predicates (`clause/2`, `predicate_property/2`) to analyze the loaded code.
3.  Feed this analysis to the appropriate compiler backend (e.g., `recursive_compiler.pl`).

Our playbook compiler adheres to this philosophy.

## 3. The Orchestration Workflow

The `playbook_compiler.pl` does not transpile logic itself. It orchestrates the following process:

1.  **Extract Prolog Text:** It performs a simple text extraction to pull the raw Prolog code from ` ```prolog ... ``` ` blocks within the input Markdown playbook.

2.  **Create Temporary Knowledge Base:** It writes this extracted Prolog code to a temporary `.pl` file.

3.  **Consult:** It uses `consult/1` to load the rules from the temporary file into the Prolog database. At this point, the playbook's logic (e.g., `strategy_selection/2`) becomes accessible to the introspection system.

4.  **Invoke UnifyWeaver Core:** It calls the main `compiler_driver:compile/3` predicate, pointing it at the key predicates loaded from the playbook.

5.  **Integrate Artifacts:** The `compiler_driver` produces one or more `.sh` files containing the pure-bash, transpiled logic. The `playbook_compiler` then reads these artifacts.

6.  **Assemble Final Output:** Using the `template_system.pl`, the orchestrator assembles the final output by injecting the transpiled bash logic, tool function wrappers, and other components into the appropriate templates.

## 4. Configurable Compilation Modes

To provide maximum flexibility, the compiler will support multiple, configurable compilation modes via an `analysis_mode(Mode)` option. The primary modes are:

### a. `analysis_mode(pre_compiled)`
*   **Description:** The default mode. The orchestration described in Section 3 is performed once at compile-time.
*   **Output:** A simple, highly efficient bash script with the decision logic "baked in."
*   **Use Case:** Production environments where runtime performance is critical.

### b. `analysis_mode(runtime_prolog)`
*   **Description:** The compiler generates a bash script that contains the raw Prolog rules embedded within it.
*   **Output:** A bash script that uses the "Prolog as a Service" pattern. On execution, it invokes `swipl` to run the analysis and determine the correct strategy just-in-time.
*   **Use Case:** Development and situations requiring maximum runtime flexibility, where input parameters (like budget) change frequently.

### c. `analysis_mode(self_transpiling)`
*   **Description:** A hybrid JIT (Just-in-Time) compilation model. The compiler produces a script similar to `runtime_prolog`.
*   **Output:** A bash script that, on its *first run*, invokes the UnifyWeaver transpiler on its own embedded rules to generate a pure-bash version of itself, which it then uses for all subsequent runs.
*   **Use Case:** Deploying flexible agents that can self-optimize into a high-performance, zero-dependency script in the target environment.

## 5. Next Steps

The immediate next step in implementing this architecture is to build out the orchestration logic in `playbook_compiler.pl`, specifically:

1.  Implement the logic to save the extracted Prolog blocks to a temporary file.
2.  Add the call to `consult/1` to load the temporary file.
3.  Add the call to `compiler_driver:compile/3` to trigger the existing UnifyWeaver transpiler.
