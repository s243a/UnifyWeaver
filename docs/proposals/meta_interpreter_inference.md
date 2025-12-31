# Proposal: Meta-Interpreter for Pipeline Step Inference

## Context
UnifyWeaver currently allows users to define pipelines explicitly using `generate_pipeline/3`, where the list of `step/4` terms must be manually constructed. Users also define high-level declarative logic (e.g., `process_data/2`) and map predicates to targets using `declare_target/2`.

## Problem
There is a gap between the high-level declarative logic and the concrete pipeline definition. Users have to effectively "compile" their Prolog logic into the `Steps` list manually, which is redundant and error-prone. The system should be able to infer the execution steps directly from the high-level goal and the target declarations.

## Proposed Solution
We propose implementing a **Meta-Interpreter** within the UnifyWeaver compiler. This component will:
1.  Analyze a high-level Prolog goal (e.g., `process_data(Input, Output)`).
2.  Introspect the goal's body to identify sub-goals (logical steps).
3.  Consult `target_mapping` declarations to resolve each sub-goal to a specific target, script file, and step name.
4.  Automatically construct the list of `step/4` terms required by `generate_pipeline/3`.

This enables users to drive pipeline generation directly from their business logic, treating `generate_pipeline/3` as a lower-level assembly instruction that is automatically invoked.

## Proof of Concept
A working proof-of-concept has been implemented in:
`examples/glue/my_pipeline_example.pl`

This example demonstrates:
-   **`infer_steps_from_goal/2`**: A predicate that simulates the internal compiler logic.
-   **Integration**: How `generate_process_data_pipeline_inferred/1` uses inference to build the pipeline dynamically.

## Next Steps
1.  Formalize `infer_steps_from_goal/2` into the core `unifyweaver` compiler module.
2.  Extend inference to handle more complex Prolog structures (e.g., control flow, failure handling).
3.  Expose a top-level predicate, e.g., `compile_goal_to_pipeline(+Goal, +Options, -Script)`, to streamline the user experience.
