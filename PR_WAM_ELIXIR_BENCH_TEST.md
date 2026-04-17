# Title
feat(wam-elixir): implement benchmark harness and end-to-end integration tests

# Description
This PR introduces a formal benchmarking harness and comprehensive end-to-end integration tests for the WAM-to-Elixir lowered emitter. It also includes several critical bug fixes and architectural refinements to support complex, recursive Prolog workloads.

### Key Enhancements

*   **Benchmarking Infrastructure**:
    *   **`benchmark_wam_elixir_effective_distance.py`**: A new Python driver that automates building and timing the WAM-Elixir target against standard Wikipedia graph scales (300, 1k, 5k).
    *   **`generate_wam_elixir_effective_distance_benchmark.pl`**: Generates a complete Elixir project for the `effective_distance.pl` workload, including a custom Elixir driver that computes per-seed weight sums.
*   **Backtracking & Choice Points**:
    *   Updated the lowered emitter to push choice points containing Elixir function references (`&clause_name/1`) to the `state.choice_points` list.
    *   Modified `WamRuntime.backtrack/1` to correctly pop and execute these function references, enabling the driver to backtrack into lowered predicates (unlocking all-results queries).
*   **Core Runtime Robustness**:
    *   **Infinite Loop Protection**: Added self-reference checks in `deref_var/2` to prevent infinite recursion on unbound WAM variables.
    *   **Arithmetic & Built-ins**: Expanded `execute_builtin/3` to support `length/2`, `</2`, and negation (`\+/1`).
    *   **Heap Evaluation**: Implemented `eval_arith/2` with recursive heap traversal, allowing the WAM runtime to evaluate compound arithmetic expressions (e.g., `Hops + 1`) stored on the heap.
*   **Elixir Compatibility**:
    *   **CamelCase Modules**: Implemented `camel_case/2` utility to ensure all generated Elixir module names are syntactically valid (e.g., `WamPredLow.CategoryAncestor` instead of `WamPredLow.category_ancestor`).
    *   **PC Management**: Fixed `pc` increment logic in lowered mode to treat `state.pc` as an opaque identifier when it contains function references, preventing arithmetic errors during instruction execution.

### Bug Fixes
*   Resolved multiple unbound variable warnings and bugs in `wam_elixir_target.pl`.
*   Corrected `get_list` unification logic for native Elixir lists to properly push `[head, tail]` to the unification stack.

### Testing
*   **E2E Validation**: Successfully verified the full pipeline (compile → emit → run) using recursive predicates (`ancestor/2`, `member/2`).
*   **Backtracking Test**: Confirmed that `WamRuntime.backtrack(state)` correctly finds subsequent results in lowered mode.

---
*Co-authored-By: Claude Opus 4.6 (1M context)*
