# PR: feat: WAM-Rust runtime enhancements (relational member/2, modular builtins, integration tests)

## Summary
This PR significantly matures the WAM-to-Rust runtime and the core Prolog WAM emulator by modularizing builtin execution, implementing relational backtracking for builtins, and adding robust end-to-end execution tests.

### Key Changes
#### Rust WAM Runtime (Generated)
- **Modular Builtins**: Refactored the WAM dispatcher into categorized helpers (Arithmetic, I/O, Type, Term), improving maintainability.
- **Relational `member/2`**: Implemented non-deterministic `member/2` using a new `builtin_state` mechanism in ChoicePoints. The VM can now automatically backtrack through builtins.
- **Runtime Testing**: Introduced `tests/test_wam_rust_runtime.pl`, verifying that generated Rust code compiles and executes correctly via `cargo test`.
- **Brace-Balance Check**: Added a new unit test to ensure generated Rust code is syntactically valid regarding braces.
- **VM Correctness**: Fixed nested structure construction, improved `deref_heap` for complex bindings, and standardized environment register detection.

#### Prolog WAM Emulator (`wam_runtime.pl`)
- **Robust Assoc Deletion**: Replaced tombstone-based deletion with proper `del_assoc/4`, preventing value leakage.
- **Correct Arithmetic**: Fixed division semantics (float `/` vs integer `//`) and implemented dereferencing chains for arithmetic operands.
- **Relational Integrity**: Implemented WAM-level choice points for `member/2` in the emulator, matching the new Rust behavior.
- **Symbolic Query Mapping**: Added mapping of query variables to symbolic atoms (`_Q1`) to allow reliable results extraction during backtracking.

## Test plan
- [x] All 23 WAM-Rust target code-gen tests pass.
- [x] New `check_brace_balance` test ensures valid Rust syntax generation.
- [x] New runtime integration tests pass, verifying:
    - Fact lookup and recursive rule execution.
    - Non-deterministic `member/2` backtracking.
    - Correct term reconstruction on the heap.
- [x] Prolog emulator tests (`wam_runtime:test_wam_runtime`) pass for basic and relational goals.
