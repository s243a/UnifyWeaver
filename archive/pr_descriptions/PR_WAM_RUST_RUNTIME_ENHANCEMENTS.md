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
- **Refactored `deref_heap`**: Rewrote heap dereferencing with clearer structure and proper guards for non-atom entries and zero-arity functors.
- **Accumulator-based `parse_lines`**: Converted label collection from post-recursion `put_assoc` to an accumulator pattern for correctness with tail recursion.
- **Generalized indexing instructions**: `switch_on_constant`, `switch_on_structure`, and `switch_on_constant_a2` now share a single handler supporting both list and variadic argument forms.
- **`extract_bindings` refactor**: Takes a full `wam_state` instead of separate registers and heap, with a dedicated `extract_bindings_iter` helper.
- **Improved `normalize_line`**: Handles list-form lines and splits on all whitespace characters (tabs, newlines, carriage returns).
- **`get_arity` cleanup**: Switched from `sub_string`/`number_string` to `sub_atom`/`atom_number` for consistency.
- **Negation-as-failure stub**: `\+/1` builtin now throws an explicit `not_supported` error instead of silently failing.
- **Bug fix: preserved `get_reg_val` soft lookup**: Restored the `get_reg_val` wrapper that returns `unbound_missing` for absent registers, which is required by step instructions (`put_list`, `put_structure`, etc.) while keeping `get_reg` as a hard (failing) lookup that `deref_wam`'s register chain-following depends on.

## Test plan
- [x] All 23 WAM-Rust target code-gen tests pass.
- [x] New `check_brace_balance` test ensures valid Rust syntax generation.
- [x] New runtime integration tests pass, verifying:
    - Fact lookup and recursive rule execution.
    - Non-deterministic `member/2` backtracking.
    - Correct term reconstruction on the heap.
- [x] Prolog emulator tests (`wam_runtime:test_wam_runtime`) pass for basic and relational goals.
