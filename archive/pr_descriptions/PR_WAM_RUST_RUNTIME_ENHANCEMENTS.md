# feat: WAM-Rust runtime enhancements

## Summary

- Mature the WAM-to-Rust runtime and Prolog WAM emulator with modular builtins, relational backtracking, and hardened core mechanics
- Implement non-deterministic `member/2` in both the Prolog emulator and generated Rust VM using choice-point-based backtracking
- Fix 12 correctness issues across unification, trail management, heap dereferencing, arithmetic, and register handling (identified and resolved over 4 review rounds)

### Rust WAM Runtime (Generated)

- **Modular Builtins**: Refactored the WAM dispatcher into categorized helpers (Arithmetic, I/O, Type, Term)
- **Relational `member/2`**: Non-deterministic `member/2` via a `builtin_state` mechanism in ChoicePoints, enabling automatic backtracking through builtins
- **Runtime Testing**: New `tests/test_wam_rust_runtime.pl` verifying generated Rust compiles and executes correctly via `cargo test`
- **Brace-Balance Check**: Unit test ensuring generated Rust code is syntactically valid
- **VM Correctness**: Fixed nested structure construction, improved `deref_heap` for complex bindings, standardized environment register detection

### Prolog WAM Emulator (`wam_runtime.pl`)

- **Robust Assoc Deletion**: Replaced tombstone-based deletion with proper `del_assoc/4`, preventing value leakage
- **Correct Arithmetic**: Fixed division semantics (float `/` vs integer `//`) and implemented dereferencing chains for arithmetic operands
- **Relational Integrity**: WAM-level choice points for `member/2`, matching the new Rust behavior
- **Symbolic Query Mapping**: Query variables mapped to symbolic atoms (`_Q1`) for reliable results extraction during backtracking
- **Refactored `deref_heap`**: Clearer structure with proper guards for non-atom entries, zero-arity functors, and `once/1`-guarded `sub_atom` to prevent wrong `/` matches in functors like `./2`
- **Accumulator-based `parse_lines`**: Converted label collection to an accumulator pattern for correctness with tail recursion
- **Generalized indexing instructions**: `switch_on_constant`, `switch_on_structure`, and `switch_on_constant_a2` share a single handler supporting both list and variadic forms
- **`extract_bindings` refactor**: Accepts a full `wam_state` instead of separate registers/heap, with dedicated `extract_bindings_iter` helper
- **`get_reg` / `get_reg_val` split**: `get_reg` fails on missing registers (needed by `deref_wam` chain-following); `get_reg_val` returns `'_Vunbound'` (a `_V`-prefixed sentinel recognized by `is_unbound_var`) for step instructions that must tolerate absent registers
- **`is_unbound_var` for native vars**: New `var(Val)` clause handles actual Prolog variables from `copy_term`
- **Negation-as-failure stub**: `\+/1` throws an explicit `not_supported` error instead of silently failing
- **`normalize_line` improvements**: Handles list-form lines and splits on all whitespace characters
- **`get_arity` cleanup**: Uses `sub_atom`/`atom_number` with `once/1` guards for consistency and correctness

## Test plan

- [x] All 23 WAM-Rust target code-gen tests pass
- [x] `check_brace_balance` test ensures valid Rust syntax generation
- [x] Runtime integration tests pass (fact lookup, recursive rules, non-deterministic `member/2`, heap term reconstruction)
- [x] Prolog emulator tests (`test_wam_runtime`) pass for basic and relational goals
- [x] 4 rounds of external review — all 12 identified issues resolved

🤖 Generated with [Claude Code](https://claude.com/claude-code)
