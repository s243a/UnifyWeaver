# PR: feat: WAM instruction parser for Rust and cargo_check validation

**Title:** `feat: WAM instruction parser for Rust and cargo_check validation`

## Summary

- Complete the WAM-to-Rust pipeline so generated predicates actually execute: `compile_wam_predicate_to_rust` now embeds real `Instruction::*` enum literals and label `HashMap` inserts instead of a TODO stub, and calls `vm.run()` for execution
- Add `wam_code_to_rust_instructions/3` that parses WAM instruction strings line by line, converting each to a Rust enum literal — handles all 28 instruction types (constants with `Value::Atom`/`Value::Integer` wrapping, registers with `.to_string()`, labels, arity extraction)
- Add `wam_line_to_rust_instr/2` with clauses for every WAM instruction: head unification (8), body construction (8), control (6), choice points (3), and indexing (3)
- Enhance `parse_instructions` in `state.rs.mustache` to fully implement indexing instruction parsing (`switch_on_constant`, `switch_on_structure`, `switch_on_constant_a2`) and support Boolean values (`true`, `false`) during runtime WAM loading
- Improve parser resilience: added guards against empty input in `parse_single_instruction` and implemented error propagation (`ok()?`) for arity parsing in `call` and `builtin_call`
- Update stale comments in `state.rs.mustache` to reflect that Phase 2 (step/run transpilation) is now live
- Add `cargo_check_project/2` that runs `cargo check` on a generated Rust project directory, returning `ok`, error(ExitCode, Output), or `not_available` — handles missing cargo and shell errors gracefully

## Test plan

- [x] All 28 existing WAM tests pass (7 compiler + 21 E2E) — no regressions
- [x] All 23 WAM-Rust target tests pass (6 Phase 2+3 + 6 Phase 4+5 + 3 project gen + 8 parser/cargo/resilience)
- [x] Instruction parser generates real `Instruction::GetConstant`, `Instruction::Proceed`, etc. — no TODOs in output
- [x] Runtime `parse_instructions` in Rust correctly parses indexing dispatch tables (e.g., `alice:default, bob:L1`)
- [x] Boolean values `true`/`false` are correctly parsed into `Value::Bool`
- [x] Parser handles malformed input (empty lines, invalid arity) gracefully without panicking
- [x] Label map correctly extracts predicate and clause labels with PC indices
- [x] Resistant predicate (`test_resistant/3`) generates full WAM code with `TryMeElse`, `Allocate`, `Call`, `vm.run()`
- [x] `cargo_check_project` handles nonexistent directories and missing cargo gracefully
- [x] Exit code 0 on success, 1 on failure (CI-compatible)

Generated with [Claude Code](https://claude.com/claude-code)
