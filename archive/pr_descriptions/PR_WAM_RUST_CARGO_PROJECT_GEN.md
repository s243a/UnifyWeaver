# PR: feat: write_wam_rust_project generates complete Cargo crate

**Title:** `feat: write_wam_rust_project generates complete Cargo crate`

## Summary

- Add `write_wam_rust_project/3` that generates a compilable Cargo crate from a list of Prolog predicates, producing `Cargo.toml`, `src/value.rs`, `src/instructions.rs`, `src/state.rs` (with transpiled WAM runtime), and `src/lib.rs` (with compiled predicates)
- Each predicate is attempted via native Rust lowering first, falling back to WAM compilation — the strategy (`native`/`wam`/`failed`) is recorded as a comment in the generated code
- Add `compile_predicates_for_project/3` that iterates predicates and dispatches to `rust_target` or `wam_rust_target` with error handling
- Add `read_template_file/2` for loading Mustache templates from disk and rendering with date/module placeholders
- Options: `module_name(Name)` for crate naming, `wam_fallback(Bool)` for exclusion control, `include_runtime(Bool)` to toggle transpiled runtime inclusion

## Test plan

- [x] All 28 existing WAM tests pass (7 compiler + 21 E2E) — no regressions
- [x] All 15 WAM-Rust target tests pass (6 Phase 2+3 + 6 Phase 4+5 + 3 project gen)
- [x] Verified generated crate has all 5 files (Cargo.toml, value.rs, instructions.rs, state.rs, lib.rs)
- [x] Verified Cargo.toml contains correct module name and edition
- [x] Verified state.rs contains transpiled `impl WamState` with `fn step`
- [x] Verified lib.rs contains compiled predicate code for WAM-fallback predicates
- [x] Generated test directories cleaned up after each test
- [x] Exit code 0 on success, 1 on failure (CI-compatible)

Generated with [Claude Code](https://claude.com/claude-code)
