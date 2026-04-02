# PR: feat: Phase 0+1 — Rust WAM bindings and crate templates

**Title:** `feat: Phase 0+1 — Rust WAM bindings and crate templates`

## Summary

Implements Phases 0 and 1 of the WAM-to-Rust transpilation plan (see `docs/design/WAM_RUST_TRANSPILATION_IMPLEMENTATION_PLAN.md`).

**Phase 0 — Rust WAM Bindings:**
- Add `rust_wam_bindings.pl` with 25+ binding declarations mapping Prolog builtins used by `wam_runtime.pl` to idiomatic Rust equivalents
- Assoc operations: `get_assoc/3` → `HashMap::get`, `put_assoc/4` → `HashMap::insert`, `empty_assoc/1` → `HashMap::new()`
- List operations: `nth0/3` / `nth1/3` → Vec indexing, `append/3` → Vec extend, `member/2` → `.iter().any()`, `length/2` → `.len()`
- Type checks: `atom/1` → `matches!(val, Value::Atom(_))`, `var/1` → `.is_unbound()`, `compound/1` → `.is_compound()`
- Term manipulation: `=../2` → `.univ()`, `functor/3` → `.functor()`
- String/atom ops: `atom_string/2`, `atom_concat/3`, `split_string/4`, `number_string/2`, `sub_atom/5`
- Format: `format/2,3` → `format!()` macro
- Type mapping table (`rust_wam_type_map/2`): `assoc` → `HashMap<String, Value>`, `list` → `Vec<Value>`, etc.

**Phase 1 — Mustache Templates:**
- `value.rs`: `Value` enum (8 variants) with helper methods (`is_unbound`, `as_int`, `univ`, `make_str`), `PartialEq` and `Display` impls
- `instructions.rs`: `Instruction` enum (25+ variants) covering the full WAM instruction set including indexing
- `state.rs`: `WamState` struct mirroring the 9-field Prolog state tuple, with `TrailEntry`, `ChoicePoint`, `StackEntry` types, Yi-aware register access, trail management, heap operations, and `deref_heap` reconstruction
- `lib.rs`: Module layout with `{{predicates_code}}` placeholder for transpiled predicate code
- `Cargo.toml`: Package metadata template
- Register `rust_wam_cargo` and `rust_wam_lib` inline templates in `template_system.pl`

## Test plan

- [x] All 28 existing WAM tests pass (7 compiler + 21 E2E) — no regressions
- [x] Bindings module loads and queries correctly (`rust_wam_binding/5`, `rust_wam_type_map/2`)
- [x] Templates registered in template_system.pl without conflicts
- [x] Mustache templates contain valid Rust syntax with `{{placeholder}}` markers

Generated with [Claude Code](https://claude.com/claude-code)
