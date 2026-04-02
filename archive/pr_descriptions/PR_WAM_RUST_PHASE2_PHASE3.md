# PR: feat: Phase 2+3 — WAM step_wam match arms and helper functions in Rust

**Title:** `feat: Phase 2+3 — WAM step_wam → Rust match and helper transpilation`

## Summary

Implements Phases 2 and 3 of the WAM-to-Rust transpilation plan.

**Phase 2 — `step_wam/3` → Rust `match` expression:**
- Add `wam_rust_target.pl` with `compile_step_wam_to_rust/2` that generates a complete `fn step(&mut self, instr: &Instruction) -> bool` method
- 28 `wam_instruction_arm/2` facts map each `Instruction` enum variant to its Rust implementation
- Head unification (8): `GetConstant`, `GetVariable`, `GetValue`, `GetStructure` (read+write mode), `GetList` (read+write+heap-ref), `UnifyVariable`, `UnifyValue` (with unbound unification), `UnifyConstant`
- Body construction (8): `PutConstant`, `PutVariable`, `PutValue`, `PutStructure`, `PutList`, `SetVariable`, `SetValue`, `SetConstant`
- Control (6): `Allocate`, `Deallocate`, `Call`, `Execute`, `Proceed`, `BuiltinCall`
- Choice points (3): `TryMeElse`, `TrustMe`, `RetryMeElse`
- Indexing (3): `SwitchOnConstant`, `SwitchOnStructure`, `SwitchOnConstantA2`

**Phase 3 — Helper predicates → Rust methods:**
- `run()`: main fetch-step-backtrack execution loop
- `backtrack()`: restore from top choice point without popping (trust_me/retry_me_else handle stack management)
- `unwind_trail()`: undo bindings recorded since saved trail state
- `execute_builtin()`: dispatch for `is/2`, 6 comparison ops, `==/2`, `true/0`, `fail/0`, `!/0`, 8 type check predicates
- `eval_arith()` + `eval_arith_compound()`: recursive arithmetic evaluation with heap reference dereferencing
- `compile_wam_predicate_to_rust/4`: WAM-fallback wrapper function generator

**Assembly:**
- `compile_wam_runtime_to_rust/2` combines Phase 2 + Phase 3 into a complete `impl WamState { ... }` block

## Test plan

- [x] All 28 existing WAM tests pass (7 compiler + 21 E2E) — no regressions
- [x] All 6 new WAM-Rust target tests pass: step arm generation, helper functions, full runtime assembly, all 28 instruction types covered, builtin dispatch completeness, predicate wrapper generation
- [x] Generated Rust contains valid match arm structure for all instruction types
- [x] Exit code 0 on success, 1 on failure (CI-compatible)

Generated with [Claude Code](https://claude.com/claude-code)
