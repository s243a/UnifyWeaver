# PR: feat: Phase 4+5 — WAM fallback in Rust target with configurable exclusion

**Title:** `feat: Phase 4+5 — WAM fallback in Rust target with exclusion control`

## Summary

Completes the WAM-to-Rust transpilation implementation plan (Phases 0–5).

**Phase 4 — WAM fallback integration:**
- Add final clause to `compile_predicate_to_rust_normal` that catches predicates failing all native lowering tiers and compiles them via WAM → Rust wrapper
- Diagnostic message: `WAM fallback: pred/arity resists native lowering, using WAM compilation`
- Import `wam_target` and `wam_rust_target` modules at the top of `rust_target.pl`

**Configurable exclusion:**
- Per-call: `compile_predicate_to_rust(P, [wam_fallback(false)], Code)` — explicitly disables WAM fallback
- Global: `set_prolog_flag(rust_wam_fallback, false)` — disables for all compilations
- Default: enabled — WAM fallback is the safety net
- This lets users see how far native lowering can push without WAM, useful for benchmarking and identifying predicates that could benefit from new native patterns

**Phase 5 — End-to-end testing:**
- `test_resistant/3` (multi-clause rule with body calls to `test_resistant_helper/2`) resists all native tiers and triggers WAM fallback
- Verified WAM fallback disabled via `wam_fallback(false)` option — compilation fails as expected
- Verified native lowering still preferred for `test_simple_fact/2` even with fallback enabled
- Verified global `rust_wam_fallback` Prolog flag controls fallback
- Validated generated Rust wrapper has `fn test_resistant`, `WamState`, `set_reg` structure
- Validated full `impl WamState` runtime block with step/run/backtrack/eval_arith

## Test plan

- [x] All 28 existing WAM tests pass (7 compiler + 21 E2E) — no regressions
- [x] All 12 WAM-Rust target tests pass (6 Phase 2+3 + 6 Phase 4+5)
- [x] WAM fallback triggers only when all native tiers fail
- [x] WAM fallback respects both per-call option and global Prolog flag
- [x] Native lowering still preferred for simple predicates
- [x] Exit code 0 on success, 1 on failure (CI-compatible)

## Implementation plan status

| Phase | Description | Status |
|-------|------------|--------|
| 0 | Rust binding registry | DONE (#1155) |
| 1 | Mustache templates | DONE (#1155) |
| 2 | step_wam → Rust match | DONE (#1156) |
| 3 | Helper predicates → Rust | DONE (#1156) |
| 4 | WAM fallback integration | DONE (this PR) |
| 5 | E2E testing | DONE (this PR) |

Generated with [Claude Code](https://claude.com/claude-code)
