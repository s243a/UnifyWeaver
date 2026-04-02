# PR: docs: WAM-to-Rust transpilation design (philosophy, spec, impl plan)

**Title:** `docs: WAM-to-Rust transpilation design documents`

## Summary

- Add **Philosophy** document establishing the "transpile, don't rewrite" principle — UnifyWeaver compiles its own `wam_runtime.pl` to Rust using its existing compilation pipeline, maintaining a single source of truth. Templates provide idiomatic Rust crate structure, native lowering fills function bodies, both applied recursively via `compile_expression/6`. The WAM runtime's predicates are themselves mostly native-lowerable.
- Add **Specification** defining concrete Rust types (`Value` enum, `WamState` struct, `Instruction` enum with 25+ variants), predicate classification strategy (native vs WAM vs builtin), interop calling convention (native ↔ WAM-compiled), builtin mapping table (16 Prolog builtins → Rust equivalents), and target capability matrix
- Add **Implementation Plan** with 6 phases: binding registry (Phase 0), Mustache templates (Phase 1), `step_wam/3` → Rust `match` transpilation (Phase 2), helper predicate lowering (Phase 3), WAM fallback integration in Rust target (Phase 4), E2E testing (Phase 5). Includes dependency graph, effort/risk per phase, and metrics table. Architecture extends to WAT/JVM/C downstream targets.
- Archive PR description for PR #1152

## Test plan

- [x] Documents follow established design doc style (philosophy → specification → implementation plan)
- [x] Cross-references existing architecture (clause_body_analysis, template_system, compile_expression/6, fallback_chain)
- [x] Phase dependencies are consistent and parallelizable where noted
- [x] Implementation plan references existing code paths (`compile_predicate_to_rust_normal`, `is_builtin_pred`, `switch_on_structure`)

Generated with [Claude Code](https://claude.com/claude-code)
