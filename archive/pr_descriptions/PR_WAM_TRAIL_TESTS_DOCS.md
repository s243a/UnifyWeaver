# PR: docs: document unify_* instructions in Book 17 ISA

**Title:** `docs: document unify_variable, unify_value, unify_constant in Book 17`

## Summary

- Add `unify_variable(Xn)`, `unify_value(Xn)`, and `unify_constant(C)` to the head unification section of Chapter 2 (ISA)
- These instructions are emitted by the compiler inside `get_structure` sequences but were previously undocumented
- Clarify that `get_structure(F/N, Ai)` must be followed by N `unify_*` instructions for sub-argument matching

## Test plan

- [x] Documentation matches compiler implementation in `compile_unify_arguments/4`
- [x] Instruction descriptions are consistent with runtime `step_wam` clauses

## Companion PR

- Main repo: `feat: add WAM trail unwinding and CI-compatible test harness` (same branch name)

Generated with [Claude Code](https://claude.com/claude-code)
