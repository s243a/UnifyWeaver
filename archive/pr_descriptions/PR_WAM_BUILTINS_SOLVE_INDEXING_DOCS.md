# PR: docs: add builtin_call, switch_on_constant to Book 17 ISA

**Title:** `docs: add builtin_call and switch_on_constant to Book 17 ISA`

## Summary

- Add `builtin_call(P/N, Arity)` to the Control section of Chapter 2 (ISA) — evaluates built-in predicates inline without jumping to compiled code
- Add new "Indexing" section (section 5) documenting `switch_on_constant(Key1:Label1, ...)` — first-argument dispatch that skips inapplicable clauses

## Test plan

- [x] Documentation matches compiler emission and runtime `step_wam` clauses
- [x] Instruction descriptions consistent with `is_builtin_pred` and `build_first_arg_index`

## Companion PR

- Main repo: `feat: WAM builtins, solve_wam, indexing, and clause_body_analysis refactor` (same branch name)

Generated with [Claude Code](https://claude.com/claude-code)
