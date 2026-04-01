# PR: docs: add switch_on_structure and switch_on_term to Book 17 ISA

**Title:** `docs: add switch_on_structure and switch_on_term to Book 17 ISA`

## Summary

- Add `switch_on_structure(F/N1:Label1, ...)` documentation to the Indexing section — compound first-argument dispatch by functor/arity
- Add `switch_on_term(constant:..., structure:...)` documentation — type-based dispatch for predicates with mixed first argument types

## Test plan

- [x] Documentation matches compiler `classify_first_args` / `build_structure_index` / `format_switch_on_term` behavior
- [x] Instruction descriptions consistent with runtime `step_wam` clauses

## Companion PR

- Main repo: `feat: list builtins, findall_wam, and structure/term indexing` (same branch name)

Generated with [Claude Code](https://claude.com/claude-code)
