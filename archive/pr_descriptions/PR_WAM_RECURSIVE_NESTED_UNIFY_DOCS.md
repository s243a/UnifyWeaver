# PR: docs: document read/write mode for get_structure and unify_* in Book 17

**Title:** `docs: document read/write mode for get_structure and unify_* in Book 17`

## Summary

- Update `get_structure` documentation in Chapter 2 (ISA) to describe dual-mode operation: read mode for matching existing compound terms, write mode for constructing new structures when the register is unbound
- Update `unify_variable`, `unify_value`, `unify_constant` descriptions to explain both read mode (matching sub-arguments) and write mode (pushing values onto the heap)

## Test plan

- [x] Documentation matches runtime `step_wam` clause behavior for both modes
- [x] Descriptions consistent with compiler emission in `compile_unify_arguments`

## Companion PR

- Main repo: `feat: recursive E2E, nested compound unification, and write mode` (same branch name)

Generated with [Claude Code](https://claude.com/claude-code)
