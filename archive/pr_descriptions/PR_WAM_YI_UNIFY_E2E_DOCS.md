# PR: docs: update Book 17 for Yi permanent variables and correct WAM output

**Title:** `docs: update Book 17 for Yi permanent variables and correct WAM output`

## Summary

- Update Yi register documentation in Chapter 2 (ISA) from "planned enhancement" to fully implemented, with explanation of automatic permanent variable detection and environment frame storage
- Update `grandparent/2` and `ancestor/2` compilation examples in Chapter 3 to show correct Yi register allocation with annotated WAM output
- Document the `allocate`-before-head and `deallocate`-after-Yi-reads placement rationale

## Test plan

- [x] Documentation examples match actual compiler output
- [x] Yi register descriptions consistent with runtime `get_reg`/`put_reg` behavior

## Companion PR

- Main repo: `feat: Yi permanent variables, unify_* runtime, and E2E rule test` (same branch name)

Generated with [Claude Code](https://claude.com/claude-code)
