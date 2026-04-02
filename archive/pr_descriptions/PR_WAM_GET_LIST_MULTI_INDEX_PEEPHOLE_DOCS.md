# PR: docs: add get_list, put_list, switch_on_constant_a2 to Book 17 ISA

**Title:** `docs: add get_list, put_list, switch_on_constant_a2 to Book 17 ISA`

## Summary

- Document `get_list(Ai)` in the Head Unification section — list-specific sugar for `get_structure('./2', Ai)` that decomposes `[H|T]` in read mode or constructs in write mode
- Document `put_list(Ai)` in the Body Construction section — list-specific sugar for `put_structure('./2', Ai)`
- Document `switch_on_constant_a2` in the Indexing section — second-argument dispatch when first-argument indexing is not applicable

## Test plan

- [x] Documentation matches compiler emission and runtime `step_wam` clauses
- [x] Instruction descriptions consistent with `is_list_term` detection in compiler

## Companion PR

- Main repo: `feat: get_list/put_list, second-arg indexing, and peephole optimizer` (same branch name)

Generated with [Claude Code](https://claude.com/claude-code)
