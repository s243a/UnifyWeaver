# PR: docs: document put_structure and set_* instructions in Book 17

**Title:** `docs: document put_structure and set_* instructions in Book 17`

## Summary

- Add `put_structure`, `set_variable`, `set_value`, `set_constant` documentation to Chapter 2 (ISA) with descriptions of heap construction semantics
- Add compound body argument compilation example (`wrap/1 :- check(pair(X, done))`) to Chapter 3 with annotated WAM output
- Update README instruction table to include `set_*` instructions under Body Construction

## Test plan

- [x] Documentation matches actual compiler output (verified against `wam_target.pl`)
- [x] Examples are consistent with the ISA definitions

## Companion PR

- Main repo: `feat: add put_structure compound term support and CP safety docs` (same branch name)

Generated with [Claude Code](https://claude.com/claude-code)
