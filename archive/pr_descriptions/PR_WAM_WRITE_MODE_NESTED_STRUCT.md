# PR: feat: write mode E2E, nested put_structure, and heap-ref get_structure

**Title:** `feat: write mode E2E, nested put_structure, and heap-ref get_structure`

## Summary

- Add E2E test for write mode: `e2e_lookup(255)` builds `rgb(255,0,0)` via `put_structure` on the heap, then `e2e_color/2` reads it back via `get_structure` on a heap ref — validates the full put_structure → get_structure round-trip
- Add E2E test for nested compound heads: `e2e_nested(pair(a,b), yes)` exercises `get_structure` on a non-Ai register (Xn) via the compiler's nested `get_structure` + `unify_*` emission
- Compiler now recursively emits `put_structure` + `set_*` for nested compound sub-arguments in body construction (e.g., `box(inner(X, done))` emits `put_structure box/1` + `set_variable X3` + `put_structure inner/2, X3` + `set_value X1` + `set_constant done`)
- Runtime `get_structure` read mode now handles `ref(Addr)` values from heap-constructed structures — looks up `str(F/N)` at the heap address and extracts sub-arguments via `heap_subargs/4`
- Add compiler test for nested `put_structure` emission
- Archive PR descriptions for PR #1121

## Test plan

- [x] All 7 WAM target compiler tests pass (facts, single clause, recursion, put_structure, nested put_structure, compound head, module)
- [x] All 7 E2E tests pass (fact execution, backtracking, grandparent, recursive ancestor, compound head, nested compound head, write mode)
- [x] Verified `e2e_lookup(255)` succeeds through heap-ref round-trip
- [x] Verified nested `put_structure` output: `put_structure box/1` + `put_structure inner/2, X3`
- [x] Exit code 0 on success, 1 on failure (CI-compatible)

Generated with [Claude Code](https://claude.com/claude-code)
