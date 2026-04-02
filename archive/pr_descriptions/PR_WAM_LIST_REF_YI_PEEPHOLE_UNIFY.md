# PR: feat: get_list heap-ref support, Yi peephole rules, and unify_value unification

**Title:** `feat: get_list heap-ref, Yi peephole, and unify_value unification`

## Summary

- Add `ref(Addr)` handling to `get_list` read mode — when a register holds a heap reference to a `./2` structure (from `put_list`), look up head and tail on the heap via `heap_subargs`, completing the `put_list` -> `get_list` round-trip through the heap
- Extend `unify_value` from strict `==` equality to full unification: if `Xn` is unbound, bind it to the sub-argument; if the sub-argument is unbound, succeed unconditionally
- Extend `unify_constant` to accept unbound sub-arguments (unifies with any constant)
- Add `match_put_variable_put_value` peephole rule to eliminate redundant `put_variable Xn, Ai` + `put_value Xn, Ai` identity pairs (works for both Xi and Yi registers)
- Archive PR descriptions for PR #1144

## Test plan

- [x] All 7 WAM target compiler tests pass
- [x] All 20 E2E tests pass — including new tests for `get_list` head matching (`e2e_cons/3`) and `unify_value` with unbound variables (`e2e_pair_first/2`)
- [x] Verified `e2e_cons(a, [b,c], [a,b,c])` succeeds via `get_list` decomposition + `unify_value`
- [x] Verified `e2e_pair_first(pair(hello, world), hello)` succeeds with unbound second sub-arg
- [x] Exit code 0 on success, 1 on failure (CI-compatible)

Generated with [Claude Code](https://claude.com/claude-code)
