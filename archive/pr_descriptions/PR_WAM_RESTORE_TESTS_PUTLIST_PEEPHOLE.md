# PR: fix: restore dropped tests, add put_list E2E, fix peephole and heap deref

**Title:** `fix: restore dropped tests, put_list E2E, peephole safety, heap deref`

## Summary

- Restore E2E tests dropped during prior merge: `get_list_match` (cons/3 head matching with execution) and `unify_value_unbound` (pair_first/2 with unbound sub-argument), plus test data for `e2e_cons/3` and `e2e_pair_first/2`
- Add `put_list` body construction E2E test: `e2e_wrap_in_list(X)` constructs `[X]` via `put_list` + `set_value` + `set_constant []`, then passes the heap-constructed list to `member/2` — validates the full put_list -> builtin round-trip
- Fix peephole optimizer safety: `match_get_put_passthrough` now verifies the register is not referenced by any subsequent instruction before eliminating the get/put pair, preventing removal of bindings needed by later `set_value`/`set_constant` instructions
- Add `deref_heap/3` to reconstruct Prolog terms from `ref(Addr)` heap references, including list reconstruction (`./2` -> `[H|T]`). List builtins (`member/2`, `append/3`, `length/2`) now dereference heap refs before delegating to native Prolog operations

## Test plan

- [x] All 7 WAM target compiler tests pass
- [x] All 21 E2E tests pass — including 3 restored/new tests
- [x] Verified `e2e_cons(a, [b,c], [a,b,c])` succeeds via `get_list` decomposition
- [x] Verified `e2e_pair_first(pair(hello, world), hello)` succeeds with unbound second sub-arg
- [x] Verified `e2e_wrap_in_list(hello)` succeeds through full `put_list` -> heap -> `deref_heap` -> `member/2` pipeline
- [x] Verified peephole no longer eliminates `get_variable X1, A1` when X1 is used by later `set_value X1`
- [x] Exit code 0 on success, 1 on failure (CI-compatible)

Generated with [Claude Code](https://claude.com/claude-code)
