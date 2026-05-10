# PR Title

Index Lua WAM external fact sources

# PR Description

## Summary

- Add first-argument indexing for loaded Lua WAM external `P/2` fact sources.
- Cache external fact source loads as `{ rows, arg1_index }` instead of a flat row list.
- Teach `CallFactStream` to use the external source `arg1_index` when `A1` is already bound, falling back to the full row scan only when needed.
- Expand external fact source tests to cover multiple first-argument groups and assert the cached index shape.

## Verification

- `swipl -q -g run_tests -t halt tests/test_wam_lua_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_target.pl`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_lua_target), halt"`

## Notes

This aligns Lua external fact-source behavior with Haskell's `fsLookupArg1` path and the R grouped-by-first fact-source backend, while keeping the Lua implementation dependency-free.
