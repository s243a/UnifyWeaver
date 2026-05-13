# PR Title

Add Lua WAM IO builtins

# PR Description

## Summary

- add Lua WAM runtime support for `write/1`, `display/1`, and `nl/0`
- add term formatting for atoms, numbers, variables, lists, and compound terms
- add Lua generator end-to-end coverage for generated `write/1` plus `nl/0`
- add direct Lua runtime coverage for `display/1`

## Verification

- `swipl -q -g run_tests -t halt tests/test_wam_lua_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_target.pl`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_lua_target), halt"`
