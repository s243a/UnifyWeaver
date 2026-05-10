# PR Title

Add Lua WAM univ builtin

# PR Description

## Summary

- add Lua WAM runtime support for `=../2` (`univ`) in decompose mode
- add Lua WAM runtime support for `=../2` compose mode from proper WAM cons-list terms
- add Lua generator end-to-end coverage for struct, atom, and list decomposition, struct/atom composition, and mismatch failure

## Verification

- `swipl -q -g run_tests -t halt tests/test_wam_lua_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_target.pl`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_lua_target), halt"`
