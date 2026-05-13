# PR Title

Add Lua WAM term inspection builtins

# PR Description

## Summary

- add Lua WAM runtime support for `functor/3` in read and construct modes
- add Lua WAM runtime support for `arg/3` over compound terms and WAM cons-list cells
- add Lua generator end-to-end coverage for functor read mode, atom arity, construct mode, struct argument extraction, list argument extraction, and out-of-range failure

## Verification

- `swipl -q -g run_tests -t halt tests/test_wam_lua_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_target.pl`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_lua_target), halt"`
