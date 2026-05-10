# PR Title

Add Lua WAM control builtins

# PR Description

## Summary

- add Lua WAM runtime support for `true/0`, `fail/0`, `!/0`, `\+/1`, and `CutIte`
- evaluate negated goals in an isolated substate so bindings inside `\+/1` do not leak
- add Lua generator end-to-end coverage for cut over alternatives, if-then-else soft cut, negation success/failure, and negated `member/2`

## Verification

- `swipl -q -g run_tests -t halt tests/test_wam_lua_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_target.pl`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_lua_target), halt"`
