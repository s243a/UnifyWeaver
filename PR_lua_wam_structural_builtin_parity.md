# PR Title

Add Lua WAM structural and builtin parity

# PR Description

## Summary

- add Lua WAM runtime support for `member/2` with choice-point backtracking over list alternatives
- add Lua WAM runtime support for `length/2` over WAM cons-list terms
- add Rust/Haskell parity builtins for Lua WAM type checks and comparisons: `atom/1`, `integer/1`, `float/1`, `number/1`, `compound/1`, `var/1`, `nonvar/1`, `is_list/1`, `==/2`, `=:=/2`, `=\=/2`, `>/2`, `</2`, `>=/2`, and `=</2`
- add Lua generator end-to-end coverage for structural, type, and comparison builtin behavior

## Verification

- `swipl -q -g run_tests -t halt tests/test_wam_lua_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_target.pl`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_lua_target), halt"`
