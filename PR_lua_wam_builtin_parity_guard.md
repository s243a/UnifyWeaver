# PR Title

Add Lua WAM builtin parity guard test

# PR Description

## Summary

- add a lightweight guard test for the documented Lua WAM builtin parity surface
- assert that the runtime template still contains the main builtin handlers
- assert that the Lua generator suite still contains e2e test groups for structural, type/comparison, term, univ, copy, control, and IO builtins
- assert that the Lua parity audit doc still records the major covered builtins

## Verification

- `swipl -q -g run_tests -t halt tests/test_wam_lua_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_target.pl`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_lua_target), halt"`
