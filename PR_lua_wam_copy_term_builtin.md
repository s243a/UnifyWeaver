# PR Title

Add Lua WAM copy_term builtin

# PR Description

## Summary

- add Lua WAM runtime support for `copy_term/2`
- copy terms with fresh variables while preserving sharing for repeated source variables
- add a self-unify guard for identical unbound variables to avoid self-binding dereference loops
- add Lua generator end-to-end coverage for ground copies, freshness, shared variables, and independent variables

## Verification

- `swipl -q -g run_tests -t halt tests/test_wam_lua_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_target.pl`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_lua_target), halt"`
