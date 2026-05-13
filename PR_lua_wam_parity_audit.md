# PR Title

Document Lua WAM parity audit

# PR Description

## Summary

- add a Lua WAM parity audit under `docs/design`
- record the runtime and builtin areas now covered by Lua WAM e2e tests
- document notable semantics for `copy_term/2`, `\+/1`, `CutIte`, and IO behavior
- clarify known non-gaps, including Rust's unimplemented `append/3` and out-of-scope live-store/read predicates

## Verification

- `swipl -q -g run_tests -t halt tests/test_wam_lua_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_target.pl`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_lua_target), halt"`
