# PR Title

Add Go WAM builtin parity fixes

# PR Description

## Summary

- Fix Go WAM builtin dispatch for `=</2`.
- Add Go WAM runtime support for `is_list/1`, `display/1`, `functor/3`, `arg/3`, `=../2`, and `copy_term/2`.
- Add runtime helpers for term functor/arity extraction, univ conversion, structure construction, and sharing-preserving fresh `copy_term/2`.
- Extend the generated Go WAM builtin E2E test to compile and execute the new builtin coverage.
- Update the Go parity audit to mark the small builtin and term builtin parity slices complete.

## Verification

- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_target.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_python_target.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_lua_generator.pl`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), halt"`
- `git diff --check`
