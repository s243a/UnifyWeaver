# PR Title

Fix Go WAM atom_number/2 reverse-mode atom output

# PR Description

## Summary

- Fix Go WAM `atom_number/2` reverse mode to emit plain interned numeric atoms.
- Restore compatibility with current generated atom literals such as `"42"` and `"3.5"`.
- Fix the post-merge generated Go WAM builtin regression where `ATOM_NUMBER_SUCCESS` became `ATOM_NUMBER_FAILURE`.

## Verification

- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), halt"`
- `git diff --check`
