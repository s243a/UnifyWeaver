# PR Title

Add Go WAM memberchk builtin parity

# PR Description

## Summary

- Add deterministic `memberchk/2` support to the generated Go WAM runtime.
- Route Go WAM `call`/`execute` instructions for `memberchk/2` through `BuiltinCall` without broadening the shared WAM builtin table for other targets.
- Extend generated Go WAM builtin E2E coverage to verify first-match commitment and negative membership failure.
- Update the Go WAM parity audit to include `memberchk/2` in the structural builtin surface.

## Verification

- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl tests/test_wam_go_generator.pl tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), halt"`
- `git diff --check`
