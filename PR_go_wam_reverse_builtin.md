# PR Title

Add Go WAM reverse builtin parity

# PR Description

## Summary

- Add deterministic `reverse/2` support to the generated Go WAM runtime.
- Route Go WAM `call`/`execute` instructions for `reverse/2` through `BuiltinCall` using the existing Go-local direct builtin path.
- Extend generated Go WAM builtin E2E coverage for both forward and reverse-list binding modes.
- Update the Go WAM parity audit to record the expanded Clojure/R/C++ list-builtin parity surface.

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
