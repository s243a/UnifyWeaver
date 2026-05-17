# PR Title

Add Go WAM numlist builtin parity

# PR Description

## Summary

- Add deterministic `numlist/3` support to the generated Go WAM runtime.
- Route Go WAM `call`/`execute` instructions for `numlist/3` through `BuiltinCall` using the Go-local direct builtin path.
- Cover `numlist(+Low,+High,-List)` for closed integer range construction, singleton range construction, and `Low > High` failure.
- Update the Go WAM parity audit for the expanded Clojure/R range-list builtin surface.

## Verification

- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl tests/test_wam_go_generator.pl tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), halt"`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), wam_go_target:wam_instruction_to_go_literal(call(numlist/3,3), X), writeln(X), halt"`
- `git diff --check`
