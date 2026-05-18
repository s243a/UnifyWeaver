# PR Title

Add Go WAM select builtin

# PR Description

## Summary

- Adds generated Go WAM runtime support for `select/3`.
- Registers `select/3` as a direct Go WAM builtin so compiled calls emit `BuiltinCall` instructions.
- Adds choicepoint-backed `select/3` enumeration so `findall/3` can observe each selected element/rest pair.
- Extends the generated Go builtin E2E test to cover direct first/middle/last selection, missing-element failure, empty-list failure, malformed-list failure, and `findall/3` enumeration.
- Updates the Go WAM parity audit to record the structural-list parity closure against the Clojure/C++ surface.

## Verification

- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`
- direct emitter check for `select/3`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), halt"`
- `git diff --check`
