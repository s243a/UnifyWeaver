# PR Title

Add Go WAM delete builtin

# PR Description

## Summary

- Adds generated Go WAM runtime support for deterministic `delete/3` over proper lists.
- Registers `delete/3` as a direct Go WAM builtin so compiled calls emit `BuiltinCall` instructions.
- Extends the generated Go builtin E2E test to cover removing one matching atom, removing no matches, removing all matches, and malformed-list failure.
- Updates the Go WAM parity audit to record the structural-list parity closure against the Clojure/C++ surface.

## Verification

- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`
- direct emitter check for `delete/3`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), halt"`
- `git diff --check`
