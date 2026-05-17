# PR Title

Add Go WAM sort and msort builtins

# PR Description

## Summary

- Adds generated Go WAM runtime support for deterministic `sort/2` and `msort/2`.
- Registers both predicates as direct Go WAM builtins so compiled calls emit `BuiltinCall` instructions.
- Extends the generated Go builtin E2E test to cover sorted output, duplicate removal for `sort/2`, duplicate preservation for `msort/2`, mixed atom/integer ordering, and malformed-list failure.
- Updates the Go WAM parity audit to record the closed structural-list parity gap against the broader cross-target baseline.

## Verification

- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`
- direct emitter checks for `call(sort/2,2)` and `call(msort/2,2)`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), halt"`
- `git diff --check`
