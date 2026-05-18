# PR Title

Add Go WAM succ builtin

# PR Description

## Summary

- Adds generated Go WAM runtime support for bidirectional `succ/2`.
- Registers `succ/2` as a direct Go WAM builtin so compiled calls emit `BuiltinCall` instructions.
- Extends the generated Go builtin E2E test to cover forward binding, reverse binding, matching integer pairs, mismatch failure, negative predecessor failure, non-positive successor failure, and both-unbound failure.
- Updates the Go WAM parity audit to record the expanded successor arithmetic surface against the sibling Clojure and Elixir baseline.

## Verification

- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`
- direct emitter check for `succ/2`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), halt"`
- `git diff --check`
