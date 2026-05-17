# PR Title

Add Go WAM term-order comparison builtins

# PR Description

## Summary

- Adds generated Go WAM runtime support for `@</2`, `@=</2`, `@>/2`, `@>=/2`, and `compare/3`.
- Registers the term-order predicates as direct Go WAM builtins so compiled calls emit `BuiltinCall` instructions.
- Extends the generated Go builtin E2E test to cover atom ordering, integer ordering, mixed atom/integer ordering, and `compare/3` results.
- Updates the Go WAM parity audit to record the expanded term-order comparison surface against the Clojure and Haskell-oriented baseline.

## Verification

- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`
- direct emitter checks for `@</2`, `@=</2`, `@>/2`, `@>=/2`, and `compare/3`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), halt"`
- `git diff --check`
