# PR Title

Add Go WAM last builtin parity

# PR Description

## Summary

- Add deterministic `last/2` support to the generated Go WAM runtime.
- Route Go WAM `call`/`execute` instructions for `last/2` through `BuiltinCall` using the Go-local direct builtin path.
- Fix generated cons-chain traversal so nested heap-tail placeholders resolve to the actual following cons cells.
- Add list-aware unification between flat Go list values and generated cons-chain list terms.
- Extend generated Go WAM builtin E2E coverage for non-empty list success and empty-list failure.
- Update the Go WAM parity audit for the expanded Clojure/R/C++ list-builtin surface.

## Verification

- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl tests/test_wam_go_generator.pl tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), halt"`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), wam_go_target:wam_instruction_to_go_literal(call(last/2,2), X), writeln(X), halt"`
- `git diff --check`
