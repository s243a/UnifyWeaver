# Add Go WAM =/2 builtin support

## Description

Adds explicit Go WAM runtime handling for the `=/2` builtin by routing it through normal WAM unification. This closes the documented Go parity gap where `\=/2` existed but `=/2` did not have a direct builtin handler.

The generated Go builtin E2E now forces an emitted `Op: "=/2"` and verifies both successful structural binding and use under negation with `\+ =(a, b)`.

## Tests

```sh
swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl
swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl
swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), halt"
```
