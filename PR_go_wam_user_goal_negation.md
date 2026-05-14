# Add Go WAM user-goal negation

## Description

Expands Go WAM `\+/1` beyond builtin-shaped goals by resolving non-builtin compound goals through shared WAM labels and running them in an isolated cloned VM. Existing builtin negation behavior is preserved, while user predicates can now be probed without leaking bindings, trail entries, stack frames, or choicepoints back into the caller.

This narrows the Go control parity gap against the Haskell/Python isolated-goal negation behavior. The remaining control gap is the parallel/race-style negation path.

## Tests

```sh
swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl
swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl
swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), halt"
```
