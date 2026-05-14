# Add Go WAM set aggregate support

## Description

Adds Go WAM aggregate finalization for `set` and `setof`, deduplicating collected values in first-seen order to match the current Haskell `nub` behavior. The same runtime helper now also treats `bag` and `bagof` as list-producing aggregate aliases.

This closes the documented Go aggregate parity gap against the Rust/Haskell aggregate baseline and adds generated Go E2E coverage for `aggregate_all(set(X), member(X, [a,b,a]), S)`.

## Tests

```sh
swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl
swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl
swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), halt"
```
