# Add Go WAM member/2 choicepoint enumeration

## Description

Adds builtin choicepoint support for Go WAM `member/2`, allowing it to enumerate later list members on backtracking instead of stopping at the first unifiable element. This lets aggregate collection paths such as `findall(X, member(X, [a,b]), L)` observe all solutions.

The change also teaches the Go WAM list walker to treat compiler-emitted cons structures such as `[|]/2` as list tails, and makes negated builtin dispatch clean up bindings and choicepoints after probing the inner goal.

## Tests

```sh
swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl
swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl
swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), halt"
```
