# PR Title

feat(wam-clojure): add memberchk/2 builtin

# PR Description

## Summary

- Adds Clojure WAM support for `memberchk/2`.
- Direct-lowers `memberchk/2` to a deterministic runtime helper.
- Reuses the existing proper-list traversal used by `member/2`.
- Avoids choicepoint creation so `memberchk/2` commits to the first successful match.
- Adds lowered-emitter and runtime smoke coverage.

## Semantics

`memberchk/2` scans a proper list in the second argument and succeeds on the first element that unifies with the first argument.

Unlike `member/2`, this implementation is deterministic: it advances immediately on the first match and does not leave alternatives for backtracking.

Unsupported cases such as unbound or improper input lists fail cleanly.

## Validation

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
