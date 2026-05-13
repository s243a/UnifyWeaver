# feat(wam-clojure): lower member builtin

## Summary

Adds direct lowered Clojure WAM support for `member/2`.

## What Changed

- Marks `member/2` as a direct Clojure lowered builtin.
- Emits lowered `member/2` calls through `runtime/apply-member-solution` instead of routing through the generic WAM step loop.
- Adds Clojure runtime member choice points so backtracking can resume at the next list tail.
- Generalizes foreign choice activation into `activate-choice` so both foreign stream choices and member choices share the backtrack restore path.
- Adds generator, lowered-emitter, and smoke coverage for emitted member support.

## Validation

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- Targeted generated Clojure runtime checks:
  - `wam_member_guard/2` succeeds for first element.
  - `wam_member_guard/2` succeeds for later element.
  - `wam_member_guard/2` fails for missing element.
  - `wam_member_backtrack_b/0` succeeds by backtracking from `a` to `b`.
  - `wam_member_unbound_list/1` fails for an unbound list argument.

## Notes

The full `tests/test_wam_clojure_runtime_smoke.pl` run was attempted with a 240 second timeout in Termux. It generated the project successfully but timed out during the broad multi-invocation Java smoke phase, so targeted Java checks were used for this feature path.
