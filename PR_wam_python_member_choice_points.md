# PR Title

Add Python WAM member choice points

# PR Description

## Summary

- Make Python WAM `member/2` enumerate list members via builtin choice points instead of returning only the first solution.
- Pass builtin resume PCs through normal runtime execution and aggregate body execution so backtracking consumers can resume `member/2`.
- Add generated-project E2E coverage proving `member/2` enumeration through aggregate collection.
- Update the Python WAM parity audit to mark the remaining structural builtin gap closed.

## Verification

- `swipl -q -g run_tests -t halt tests/test_wam_python_target.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_python_effective_distance_smoke.pl`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_python_target), halt"`
- `git diff --check`
