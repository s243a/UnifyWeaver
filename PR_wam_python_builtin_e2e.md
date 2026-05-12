# PR Title

Add Python WAM builtin generated-project E2E tests

# PR Description

## Summary

- adds generated-project E2E coverage for Python WAM builtin parity
- verifies newly covered builtins through the packaged static runtime copied into generated Python projects
- updates the Python WAM parity audit to mark generated-project builtin coverage complete

## Details

This PR extends `tests/test_wam_python_target.pl` with E2E tests that build a real Python WAM project via `write_wam_python_project/3` and execute `main.py`.

The generated-project tests cover:

- term builtins: `functor/3`, `arg/3`, `=../2`
- copying, control, and IO: `copy_term/2`, `\+/1`, `write/1`, `nl/0`
- type and comparison builtins: `float/1`, `number/1`, `compound/1`, `is_list/1`, `==/2`, `=:=/2`, `=\=/2`

This complements the static-runtime parity guard by proving the generated project path exercises the packaged `WamRuntime.py` behavior.

## Verification

```sh
python3 -m py_compile src/unifyweaver/targets/wam_python_runtime/WamRuntime.py
swipl -q -g run_tests -t halt tests/test_wam_python_target.pl
swipl -q -g run_tests -t halt tests/test_wam_python_effective_distance_smoke.pl
swipl -q -g "use_module(src/unifyweaver/targets/wam_python_target), halt"
```
