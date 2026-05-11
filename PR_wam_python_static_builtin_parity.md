# PR Title

Add Python WAM static runtime builtin parity

# PR Description

## Summary

- documents Python WAM parity against the Lua/Rust/Haskell baseline
- adds missing baseline builtins to the packaged static Python WAM runtime
- adds a parity guard so the static runtime cannot silently drift from the baseline

## Details

The generated Python WAM project path copies `src/unifyweaver/targets/wam_python_runtime/WamRuntime.py`, so this PR focuses on that packaged runtime rather than the generated fallback runtime.

The static runtime now covers the Lua/Rust/Haskell builtin baseline for:

- type checks: `atom/1`, `integer/1`, `float/1`, `number/1`, `compound/1`, `var/1`, `nonvar/1`, `is_list/1`
- comparison and unification: `==/2`, `=/2`, `\=/2`, `=:=/2`, `=\=/2`, `</2`, `>/2`, `=</2`, `>=/2`
- term operations: `functor/3`, `arg/3`, `=../2`, `copy_term/2`
- control and IO: `true/0`, `fail/0`, `!/0`, `\+/1`, `write/1`, `display/1`, `nl/0`

`\\+/1` now evaluates callable goals against an isolated copy of the current state so bindings from the negated goal do not leak into the caller.

## Verification

```sh
python3 -m py_compile src/unifyweaver/targets/wam_python_runtime/WamRuntime.py
swipl -q -g run_tests -t halt tests/test_wam_python_target.pl
swipl -q -g run_tests -t halt tests/test_wam_python_effective_distance_smoke.pl
```
