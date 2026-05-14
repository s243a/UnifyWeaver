# PR Title

feat(wam-clojure): add term identity builtins

# PR Description

## Summary

- Adds Clojure WAM lowered-emitter support for `==/2` and `\==/2`.
- Adds runtime `term-identical?`, which dereferences terms but never binds.
- Wires interpreted runtime dispatch for `"==/2"` and `"\\==/2"`.
- Adds lowered-emitter coverage for both builtins.
- Adds runtime smoke coverage for atoms, structures, same variable identity, distinct variable non-identity, alias identity, and non-binding failure behavior.

## Scope

This PR implements strict term identity and non-identity only. It does not implement standard term ordering or comparison builtins such as `@</2`, `@=</2`, `@>/2`, or `@>=/2`.

The key semantic distinction from `=/2` and `\=/2` is that `==/2` and `\==/2` inspect current term identity after dereferencing but do not unify or mutate bindings.

## Validation

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
