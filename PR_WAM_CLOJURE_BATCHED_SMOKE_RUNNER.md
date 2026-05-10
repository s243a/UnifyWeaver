# test(wam-clojure): batch runtime smoke checks

## Summary

Adds a batched Clojure WAM smoke-test entrypoint so the runtime smoke suite can validate many predicates through one JVM process.

## Why

The Clojure runtime smoke test previously launched a fresh JVM for every predicate assertion. On Termux this made the full smoke suite slow enough to time out, even when the generated Clojure code was correct. Batching the checks keeps validation practical in the local mobile environment and reduces friction for future Clojure lowered-builtin work.

## What Changed

- Added generated Clojure `--batch` mode that reads EDN test cases from stdin.
- Preserved the existing single-predicate CLI behavior.
- Refactored `tests/test_wam_clojure_runtime_smoke.pl` to collect expected predicate checks into `smoke_cases/1`.
- Runs all smoke predicate checks through one Java process.
- Added generator assertions for the new batch entrypoint.

## Validation

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`

## Result

The full Clojure runtime smoke suite now completes successfully in about 4 seconds in Termux instead of timing out due to repeated JVM startup.
