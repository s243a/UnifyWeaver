# feat(wam-clojure): add copy_term/2 builtin support

## Summary

Adds Clojure WAM runtime and lowered-emitter support for `copy_term/2`.

## What Changed

- Added a sharing-preserving `copy-term` walker to the generated Clojure WAM runtime.
- Routed interpreted `copy_term/2` builtin calls through `apply-copy-term-solution`.
- Registered `copy_term/2` as a direct lowered Clojure builtin.
- Added lowered-emitter coverage proving `copy_term/2` bypasses generic `runtime/step`.
- Extended the Clojure runtime smoke suite with ground copy, repeated-variable sharing, and independent-variable copy cases.

## Validation

```sh
git diff --check
swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl
swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl
timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl
```

## Notes

- This keeps the scope focused on `copy_term/2` parity for the Clojure hybrid WAM path.
- Broader generated-build cyclic-term behavior is left unchanged; this PR only validates `copy_term/2` semantics independently of that older runtime edge case.
