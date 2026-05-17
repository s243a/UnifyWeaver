# PR Title

feat(wam-clojure): add forward atom concat builtin

# PR Description

## Summary

- Adds Clojure WAM support for deterministic forward `atom_concat/3`.
- Adds `string_concat/3` as an alias over the same atom/string representation.
- Direct-lowers concat calls to a runtime helper instead of routing through the interpreted builtin path.
- Interns the concatenated output atom in the runtime intern context.
- Adds lowered-emitter and runtime smoke coverage for success, mismatch, runtime-created output atoms, alias behavior, and unsupported reverse/unbound mode.

## Semantics

The implementation supports:

- `atom_concat(+Left, +Right, ?Out)`
- `string_concat(+Left, +Right, ?Out)`

Unsupported reverse/split modes fail cleanly for now. This keeps the branch deterministic and avoids introducing nondeterministic split enumeration in the same change.

## Validation

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
