# PR Title

feat(wam-clojure): split atom_concat outputs

# PR Description

## Summary

- Extend the lowered Clojure WAM runtime handling for `atom_concat/3` and `string_concat/3` beyond fully-bound concatenation.
- Add deterministic one-sided split support for cases where exactly one input side is unbound and the output text is bound.
- Keep both-inputs-unbound split mode conservative because SWI-Prolog produces multiple possible splits.
- Add runtime smoke coverage and generated-code assertions for the new atom and string split paths.

## Behavior

Supported deterministic modes now include:

- `atom_concat(A, o, foo)` deriving `A = fo`
- `atom_concat(fo, B, foo)` deriving `B = o`
- `string_concat(A, o, foo)` deriving `A = fo`
- `string_concat(fo, B, foo)` deriving `B = o`

The ambiguous mode remains unsupported by the lowered fast path:

- `atom_concat(A, B, foo)` still fails conservatively because it has multiple valid splits.

## Validation

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
