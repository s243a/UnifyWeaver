# PR Title

feat(wam-clojure): add atom and number text conversions

# PR Description

## Summary

- Adds Clojure WAM support for `atom_codes/2`, `atom_chars/2`, `number_codes/2`, and `number_chars/2`.
- Direct-lowers these conversion builtins to a shared runtime helper instead of routing through the interpreted builtin path.
- Supports deterministic forward and reverse modes.
- Updates the runtime intern context when reverse atom/char conversion creates atoms at runtime.
- Adds lowered-emitter and runtime smoke coverage for success, mismatch, reverse construction, and unsupported unbound/unbound mode.

## Semantics

The implementation supports:

- `atom_codes(+Atom, ?Codes)`
- `atom_codes(?Atom, +Codes)`
- `atom_chars(+Atom, ?Chars)`
- `atom_chars(?Atom, +Chars)`
- `number_codes(+Number, ?Codes)`
- `number_codes(?Number, +Codes)`
- `number_chars(+Number, ?Chars)`
- `number_chars(?Number, +Chars)`

Unsupported or ambiguous modes fail cleanly rather than raising ISO-style errors.

## Notes

- Code-list parsing accepts raw integer code points and decimal atom/string code-point literals to match current Clojure WAM literal normalization.
- `number_chars/2` smoke coverage avoids source-level quoted digit atoms in generated WAM text because that path currently emits Clojure-hostile escape sequences.

## Validation

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
