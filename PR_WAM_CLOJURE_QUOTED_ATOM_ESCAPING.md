# PR Title

fix(wam-clojure): escape quoted atom constants in generated code

# PR Description

## Summary

- Normalizes WAM single-quoted atom tokens before emitting Clojure literals.
- Strips the WAM quoted-numeric atom marker so atoms like `'4'` become the runtime atom text `4`, not an invalid generated Clojure escape sequence.
- Hardens Clojure string literal emission for backslashes, quotes, and control characters.
- Restores the `number_chars/2` reverse smoke fixture to use quoted digit atoms directly instead of the `char_code/2` workaround.

## Why

The Clojure lowered emitter previously rendered Prolog `~q` output into Clojure source. For quoted numeric atoms this could produce invalid Clojure such as `"'\x1\4'"`-style syntax. This PR makes quoted atom normalization explicit and keeps generated source valid.

## Testing

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
