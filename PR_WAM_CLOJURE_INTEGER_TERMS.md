# PR Title

fix(wam-clojure): preserve integer terms in WAM runtime

# PR Description

## Summary

- Emit bare numeric WAM constants as Clojure numeric literals while preserving quoted numeric atoms as atoms.
- Return integer terms from Clojure WAM runtime helpers that model Prolog integer outputs.
- Add regression coverage for numeric WAM token emission.

## Details

This fixes Clojure WAM smoke regressions where builtins such as `functor/3`, `compound_name_arity/3`, `atom_codes/2`, `char_code/2`, `string_code/3`, `atom_length/2`, and `sub_atom/5` produced atom-like numeric terms instead of integers.

The generator now distinguishes unquoted numeric WAM tokens like `2` from quoted atom tokens like `'2'`, so WAM constants retain their intended Prolog type in generated Clojure EDN.

## Validation

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
