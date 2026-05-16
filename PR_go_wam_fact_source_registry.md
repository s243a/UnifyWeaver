# PR Title

Add Go WAM atom fact-source registry

# PR Description

## Summary

- Adds a Go WAM `AtomFact2Source` interface with `Scan` and `LookupArg1`.
- Registers inline indexed atom facts and TSV-backed atom facts through the same source registry.
- Keeps `IndexedAtomFactPairs` populated so existing native graph/kernel paths continue to work unchanged.
- Routes `call_indexed_atom_fact2` through registered fact sources before falling back to stored pairs.
- Updates the Go WAM parity audit to mark the Haskell-shaped fact-source abstraction as present while leaving concrete LMDB support as the remaining direct-fact parity gap.
- Fixes Go generator test isolation so combined Prolog test runs do not inherit stale atom-intern state from earlier suites.

## Verification

```sh
swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl
swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl
swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl
swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl tests/test_wam_go_generator.pl tests/test_go_wam_builtins.pl
git diff --check
```

## Follow-Up

- Add a concrete LMDB-backed `AtomFact2Source` implementation if Go should close the remaining direct-fact parity gap against Haskell.
