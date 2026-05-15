# Add Go WAM indexed atom fact dispatch

## Description

Adds `call_indexed_atom_fact2` support to the Go WAM target. The Go instruction set now includes `CallIndexedAtomFact2`, the WAM text parser emits it, and the runtime executes indexed atom fact lookups with stream-style backtracking over multiple matches.

This narrows the Go direct fact dispatch parity gap against the Rust/Haskell baseline by covering the inline indexed fact instruction path. External TSV/LMDB FactSource adapters remain a separate follow-up.

The generated Go runtime now also writes a minimal `internAtom` helper even when a project has no compile-time atom literals, because fact dispatch and foreign graph helpers can construct atoms dynamically.

## Tests

```sh
swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl
swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl
swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl
swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), halt"
```
