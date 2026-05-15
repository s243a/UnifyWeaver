# Add Go WAM TSV atom fact source

## Description

Adds a TSV-backed atom fact source path for the Go WAM target. The runtime can now load a headered two-column TSV file into the indexed atom fact table with `registerTsvAtomFact2`, and `call_indexed_atom_fact2` can query those dynamically loaded facts with normal stream-style backtracking.

This narrows the Go external fact-source parity gap against the Rust/Haskell baseline. TSV-backed atom facts are now covered; LMDB remains a separate follow-up.

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
