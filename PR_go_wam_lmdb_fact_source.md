# PR Title

Add Go WAM LMDB atom fact-source adapter

# PR Description

## Summary

- Adds `register_lmdb_atom_fact2(...)` setup generation for Go WAM foreign setup specs.
- Adds generated Go `registerLmdbAtomFact2` and an `lmdbAtomFact2Source` implementing the existing `AtomFact2Source` interface.
- Dispatches LMDB-backed `Scan` and `LookupArg1` through the existing `lmdb_relation_artifact` helper command, avoiding a new generated-project Go dependency.
- Supports `UW_LMDB_RELATION_ARTIFACT_BIN` so deployments or tests can provide the concrete helper path.
- Adds generated-Go E2E coverage using a mock LMDB artifact helper.
- Updates the Go parity audit to mark direct external atom fact-source support as covered for the current baseline, while leaving an optional native Go LMDB binding as a future dependency-policy decision.

## Verification

```sh
swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl
swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl
swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl
swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl tests/test_wam_go_generator.pl tests/test_go_wam_builtins.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl
swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), halt"
git diff --check
```

## Follow-Up

- Consider an optional native Go LMDB build-tag implementation if the project settles on a Go LMDB dependency policy.
