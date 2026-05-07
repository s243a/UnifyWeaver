# perf: Pack WAM C instruction payloads

## Summary

This PR replaces the WAM-C runtime's wide `Instruction` struct with a tagged payload union keyed by `WamInstrTag`. Generated instruction setup now writes only the payload arm each instruction uses, and runtime dispatch reads through those typed payload arms.

## What Changed

- Adds `InstructionPayload` and tag-specific payload structs in the WAM-C runtime header.
- Updates WAM-C instruction literal emission and parsed WAM line emission to generate `.as.<payload>` initializers.
- Updates `step_wam` to read constants, register pairs, functors, predicates, choice targets, and switch tables through the correct payload arm.
- Keeps switch hash-table allocation and cleanup tied to the switch payload.
- Updates WAM-C tests for the packed payload field names.
- Updates `WAM_C_TARGET_NEXT_STEPS.md` to mark classic recursive coverage complete, record packed instruction layout as ready, and recommend memory-lifecycle sanitizer coverage next.

## Validation

- `swipl -q -g run_tests -t halt tests/test_wam_c_target.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_c_effective_distance_benchmark.pl`
- `python3 examples/benchmark/benchmark_effective_distance_matrix.py --scales dev --targets prolog-accumulated,c-wam-accumulated,c-wam-accumulated-no-kernels --repetitions 1`
- `git diff --check`

## Follow-Up

Next recommended branch: `test/wam-c-memory-lifecycle-asan`.
