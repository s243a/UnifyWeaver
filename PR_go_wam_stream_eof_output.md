# PR Title

Add Go WAM stream EOF and helper output builtins

# PR Description

## Summary

- Registers `at_end_of_stream/1`, `write_to_stream/2`, and `nl_to_stream/1` as direct Go WAM builtins.
- Adds Go WAM runtime support for non-consuming file-backed EOF checks and simple helper output to writable streams.
- Extends the generated Go builtin E2E test to cover EOF before and after reads, closed-stream failure, helper output, and read-stream output failure.
- Updates the Go WAM parity audit to record the Python/C++-aligned stream EOF and helper-output surface.

## Tests

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_foreign_lowering.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), halt"`

## Push Command

```sh
git push -u origin feat/wam-go-stream-eof-output
```
