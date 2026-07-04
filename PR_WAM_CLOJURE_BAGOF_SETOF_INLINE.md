# PR Title

feat(wam-clojure): inline bagof/setof aggregate witnesses

# Description

## Summary

- Adds Clojure WAM support for the 4-argument `begin_aggregate` form emitted by inline `bagof/3` and `setof/3` compilation.
- Emits aggregate witness registers into generated Clojure bytecode as `:witnesses [...]`.
- Captures witness values per aggregate solution and groups `bagof`/`setof` results by witness using existing WAM term ordering.
- Adds aggregate-result choicepoints so generated predicates can backtrack across witness groups.
- Keeps witness aggregates runtime-mediated in the lowered Clojure emitter instead of accidentally dropping unsupported aggregate instructions.

## Behavior Covered

- No-witness `bagof/3` preserves duplicates and solution order.
- No-witness `setof/3` sorts and deduplicates results.
- Empty `bagof/3` fails, unlike `findall/3`.
- Grouped `bagof/3` and `setof/3` produce one bag/set per witness group and allow backtracking to later groups.

## Validation

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
- `swipl -q -s tests/test_wam_clojure_lowered_t4.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_lowered_ite_exec.pl`
- `swipl -q -g run_tests -t halt tests/core/test_clojure_native_lowering.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_benchmark_generator.pl`
