# feat(wam-clojure): lower append builtin concat mode

## Summary

Adds direct lowered Clojure WAM support for deterministic `append/3` concat mode.

## What Changed

- Marks `append/3` as a direct Clojure lowered builtin.
- Emits lowered `append/3` calls through Clojure runtime list helpers instead of the generic WAM step loop.
- Adds `runtime/proper-list-items` to decompose proper list terms into item vectors.
- Adds runtime dispatch for `append/3`.
- Adds generator, lowered-emitter, and batched runtime smoke coverage.

## Semantics

This PR intentionally implements the conservative deterministic mode:

- `append(+ListA, +ListB, ?Out)`
- Proper-list inputs are concatenated and unified with `Out`.
- Improper lists and unbound first/second list inputs fail.
- Split/generative modes, such as `append(A, B, [a,b,c])`, remain out of scope for this PR.

This matches the cautious mode already used by the R WAM runtime and leaves Scala-style split enumeration for a separate follow-up.

## Validation

- `git diff --check`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`
