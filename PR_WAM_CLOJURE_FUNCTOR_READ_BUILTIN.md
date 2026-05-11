# feat(wam-clojure): add read-mode functor/3 builtin support

## Summary

Adds read-mode `functor/3` support to the Clojure WAM runtime and lowered-emitter path.

## What Changed

- Added Clojure runtime helpers to inspect a term and derive its functor name and arity.
- Added interpreted runtime dispatch for `functor/3`.
- Registered `functor/3` as a direct lowered Clojure builtin.
- Added lowered-emitter coverage proving `functor/3` bypasses generic `runtime/step`.
- Extended the Clojure runtime smoke suite for structure, atom, number, and mismatch cases.

## Supported Scope

This PR supports read mode:

```prolog
functor(+Term, ?Name, ?Arity)
```

Construct mode is intentionally left for a follow-up:

```prolog
functor(?Term, +Name, +Arity)
```

## Validation

```sh
git diff --check
swipl -q -g run_tests -t halt tests/test_wam_clojure_lowered_emitter.pl
swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl
timeout 240 swipl -q -s tests/test_wam_clojure_runtime_smoke.pl
```
