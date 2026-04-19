# PR Title

`feat(wam-clojure): add hybrid WAM runtime baseline`

# PR Description

## Summary

This PR introduces the first runnable hybrid/WAM target for Clojure.

It establishes the same broad architectural shape used by the more mature
Haskell and Rust hybrid WAM targets:

- a shared WAM instruction table for project generation
- one-time label/control-flow resolution at load time
- per-predicate wrappers that dispatch by start PC
- a generated runtime namespace plus standalone project scaffold

It also moves the Clojure target beyond generator-only scaffolding by adding a
minimal interpreted runtime with end-to-end smoke coverage.

## What’s Included

- new `wam_clojure` target registration and target-module wiring
- new `write_wam_clojure_project/3` generator
- new Clojure WAM templates:
  - `deps.edn`
  - `project.clj`
  - runtime namespace
- shared-table code generation for multiple predicates
- one-time resolution of:
  - `call`
  - `execute`
  - `jump`
  - choice-point labels
  - `switch_on_constant`
- runtime support for:
  - basic call/execute flow
  - indexed multi-clause dispatch
  - choice points and backtracking
  - explicit environment frames for `Y` slots
  - clause cut barriers and `cut_ite`
  - read-mode compound/list matching
  - write-mode compound/list construction
- focused generator tests
- standalone end-to-end smoke runner for generated Clojure WAM projects
- documentation update in `docs/design/WAM_PERF_OPTIMIZATION_LOG.md`

## Design Notes

This PR intentionally targets a solid baseline rather than full parity.

The most important architectural choices already match the Haskell/Rust path:

- shared code/label space instead of per-predicate isolated blobs
- pre-resolved control flow instead of repeated runtime label lookup
- environment-aware cut semantics instead of clearing all choice points

The runtime is still a simplified implementation in one important respect:

- it is primarily bindings-centric and does not yet have full heap/trail
  semantics like the more mature targets

That is the next major parity boundary.

## Validation

Passed locally:

- `swipl -q -g run_tests -t halt tests/test_wam_clojure_generator.pl`
- `swipl -q -g run_tests -t halt tests/core/test_clojure_native_lowering.pl`
- `swipl -q -s tests/test_wam_clojure_runtime_smoke.pl`

## Current Limitations

- no foreign/kernel lowering yet
- no full heap/trail runtime model yet
- choice-point snapshots are still heavier than the Haskell/Rust versions
- plunit-managed JVM subprocess execution remains unreliable in this Termux
  environment, so generated-project runtime coverage currently lives in the
  standalone smoke runner

## Follow-up Work

1. add proper heap/trail semantics
2. reduce choice-point snapshot cost further
3. split hot runtime state from cold code/context data
4. start foreign/kernel lowering for Clojure hybrid WAM
