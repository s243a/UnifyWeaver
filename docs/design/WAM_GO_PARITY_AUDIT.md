# WAM Go Parity Audit

This note compares the Go hybrid WAM target against the Rust and Haskell WAM
targets, using the Lua and Python parity audits as secondary references for the
current cross-target builtin/runtime baseline.

## Verified Runtime Surface

| Area | Go support | Rust/Haskell baseline | Status |
| --- | --- | --- | --- |
| Choice points and backtracking | `try_me_else`, `retry_me_else`, `trust_me`, indexed alternatives, foreign stream retries, indexed atom fact streams, `member/2` builtin retries | Choice point stack with normal, builtin, and fact-stream resume states | Present for current resume-state baseline |
| Direct fact dispatch | `call_indexed_atom_fact2`, `AtomFact2Source` registry, TSV-backed and LMDB-artifact atom fact loading, foreign/native kernel registration, indexed atom/weighted fact tables | `call_indexed_atom_fact2`, inline/external fact stream paths | Present for current external atom fact-source baseline |
| Aggregates | `begin_aggregate`, `end_aggregate`; `collect`, `bag`, `bagof`, `count`, `sum`, `min`, `max`, `set`, `setof` | `findall/3`, `aggregate_all/3` count/sum/min/max/set families | Present for current aggregate baseline |
| Structural builtins | `member/2`, `length/2`, `append/3` | `member/2`, `length/2`; Rust `append/3` is explicitly unimplemented | Present for current baseline structural checks |
| Type builtins | `var/1`, `nonvar/1`, `atom/1`, `integer/1`, `float/1`, `number/1`, `compound/1`, `atomic/1`, `is_list/1` | Includes `is_list/1` in the current baseline | Present for current baseline type checks |
| Comparison builtins | `==/2`, `\==/2`, `\=/2`, `=:=/2`, `=\=/2`, `</2`, `>/2`, `=</2`, `>=/2` | Includes `=</2` | Present for current baseline comparisons |
| Unification builtin | `=/2`, `\=/2` | `=/2`, `\=/2` | Present |
| Term inspection | `functor/3`, `arg/3` | `functor/3`, `arg/3` | Present |
| Univ | `=../2` compose/decompose | `=../2` compose/decompose | Present |
| Copying | `copy_term/2` with fresh variables and preserved sharing | `copy_term/2` with fresh variables and preserved sharing | Present |
| Control | `true/0`, `fail/0`, `!/0`, `\+/1`, `CutIte` | Same baseline, with broader isolated-goal NAF in Haskell/Python | Present for current baseline, including isolated user-goal NAF and race-to-true over multi-clause WAM targets |
| IO | `write/1`, `display/1`, `nl/0` | `write/1`, `display/1`, `nl/0` | Present |

## Immediate Findings

- `tests/test_wam_go_generator.pl` had stale expectations for atom literals.
  The Go target now emits `internAtom("...")` instead of raw
  `&Atom{Name: "..."}` literals, so the assertions needed to follow the
  current intern-table design.
- The Go WAM runtime has a substantial execution core; `set` aggregate results
  are now deduplicated like Haskell's `nub` behavior.
- `member/2` now pushes builtin choice points for later list members, so
  `findall/3` can collect every unifiable element.
- Go WAM choice points now have explicit generated-runtime coverage for normal
  clause retries, indexed alternatives, foreign stream retries, indexed atom
  fact streams, and `member/2` builtin retries.
- `=</2`, `is_list/1`, and `display/1` are now covered by the generated Go
  WAM builtin E2E test.
- `=/2` and `\=/2` are now covered by the generated Go WAM builtin E2E test.
- `functor/3`, `arg/3`, `=../2`, and `copy_term/2` are now covered by the
  generated Go WAM builtin E2E test.
- `aggregate_all(set(X), Goal, Set)` is now covered by the generated Go WAM
  builtin E2E test.
- `\+/1` over user predicates is now covered by the generated Go WAM builtin
  E2E test for both failing and succeeding inner goals.
- `\+/1` now dispatches multi-clause WAM targets through a `runNegationParallel`
  helper that races isolated clause bodies and fails fast when any branch
  succeeds, matching the Haskell race-to-true control shape.
- `call_indexed_atom_fact2` is now parsed and executed by the Go WAM runtime,
  including backtracking over multiple indexed atom facts.
- Headered two-column TSV files can now be loaded into Go WAM indexed atom
  fact tables and queried through `call_indexed_atom_fact2`.
- Inline and TSV atom fact pairs now register through a Go `AtomFact2Source`
  interface with `Scan` and `LookupArg1`, matching the Haskell FactSource shape
  while preserving the existing indexed pair table used by native kernels.
- LMDB relation artifacts can now be registered as Go `AtomFact2Source`
  adapters via the existing `lmdb_relation_artifact` helper command. The helper
  path defaults to `lmdb_relation_artifact` and can be overridden with
  `UW_LMDB_RELATION_ARTIFACT_BIN`, so tests and deployments can provide the
  concrete LMDB reader without adding a generated-project Go dependency.

## Recommended Follow-Up Order

1. Continue broadening generated Go WAM E2E coverage for any remaining
   cross-target builtin edge cases.
2. If Go should avoid a helper process for LMDB, add an optional native Go
   LMDB build-tag path once a dependency policy is settled.

## Verification Commands

Use these checks after touching Go WAM parity:

```sh
swipl -q -g run_tests -t halt tests/test_wam_go_generator.pl
swipl -q -g run_tests -t halt tests/test_go_wam_builtins.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase1.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase2.pl
swipl -q -g run_tests -t halt tests/test_wam_go_lowered_phase3.pl
swipl -q -g "use_module(src/unifyweaver/targets/wam_go_target), halt"
```
