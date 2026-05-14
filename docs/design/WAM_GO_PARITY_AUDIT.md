# WAM Go Parity Audit

This note compares the Go hybrid WAM target against the Rust and Haskell WAM
targets, using the Lua and Python parity audits as secondary references for the
current cross-target builtin/runtime baseline.

## Verified Runtime Surface

| Area | Go support | Rust/Haskell baseline | Status |
| --- | --- | --- | --- |
| Choice points and backtracking | `try_me_else`, `retry_me_else`, `trust_me`, indexed alternatives, foreign stream retries, `member/2` builtin retries | Choice point stack with normal, builtin, and fact-stream resume states | Partial |
| Direct fact dispatch | Foreign/native kernel registration and indexed atom/weighted fact tables | `call_indexed_atom_fact2`, inline/external fact stream paths | Partial |
| Aggregates | `begin_aggregate`, `end_aggregate`; `collect`, `bag`, `bagof`, `count`, `sum`, `min`, `max`, `set`, `setof` | `findall/3`, `aggregate_all/3` count/sum/min/max/set families | Present for current aggregate baseline |
| Structural builtins | `member/2`, `length/2`, `append/3` | `member/2`, `length/2`; Rust `append/3` is explicitly unimplemented | Present for current baseline structural checks |
| Type builtins | `var/1`, `nonvar/1`, `atom/1`, `integer/1`, `float/1`, `number/1`, `compound/1`, `atomic/1`, `is_list/1` | Includes `is_list/1` in the current baseline | Present for current baseline type checks |
| Comparison builtins | `==/2`, `\==/2`, `\=/2`, `=:=/2`, `=\=/2`, `</2`, `>/2`, `=</2`, `>=/2` | Includes `=</2` | Present for current baseline comparisons |
| Unification builtin | `=/2`, `\=/2` | `=/2`, `\=/2` | Present |
| Term inspection | `functor/3`, `arg/3` | `functor/3`, `arg/3` | Present |
| Univ | `=../2` compose/decompose | `=../2` compose/decompose | Present |
| Copying | `copy_term/2` with fresh variables and preserved sharing | `copy_term/2` with fresh variables and preserved sharing | Present |
| Control | `true/0`, `fail/0`, `!/0`, `\+/1`, `CutIte` | Same baseline, with broader isolated-goal NAF in Haskell/Python | Partial: `\+/1` handles builtin-shaped and sequential user goals; no parallel race path |
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
- `=</2`, `is_list/1`, and `display/1` are now covered by the generated Go
  WAM builtin E2E test.
- `=/2` and `\=/2` are now covered by the generated Go WAM builtin E2E test.
- `functor/3`, `arg/3`, `=../2`, and `copy_term/2` are now covered by the
  generated Go WAM builtin E2E test.
- `aggregate_all(set(X), Goal, Set)` is now covered by the generated Go WAM
  builtin E2E test.
- `\+/1` over user predicates is now covered by the generated Go WAM builtin
  E2E test for both failing and succeeding inner goals.

## Recommended Follow-Up Order

1. Continue broadening generated Go WAM E2E coverage for control and fact-source
   behavior that remains marked partial.
2. Compare Go direct fact dispatch against Rust/Haskell external fact stream
   paths if Go should close the remaining direct-fact parity gap.

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
