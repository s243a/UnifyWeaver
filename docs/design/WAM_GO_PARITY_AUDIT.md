# WAM Go Parity Audit

This note compares the Go hybrid WAM target against the Rust and Haskell WAM
targets, using the Lua and Python parity audits as secondary references for the
current cross-target builtin/runtime baseline.

## Verified Runtime Surface

| Area | Go support | Rust/Haskell baseline | Status |
| --- | --- | --- | --- |
| Choice points and backtracking | `try_me_else`, `retry_me_else`, `trust_me`, indexed alternatives, foreign stream retries | Choice point stack with normal, builtin, and fact-stream resume states | Partial |
| Direct fact dispatch | Foreign/native kernel registration and indexed atom/weighted fact tables | `call_indexed_atom_fact2`, inline/external fact stream paths | Partial |
| Aggregates | `begin_aggregate`, `end_aggregate`; `collect`, `count`, `sum`, `min`, `max` | `findall/3`, `aggregate_all/3` count/sum/min/max/set families | Partial: no `set` aggregate result |
| Structural builtins | `member/2`, `length/2`, `append/3` | `member/2`, `length/2`; Rust `append/3` is explicitly unimplemented | Partial: `member/2` is first-solution only |
| Type builtins | `var/1`, `nonvar/1`, `atom/1`, `integer/1`, `float/1`, `number/1`, `compound/1`, `atomic/1`, `is_list/1` | Includes `is_list/1` in the current baseline | Present for current baseline type checks |
| Comparison builtins | `==/2`, `\==/2`, `\=/2`, `=:=/2`, `=\=/2`, `</2`, `>/2`, `=</2`, `>=/2` | Includes `=</2` | Present for current baseline comparisons |
| Unification builtin | `\=/2` | `=/2`, `\=/2` | Partial: no explicit `=/2` builtin handler |
| Term inspection | Not present in `executeBuiltin/2` | `functor/3`, `arg/3` | Gap |
| Univ | Not present in `executeBuiltin/2` | `=../2` compose/decompose | Gap |
| Copying | Not present in `executeBuiltin/2` | `copy_term/2` with fresh variables and preserved sharing | Gap |
| Control | `true/0`, `fail/0`, `!/0`, `\+/1`, `CutIte` | Same baseline, with broader isolated-goal NAF in Haskell/Python | Partial: `\+/1` only dispatches builtin-shaped goals |
| IO | `write/1`, `display/1`, `nl/0` | `write/1`, `display/1`, `nl/0` | Present |

## Immediate Findings

- `tests/test_wam_go_generator.pl` had stale expectations for atom literals.
  The Go target now emits `internAtom("...")` instead of raw
  `&Atom{Name: "..."}` literals, so the assertions needed to follow the
  current intern-table design.
- The Go WAM runtime has a substantial execution core, but its builtin set is
  behind Rust/Haskell/Lua/Python for term inspection and copying.
- `member/2` succeeds on the first unifiable element and does not push builtin
  choice points for later list members. This mirrors the Python gap that was
  recently closed.
- `=</2`, `is_list/1`, and `display/1` are now covered by the generated Go
  WAM builtin E2E test.

## Recommended Follow-Up Order

1. Add a focused generated-project E2E parity suite for Go WAM builtins, similar
   to the Python and Lua generated-project tests.
2. Add term builtin parity: `functor/3`, `arg/3`, `=../2`, and `copy_term/2`.
3. Upgrade `member/2` to enumerate through builtin choice points so aggregate
   collection can observe all list members.
4. Add `set` aggregate support if Go is expected to match the current Lua/Python
   aggregate parity surface.

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
