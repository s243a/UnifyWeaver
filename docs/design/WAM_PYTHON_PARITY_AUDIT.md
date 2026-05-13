# WAM Python Parity Audit

This note compares the Python hybrid WAM target against the Lua parity
baseline, using Rust and Haskell as the reference targets for builtin and
runtime behavior.

## Packaged Runtime Surface

`write_wam_python_project/3` copies
`src/unifyweaver/targets/wam_python_runtime/WamRuntime.py` into generated
projects as `wam_runtime.py`. That static runtime is therefore the runtime that
matters for parity.

| Area | Python support | Lua/Rust/Haskell baseline | Status |
| --- | --- | --- | --- |
| Direct fact dispatch | `call_indexed_atom_fact2`, category ancestor helpers | Indexed fact paths | Present |
| Aggregates | `begin_aggregate`, `end_aggregate` runtime paths | `findall/3`, `aggregate_all/3` families | Present, but not covered by a parity guard |
| Structural builtins | `member/2`, `length/2` | `member/2`, `length/2` | Partial: `member/2` returns only the first solution |
| Type builtins | `atom/1`, `integer/1`, `float/1`, `number/1`, `compound/1`, `var/1`, `nonvar/1`, `is_list/1` | Same baseline set | Present |
| Comparison builtins | `==/2`, `=:=/2`, `=\=/2`, `</2`, `>/2`, `=</2`, `>=/2` | Same baseline set | Present |
| Unification builtins | `=/2`, `\=/2` | `=/2`, `\=/2` in comparable runtimes | Present |
| Term inspection | `functor/3`, `arg/3` | `functor/3`, `arg/3` | Present |
| Univ | `=../2` compose and decompose | `=../2` compose and decompose | Present |
| Copying | `copy_term/2` with fresh variables and preserved sharing | `copy_term/2` with fresh variables and preserved sharing | Present |
| Control | `true/0`, `fail/0`, `!/0`, `\+/1`, `cut_ite` opcode | `true/0`, `fail/0`, `!/0`, `\+/1`, `CutIte` | Present |
| IO | `write/1`, `display/1`, `nl/0` output behavior | `write/1`, `display/1`, `nl/0` output behavior | Present |

## Runtime Source Of Truth

`compile_wam_runtime_to_python/2` now reads the packaged static runtime source
instead of assembling a second runtime from Prolog string fragments. Generated
projects also copy the same `WamRuntime.py`, so tests and generated projects
now share one Python WAM runtime surface.

## Remaining Follow-Up

The packaged static runtime now carries the Lua/Rust/Haskell builtin baseline.
The remaining parity work is narrower:

1. Decide whether `member/2` should enumerate all solutions via builtin choice
   points, or remain first-solution-only like the current packaged runtime.

Completed follow-up:

- Generated-project E2E tests now exercise term builtins, copy/NAF/IO, and
  type/comparison builtins through the packaged static runtime.
- `compile_wam_runtime_to_python/2` now returns the packaged static runtime,
  removing the separate fallback runtime surface.

## Verification Commands

Use these checks after touching Python WAM runtime parity:

```sh
swipl -q -g run_tests -t halt tests/test_wam_python_target.pl
swipl -q -g run_tests -t halt tests/test_wam_python_effective_distance_smoke.pl
swipl -q -g "use_module(src/unifyweaver/targets/wam_python_target), halt"
```
