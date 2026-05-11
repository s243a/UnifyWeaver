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
| Type builtins | `atom/1`, `integer/1`, `number/1`, `var/1`, `nonvar/1` | Also `float/1`, `compound/1`, `is_list/1` | Gap |
| Comparison builtins | `=:=/2`, `=\=/2`, `</2`, `>/2`, `=</2`, `>=/2` | Also term equality `==/2` | Gap |
| Unification builtins | Not in static runtime builtin dispatch | `=/2`, `\=/2` in comparable runtimes | Gap |
| Term inspection | Not in static runtime builtin dispatch | `functor/3`, `arg/3` | Gap |
| Univ | Not in static runtime builtin dispatch | `=../2` compose and decompose | Gap |
| Copying | Not in static runtime builtin dispatch | `copy_term/2` with fresh variables and preserved sharing | Gap |
| Control | `true/0`, `fail/0`, `!/0`, `\+/1`, `cut_ite` opcode | `true/0`, `fail/0`, `!/0`, `\+/1`, `CutIte` | Partial: `\+/1` has a `member/2` fast path and otherwise succeeds by default |
| IO | `write/1`, `writeln/1`, `print/1`, `nl/0` dispatch returns success only | `write/1`, `display/1`, `nl/0` output behavior | Gap |

## Generated Fallback Runtime

`compile_wam_runtime_to_python/2` emits a fallback runtime only when the static
runtime file cannot be copied. Its builtin dispatch is not equivalent to the
packaged static runtime:

- It includes `=/2`, `\=/2`, `float/1`, `compound/1`, `write/1`, `writeln/1`,
  `nl/0`, `copy_term/2`, `functor/3`, and `=../2`.
- It does not include the static runtime's `member/2`, `length/2`, `true/0`,
  `fail/0`, `!/0`, or `\+/1` dispatch.
- It does not include `arg/3`, `is_list/1`, `==/2`, or `display/1`.

This split is a parity risk: tests that inspect the generated fallback runtime
can pass while generated Python projects still run against the static runtime.

## Highest-Value Next Slice

Bring the static Python runtime up to the Lua/Rust/Haskell builtin baseline
before changing lowered codegen:

1. Add static-runtime dispatch for `float/1`, `compound/1`, `is_list/1`,
   `==/2`, `=/2`, `\=/2`, `functor/3`, `arg/3`, `=../2`, `copy_term/2`,
   `display/1`, and real `write/1`/`nl/0` output.
2. Replace the default-success `\+/1` fallback with isolated sub-execution
   behavior, matching the Lua audit's non-leaking binding semantics.
3. Add a Python parity guard test similar to `lua_parity_guard` that checks the
   packaged static runtime for every baseline builtin.
4. Reconcile or retire the generated fallback runtime so it cannot drift from
   `WamRuntime.py`.

## Verification Commands

Use these checks after touching Python WAM runtime parity:

```sh
swipl -q -g run_tests -t halt tests/test_wam_python_target.pl
swipl -q -g run_tests -t halt tests/test_wam_python_effective_distance_smoke.pl
swipl -q -g "use_module(src/unifyweaver/targets/wam_python_target), halt"
```
