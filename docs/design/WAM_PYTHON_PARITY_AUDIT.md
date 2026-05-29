# WAM Python Parity Audit

This note compares the Python hybrid WAM target against the Lua parity
baseline, using Rust and Haskell as the reference targets for builtin and
runtime behavior. It also records the Python target's readiness for the
cross-target ISO-error design described in
`WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md`.

## Packaged Runtime Surface

`write_wam_python_project/3` copies
`src/unifyweaver/targets/wam_python_runtime/WamRuntime.py` into generated
projects as `wam_runtime.py`. That static runtime is therefore the runtime that
matters for parity.

| Area | Python support | Lua/Rust/Haskell baseline | Status |
| --- | --- | --- | --- |
| Direct fact dispatch | `call_indexed_atom_fact2`, category ancestor helpers | Indexed fact paths | Present |
| Aggregates | `begin_aggregate`, `end_aggregate` runtime paths | `findall/3`, `aggregate_all/3` families | Present, covered by generated-project E2E |
| Structural builtins | `member/2`, `length/2` | `member/2`, `length/2` | Present: `member/2` enumerates via builtin choice points; `length/2` covers fixed and generative modes |
| Type builtins | `atom/1`, `integer/1`, `float/1`, `number/1`, `compound/1`, `var/1`, `nonvar/1`, `is_list/1` | Same baseline set | Present |
| Comparison builtins | `==/2`, `=:=/2`, `=\=/2`, `</2`, `>/2`, `=</2`, `>=/2` | Same baseline set | Present |
| List ordering | `sort/2`, `msort/2`, `keysort/2` | Same baseline set where present | Present: generated-project E2E covers numeric, atom, duplicate, empty-list, and keyed-pair cases |
| Unification builtins | `=/2`, `\=/2` | `=/2`, `\=/2` in comparable runtimes | Present |
| Term inspection | `functor/3`, `arg/3` | `functor/3`, `arg/3` | Present |
| Univ | `=../2` compose and decompose | `=../2` compose and decompose | Present |
| Copying | `copy_term/2` with fresh variables and preserved sharing | `copy_term/2` with fresh variables and preserved sharing | Present |
| Control | `true/0`, `fail/0`, `!/0`, `\+/1`, `cut_ite` opcode | `true/0`, `fail/0`, `!/0`, `\+/1`, `CutIte` | Present |
| IO | `write/1`, `display/1`, `nl/0` output behavior | `write/1`, `display/1`, `nl/0` output behavior | Present |

## ISO Error Readiness

Python is now a partial ISO-error adopter. It has the Prolog-level
`catch/3` and `throw/1` substrate, ISO error-term constructors,
`throw_iso_error`, the config/rewrite/audit plumbing, and ISO/lax variants for
arithmetic assignment, arithmetic comparison, successor builtins, and lax
IEEE-754 float zero division. It should not yet be described as fully ISO-error
compatible until any remaining concrete builtins with ISO/lax behavior are
swept.

Current state:

| Component | Python status | Notes |
| --- | --- | --- |
| Prolog `catch/3` / `throw/1` | Present | Packaged runtime has side-stack catcher frames and generated-project E2E coverage. |
| ISO error constructors | Present | Runtime builders exist for `instantiation_error`, `type_error/2`, `domain_error/2`, and `evaluation_error/1`. |
| `throw_iso_error` helper | Present | Wraps `error(ErrorTerm, Context)` and routes through `throw/1`. |
| `is_iso/2` / `is_lax/2` | Present | ISO mode throws structured errors; `is/2` and `is_lax/2` preserve lax failure. |
| ISO/lax arithmetic compares | Present | Six comparison variants now follow ISO/lax three-form dispatch. |
| `succ/2` and ISO/lax variants | Present | Lax `succ/2`/`succ_lax/2` fail silently; `succ_iso/2` throws structured instantiation, type, and domain errors. |
| Lax IEEE-754 float divide behavior | Present | `is_lax/2` and default-lax `is/2` return `inf`, `nan`, or `-inf` for float zero division; integer zero division still fails silently. |
| Per-predicate ISO config loader | Present | Supports `iso_errors_config(File)`, `iso_errors(Default)`, and `iso_errors(PI, Mode)` options. |
| Per-predicate default rewrite | Present | `is/2` now rewrites to `is_iso/2` or `is_lax/2`; text-level rewrite feeds interpreter and lowered emission. |
| ISO audit predicate | Present | `wam_python_iso_audit/3` reports builtin call sites using the shared audit shape. |

The packaged runtime is the main implementation surface to update:

```text
src/unifyweaver/targets/wam_python_runtime/WamRuntime.py
```

The older string-assembled helper path in `wam_python_target.pl` still contains
a smaller builtin dispatcher, but generated projects copy the packaged runtime.
ISO work should avoid growing the legacy helper path unless a test proves it is
still load-bearing for a supported emit mode.

The existing arithmetic behavior is lax by construction. For example,
`_execute_builtin/4` wraps `eval_arith/2` in `try/except WAMError` for `is/2`
and the arithmetic comparisons, returning `False` on malformed arithmetic
instead of throwing a structured Prolog error. That is a good starting point for
`*_lax` aliases, but not enough for ISO mode.

## Recommended ISO Port Sequence

Porting `is_iso/2` first was premature before `catch/3` / `throw/1` existed.
That substrate and the ISO error helpers are now present, so the remaining
sequence is:

1. Sweep any remaining arithmetic builtins that need ISO/lax behavior.
2. Extend the same three-form pattern to the next concrete builtin family that
   needs ISO-mode errors.

The successor variants now have explicit-lax bypass coverage. The C++ and
Elixir ISO tests remain the direct template for any next Python E2E coverage.

## Runtime Source Of Truth

`compile_wam_runtime_to_python/2` now reads the packaged static runtime source
instead of assembling a second runtime from Prolog string fragments. Generated
projects also copy the same `WamRuntime.py`, so tests and generated projects
now share one Python WAM runtime surface.

## Remaining Follow-Up

The packaged static runtime now carries the Lua/Rust/Haskell builtin baseline.
No Python WAM builtin parity gaps are currently tracked here. The current
Python WAM target file also runs without the earlier plunit choicepoint-warning
noise in the registry and ITE detection tests.

Completed follow-up:

- Generated-project E2E tests now exercise term builtins, copy/NAF/IO, and
  type/comparison builtins through the packaged static runtime.
- Generated-project E2E tests now verify `member/2` enumeration through
  aggregate collection.
- Generated-project E2E tests now verify `sort/2`, `msort/2`, and `keysort/2`
  through the packaged static runtime.
- Python WAM registry and ITE tests now use deterministic assertions where the
  test only needs the first valid proof, keeping full-suite output warning-free.
- `compile_wam_runtime_to_python/2` now returns the packaged static runtime,
  removing the separate fallback runtime surface.

## Verification Commands

Use these checks after touching Python WAM runtime parity. On current `main`,
`tests/test_wam_python_target.pl` passes 166/166 without choicepoint warnings:

```sh
swipl -q -g run_tests -t halt tests/test_wam_python_target.pl
swipl -q -g run_tests -t halt tests/test_wam_python_effective_distance_smoke.pl
swipl -q -g "use_module(src/unifyweaver/targets/wam_python_target), halt"
```
