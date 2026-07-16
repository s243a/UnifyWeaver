# WAM Python Target — Status

Living summary of the hybrid WAM-Python backend
(`wam_python_target.pl` + `wam_python_lowered_emitter.pl` + packaged
`wam_python_runtime/`). Distinct from the **non-WAM** direct Python
compiler documented in [`PYTHON_TARGET.md`](PYTHON_TARGET.md)
(`python_target.pl`).

Companion docs:

- [`design/WAM_PYTHON_PARITY_AUDIT.md`](design/WAM_PYTHON_PARITY_AUDIT.md) — parity + partial ISO.
- [`PYTHON_TARGET.md`](PYTHON_TARGET.md) — non-WAM sibling compiler.
- [`design/WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md`](design/WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md).
- [`WAM_HYBRID_TARGETS_COMPARISON.md`](WAM_HYBRID_TARGETS_COMPARISON.md).

## Role

**Scripting-embed parity surface.** A packaged `WamRuntime.py` that
hosts the WAM inside CPython, tracking the parity audit and a partial
ISO error stack.

## Codegen surface

| Module | Approx. lines |
|---|---:|
| `src/unifyweaver/targets/wam_python_target.pl` | ~2.8k |
| `src/unifyweaver/targets/wam_python_lowered_emitter.pl` | ~1.3k |
| `src/unifyweaver/targets/wam_python_runtime/` (packaged `WamRuntime.py`) | ~3.6k |
| Dedicated tests | ~8 files (~194 plunit cases) |

## What's shipped

**Dual lowering.** WAM instruction VM plus lowered emitter.

**Partial ISO errors.** `catch`/`throw`, `is_iso`/`is_lax`, six
arithmetic-compare ISO/lax variants, `succ` family — a partial adopter
of the three-form contract (not full, per the cross-target ISO status).

**Interpreter-level graph ops.** Indexed-fact and
`base_category_ancestor*` operations at the interpreter level — **no**
FFI graph-kernel set like Rust/Go/C.

**Conformance.** Registered `conformance_target(python)` and green;
opt-in (needs a `python3` per-program build), not default CI.

**Runtime parser.** Compiled `prolog_term_parser` available as an
opt-in mode; no native default.

## Gaps (relative to Rust / Haskell / F#)

- **No FFI graph-kernel set** — graph ops stay interpreter-bound.
- **No LMDB / memory-mapped fact source** (zero LMDB surface).
- **ISO adoption is partial** — remaining concrete builtins must adopt
  three-form keys before Python is "fully ISO-compatible".
- No native runtime-parser default (compiled opt-in only).

## Path forward

1. Complete ISO three-form adoption across remaining builtins.
2. Add an FFI (or C-extension) graph-kernel path if perf-class
   graph work is wanted.
3. LMDB / mmap fact source for >~100k facts.
4. Effective-distance cross-target matrix row.

## Document status

Fleet-aligned snapshot; source-verified line counts, the
interpreter-only kernel story, partial-ISO surface, and opt-in
conformance registration against `wam_python_target.pl`, the parity
audit, and the conformance harness (2026-07-11).
