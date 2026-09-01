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

## Whole-program exercise (A2, 2026-09): known / suspected deficiencies

The peerhailer CLI-parser exercise (see
[`WAM_FLEET_GAPS.md`](WAM_FLEET_GAPS.md)) found three JS-WAM runtime bug
classes that are fleet-wide suspects. Python's audit:

| # | Deficiency | Status | Evidence / reason |
|---|---|---|---|
| A1 | `sub_string/5` builtin missing | **verified missing** | not in `_execute_builtin` or any dispatch arm |
| A2 | Y-register clobber across `Call` of a no-`Allocate` fact | **verified (structural)** | X window is 128 slots (`A=1..128, X=129..256, Y=301..428`, `wam_python_target.pl:2205-2208`), so with X_k→128+k, **X173 aliases Y1**; ids ≥ 301 are written into the *current env frame's* `perm_vars` (`wam_python_runtime/WamRuntime.py:171-194`) — i.e. the caller's frame when the callee is a no-`Allocate` fact. The 128-slot window is wider than JS/Lua/Go's 100, so the cli_args registry (~X120) squeaks by, but the class is the same and a slightly larger fact trips it |
| A3 | `Execute` of a builtin doesn't return to the continuation | **verified** | `execute` with no label simply `return False` — no builtin fallback, and unlike `call` not even the foreign-predicate fallback (`wam_python_target.pl:400-406`); a last-goal builtin outside `is_builtin_pred/2` silently fails |
| A4 | String fidelity | **rung 0** | the term model (`WamRuntime.py:16-52`) has Atom/Compound/Var/Int/Float/Ref — no string type; D37's double-quoted literals intern as atoms |

Pattern lane: `python_target.pl` compiles facts from
`clause(Head, true)` (`:2923-2931`) — the G-A3-8 execute-at-compile-time
hazard is absent. The G-A3 machinery (cross-calls by output count,
multi-output tuples, semidet sentinel, compound runtime data) has no
analogue; presume those gaps present until `examples/cli_args/` is
attempted through the Python pattern lane.

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
audit, and the conformance harness (2026-07-11). 2026-09-01: added the
whole-program (A2) deficiency audit — Y-aliasing threshold, silent
`execute` failure and `sub_string/5` verified by source reading; see
[`WAM_FLEET_GAPS.md`](WAM_FLEET_GAPS.md).
