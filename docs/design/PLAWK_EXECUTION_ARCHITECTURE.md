<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (@s243a)
-->

# PLAWK Execution Architecture: Compiled, Interpreted, and (Not Yet) JIT

A recurring question about the hybrid WAM → LLVM target is where it sits
on the classic compiled / JIT / interpreted spectrum. The system is a
mix, but the LLVM layer complicates the description less than it might
appear to — because LLVM is used purely as an **ahead-of-time backend**,
never as a JIT. There is no JIT anywhere in the system today. The
accurate one-line description:

> **Two execution tiers inside one AOT-compiled native binary:** PLAWK
> record loops are straight native code, and transpiled Prolog runs as
> WAM bytecode on a natively compiled interpreter linked into the same
> executable.

## Tier 1 — PLAWK programs: fully AOT-compiled

The pipeline is:

```
awk-like source
  → plawk_parse_string/2            (Prolog parser, build time)
  → AST
  → plawk_program_native_driver_ir  (Prolog codegen emits LLVM textual IR)
  → appended to write_wam_llvm_project/3 output
  → clang                           (AOT, build time)
  → native executable
```

Everything above the executable happens at **build** time. SWI-Prolog
acts as the compiler frontend and does not exist at runtime — the
shipped binary has no Prolog dependency.

The record loop in the running binary is straight machine code. For a
binary-records program like

```awk
BEGIN { BINFMT = "i64 f64" }
$1 > 100 { sum += float($2) }
END { print sum }
```

the hot loop is literally `read_record → load i64 at offset 0 →
compare → load double at offset 8 → fadd → branch`. No dispatch loop,
no boxing, no string machinery, no interning — the same instructions
you would write by hand in C. The test suites assert this shape
directly (`\+ sub_atom(DriverIR, '@run_loop')`: the bytecode
interpreter must not appear in a lowered driver).

This is what the performance numbers measure:

| workload (2M records) | time |
|---|---|
| plawk, binary records | 0.040s |
| plawk, text mode (transient lines) | 0.156s |
| mawk on the equivalent text | 0.225s |

and why streaming runs in constant memory: state lives in native SSA
slots (i64 or double loop phis) and a fixed record buffer, not on a
managed heap.

## Tier 2 — transpiled Prolog: bytecode on an AOT-compiled interpreter

When Prolog predicates are compiled through `write_wam_llvm_project/3`,
most clauses become **WAM bytecode** stored as data (`@module_code`)
inside the same binary. At runtime `run_loop` — a WAM bytecode
interpreter that is itself AOT-compiled native code — executes them,
with full unification, backtracking, and the heap/trail machinery.

So Prolog code is *interpreted*, but the interpreter is native and
statically linked, not hosted on anything else. Two refinements:

- **Native lowering (the "hybrid" in hybrid WAM/LLVM):** certain
  deterministic clause shapes skip bytecode entirely and are emitted as
  dedicated LLVM functions. Those clauses execute as tier-1 code.
- **Bytecode as fallback:** anything the lowering pass cannot prove
  deterministic ships as bytecode and runs on `run_loop`.

## The bridge: native → interpreter foreign calls

When a PLAWK rule calls a compiled Prolog predicate
(`ok($1) { hits++ }` or `score += weight($2, 10)`), compiled native
code sets up WAM argument registers, invokes the interpreter through a
per-predicate wrapper (`@plawk_foreign_guard_<name>_<arity>` /
`@plawk_foreign_call_<name>_<arity>`), and reads the result back. Each
call saves and restores the VM heap top and rewinds the arena, so
per-record foreign calls run in **constant memory at roughly 0.2µs per
call** (bytecode-interpreted WAM dispatch).

The direction of the bridge is worth noticing: a PLAWK program with
foreign calls is native code that occasionally *steps down* into
interpretation — the inverse of awk-with-extensions or
CPython-with-C-modules, which are interpreters that occasionally call
native helpers.

## Where LLVM sits

LLVM is just the code generator. Prolog emits textual IR; clang turns
it into an executable; this finishes before the program ever runs. LLVM
adds no runtime layer, no runtime dependency (beyond libc), and no JIT
tier — its complication is confined to build time.

## What would make it a JIT

A true JIT is a **Phase 5 plan**, not a present capability: the
`wam_runtime_parser_capability` item — runtime-loaded DCG grammar rules
compiled to native code while the program runs. If that lands, the
honest description becomes "AOT-compiled with a JIT tier." Today it is
"an AOT-compiled native binary containing an AOT-compiled bytecode
interpreter for the Prolog parts."

## Summary table

| layer | execution model | analogue |
|---|---|---|
| PLAWK record loops, guards, prints, scalar/assoc state, binary readers/writers | AOT-compiled native code | C |
| natively lowered Prolog clauses | AOT-compiled native code | C |
| general transpiled Prolog | WAM bytecode on a native interpreter | CPython, with the interpreter statically linked and specialized |
| foreign calls from PLAWK | native → interpreter bridge, ~0.2µs, constant memory | JNI-style downcall |
| JIT | none today; planned (Phase 5 runtime DCG compilation) | — |
