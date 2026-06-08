<!--
SPDX-License-Identifier: MIT
Copyright (c) 2026 John William Creighton (s243a)
-->

# plawk — Prolog awk: a compiled, binary-record awk/bash-like DSL

> **Status:** design phase. This directory is the home of a planned UnifyWeaver
> submodule. No runtime code yet — the canonical specification lives in the
> design triad under [`docs/design/`](../../../docs/design/) (links below).

## What this is

A small **awk-like surface language** whose pattern–action programs are lowered
to a deterministic Prolog core (`process_all/4` + Reader/Handler/Writer) and
compiled through UnifyWeaver's **hybrid WAM → LLVM** target to a native binary.

The point is to keep awk's ergonomic *familiar surface* while replacing its
interpreted, string-centric runtime with a **compiled, binary/typed-record**
interior — so stream-processing programs can sit in the same native binary as
UnifyWeaver's graph algorithms with **no text serialization between stages**.

## Why it's mostly already built

The LLVM hybrid WAM target (`src/unifyweaver/targets/wam_llvm_target.pl`) has
accumulated **~162 builtins**. Categorized, they already cover:

- **awk's expression half** — arithmetic/compare, `split_string/4`, `sub_atom/5`,
  `atom_*`/`string_*`, `functor`/`arg`/`=..`.
- **awk's associative-array half** — `sort`/`msort`/`keysort`, `nth0/1`, `length`,
  `sum_list`/`max_list`, `intersection`/`union`/`subtract`, `pairs_keys_values`.
- **bash's half** — `shell/1,2`, `mkfifo/2`, `kill/2`, the POSIX id/limit/priority
  syscalls, and a full filesystem layer (`copy_file`, `directory_files`,
  `realpath`, `symlink`, `path_join`, `working_directory`, …).

So this submodule is **a frontend, not a new backend**: surface parser →
Prolog core → existing WAM→LLVM pipeline + those builtins as its standard library.

## Naming note (collision)

The name **`plawk`** = "**Prolog awk**": an awk-like surface compiled through
Prolog → WAM → LLVM. It also disambiguates from the existing **AWK target**
(`docs/AWK_TARGET_STATUS.md`, `src/unifyweaver/targets/awk_target.pl`), which
goes the *opposite* direction — it *emits* awk scripts *from* Prolog facts. The
two are complementary: one emits awk for portability, `plawk` consumes an
awk-like surface for native performance.

## Planned structure

```
plawk/
  README.md          this file
  core/              process_all/4, reader/handler/writer, item_field/3   (Phase 0)
  parser/            awk-like surface → AST → Prolog core                 (Phase 2)
  codegen/           AST → Prolog-core source emission                    (Phase 2)
```

## Design documents

The original single design doc has been split into UnifyWeaver's standard
philosophy / specification / implementation-plan triad, **reconciled against the
existing codebase**:

- [`docs/design/PLAWK_PHILOSOPHY.md`](../../../docs/design/PLAWK_PHILOSOPHY.md)
- [`docs/design/PLAWK_SPECIFICATION.md`](../../../docs/design/PLAWK_SPECIFICATION.md)
- [`docs/design/PLAWK_IMPLEMENTATION_PLAN.md`](../../../docs/design/PLAWK_IMPLEMENTATION_PLAN.md)

The implementation plan's **§ Codebase reconciliation** captures the three audit
findings: modes are already consumed by codegen, `:- det` is not, and a buffered
streaming reader is the one builtin that must be added before the awk loop is real.
