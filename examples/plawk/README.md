<!--
SPDX-License-Identifier: MIT
Copyright (c) 2026 John William Creighton (s243a)
-->

# plawk - Prolog awk: a compiled, binary-record awk/bash-like DSL

> **Status:** design / prototype phase. This directory is the home of a planned
> UnifyWeaver-powered application or future submodule. The canonical
> specification lives in the design triad under [`docs/design/`](../../docs/design/)
> (links below).

## What this is

A small **awk-like surface language** whose pattern-action programs are lowered
to a deterministic Prolog core (`process_all/4` + Reader/Handler/Writer) and
compiled with UnifyWeaver's **hybrid WAM -> LLVM** target to a native binary.

The point is to keep awk's ergonomic *familiar surface* while replacing its
interpreted, string-centric runtime with a **compiled, binary/typed-record**
interior - so stream-processing programs can sit in the same native binary as
UnifyWeaver's graph algorithms with **no text serialization between stages**.

This is intentionally an application/frontend rather than core UnifyWeaver
machinery. Building it should expose the next set of representation and runtime
gaps that UnifyWeaver needs to support.

## Why it is mostly already built

The LLVM hybrid WAM target (`src/unifyweaver/targets/wam_llvm_target.pl`) has
accumulated **~162 builtins**. Categorized, they already cover:

- **awk's expression half** - arithmetic/compare, `split_string/4`, `sub_atom/5`,
  `atom_*`/`string_*`, `functor`/`arg`/`=..`.
- **awk's associative-array half** - `sort`/`msort`/`keysort`, `nth0/1`, `length`,
  `sum_list`/`max_list`, `intersection`/`union`/`subtract`, `pairs_keys_values`.
- **bash's half** - `shell/1,2`, `mkfifo/2`, `kill/2`, the POSIX id/limit/priority
  syscalls, and a full filesystem layer (`copy_file`, `directory_files`,
  `realpath`, `symlink`, `path_join`, `working_directory`, ...).

So this is a frontend application: surface parser -> Prolog core -> existing
WAM -> LLVM pipeline + those builtins as its standard library.

## Naming note

The name **`plawk`** = "**Prolog awk**": an awk-like surface compiled through
Prolog -> WAM -> LLVM. It also disambiguates from the existing **AWK target**
(`docs/AWK_TARGET_STATUS.md`, `src/unifyweaver/targets/awk_target.pl`), which
goes the opposite direction: it emits awk scripts from Prolog facts. The two
are complementary.

## Planned structure

```text
plawk/
  README.md          this file
  core/              process_all/4, reader/handler/writer, item_field/3   (Phase 0)
  parser/            awk-like surface -> AST -> Prolog core               (Phase 2)
  codegen/           AST -> Prolog-core source emission                   (Phase 2)
```

## Run the Phase 0 prototype

```bash
swipl -q -s examples/plawk/core/plawk_core_tests.pl -g run_tests -t halt
swipl -q -s examples/plawk/demo/count_errors.pl -t halt
swipl -q -s examples/plawk/demo/print_error_fields.pl -t halt
swipl -q -s tests/test_plawk_compiled_stream_core.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
swipl -q -s tests/test_plawk_native_outer_loop_driver.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
swipl -q -s tests/test_plawk_native_stream_loop_driver.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
swipl -q -s tests/test_plawk_native_counter_stream_loop_driver.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
```

The demo prints the record count and the lines whose first field is `ERROR`.

For a walkthrough of the current Prolog-core syntax and how it maps to awk
concepts like `$0`, `$1`, `NR`, `NF`, `FS`, `OFS`, and `print`, see
[`TUTORIAL.md`](TUTORIAL.md).

## Design documents

- [`docs/design/PLAWK_PHILOSOPHY.md`](../../docs/design/PLAWK_PHILOSOPHY.md)
- [`docs/design/PLAWK_SPECIFICATION.md`](../../docs/design/PLAWK_SPECIFICATION.md)
- [`docs/design/PLAWK_IMPLEMENTATION_PLAN.md`](../../docs/design/PLAWK_IMPLEMENTATION_PLAN.md)

The implementation plan's **Codebase reconciliation** section captures the
three audit findings: modes are already consumed by codegen, `:- det` is not,
and a buffered streaming reader is the one builtin that must be added before the
awk loop is real.
