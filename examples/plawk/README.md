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
  parser/            awk-like surface -> AST                              (Phase 2)
  codegen/           AST -> native WAM/LLVM driver fragments              (Phase 2)
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
swipl -q -s tests/test_plawk_native_output_stream_loop_driver.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
swipl -q -s tests/test_plawk_native_lowered_handler_stream_loop_driver.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
swipl -q -s tests/test_plawk_surface_prefix_print.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
```

The demo prints the record count and the lines whose first field is `ERROR`.
The first Phase 2 surface smokes parse `/^ERROR/ { print $0 }` and
`$1 == "ERROR" { print $0 }`, plus selected-field actions such as
`$1 == "ERROR" { print $2, $3 }`. Rule prints can include native `NR` and
`NF`, e.g. `$1 == "ERROR" { print NR, NF, $2, $3 }`, plus native field lengths
such as `$1 == "ERROR" { print length($2), $2 }` and native byte substrings such
as `$1 == "ERROR" { print substr($2, 1, 3) }`, and native byte searches such as
`$1 == "ERROR" { print index($2, "sk") }`. Rule prints can also emit ASCII
case-mapped field slices such as `$1 == "ERROR" { print tolower($2), toupper($0) }`.
Explicit numeric field coercion is available as `int($N)`, e.g.
`$1 == "ERROR" { print $3, int($3) }`; failed numeric parses print `0`.
The first arithmetic composition forms add or subtract a non-negative integer
constant from native `i64` primaries such as `NR`, `NF`, `length($N)`,
`int($N)`, and `index($N, "literal")`, e.g.
`$1 == "ERROR" { print NR - 1, int($3) + 1, index($2, "sk") + 1 }`.
Numeric field guards use the shared WAM/LLVM `i64` comparison helper, so forms
such as `$3 > 100 { print $1, $3 }` and `$2 <= -5 { cold++ }` stay in the native
streaming loop.
Scalar state works with `$1 ==
"ERROR" { count++ } END { print count }`. Multiple scalar increments
compile to indexed native slots, e.g. `{ errors++; matches++ }`, and multiple
guarded rules can update shared scalar slots before an `END` print. Scalar slots
also support native `+=` with integer constants and field lengths, e.g.
`$1 == "ERROR" { bytes += length($0); hits += 2 } END { print bytes, hits }`.
They also support numeric field expressions through the same shared field parser,
e.g. `$1 == "ERROR" { bytes += $3; last = $3 } END { print bytes, last }`.
Plain scalar assignment uses the same native slot path and preserves source
order with later updates, e.g. `$1 == "ERROR" { last_len = length($0); hits++ }
END { print hits, last_len }`. The current assignment expression subset is
integer literals, `NR`, `NF`, `length($N)`, `index($N, "literal")`, numeric
`$N`, explicit `int($N)`, and native scalar `i64` primary `+/- K` forms such as
`NF + K`, `length($N) - K`, `int($N) + K`, and
`index($N, "literal") + K`.
Scalar slot updates can also sit behind native `if/else` guards, e.g.
`{ if ($1 == "ERROR") { errors++; last_len = length($0) } else { non_errors++ } }
END { print errors, non_errors, last_len }`. The first branch slice supports
field-equality conditions, scalar updates, field-key associative increments,
selected-field and string-literal `print` including `NR`, and terminal `next`/`break` inside
branches. The native lowering evaluates each source `if` guard once, threads
every scalar slot through the then/else bodies, emits associative table
increments and branch-local prints only on the selected branch, rejoins scalar
slots with per-slot LLVM phis, routes selected branch-local `next` paths to the
stream loop continuation, and routes selected branch-local `break` paths to the
stream close path before `END`.
Terminal `next` is supported in native rule chains, so `$1 == "DEBUG" {
skipped++; next } { total++ } END { print total, skipped }` skips the later
rule for matching records. Terminal `break` is supported in the same native
rule-chain shape and closes the stream before running `END`, e.g. `$1 == "ERROR"
{ hits++; break } { total++ } END { print hits, total }`. If `next` or `break`
appears before the end of a rule body or branch, later actions in that same
action list are treated as unreachable tail actions and skipped by native
codegen. The first
associative-count surface now supports multiple source arrays, e.g.
`{ counts[$1]++; by_component[$2]++ } END { print counts["ERROR"], by_component["disk"] }`.
Codegen allocates one WAM/LLVM runtime interned-atom-keyed `i64` table per
source array rather than specializing the `END` keys to fixed slots. Guarded
associative count rules lower through the same native rule-chain shape as scalar
counters, so `$1 == "ERROR" { by_component[$2]++ }` only updates on matches. Each table
grows and rehashes as needed, and the native stream reader grows its line buffer
without consuming WAM arena space per record. `BEGIN` print clauses can emit
literal report headers before the stream opens. The default space `FS` uses
AWK-style whitespace splitting, so leading whitespace is ignored and whitespace
runs do not create empty fields. The first `BEGIN` assignment slice supports
explicit single-byte `FS` values such as `BEGIN { FS = ":" }` for native field
equality, selected-field printing, and associative key extraction. Single-byte
`OFS` values such as `BEGIN { OFS = "," }` drive comma-separated `print` fields;
the native path emits separator bytes directly, so values such as `%` are data,
not `printf` formats. Mixed scalar/associative state is
supported in the same native loop, e.g. `{ total++; counts[$1]++ }` with an
`END` print of both `total` and `counts["ERROR"]`. `END` print fields can also
include literal labels such as `print "total", total, "errors", counts["ERROR"]`.

For a walkthrough of the current Prolog-core syntax and how it maps to awk
concepts like `$0`, `$1`, `NR`, `NF`, `FS`, `OFS`, and `print`, see
[`TUTORIAL.md`](TUTORIAL.md).

## Design documents

- [`docs/design/PLAWK_PHILOSOPHY.md`](../../docs/design/PLAWK_PHILOSOPHY.md)
- [`docs/design/PLAWK_SPECIFICATION.md`](../../docs/design/PLAWK_SPECIFICATION.md)
- [`docs/design/PLAWK_IMPLEMENTATION_PLAN.md`](../../docs/design/PLAWK_IMPLEMENTATION_PLAN.md)

The implementation plan's **Codebase reconciliation** section captures the
original audit findings and the current compiled-stream boundary. The buffered
streaming reader now exists in the WAM/LLVM target; the remaining work is to
grow the surface parser/codegen until ordinary awk-like programs lower through
the same native streaming path.
