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
swipl -q -s tests/test_plawk_surface_forin_end_print.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
swipl -q -s tests/test_plawk_surface_stdin_input.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
swipl -q -s tests/test_plawk_surface_arith_exprs.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
swipl -q -s tests/test_plawk_surface_pattern_combinators.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
swipl -q -s tests/test_plawk_surface_regex_match.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
swipl -q -s tests/test_plawk_surface_end_scalar_exprs.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
swipl -q -s tests/test_plawk_surface_else_if.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
swipl -q -s tests/test_plawk_surface_prolog_calls.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
swipl -q -s tests/test_plawk_surface_float_exprs.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
swipl -q -s tests/test_wam_llvm_atom_intern_scaling.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
swipl -q -s tests/test_plawk_transient_line_records.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
swipl -q -s tests/test_plawk_binary_records.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
swipl -q -s tests/test_plawk_binary_assoc.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
swipl -q -s tests/test_plawk_float_slots.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
swipl -q -s tests/test_plawk_binfmt_strings.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
```

The demo prints the record count and the lines whose first field is `ERROR`.
The first Phase 2 surface smokes parse `/^ERROR/ { print $0 }`,
literal contains-pattern rules such as `/disk/ { print $0 }`, and
`$1 == "ERROR" { print $0 }`, plus selected-field actions such as
`$1 == "ERROR" { print $2, $3 }`. Rule prints can include native `NR` and
`NF`, e.g. `$1 == "ERROR" { print NR, NF, $2, $3 }`, plus native field lengths
such as `$1 == "ERROR" { print length($2), $2 }` and native byte substrings such
as `$1 == "ERROR" { print substr($2, 1, 3) }`, and native byte searches such as
`$1 == "ERROR" { print index($2, "sk") }`. Rule prints can also emit ASCII
case-mapped field slices such as `$1 == "ERROR" { print tolower($2), toupper($0) }`.
Explicit numeric field coercion is available as `int($N)`, e.g.
`$1 == "ERROR" { print $3, int($3) }`; failed numeric parses print `0`.
Arithmetic expressions support general `+`, `-`, `*`, `/`, and `%` between
native `i64` operands with awk precedence (`* / %` bind tighter than `+ -`,
both associate left) and parentheses, e.g.
`{ print ($2 + $3) * $4, $2 + $3 * $4 }`. Operands are integer literals,
`NR`, `NF`, `length($N)`, `int($N)`, `index($N, "literal")`, and bare numeric
fields such as `$3`, which coerce like `int($3)` inside arithmetic (a bare
`$N` on its own still prints as a byte slice). Division and modulo are
guarded: a zero divisor yields `0`, and `INT64_MIN / -1` wraps to
`INT64_MIN` (`% -1` yields `0`) instead of trapping.
Expressions become double-typed when any operand is a float literal or a
`float($N)` coercion: `{ print $2 / 2.0 }` prints `3.5` where
`{ print $2 / 2 }` prints `3`. Float literals are kept exact as integer
ratios (`1.5` emits `fdiv double 15.0, 10.0`, giving the correctly rounded
double), `float($N)` parses with `strtod` semantics (leading number,
trailing text ignored, `0` when non-numeric), and `i64` operands promote
with `sitofp`. Double results print with `%g` (so `2.0 * 10` prints `20`)
and `printf` accepts `%f`/`%g`/`%e` with optional precision such as `%.2f`.
IEEE semantics apply to float division (no zero-divisor guard). Doubles are
expression-level only in this slice: scalar slots, guards, and `END`
expressions stay `i64`, and assigning a double expression to a scalar is
rejected at codegen; typed double slots are the documented follow-up.
Rule actions can also use basic `printf` forms, e.g.
`$1 == "ERROR" { printf "%s=%s\n", $2, $3 }`. `printf` does not add `OFS` or
an implicit newline; supported native formats are `%%`, `%s` for strings and
field slices, and `%d`/`%i`/`%ld` for native `i64` values. Field slices are
lowered allocation-free by rewriting `%s` to `%.*s` and passing the slice length
and pointer to the native vararg call.
Numeric field guards use the shared WAM/LLVM `i64` comparison helper, so forms
such as `$3 > 100 { print $1, $3 }` and `$2 <= -5 { cold++ }` stay in the native
streaming loop. Patterns compose with `&&`, `||`, and `!` using awk precedence
(`!` over `&&` over `||`, parentheses group), e.g.
`$1 == "ERROR" && $3 > 100 { print $0 }`, `/^ERROR/ || /^WARN/ { print $1 }`,
and `!/disk/ { print $0 }`. Combined guards lower to straight-line bitwise
`i1` ops over the existing native guard helpers — no extra branches — and the
same combinators work in `if (...)` conditions.
POSIX ERE matching is available through awk's match operators `$N ~ /re/` and
`$N !~ /re/` (with `$0` for the whole record) and through bare `/re/` patterns
containing ERE metacharacters, e.g. `$2 ~ /^d[io]sk$/ { print $0 }` and
`/ERROR|WARN/ { print $1 }`. Bare patterns with no metacharacters keep their
existing fast native lowerings (`/^ERROR/` stays a prefix check, `/disk/` a
byte search); `\/` escapes a literal slash. Regexes are compile-time constants
compiled once per match site with libc `regcomp` (`REG_EXTENDED`) and cached;
a pattern that fails to compile never matches. Match operators work in rule
guards, `if` conditions, and `&&`/`||`/`!` combinations.
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
`else` is optional (`{ if ($1 == "ERROR") { errors++ } }`), and `else if`
chains parse as nested conditionals with awk semantics, e.g.
`{ if ($3 > 100) { big++ } else if ($3 > 10) { mid++ } else { small++ } }`;
`if` bodies can also nest further `if` statements.
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
not `printf` formats. Compiled binaries take their input the awk way: passing the
`stdin_or_argv` sentinel instead of a compile-time path emits a
`main(argc, argv)` that opens `argv[1]` at runtime, treats `-` as stdin, and
defaults to stdin when no argument is given, so `./prog file.txt`,
`./prog < file.txt`, and `cat file.txt | ./prog` all work.
Mixed scalar/associative state is
supported in the same native loop, e.g. `{ total++; counts[$1]++ }` with an
`END` print of both `total` and `counts["ERROR"]`. `END` print fields can also
include literal labels such as `print "total", total, "errors", counts["ERROR"]`.
`END` prints also take native `i64` arithmetic over final scalar values, `NR`
(the final record count), and integer literals, so canonical reports such as
`{ sum += $2 } END { print "avg", sum / NR }` lower natively; `NR` reads the
loop-head record phi, and empty input divides to `0` through the shared
guarded division. Scalar variables can also be read inside rule-body update
expressions, e.g. `{ avg = $2 / 2; total += avg }` or `{ x = x + 2 }`;
assignments apply in source order, a read before any write sees `0` (awk's
uninitialized-variable semantics), and reads work across rules within the
same record.
`END` can also iterate an associative array with the canonical awk report
idiom, e.g. `{ counts[$1]++ } END { for (k in counts) print k, counts[k] }`.
The loop lowers to a native walk over the runtime table's occupied slots via
`wam_assoc_i64_iter_next`, maps each key id back to its text with
`wam_atom_to_string`, and prints the loop key, same-array values (direct slot
reads), other-array lookups such as `errs[k]` (hash lookups, `0` for missing
keys), and string literal labels. Iteration order follows the hash table's
slot order and, as in awk, is unspecified.

Compiled Prolog predicates in the same binary are callable from PLAWK — the
hybrid's headline capability. A named predicate is a rule guard
(`plawk_is_error($1) { print $0 }` matches when the predicate succeeds) or an
`i64` expression (`{ total += severity_rank($1) }` calls `severity_rank/2`
with a trailing output argument and yields its integer binding, `0` on
failure). Arguments are field atoms (`$0` is the whole record), string
literal atoms, and integers; guards compose with `&&`/`||`/`!` and `if`
conditions, and calls compose with native arithmetic. Codegen collects the
called predicates and emits one wrapper per shape around a lazily created
shared `%WamState`; each wrapper runs `wam_prepare_call` + `run_loop`, then
restores the VM heap top and rewinds the arena, so per-record foreign calls
run in constant memory at roughly 5µs per call (bytecode-interpreted WAM
dispatch). Programs with foreign calls use
`plawk_program_native_driver_ir/4` with `wam_vm(InstrCount, LabelCount)`
from `wam_llvm_last_compile_counts/2` after `write_wam_llvm_project/3`
compiled the predicates into the module.

The first binary/typed-record slice is in:
`BEGIN { BINFMT = "i64 i64 f64" }` switches the program to fixed-layout
binary records (`$1..$N`; `i64`/`f64` fields are 8 native-endian bytes,
`sN` fields are N fixed bytes, offsets and record size follow from the
declared layout). Field access compiles to a typed load at a compile-time
offset — no field splitting, no numeric parsing, no interning — so
`BEGIN { BINFMT = "i64 i64" } $1 > 100 { sum += $2 } END { print sum }`
is a load-compare-add loop. `i64` fields work in guards, arithmetic,
scalar updates, and prints; `f64` fields load as native doubles for
prints and `float($N)` double expressions. Scalar accumulators are typed
by inference: `sum += float($2) * 1.5` makes `sum` a native double slot
(double loop phis, `fadd` updates, `%g` END prints) while `n++` in the
same program stays i64 — a scalar becomes double when any update
assigns it a float-typed expression or reads an already-double scalar
(fixpoint), and i64 operands promote via `sitofp` at the update site.
This works in text mode (`float($N)` = strtod) and binary mode
(`float($N)` = native f64 field load), through `if`/`else`,
`next`/`break`, and rule chains. `NF` is a compile-time
constant; `NR`, `if/else`, `next`/`break`, `printf`, and END reports
compose unchanged. Associative arrays keyed by i64 fields work in
binary mode: `{ counts[$1]++ }` uses the raw field value as the table key
(no interning anywhere in the record loop), `END { for (k in counts) print
k, counts[k] }` prints keys numerically, and END lookups take integer
literals (`print counts[5], counts[-3]`); integer keys are binary-only
(in text mode they would collide with atom ids, so they are rejected
there). Fixed-width string fields are in: with
`BINFMT = "s8 i64"`, `print $1` emits the bytes up to the first NUL or
the field width (strnlen + `%.*s`, no copying), and `$1 == "ERR"`
compiles to a memcmp plus a NUL check at the key length (skipped for
full-width keys; oversized keys fold to constant false). String fields
are print/equality only: arithmetic, `float()`, numeric compares, and
assoc keys on `sN` fields are rejected. Remaining text-shaped forms
($0, regex, substr/length/index/case, string assoc keys, foreign calls)
are rejected at codegen in binary mode. A trailing partial record exits with the read
error code. Measured on 2M records: 0.040s for the binary program vs
0.225s for mawk on the equivalent text (5.6x) and 0.156s for plawk's own
text mode.

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
