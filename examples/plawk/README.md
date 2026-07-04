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

## The plawk CLI

`examples/plawk/bin/plawk` compiles `.plawk` programs to native
binaries in one step (requires swipl + clang on PATH; run from
anywhere -- the script locates its own modules):

```bash
examples/plawk/bin/plawk build program.plawk -o program   # compile
./program input.txt                                       # like awk
producer | ./program                                      # stdin
examples/plawk/bin/plawk run program.plawk input.txt      # build + run
```

The produced binary follows the awk input convention: first argument
is the input file, `-` or no argument reads stdin. `--keep-ll` keeps
the intermediate LLVM IR next to the output. Exit codes: 2 parse
error, 3 compile error (including a program that calls a predicate no
`@prolog` block or `function` defines), 4 clang failure; `run`
propagates the program's own exit status. Compilation uses
`clang -O2`.

## Benchmarks

`examples/plawk/bench/bench.sh` generates text and binary workloads,
verifies plawk's output is byte-identical to the system awk on every
text job (a hard gate before any timing), then reports best-of-3 wall
times. One run on this repository's CI-like container (4-core Xeon
2.80GHz, mawk 1.3.4 -- the fast awk -- as the baseline, `clang -O2`),
N = 2,000,000 records (~29MB text / 32MB binary):

| workload | plawk | mawk | speedup |
|---|---|---|---|
| W1 filter-count (text) | 104 ms | 180 ms | 1.7x |
| W2 aggregate (text) | 139 ms | 309 ms | 2.2x |
| W3 aggregate (binary records) | 17 ms | 234 ms | 13.8x |
| W4 group-by (text) | 117 ms | 186 ms | 1.6x |

W3 is the thesis workload: the same aggregation over `i64 f64` binary
records instead of their text encoding -- no field splitting, no
number parsing, just typed loads at fixed offsets. Text-mode wins are
honest but modest (mawk is very fast at what it does); the binary
representation is where the design pays. Numbers are
environment-relative; rerun the script for yours (`N=... sh
examples/plawk/bench/bench.sh`).

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
swipl -q -s tests/test_plawk_binary_writers.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
swipl -q -s tests/test_plawk_forin_writebin.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
swipl -q -s tests/test_plawk_outfmt_strings.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
swipl -q -s tests/test_plawk_varlen_records.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
swipl -q -s tests/test_plawk_varlen_writers.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
swipl -q -s tests/test_plawk_tagged_unions.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
swipl -q -s tests/test_plawk_bounded_rep.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
swipl -q -s tests/test_plawk_rep_strings.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
swipl -q -s tests/test_plawk_union_rep.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
swipl -q -s tests/test_plawk_tier2_blob.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
swipl -q -s tests/test_plawk_f64_foreign.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
swipl -q -s tests/test_plawk_union_writebin.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
swipl -q -s tests/test_plawk_union_assoc.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
swipl -q -s tests/test_plawk_rep_writer.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
swipl -q -s tests/test_plawk_union_out.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
swipl -q -s tests/test_plawk_multiline.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
swipl -q -s tests/test_plawk_prolog_blocks.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
swipl -q -s tests/test_plawk_functions.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
swipl -q -s tests/test_plawk_cli.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
swipl -q -s tests/test_plawk_bench_smoke.pl -g "setenv('UW_SMOKE_TMPDIR', '/mnt/c/Users/johnc/Scratch'),run_tests" -t halt
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
Programs are multi-line awk: `#` comments run to end of line,
statements separate on newlines as well as `;` (with awk/C semantics
after a compound statement's closing brace -- no separator needed),
and trailing separators before `}` are harmless. A program can carry
its Prolog with it: `@prolog ... @end` blocks hold ordinary clauses
(DCG rules included) that compile into the same binary and are
callable through the foreign bridge --

```awk
@prolog
weight(I, F, R) :- R is I * F.
hot(X) :- X > 100.
@end
BEGIN { BINFMT = "i64 f64" }
hot($1) { wsum += float(weight($1, $2)) }
END { print wsum }
```

Markers sit alone on their line; the heredoc-style tagged form
(`@prolog-TAG ... @end-TAG`, exact tag match) fences Prolog text that
itself contains an `@end`-shaped line. `plawk_parse_source/3` returns
program + clauses, and `plawk_prolog_block_preds/2` installs them for
`write_wam_llvm_project/3`. awk-style expression functions are sugar
over the same bridge: `function scale(a, b) { return a * b + 1 }`
desugars at parse time to the Prolog clause
`scale(A, B, R) :- R is A * B + 1` (awk precedence, `%` maps to mod,
float literals allowed) and is called like any bridged predicate --
`scale($1, $2)` as an integer expression, `float(scale($1, $2))` to
keep fractions.
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
`next`/`break`, and rule chains. END arithmetic composes with double
slots: `END { print sum / NR }` promotes the whole expression to double
(IEEE `fdiv`, no divide-by-zero guard, `%g` print), and float literals
in END expressions (`print n * 1.5`) do the same. `NF` is a compile-time
constant; `NR`, `if/else`, `next`/`break`, `printf`, and END reports
compose unchanged. Associative arrays keyed by i64 fields work in
binary mode: `{ counts[$1]++ }` uses the raw field value as the table key
(no interning anywhere in the record loop), `END { for (k in counts) print
k, counts[k] }` prints keys numerically, and END lookups take integer
literals (`print counts[5], counts[-3]`); integer keys are binary-only
(in text mode they would collide with atom ids, so they are rejected
there). Variable-length records are in via `lpsN`
(length-prefixed string, 8-byte length + up to N payload bytes):
`BINFMT = "i64 lps16"` switches to a field-by-field varlen read loop
that materializes each record into the same fixed access layout, so an
`lpsN` field behaves exactly like `sN` downstream (prints, equality
guards, `sN` OUTFMT passthrough) while numeric fields keep guards,
arithmetic, and assoc keys. Clean EOF is only legal at a record
boundary; oversized lengths, truncated payloads, and mid-record EOF
exit with the read-error code. Tier-2 payloads connect the two worlds:
`BINFMT = "i64 blob32"` declares a length-prefixed binary payload whose
only consumer is a compiled Prolog predicate - the loop frames records
natively, and `total += payload_sum($2)` hands the payload bytes to a
WAM-compiled DCG through the ~0.2us foreign bridge (constant memory:
the payload rides the shared transient atom, no interning). i64 fields
marshal as WAM integers and f64 fields as WAM floats in binary mode, so
foreign guards and calls work over binary records generally; a
double-returning call is spelled `wsum += float(score($1, $2))` (the
float(...) wrapper selects a {double, ok} bridge that accepts Integer
or Float results, keeping fractions that the i64 spelling would
truncate; a failed call contributes 0.0); the same spelling works in
print position (`print $1, float(score($1, $2))`). One blob argument per call;
payloads are NUL-free byte strings.
Runtime-loaded grammars (JIT) go one step further than the compiled
bridge: `BEGIN { DYNLOAD = "file.wamo" }` names a WAM object built ahead
of time with `write_wam_object/3`, and `dyncall(args...)` invokes its
entry at runtime, yielding an i64 (0 on load/call failure). The object
loads lazily on the first `dyncall` and is reused; swapping the `.wamo`
file changes behaviour with **no rebuild of the plawk binary**. `dyncall`
is a reserved form (never a compiled predicate), so the spelling marks
the JIT boundary. A grammar predicate has arity N+1 (N inputs read as
`A0..A_{N-1}`, output in `A_N`). Example:
`BEGIN { BINFMT = "i64" ; DYNLOAD = "square.wamo" } { total += dyncall($1) } END { print total }`
sums `X*X` over i64 records; overwrite `square.wamo` with a doubling
grammar and the same binary sums `X*2`.
`dyncall_at(Source, args...)` is the dynamic-source form: `Source` (a
field or string) names the `.wamo` at runtime, so a program chooses its
grammar per record — e.g. `{ total += dyncall_at($1) }` picks the grammar
named in column 1 of each line. Object management is set by
`BEGIN { DYNCACHE = "on" | "mtime" | "off" }` (default `on`): `on` caches
each distinct grammar (load once, reuse); `mtime` also keys on the file's
modification time, so recompiling a `.wamo` busts the cache and the new
definition takes effect with no rebuild (query/userspace redefinition);
`off` reloads and frees every call (always current, no cache).
`float(dyncall(...))` / `float(dyncall_at(...))` read the grammar's output
as a double (keeping fractions), mirroring `float(name(args))` for
compiled predicates — needed when a grammar returns a Float (e.g.
`R is X / 2`), which the integer form cannot read (it yields 0).
`blob(dyncall(...))` / `blob(dyncall_at(...))` read an Atom output as
**opaque bytes** (a byte slice) for `print` — a grammar that emits text or
encoded output rather than a number, e.g.
`{ print blob(dyncall($1)) }`.
A single `.wamo` can expose several named entries (built with
`write_wam_object(Preds, [wamo_entries([P/A, ...])], File)`);
`dyncall@name(args...)` selects one by name at the call site — e.g.
`{ s += dyncall@square($1) ; c += dyncall@cube($1) }` calls two entries of
one `DYNLOAD` object. The `@name` is fixed at compile time, so the entry's
address is resolved once at startup and reused (a name no entry exposes
yields 0). Bounded repetition handles records containing a
list: `BINFMT = "i64 rep4(i64 f64)"` declares an 8-byte element count
(at most 4) followed by that many (i64, f64) elements. The count is an
ordinary i64 field, element slots are flat addressable fields
(zero-filled past the count), and `foreach { ... }` runs its block once
per element with `$1..$M` meaning the current element's fields - a
real runtime loop (loop-carried phis per scalar slot; the current
element is staged into a hidden slot at the end of the record buffer
so field accesses stay compile-time offsets), so code size is
independent of the cap and rep64 costs the same IR as rep4. The wire
read is one bulk count*elemsize read for fixed-width elements; elements
may also contain `lpsN` strings (`BINFMT = "i64 rep4(lps8 i64)"`), in
which case the reader loops, parsing one variable-length element at a
time into its fixed in-memory slot group, so `foreach` string guards
and prints work unchanged. Oversized counts, oversized element strings,
and truncated element regions exit with the read-error code. Tagged unions let one stream carry several
record kinds: `BINFMT = "case(i64 f64 | lps16 i64)"` declares that
every record starts with an 8-byte tag selecting an arm layout, and
`case K { ... }` blocks hold ordinary pattern-action rules whose
`$1..$N` are typed by arm K. Scalars (including doubles), `NR`,
`next`/`break`, `if`/`else`, and the END report are shared across
arms; the reader is a native tag switch dispatching per-arm field
reads into one record buffer sized to the widest arm. Unknown tags
and truncated arms exit with the read-error code; arms with no case
block are still read and skipped, keeping the stream framed. Rules may
also lead with a tag guard instead: `TAG == 1 && $1 == "boom" { events++ }`
is pure sugar for the same rule inside `case 1 { ... }` (identical IR);
every rule must then lead with `TAG == K`, and tag tests under `||`/`!`
or in non-leftmost position are rejected. Arms can carry their own
repetition: `BINFMT = "case(i64 rep4(lps8 i64) | i64)"` gives arm 0 an
element list, and `foreach` inside that arm's rules (either spelling)
iterates it -- element types, staging, and buffer sizing all resolve
per arm. writebin also works inside case blocks: OUTFMT stays
program-wide (one output layout regardless of arm) while each rule's
source fields type against its own arm, so a two-arm stream can be
normalized into one fixed layout -- a pure per-arm normalizer needs no
END and no scalars at all. Assoc group-bys work across arms too:
`case 0 { { counts[$1]++ } } case 1 { { counts[$2]++ } }` counts into
one shared table (keys are raw i64 field values typed per arm), with
the usual END reports -- `for (k in counts) print k, counts[k]` or
integer lookups.
`lpsN` also works in OUTFMT: writebin emits the
8-byte length plus exactly the payload bytes (no padding), sourcing
from literals, `sM`/`lpsM` input fields, or text-mode slices clamped to
the cap - writer output is byte-compatible with the `lpsN` reader, so
varlen plawk-to-plawk pipelines round-trip. `repK(...)` works in
OUTFMT as a passthrough: the writebin argument names the input rep's
count field (`OUTFMT = "i64 rep4(i64 f64)"` with `writebin $1, $2`),
and the writer emits the live count plus one bulk copy of the live
elements - so guarded rules make byte-exact stream filters. Elements
with `lpsN` strings pass through too: the writer loops over the live
elements, recovering each string's live length from its NUL-padded
slot and emitting the length prefix plus exactly those bytes. The
input rep's cap and element layout must match the output slot exactly.
Output can be tagged too: `OUTFMT = "case(i64 | i64 lps8)"` declares a
union output, and `writebin case K, args` emits the 8-byte tag K then
arm K's slots -- byte-compatible with the union reader, so a plawk
program can split, retag, or normalize a stream that another plawk
program consumes directly. See
[`docs/design/PLAWK_DCG_BINARY_READERS.md`](../../docs/design/PLAWK_DCG_BINARY_READERS.md)
for the grammar-to-native-reader lowering design this is the first
slice of. Fixed-width string fields are in: with
`BINFMT = "s8 i64"`, `print $1` emits the bytes up to the first NUL or
the field width (strnlen + `%.*s`, no copying), and `$1 == "ERR"`
compiles to a memcmp plus a NUL check at the key length (skipped for
full-width keys; oversized keys fold to constant false). String fields
are print/equality only: arithmetic, `float()`, numeric compares, and
assoc keys on `sN` fields are rejected. Remaining text-shaped forms
($0, regex, substr/length/index/case, string assoc keys, foreign calls)
are rejected at codegen in binary mode.

Binary *writers* close the pipeline loop: `BEGIN { OUTFMT = "i64 f64" }
{ writebin $1, float($2) }` emits one fixed-layout binary record on
stdout per call (typed stores into a reused entry-block buffer, then a
buffered `fwrite`). writebin works in text mode (a text-to-binary
converter) and binary mode (a binary-to-binary transform), takes i64
expressions, NR/NF, scalar reads, and double expressions per the OUTFMT
slot types (i64 arguments promote into f64 slots), and composes with
guards, scalar updates, and `if`/`else`. A plawk-to-plawk pipeline -
converter | aggregator - runs with no text serialization between
stages. Group-by results can also leave as binary:
`END { for (k in counts) writebin k, counts[k] }` walks the table and
emits one record per group (raw i64 keys, table values, or literals;
i64 values promote into f64 output slots) - binary input mode only,
since text-mode keys are interned atom ids. OUTFMT also takes `sN` string slots: sources are string
literals that fit the width, `sM` binary input fields with `M <= N`
(memcpy + zero-fill), or text-mode field slices clamped to the width
(a missing field writes all zeros) - so text-to-binary converters
carry names alongside numbers. Rejected: writebin without OUTFMT,
argument/layout arity mismatch, oversized literals or source fields,
numeric fields into string slots, and double expressions into i64
slots. A trailing partial record exits with the read
error code. Measured on 2M records: 0.040s for the binary program vs
0.225s for mawk on the equivalent text (5.6x) and 0.156s for plawk's own
text mode.

For a walkthrough of the current Prolog-core syntax and how it maps to awk
concepts like `$0`, `$1`, `NR`, `NF`, `FS`, `OFS`, and `print`, see
[`TUTORIAL.md`](TUTORIAL.md) — which now ends with a
from-first-principles walkthrough of binary records, `lps`
length-prefixed strings, and tagged unions / `case` blocks (what a
tag, an arm, and a discriminated record are, with byte-level
diagrams).

## Design documents

- [`docs/design/PLAWK_PHILOSOPHY.md`](../../docs/design/PLAWK_PHILOSOPHY.md)
- [`docs/design/PLAWK_SPECIFICATION.md`](../../docs/design/PLAWK_SPECIFICATION.md)
- [`docs/design/PLAWK_IMPLEMENTATION_PLAN.md`](../../docs/design/PLAWK_IMPLEMENTATION_PLAN.md)
- [`docs/design/PLAWK_EXECUTION_ARCHITECTURE.md`](../../docs/design/PLAWK_EXECUTION_ARCHITECTURE.md)
- [`docs/design/PLAWK_DCG_BINARY_READERS.md`](../../docs/design/PLAWK_DCG_BINARY_READERS.md)
  — grammar-driven binary readers: the native/WAM lowering spectrum and
  the varlen (`lpsN`) reader design.
  — where the system sits on the compiled/JIT/interpreted spectrum: PLAWK
  loops are AOT native code, transpiled Prolog is WAM bytecode on a
  natively compiled interpreter in the same binary, and there is no JIT
  (yet).

The implementation plan's **Codebase reconciliation** section captures the
original audit findings and the current compiled-stream boundary. The buffered
streaming reader now exists in the WAM/LLVM target; the remaining work is to
grow the surface parser/codegen until ordinary awk-like programs lower through
the same native streaming path.
