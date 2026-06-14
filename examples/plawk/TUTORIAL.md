<!--
SPDX-License-Identifier: MIT
Copyright (c) 2026 John William Creighton (s243a)
-->

# PLAWK Tutorial

PLAWK is currently written as a small Prolog core that behaves like awk. The
planned surface syntax can become more awk-like later, but the Phase 0 examples
show the execution model directly: a reader produces records, a handler processes
each record, and state carries counters and output.

## Example: print fields from matching records

Input file:

```text
INFO boot ok
ERROR disk full
WARN cpu hot
ERROR net down
```

The awk version is concise:

```awk
$1 == "ERROR" { print NR, $2, $3 }
```

With `OFS=","`, that prints:

```text
2,disk,full
4,net,down
```

The current PLAWK core version is:

```prolog
:- initialization(main, main).

:- use_module('../core/plawk_core').

main :-
    Reader = text_file_reader('examples/plawk/demo/error_fields.txt', " "),
    Handler = print_error_fields,
    State0 = state([], [], 0, plawk_options(" ", ",")),
    process_all(Reader, Handler, State0, StateN),
    state_outputs(StateN, Lines),
    forall(member(Line, Lines), format('~s~n', [Line])).

print_error_fields(Item, State0, StateN, yes) :-
    increment_counter(State0, State1),
    (   item_field(1, Item, "ERROR")
    ->  item_field(2, Item, Component),
        item_field(3, Item, Message),
        nr(State1, RecordNumber),
        print_fields([RecordNumber, Component, Message], State1, StateN)
    ;   StateN = State1
    ).
```

Run it from the repository root:

```bash
swipl -q -s examples/plawk/demo/print_error_fields.pl -t halt
```

## How it maps to awk

| AWK idea | PLAWK core |
|---|---|
| input record | `Item` |
| `$0` | `item_field(0, Item, Line)` |
| `$1`, `$2`, ... | `item_field(1, Item, F1)`, `item_field(2, Item, F2)` |
| `NR` | `nr(State, RecordNumber)` |
| `NF` | `nf(Item, Count)` |
| `FS` | `plawk_options(FieldSeparator, OutputSeparator)` |
| `OFS` | `plawk_options(FieldSeparator, OutputSeparator)` |
| `print a, b, c` | `print_fields([A, B, C], State0, StateN)` |
| pattern guard | the condition before `->` |
| action block | the body after `->` |

The Prolog version is longer because it exposes the compiler target directly.
The future awk-like syntax should lower to code shaped like this.

## The three moving parts

### Reader

```prolog
Reader = text_file_reader(Path, FieldSeparator)
```

The reader converts each line into a `record(text, Line, Fields)` item. Phase 0
uses a whole-file SWI reader for convenience. The compiled smoke in
`tests/test_plawk_compiled_stream_core.pl` now exercises the target-side
streaming reader: it opens a file with `stream_open/2`, reads lines with
`read_line/2`, splits fields with `atom_split/3`, and runs a PLAWK-style
handler in a native LLVM binary.

### Handler

```prolog
print_error_fields(Item, State0, StateN, yes) :- ...
```

The handler is the compiled equivalent of an awk pattern-action block. It takes
the current `Item`, transforms `State0` into `StateN`, and returns `yes` to keep
processing records.

### State

```prolog
state(InputStreams, OutputStreams, Counter, UserFields)
```

The Phase 0 examples use the counter as `NR`, collect printed lines in
`OutputStreams`, and store `FS`/`OFS` in `UserFields` as `plawk_options(FS, OFS)`.

## Current surface example: count matching records

The native Phase 2 surface can compile rule prints with `NR`, `NF`, selected
fields, native field lengths such as `length($2)`, native byte substrings such as
`substr($2, 1, 3)`, and `OFS`, matching the first awk-style example above. It can
also compile a scalar counter and an `END` action:

```awk
$1 == "ERROR" { count++ } END { print count }
```

It can also compile multiple scalar increments in one action list:

```awk
$1 == "ERROR" { errors++; matches++ } END { print errors, matches }
```

Separate guarded rules can update shared scalar state too:

```awk
$1 == "ERROR" { errors++ }
$1 == "WARN"  { warnings++ }
END { print errors, warnings }
```

For the sample input above, that prints:

```text
2 1
```

The single-counter forms count two matching records:

```text
2
```

The first associative-count surface uses familiar AWK syntax:

```awk
{ counts[$1]++ }
END { print counts["ERROR"], counts["WARN"] }
```

PLAWK interns the `$1` slice for each record and updates a native WAM/LLVM
growable `i64` table keyed by that atom id. The `END` action looks up the printed
keys:

```text
2 1
```

Multiple associative arrays in one always-rule get separate runtime tables:

```awk
{ counts[$1]++; by_component[$2]++ }
END { print counts["ERROR"], by_component["disk"], by_component["cpu"] }
```

For the sample input above, that prints:

```text
2 1 1
```

Associative counts can also be guarded:

```awk
$1 == "ERROR" { by_component[$2]++ }
$1 == "WARN"  { warnings[$2]++ }
END { print by_component["disk"], warnings["cpu"] }
```

The generated native loop checks each rule's guard before entering its table
increment block. For the sample input, that prints:

```text
1 1
```

Scalar counters and associative counts can be used together:

```awk
{ total++; counts[$1]++ }
$1 == "ERROR" { errors++; by_component[$2]++ }
END { print total, errors, counts["WARN"], by_component["disk"] }
```

The generated native loop carries scalar counters as `i64` phi slots and keeps
associative arrays in runtime tables. For the sample input, that prints:

```text
4 2 1 1
```

`END` print fields can include literal labels:

```awk
{ total++; counts[$1]++ }
END { print "total", total, "errors", counts["ERROR"] }
```

That prints:

```text
total 4 errors 2
```

Scalar variables can accumulate native integer deltas and field lengths:

```awk
$1 == "ERROR" { bytes += length($0); hits += 2 }
END { print bytes, hits }
```

They can also be assigned in the native loop. Assignment preserves source order
with later updates:

```awk
$1 == "ERROR" { last_len = length($0); hits++ }
END { print hits, last_len }
```

The current assignment expression subset is integer literals and `length($N)`.

The first `if/else` surface lowers scalar updates behind field-equality guards:

```awk
{
  total++
  if ($1 == "ERROR") { errors++; last_len = length($0) }
  else { non_errors++ }
}
END { print total, errors, non_errors, last_len }
```

For now, branch bodies support scalar updates, field-key associative increments,
selected-field `print` including `NR`, terminal `next`/`break`, and
combinations of those actions. The generated native code evaluates the `if`
condition once, runs the selected branch, rejoins normal scalar slots through
LLVM phi nodes, routes selected branch-local `next` paths to the next input
record, and routes selected branch-local `break` paths to the stream close path
before `END`.

Associative increments can live inside the selected branch:

```awk
{
  if ($1 == "ERROR") { by_component[$2]++ }
  else { by_kind[$1]++ }
}
END { print by_component["disk"], by_kind["WARN"] }
```

Branches can also print selected fields:

```awk
{
  if ($1 == "ERROR") { print NR, $2, $3 }
  else { counts[$1]++ }
}
END { print counts["WARN"] }
```

Branches can use terminal `next` to skip later actions and later rules for the
current record:

```awk
{
  if ($1 == "DEBUG") { skipped++; next }
  else { seen++ }
  total++
}
END { print total, seen, skipped }
```

Branches can also use terminal `break` to stop the input stream before `END`:

```awk
{
  if ($1 == "ERROR") { hits++; break }
  else { total++ }
}
END { print hits, total }
```

A terminal `next` skips the remaining rules for the current record:

```awk
$1 == "DEBUG" { skipped++; next }
{ total++ }
END { print total, skipped }
```

The same terminal `next` behavior works for associative arrays and mixed scalar/array rules.
Terminal `break` stops the input stream before running `END`:

```awk
$1 == "ERROR" { hits++; break }
{ total++ }
END { print hits, total }
```

For now, `next` and `break` must be the last action in the rule body. Inside a
branch, `next` or `break` must be the last action in that branch.

`BEGIN` can emit literal report headers before the first input record is read:

```awk
BEGIN { print "kind", "count" }
{ total++ }
END { print "total", total }
```

For the sample input, that prints:

```text
kind count
total 4
```

By default, `FS=" "` uses AWK-style whitespace splitting: leading and trailing
whitespace is ignored, and whitespace runs do not create empty fields.

The first `substr` surface uses AWK-style 1-based starts and byte counts.

`index` returns an AWK-style 1-based byte position, or `0` if the literal is not
present:

```awk
$1 == "ERROR" { print index($2, "sk"), index($0, "disk") }
```

`tolower` and `toupper` can print ASCII case-mapped field bytes without
allocating transformed field strings:

```awk
$1 == "ERROR" { print tolower($2), toupper($0) }
```

`BEGIN` can also set an explicit single-byte field separator for the native field helpers:

```awk
BEGIN { FS = ":" }
$1 == "ERROR" { counts[$2]++ }
END { print "disk", counts["disk"], "net", counts["net"] }
```

For this input:

```text
ERROR:disk:full
WARN:cpu:hot
ERROR:net:down
ERROR:disk:again
```

that prints:

```text
disk 2 net 1
```

`BEGIN` can set the output field separator too:

```awk
BEGIN { FS = ":"; OFS = "," }
$1 == "ERROR" { print $2, $3 }
```

For the same colon-separated input, that prints:

```text
disk,full
net,down
```

The native output path writes the single-byte separator directly, so separators
such as `%` are treated as data rather than `printf` format strings.
This path keeps the streaming loop native while the WAM runtime supplies the
reader, atom helpers, and reusable associative table primitive.

## Another example: count and print matching lines

The demo in `examples/plawk/demo/count_errors.pl` is equivalent to:

```awk
{ records++ }
$1 == "ERROR" { print $0 }
END { print "records=" records }
```

Run it:

```bash
swipl -q -s examples/plawk/demo/count_errors.pl -t halt
```

Output:

```text
records=4
ERROR disk
ERROR network
```
