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

The native Phase 2 surface can now compile a scalar counter and an `END` action:

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

This path lowers the field comparison and scalar counters to native LLVM code.
Each scalar variable is an indexed `i64` slot in the streaming loop. The WAM
runtime still supplies the streaming reader and atom helpers.

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
