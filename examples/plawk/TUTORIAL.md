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
also compile literal contains-pattern rules such as `/disk/ { print $0 }`, where
the current `/.../` subset treats the body as a literal byte substring unless it
uses the existing `^` prefix fast path. It can also print explicit numeric field
coercions with `int($N)`, where missing or non-numeric fields become `0`, and
the first arithmetic composition forms add or subtract a non-negative integer
constant from native `i64` primaries such as `NR`, `NF`, `length($N)`, and
`int($N)`. It can also compile a scalar counter and an `END` action:

```awk
$1 == "ERROR" { count++ } END { print count }
```

It can also compile basic rule-local `printf` actions:

```awk
$1 == "ERROR" { printf "%s=%s\n", $2, $3 }
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

They can also accumulate numeric fields. The field parser is strict, but scalar
arithmetic uses zero when a field is missing or non-numeric:

```awk
$1 == "ERROR" { bytes += $3; last = $3 }
END { print bytes, last }
```

They can also be assigned in the native loop. Assignment preserves source order
with later updates:

```awk
$1 == "ERROR" { last_len = length($0); hits++ }
END { print hits, last_len }
```

The current assignment expression subset is integer literals, `NR`, `NF`,
`length($N)`, `index($N, "literal")`, numeric `$N`, explicit `int($N)`, and
native scalar `i64` primary `+/- K` forms such as `NF + K`, `length($N) - K`,
`int($N) + K`, and `index($N, "literal") + K`.

Numeric field guards are also native:

```awk
$3 >= 100 { big++ }
END { print big }
```

The selected field is parsed as a signed decimal `i64`; missing or non-numeric
fields simply do not match the guard.

The first `if/else` surface lowers scalar updates behind field-equality and
numeric field guards:

```awk
{
  total++
  if ($1 == "ERROR") { errors++; last_len = length($0) }
  else { non_errors++ }
}
END { print total, errors, non_errors, last_len }
```

For now, branch bodies support scalar updates, field-key associative increments,
selected-field and string-literal `print` including `NR`, terminal `next`/`break`, and
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

Branches can also print selected fields and string literals:

```awk
{
  if ($1 == "ERROR") { print "error", NR, $2, $3 }
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

If `next` or `break` appears before the end of a rule body or branch, later
actions in that same action list are unreachable and skipped by native codegen.

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

`int($N)` prints the strict signed-decimal numeric value of a field, or `0` for
missing and non-numeric fields:

```awk
$1 == "ERROR" { print $3, int($3) }
```

The current arithmetic print subset can add or subtract a non-negative integer
constant from native `i64` primaries:

```awk
$1 == "ERROR" { print NR - 1, NF + 1, length($0) - 3, index($2, "sk") + 1 }
$1 == "ERROR" { print int($3) + 1 }
$1 == "ERROR" { print int($3) - 1 }
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
Use `printf` when the rule action should control formatting directly; supported
native specifiers are `%%`, `%s`, `%d`, `%i`, and `%ld`, with field-slice `%s`
lowered as `%.*s`.
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

## Binary records, from the beginning

Everything above processed *text*: lines, split into fields by spaces.
PLAWK can also process *binary* records — and since several binary
concepts come from outside both awk and Prolog, this section builds
them up from zero.

### What a binary record is

A text record says `200 2.5` and makes the machine parse "200" from
digits every single time. A binary record stores the number 200 as the
8 bytes a CPU register already uses, so reading it is one load
instruction — no parsing. `BEGIN { BINFMT = "i64 f64" }` declares:
every record is exactly 16 bytes — an 8-byte integer, then an 8-byte
float:

```text
byte offset:   0        8        16
              +--------+--------+
              |  i64   |  f64   |     $1 = the i64, $2 = the f64
              +--------+--------+
```

After that BEGIN line, the same awk you already write —
`$1 > 100 { sum += float($2) }` — runs over these 16-byte records.
The `$N` variables just mean "the Nth declared slot" instead of "the
Nth space-separated word".

### Strings in binary records

Two flavors:

- `s8` — a **fixed** 8-byte slot. Short values are padded with zero
  bytes ("alpha" + three zeros). Simple, but always 8 bytes.
- `lps16` — a **length-prefixed string** ("lps"): the wire carries an
  8-byte length, then exactly that many bytes (up to the cap, 16).
  `"bee"` costs 8+3 = 11 bytes instead of a padded 16:

```text
              +--------+---+
              |   3    |bee|      an lps16 holding "bee"
              +--------+---+
```

Records containing an `lps` field have different sizes on the wire —
but PLAWK copies each one into a fixed-size buffer in memory, so to
your program an `lps16` field looks exactly like an `s16`. That
"variable on the wire, fixed in memory" trick is the foundation for
everything below.

### Tagged unions: one stream, several kinds of record

The name comes from C, not from awk or Prolog (Prolog's analogue is
different functors: `metric(Id,V)` vs `event(Name,Code)`; Rust calls
them enums; type theory says sum types). The situation it solves is
everyday, though: **a stream where not every record has the same
shape.** A telemetry feed might interleave *metrics* (an id and a
reading) with *events* (a name and a code). Different layouts, one
pipe.

The binary convention: every record starts with a small number — the
**tag** (or *discriminator*) — announcing which layout follows. Each
possible layout is called an **arm** (pattern-matching vocabulary:
the arms of a match). On the wire:

```text
a metric:   +--------+--------+--------+
            | tag=0  |  i64   |  f64   |
            +--------+--------+--------+
an event:   +--------+--------+-----------+--------+
            | tag=1  | len    | name bytes|  i64   |
            +--------+--------+-----------+--------+
```

In PLAWK you declare the arms in BINFMT, separated by `|`:

```awk
BEGIN { BINFMT = "case(i64 f64 | lps16 i64)" }
```

meaning: tag 0 → the record continues `i64 f64`; tag 1 → it continues
`lps16 i64`.

### case blocks: routing rules by record kind

Here is the part that is genuinely new syntax (awk has no switch;
gawk bolted one on later): you group your rules into one block per
arm, and **inside a block, everything is ordinary awk** — the block
only fixes what `$1, $2, …` mean there:

```awk
BEGIN { BINFMT = "case(i64 f64 | lps16 i64)" }
case 0 {                               # metric records: $1=i64, $2=f64
  $1 > 100 { msum += float($2) }
}
case 1 {                               # event records: $1=lps16, $2=i64
  $1 == "boom" { events++ }
}
END { print msum, events }
```

Reading it aloud: "when a record's tag is 0, treat it as a metric and
run these rules on it; when the tag is 1, treat it as an event and run
those." Scalars like `msum` and `events`, plus `NR`, `next`, `break`,
and the END report, are shared across the blocks — there is still only
one stream and one loop.

Why blocks rather than writing the tag into each guard? Because the
*types* of `$1, $2` depend on the arm: the compiler must know which arm
a rule belongs to before it can decide whether `$1 == "boom"` is a
string comparison or nonsense. With a block, that is decided by where
the rule sits on the page.

That said, for a program with only a rule or two per arm, a block per
arm is ceremony. So the guard spelling exists as **shorthand**: a rule
may lead with `TAG == K`, and it means exactly the same as putting the
rest of the rule inside `case K { ... }`:

```awk
BEGIN { BINFMT = "case(i64 f64 | lps16 i64)" }
TAG == 0 && $1 > 100 { msum += float($2) }
TAG == 1 && $1 == "boom" { events++ }
END { print msum, events }
```

This compiles to *identical* code as the block version — it is pure
sugar, not a second mechanism. The one rule: the `TAG == K` test must
come first in the guard (and every rule must have one), because it is
what tells the compiler which arm's types the rest of the rule is
checked against. `TAG == 0 || TAG == 1` is rejected for the same
reason — such a rule would need two different type checks at once.
Pick whichever spelling reads better; don't mix both in one program.

### What the compiled program actually does

Per record: read 8 bytes (the tag) → a native `switch` jumps to that
arm's reader → the arm's fields are copied to fixed positions in one
buffer → the rules run, each first checking the tag matches its block.
No interpreter, no allocation; a malformed record (unknown tag,
truncated bytes) exits with the read-error code, and record kinds you
wrote no block for are still read and skipped, so the stream never
loses its framing.

### Repetition: records that contain a list

Some records carry a variable number of sub-items — an order with up
to 4 line items, a reading with up to 8 samples. The wire convention
mirrors `lps`: a count, then that many elements:

```text
            +--------+--------+--------+--------+--------+--------+
            |  i64   | count=2| elem 1 (i64,f64)| elem 2 (i64,f64)|
            +--------+--------+--------+--------+--------+--------+
```

Declared as `BINFMT = "i64 rep4(i64 f64)"` — "an i64, then up to 4
(i64, f64) elements". To process the elements, `foreach { ... }` runs
its block once per element, and inside the block `$1, $2` mean *the
current element's* fields:

```awk
BEGIN { BINFMT = "i64 rep4(i64 f64)" }
$1 > 0 { foreach { n++ ; wsum += float($2) } }
END { print n, wsum }
```

The compiler emits a genuine native loop: each iteration copies the
current element into a fixed scratch slot and runs the block over it,
carrying your counters around the loop in registers. Constant memory,
and the code size does not depend on the cap — `rep64` compiles to
the same loop as `rep4`.

Elements can carry strings too. `BINFMT = "i64 rep4(lps8 i64)"`
declares elements that each start with a length-prefixed name of up to
8 bytes. On the wire every element can be a different size, so the
reader parses them one at a time (each string lands in a fixed 8-byte
slot, NUL-padded, exactly like a top-level `lps` field); inside
`foreach`, string fields guard and print like any other:

```awk
BEGIN { BINFMT = "i64 rep4(lps8 i64)" }
{ foreach { if ($1 == "hot") { hits++ }; total += $2 } }
END { print hits, total }
```

Repetition also composes with tagged unions: an arm of a `case(...)`
layout may carry its own `repK(...)`, and `foreach` inside that arm's
rules (block or `TAG == K` spelling) iterates that arm's elements.

### Handing payloads to Prolog

Everything so far was compiled to plain native code. But PLAWK lives
inside a Prolog compiler, and some payloads deserve a real parser. A
`blob32` field is a length-prefixed chunk of bytes that PLAWK itself
never interprets - its only use is as an argument to a compiled Prolog
predicate:

```awk
BEGIN { BINFMT = "i64 blob32" }
$1 > 0 { total += payload_sum($2) }
END { print total }
```

Here `payload_sum/2` is ordinary Prolog compiled into the same binary -
typically `atom_codes` on the payload followed by a DCG over the byte
list, with unification and backtracking intact. The division of labor:
the native loop does the framing (find each record, check lengths,
skip what doesn't match), and Prolog does the understanding. The
hand-off costs about 0.2 microseconds and no memory growth - the
payload travels in a single reused buffer.

Numbers cross the bridge in both directions and both widths: `i64`
fields arrive in Prolog as integers, `f64` fields as floats. When the
predicate's answer is itself fractional, wrap the call in `float(...)`:

```awk
BEGIN { BINFMT = "i64 f64" }
{ wsum += float(weight($1, $2)) }
END { print wsum }
```

Without the wrapper a call is an integer expression (fractions would
be truncated); with it, the result stays a double - whether the
predicate bound an integer or a float. A call that fails contributes
`0.0`, mirroring awk's forgiving arithmetic.
