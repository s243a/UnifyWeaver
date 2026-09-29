<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (@s243a)
-->

# plawk array value model: key kinds, value kinds, provenance, and inputs

**Status**: design specification (2026-09-28, revision 2). §1–§3 describe implemented
behaviour; §4–§7 are the rules for making array elements general values, landing in
the PR series of §8. Each PR updates this document. Changes to these rules are design
changes, not bug fixes.

Revision 2 follows an external design review (PR #4329): it separates physical
**encoding** from semantic **kind**, adds a present-but-unset element state, adds the
raw-integer key kind, separates external **provenance** from array **escape**, and
replaces the single "join" with explicit transfer and mixing rules.

Related: `PLAWK_SCALAR_VALUE_MODEL.md` (a scalar slot is `kind | unset`),
`PLAWK_STRNUM_DUALITY.md`, `PLAWK_ASSOC_FORIN.md`.

## 1. awk's rule

An awk array maps **string keys** to values; `a[1]` and `a["1"]` are the same element.
An element's value is a number, a string, a **strnum** (input-derived text that
compares numerically when it looks numeric), or **unset**. An element that was never
created is *absent*. Both unset and absent read as `0` numerically and `""` as a
string. gawk 5.1.0 (`LC_ALL=C`) is the oracle for every example below.

## 2. Key kinds

Every array has one key kind for the whole program. There is one runtime table type
(`%WamAssocI64Table`: i64 key, i64 value, occupied bit); key kinds are encodings over
it.

| key kind | representation | produced by |
|---|---|---|
| counted (interned text) | the interned atom id of the key text | `c[$1]++`, `c["x"]`, `c[5]` (the decimal text `"5"`) |
| positional | the raw integer position `1..n` | `split(...)`, `as array` binds |
| raw integer | an arbitrary raw i64 (negatives included) | binary-input programs (`BEGIN { BINFMT = ... }`: `c[$1]++` on an i64 field) |

On a counted table an integer key means its decimal text (`c[5]` is `c["5"]`, as in
awk); on a positional table it is the raw position. A program that uses one array
with two key kinds declines.

**Foreign keys.** The `as assoc` bridge today accepts integer keys verbatim and atom
keys as intern ids in the same table (its implementation documents the collision
risk). A value-kind declaration does not establish key safety: at a foreign boundary,
numeric keys for a counted table must be normalised to their decimal text, or a
separate raw-integer key declaration required and validated (§7).

Special variables are runtime values, not user scalars: `a[NR]`, `a[FILENAME]`,
`a[FS]`, ... parse as `special(_)` keys (the parser's one special-variable list,
`begin_special_name/1`). They have no lowering yet and decline. A user variable that
nothing assigns (`a[unassigned]`) also declines, caught on the generated IR (every
driver scalar global it references must be defined). Both used to fail in clang.

## 3. Element states: absent, present-unset, present-kind

An element is

    absent | present(unset) | present(kind)

and **existence (the occupied bit) marks presence**, never the value: a stored `0`,
`0.0` or atom id `0` is present.

| context | absent | present(unset) |
|---|---|---|
| numeric read | 0 | 0 |
| print / string read | `""` | `""` |
| `k in a` | false | **true** |

`present(unset)` arises from copying an unset value: `a["x"] = n` with `n` never
assigned, or `a["x"] = a["missing"]`. gawk:

```awk
BEGIN { if (0) n++; a["x"] = n; a["zero"] = 0
        print ("x" in a), "[" a["x"] "]", (a["x"] == 0), (a["x"] == "")
        print "[" a["zero"] "]" }
# 1 [] 1 1
# [0]
```

The occupied bit alone cannot tell `present(unset)` from `present(0)`. **Rule:** until
per-element metadata exists, a write whose value may be unset **declines**; it is
never silently materialised as a stored zero.

**Reads never create elements (deliberate dialect deviation).** In awk a read of
`a[k]` creates the element; in plawk it does not. `k in a` is non-inserting in both.
Observable where a program uses lookup as insertion, tests membership after a read,
or counts/iterates a domain populated by reads. On input `a b`:

```awk
{ c[$1] += 0; print c[$2] }
END { for (k in c) print k, c[k] }
# gawk: "", then "a 0" and "b " (order unspecified); plawk: "", then "a 0" only
```

## 4. Value kinds: encoding vs semantic kind

Two things must not be conflated:

- the **encoding**: how the value is stored in the i64 cell -- `i64`, `f64` (bit
  pattern, `@wam_assoc_f64_*`), or `atom` (an interned atom id);
- the **semantic kind**: how the value behaves in comparisons and prints -- `number`
  (counter or double), `string`, `strnum`, or `row`.

`atom` encodes three semantic kinds with different comparisons. For input `10`:

```awk
{ a["input"] = $1; a["literal"] = "10"
  print (a["input"] > 9), (a["literal"] > 9) }
# gawk: 1 0   -- same bytes, strnum vs string
```

An array has **one semantic kind** (no per-element metadata), so its inputs must agree
(§5). The encoding follows from the kind.

## 5. Input kinds and the mixing rules

Every value that enters an array -- every **input** -- has a semantic kind, known in
one of two ways.

### 5.1 Implied kinds (in-text inputs)

| input | semantic kind | status |
|---|---|---|
| `c[k]++`, `c[k]--`, `c[k] += 5` (integer delta) | number (counter) | supported |
| `c[k] += x` | number: counter if `x` is an integer, double otherwise (x's numeric conversion, not its spelling) | `+= $2` supported |
| `c[k] = 5`, `= n + 1` (arithmetic) | number | planned |
| `c[k] = "text"`, concatenation, `sprintf`, string builtins | string | planned |
| `c[k] = $N` (an unmodified field) | strnum | planned |
| `c[k] = $0` (unmodified record) | strnum, encoded as `atom` | row capture supported |
| `split(src, a, sep)` pieces | strnum, positional key | supported |
| `c[k] = row(...)` | row (a plawk constructor: its own contract, string comparisons) | supported |
| `c[k] = x` (a scalar copy) | x's kind | planned |
| `c[k] = d[j]` (an element copy) | d's kind | planned |
| `c[k] = cond ? A : B` | the agreement of A and B (else declines) | planned |
| a copy of a possibly-unset value | declines (§3) | -- |

**Mutated fields are not input-derived.** After `$1 = "10"`, `$1` is a string; after
`$1 = 10`, a number (gawk prints `0`, then `1`, for `a["x"] = $1; print (a["x"] > 9)`
on input `10`). A copy of a field the program assigns declines unless the assigned
kind is statically known.

Future writers must be added to this table when they land: `sub`/`gsub` element
targets, other compound assignments, `match`/`patsplit` output arrays, sort
destinations.

### 5.2 Mixing rules (one kind per array)

| inputs of one array | result |
|---|---|
| counter + double | double (numeric widening, loses nothing) |
| number + string | **declines** |
| number + strnum | **declines** (input `007`: gawk prints `a["raw"]` as `007` and `a["sum"]` as `7`; promoting the array to double loses `007`) |
| string + strnum | **declines** (the `1 0` example above) |
| row + anything else | declines |

A read never influences the kind. With per-element metadata these rules could relax;
without it, declining is the only way to keep gawk's observable behaviour.

## 6. External inputs, escape, and effect enumeration

Two properties must be kept apart:

- **provenance**: whether a value comes from outside the program (I/O, foreign code,
  a store). It can reach an array without the array being handed out:
  `getline x; a[k] = x`, `a[k] = ARGV[1]`, a copy of a foreign scalar result, a copy
  from a store-backed array -- all are local writes.
- **escape**: whether the array itself is handed out, so code outside the program can
  write it: persistent stores, Prolog/dynamic binds, and (in future) array arguments.

### 6.1 Ingress contracts

Every external value carries a contract that supplies its kind, and copies propagate
it through §5.1:

| ingress | contract |
|---|---|
| field, `$0`, `split` of input | strnum (input text) |
| getline text (planned) | strnum |
| `ARGV[i]`, `ENVIRON[k]` | strnum |
| typed foreign scalar result | its declared type |
| store / bind value | its declared value kind (§7) |
| unknown foreign result | rejected at compile time, or checked at its boundary at run time |

### 6.2 Escape

| via | forms |
|---|---|
| persistent store (loaded at open, committed at END) | `declare NAME`, `declare NAME(cols)`, `use NAME` |
| Prolog / dynamic bind (foreign code writes the table) | `as assoc(...)`, `as array(...)` |
| array argument (future) | function and foreign-call parameters are scalars today |

An escaping array's kind **must be declared** (§7); an undeclared escaping array
declines. Readers (`over` / `records of` / `rows of`) only read and are not escapes.

### 6.3 One effect enumeration

Inference, escape and `readonly` all depend on finding **every** read, write and import
of an array. They must share one enumeration of effects -- `read(A)`, `write(A, Kind)`,
`import(A, Contract)` -- that covers BEGIN, rule bodies, END, pass containers, nested
blocks, and **wrapper nodes**, and that rejects any effect-bearing node it does not
know. Today's helpers are not sufficient: `plawk_action_table_name/2` mixes reads and
writes; `plawk_scalar_nested_action/2` does not unwrap `split_count/2`, so the write in
`{ n = split($0, a, ":") }` is invisible to both; and `plawk_posarray_producer(surface,
...)` also lists an ordinary local `split`, so it cannot serve unfiltered as the bind
(escape) enumerator.

### 6.4 Static vs runtime contract failures

A contradiction the compiler can prove -- an in-text write outside an array's declared
kind -- is a **compile error** (a decline with a reason). A contract that depends on
external data -- a foreign result or a store whose content changed since the build --
is checked at run time at its boundary, and fails there. Neither silently coerces.

## 7. Declarations

| interface | today | rule |
|---|---|---|
| persistent declaration | `declare NAME` (implicitly i64), `declare NAME(col str\|i64, ...)` | legacy forms kept as documented aliases, normalised to the typed form |
| schema-derived attachment | `use NAME` | the store's schema is a fixed contract; validated at open if it differs from the build-time schema |
| dynamic bind | `as assoc` (i64 value), `as assoc(str)`, `as array` (positional, i64 value), `as array(str)` | legacy defaults kept as aliases; key kind stated explicitly (§2 foreign keys) |
| local assertion (closed program) | -- | planned: `BEGIN { declare c as assoc(KIND) }` |

Planned: value kinds `i64 | f64 | str | strnum` on every surface (`str` is a pure
string; `strnum` keeps input-derived comparison semantics across a store or bind);
`use NAME readonly`.

**Declarations do not coerce.** An assertion that narrows a kind -- e.g. counter
storage for `c[k] += $2` -- needs proof: an explicit conversion in the program
(`int($2)`) or a validated integer input schema. An unchecked declaration cannot
prove that `$2` is always an integer.

**`use NAME readonly`**: every write path to NAME (including wrapped ones such as
`split_count`, and dynamic fills) declines, the END commit is skipped, and the schema
is the sole source of the value kind. Readonly constrains writes; it does not by itself
type indirect inputs.

The existing IR guards (`plawk_f64_table_ir_ok/3`, `plawk_mixed_split_tables_ir_ok/1`)
stay until a replacement check covers the same guarantees, and enforcement lands with
each newly admitted input -- not deferred to the declaration PR.

## 8. Plan

1. Key hygiene; this document. All special-variable keys and unassigned-variable keys
   decline instead of reaching clang.
2. The shared effect enumeration (§6.3); numeric element writes (`a[k] = number`) with
   the §5.2 mixing rules and the `present(unset)` decline (§3); `a[k] += x` beside
   scalar work.
3. Element reads as expressions: conditions (`if (c[$1] > 1)`), arithmetic, END loops.
4. String and strnum element writes (`a[k] = "x"`, `a[k] = $N`, `a[k] = a[k] $2`) with
   their mixing rules, and `a[NR] = $0` with a numeric END loop (tac).
5. Declarations (§7), ingress contracts (§6.1), foreign-key normalisation (§2), with
   reason messages.

## 9. Tests

- `tests/test_plawk_array_kinds.pl`: special-variable and unassigned keys decline;
  integer-literal keys on counted tables.
- `tests/test_plawk_posarray_keyspace.pl`, `tests/test_plawk_literal_assoc_key.pl`:
  the positional/counted key-space rule.
- `tests/test_plawk_binary_assoc.pl`: raw-integer keys.
- `tests/test_plawk_forin_val_strnum.pl`: split elements compare as strnum.
- `tests/test_plawk_absent_key_read.pl`, `tests/test_plawk_mixed_rules.pl`: absent
  elements.
