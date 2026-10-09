<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (@s243a)
-->

# plawk array value model: key kinds, value kinds, provenance, and inputs

**Status**: design specification (2026-09-28, revision 3). §1–§3 describe implemented
behaviour; §4–§7 are the rules for making array elements general values, landing in
the PR series of §8. Each PR updates this document. Changes to these rules are design
changes, not bug fixes.

Revision 3 follows a second review: a field assignment inherits the RHS kind; the
static strnum category is distinguished from gawk's runtime `typeof`; a comparison is
numeric only when both operands are; binary fields keep their BINFMT types.

Revision 2 follows an external design review (PR #4329): it separates physical
**encoding** from semantic **kind**, adds a present-but-unset element state, adds the
raw-integer key kind, separates external **provenance** from array **escape**, and
replaces the single "join" with explicit transfer and mixing rules.

Related: `PLAWK_SCALAR_VALUE_MODEL.md` (a scalar slot is `kind | unset`),
`PLAWK_STRNUM_DUALITY.md`, `PLAWK_ASSOC_FORIN.md`.

## 1. awk's rule

An awk array maps **string keys** to values; `a[1]` and `a["1"]` are the same element.
An element's value is a number, a string, a **strnum**, or **unset**. A strnum is a
*numeric-string candidate* (input text, `split` pieces, `ARGV`, `ENVIRON`) whose
content looks numeric; a comparison is numeric only when **both** operands are
numeric (a number, or a numeric-looking strnum), otherwise it is a string
comparison. On input `10`, `{ print ($1 > "9"), ($1 > 9), typeof($1) }` prints
`0 1 strnum`: the string constant `"9"` forces a string comparison. An element that was never
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

plawk's `strnum` kind is **static**: it is the provenance category "numeric-string
candidate", decided at compile time, and the numeric-looking test runs at run time
(`@wam_looks_numeric`, `@wam_strnum_cmp`). So it includes input that gawk's `typeof`
reports as `string` (`abc`) as well as runtime strnum (`10`). A numeric-looking string
constant never acquires it.

## 5. Input kinds and the mixing rules

Every value that enters an array -- every **input** -- has a semantic kind, known in
one of two ways.

### 5.1 Implied kinds (in-text inputs)

| input | semantic kind | status |
|---|---|---|
| `c[k]++`, `c[k]--`, `c[k] += 5` (integer delta) | number (counter) | supported |
| `c[k] += x` | number: counter if `x` is an integer, double otherwise (x's numeric conversion, not its spelling) | `+= $2` / `+= 5` supported, with or without scalar work |
| `c[k] = 5`, `= $2 * 2`, `= 7 / 2` (literals, arithmetic, `length`, `int`) | number: counter for integer arithmetic; double once a field, a float literal or a division takes part (`plawk_set_value_kind/2`) | supported (field keys), with or without scalar work |
| `c[k] = n * 10` (a scalar inside arithmetic) | double (the scalar's numeric reading; an integral double prints as an integer) | supported beside scalar work |
| `c[k] = n` (a BARE scalar copy) | the scalar's kind -- but it may copy an unset value (§3) | declines |
| `c[k] = "text"`, concatenation, `sprintf`, string builtins | string | planned |
| `c[k] = $N` (an unmodified text-input field) | strnum | planned |
| `c[k] = $N` (a binary-input field) | its `BINFMT` type (i64 or f64 number) | planned |
| `c[k] = $0` (unmodified text record) | strnum, encoded as `atom` | row capture supported |
| `split(src, a, sep)` pieces | strnum, positional key -- for ANY source, literals included (`BEGIN { split("10 9", a); print typeof(a[1]), (a[1] > a[2]) }` prints `strnum 1`) | supported |
| `c[k] = row(...)` | row (a plawk constructor: its own contract, string comparisons) | supported |
| `c[k] = x` (a scalar copy) | x's kind | planned |
| `c[k] = d[j]` (an element copy) | d's kind | planned |
| `c[k] = cond ? A : B` | the agreement of A and B (else declines) | planned |
| a copy of a possibly-unset value | declines (§3) | -- |

**A field assignment takes the kind of its right-hand side.** Assigning a literal or
an arithmetic result replaces input provenance: after `$1 = "10"`, `$1` is a string;
after `$1 = 10`, a number (gawk prints `0`, then `1`, for
`a["x"] = $1; print (a["x"] > 9)` on input `10`). Copying an input-derived value
preserves it: on input `x 10`, `{ $1 = $2; print typeof($1), ($1 > 9) }` prints
`strnum 1`. A copy of a field whose kind the compiler cannot establish -- assigned
from different kinds on different paths -- declines.

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
| text-input field, `$0` | strnum (input text) |
| binary-input field | its `BINFMT` type |
| `split` pieces (any source) | strnum |
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

### 6.3 One effect enumeration (implemented)

Inference, escape and `readonly` all depend on finding **every** read, write and import
of an array, so they share one enumeration, `plawk_array_effects/2` (in
`plawk_native_codegen.pl`):

| effect | meaning |
|---|---|
| `read(A)` | an element read, iteration, membership test, reader pass, or a read through a row variable bound to `A` |
| `write(A, How)` | `How`: `inc`, `add(Delta)`, `delete`, `split`, `row` |
| `import(A, Via)` | `Via`: `bind` (Prolog / dynamic bind) or `store` (cache table) -- the array escapes (`plawk_array_escapes/3`) |
| `declare(A, schema)` | a store's declared row schema |
| `row_binding(R, A)` | a `rows of A as R` / `records of A as R` reader: `R["col"]` is a read of `A`, and `R` is not an array |

Effects are matched wherever they occur in the program term, so BEGIN, rule bodies,
END, pass containers, nested blocks and **wrapper nodes** are covered by construction
-- including `n = split($0, a, ":")`, which parses to `split_count(n, split_into(...))`
and whose write the older helpers (`plawk_action_table_name/2`,
`plawk_scalar_nested_action/2`) did not see. A local `split` is a write, not an import,
so it never counts as an escape.

**Completeness is checked, not assumed.** `plawk_array_unknown_uses/2` reports every
occurrence of an array's (or row variable's) `var(Name)` that is not the table
position of a recognised effect. It is empty for every program in the test suites
(2446 parsed programs); a consumer that needs every effect -- the value-kind
inference of PR 2b onward -- must decline on a non-empty result rather than miss an
effect. Running this check on the suites is what surfaced the row-variable case.

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
2. (a) The shared effect enumeration (§6.3) -- landed. (b) Numeric element writes
   (`a[k] = number`) in array-only rules: the §5.2 mixing rules enforced by
   `plawk_array_kinds_ok/1` over the enumeration, declining on unknown uses and on
   imported arrays; `@wam_assoc_f64_set` for double tables -- landed. (c) The same
   writes, and `a[k] += x`, beside scalar work (the mixed walker), scalar operands
   inside the value's arithmetic, `++` on a double table there; a bare scalar copy
   declines (§3) -- landed.
3. Element reads as expressions.
   (a) Conditions -- landed: `NAME[KEY]` is a comparison operand on either side
   (`if (c[$1] > 1)`, `if (n < c["a"])`, with `&&`/`||`). Read numerically with
   `@wam_assoc_i64_get` (absent -> 0) on **counter** tables only: every write `++`,
   `+= int` or a counter set, not imported, no unknown use. The key is interned as the
   writes intern it (field, literal, integer, SUBSEP list, or a scalar's id -- a key
   scalar is not a numeric operand and keeps its kind). Double, strnum, string, split
   and row tables, a string comparison, END conditions and loop conditions decline.
   (b) Arithmetic in rule bodies -- landed: an element read whose parent is binary
   arithmetic, or the right side of `+=` / `-=`, in the mixed walker
   (`n = c[$1] * 10`, `t += c[$1]`, `print c[$1] * 2, n`, literal and scalar keys).
   Same counter-table rule and key interning as (a). A bare copy (`n = c[$1]`)
   declines (section 3). A scalar used as an element KEY is never an unsafe strnum
   read (keys use the slot's id), so it keeps its kind.
   (c) END -- landed: the braceless guarded print `for (k in c) if (c[k] > 1) print k`
   (the same term as the braced form); arithmetic print fields over loop-keyed elements
   (`print k, c[k] * 2`, `c[k] / 4`, `c[k] + d[k]` -- the iterated table by slot, others
   by the loop key); END arithmetic over literal-key elements (`print c["a"] + c["b"]`,
   also after a for-in in a statement list). Counter tables only.
   (d-1) END control flow in a MIXED program -- landed: a top-level `if` / `while` /
   `do-while` in END of a program with arrays and scalars lowers through the scalar
   driver's END lowering (the shared sequence emitter, final slot values in) given the
   program's table plan, so element reads, literal-key element prints and the retained
   last record all work there. A field-KEYED element read at END declines (no current
   record).
   (d-2) `for (k in c)` carrying scalar state -- landed: in END, a table loop through
   the shared sequence emitter, every scalar slot carried as a head phi beside a cursor
   phi; the loop key is bound to a STRING slot per iteration, and the iterated element
   A[k] (present by construction) is read once by slot and substituted outside prints,
   so even a bare copy `max = A[k]` is sound. END assignments are typed with the rules
   (the END block joins the scalar planning as a pseudo-rule, the for-in flattened to
   `k = ""` + body; string copies propagate). The max / min idioms, counting and
   folding loops, and several former END chain boundaries (a key in a concat,
   `length` / `NF` of the last record, a printf after the loop) now compile. The
   for-in half is the last /3 driver clause, a fallback behind the dedicated for-in
   drivers. Counter tables only; break / continue / next / exit in the body decline.
   (d-3) Still open: for-in printf bodies on the dedicated drivers, guards combining
   `&&` / key comparisons, double-table reads (f64 arithmetic, `fcmp`), break /
   continue in an END for-in body, nested for-in.
4. String and strnum element writes (`a[k] = "x"`, `a[k] = $N`, `a[k] = a[k] $2`) with
   their mixing rules, and `a[NR] = $0` with a numeric END loop (tac).
5. Declarations (§7), ingress contracts (§6.1), foreign-key normalisation (§2), with
   reason messages.

## 9. Tests

- `tests/test_plawk_array_set.pl`: numeric element writes, widening, declined kinds
  and mixes.
- `tests/test_plawk_array_reads.pl`: element reads in conditions, key forms, the
  declined kinds, and the loop-context leak regression.
- `tests/test_plawk_array_effects.pl`: the effect enumeration (wrapper nodes, every
  program part, binds as imports, row variables, unknown-use detection).
- `tests/test_plawk_array_kinds.pl`: special-variable and unassigned keys decline;
  integer-literal keys on counted tables.
- `tests/test_plawk_posarray_keyspace.pl`, `tests/test_plawk_literal_assoc_key.pl`:
  the positional/counted key-space rule.
- `tests/test_plawk_binary_assoc.pl`: raw-integer keys.
- `tests/test_plawk_forin_val_strnum.pl`: split elements compare as strnum.
- `tests/test_plawk_absent_key_read.pl`, `tests/test_plawk_mixed_rules.pl`: absent
  elements.
