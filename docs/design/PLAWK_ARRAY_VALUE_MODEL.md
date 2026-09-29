<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (@s243a)
-->

# plawk array value model: key kind × value kind, and the closed-world rule

**Status**: design specification (2026-09-28). §1–§3 describe implemented behaviour;
§4–§6 are the agreed rules for making array elements general values (an array's kind
is the join of the kinds of all its inputs; external inputs must be declared),
landing in the PR series listed in §7. Each PR updates this document. Changes to these rules are
design changes, not bug fixes.

Related: `PLAWK_SCALAR_VALUE_MODEL.md` (a scalar slot is `kind | unset`),
`PLAWK_STRNUM_DUALITY.md`, `PLAWK_ASSOC_FORIN.md`.

## 1. awk's rule

An awk array maps **string keys** to values of any type; `a[1]` and `a["1"]` are the
same element. An element that was never assigned is *absent*: it reads as `0` in a
numeric context and `""` in a string context, like an unassigned scalar. (In awk a
*read* of `a[k]` also creates the element; see §3.)

## 2. plawk's model: one static kind per array

Every array is typed **statically, for the whole program**, along two axes. There is
one runtime table type (`%WamAssocI64Table`: i64 key, i64 value, occupied bit); the
kinds are encodings over it.

| axis | kind | representation | produced by |
|---|---|---|---|
| **key** | counted (interned text) | key = interned atom id of the key text | `c[$1]++`, `c["x"]`, `c[5]` (the decimal text `"5"`) |
| | positional | key = the raw integer position 1..n | `split(...)`, `as array` binds |
| **value** | counter | i64 | `c[k]++`, `c[k] += 5` |
| | double | IEEE bit pattern in the i64 cell (`@wam_assoc_f64_*`) | `s[k] += $2` (a non-integer delta) |
| | str | interned atom id | `split` pieces, `t[k] = $0`, `row(...)`, `as assoc(str)` |

A program that would use one array as two kinds **declines**. On a counted table an
integer key means its decimal text, so `c[5]` and `c["5"]` are the same element (as
in awk); on a positional table it is the raw position.

These invariants are checked on the generated IR, not assumed: a double table's
lines may only be `@wam_assoc_f64_*` value calls (`plawk_f64_table_ir_ok/3`), and a
split table in the mixed route may only be filled, iterated and read positionally
(`plawk_mixed_split_tables_ir_ok/1`).

Special variables are record values, not user scalars: `a[NR]`, `a[NF]`, `a[FNR]`
parse as `special(_)` keys. They have no lowering yet and decline (they used to
parse as a phantom user scalar and fail in clang).

## 3. Absent elements

An element is `kind | absent`, and **existence (the occupied bit) is the mark**,
never the value: a stored `0`, `0.0` or atom id `0` is a present element.

| context | absent element |
|---|---|
| numeric read | the kind's type zero (`@wam_assoc_i64_get` returns 0) |
| `print` | `""` (`@wam_assoc_i64_print` / `_f64_print` / `_str_print` probe the bit) |
| `k in a` | false, and does not create the element |

**Deliberate deviation:** plawk reads never create an element (no autovivification).
In awk `{ x = c["z"] } END { for (k in c) print k }` lists `z`; plawk does not.

## 4. An array's kind is the join of the kinds of ALL its inputs

Every value that enters an array -- every **input** -- has a kind, and the array's
value kind is the **join of the kinds of all its inputs anywhere in the program**:
program-wide and order-insensitive, never "the kind of the previous assignment". One
array is one storage, so its kind cannot change between program points. The join uses
the scalar precedence **double > strnum > string > counter**, with the same
disqualifier (a string-literal input rules out strnum). An array with both a
string-family and a numeric-family input (`c[k] = "x"` and `c[k] += 1`) declines:
there is no dual storage. A read never influences the kind.

An input's kind is known in one of two ways:

**Implied** -- by the operation or function that produces it, when the input is in
the program text:

| input | implied value kind | implied key kind |
|---|---|---|
| `c[k]++`, `c[k]--`, `c[k] += 5` (integer delta) | counter | counted |
| `c[k] += $2`, `+= x` (field / double delta) | double | counted |
| `c[k] = $2` (a field copy) | strnum | counted |
| `c[k] = "text"`, `= a b` (concat), `= sprintf(...)`, string builtins | string | counted |
| `c[k] = $0`, `= row(...)` | str (row) | counted |
| `c[k] = n + 1` (arithmetic) | counter or double, as for a scalar RHS | counted |
| `split(src, a, sep)` | str | positional |

**Declared** -- when the input comes from outside the program's text (I/O and foreign
code), the compiler cannot see the value, so its kind must be stated:

| external input | declaration |
|---|---|
| a persistent store loaded at open | `declare NAME(col type, ...)`, `use NAME` (schema from the store header) |
| a Prolog / dynamic bind | `as assoc(KIND)`, `as array(KIND)` |
| getline into an array (future) | must carry a kind when it lands |
| an array argument to a function / foreign call (future) | a kind annotation on the parameter |

So "all inputs typed" means: implied inputs type themselves; every external input
carries a declaration. §5 is how the compiler knows which inputs are external.

## 5. Knowing every input is visible: escape analysis

An in-memory plawk array lives exactly one run of the program, and the whole program
is compiled at once. Nothing outside the program can put a value into an array unless
the program hands the array out. So "every input is visible" is the same as "**the
array never escapes**" -- decidable over the program text, with no destructor,
finalizer or freeze keyword (the end of the program is the implicit finalizer).

### 5.1 The escape predicate

`plawk_array_escapes(Program, Array, Via)` holds iff one of:

| via | why it escapes | source forms | enumerator |
|---|---|---|---|
| persistent store | loaded at open (invisible input), committed at END | `declare NAME`, `declare NAME(cols)` | `plawk_program_cache_tables/2` |
| attached store | as above; schema from the store header | `use NAME` | expanded by `bin/plawk` to the same cache terms |
| dynamic / Prolog bind | foreign code writes the table | `as assoc`, `as assoc(str)`, `as array`, `as array(str)` | `plawk_posarray_producer(surface, …)` and the `dynassoc_bind*` rows of `plawk_assoc_body_action_spec` |
| array argument | the callee may write it | not possible today (function and foreign-call parameters are scalars) | add a row when array arguments land |
| getline into an array | external input | not implemented | add a row when it lands |

Not escapes: the `over TABLE` / `records of` / `rows of` readers only READ a table
(`plawk_passes_tables/2`); every in-text writer (`c[k]++`, `+=`, `=`, `delete`,
`split`, row capture) and every element read is visible, enumerated by
`plawk_action_table_name/2` and walked through branches and loops with
`plawk_scalar_nested_action/2` (top-level-only collectors are a documented source of
drift).

### 5.2 The rules

1. **Non-escaping array**: the kind is the join of all its inputs (§4), final by
   construction. A declaration, if present, is an assertion checked against the join
   (e.g. force counter over double for integer data).
2. **Escaping array**: a key-kind and value-kind declaration is **required**; an
   undeclared escaping array declines with a reason naming the route, e.g.
   `c escapes via a persistent store; declare its value kind: declare c as assoc(i64|f64|str)`.
3. **Contradiction**: an input whose kind is outside the declared kind (or outside a
   schema column's type) is a compile error -- never a coercion. The join is seeded
   with the declared kind as a fixed point.
4. **New input surfaces** (array parameters, getline into arrays) add a row to the
   escape predicate; the IR kind check re-verifies the result on the generated code.

Today's implicit defaults -- a bare `declare NAME` and `as assoc` both mean i64 --
remain as documented defaults (existing programs keep compiling); the explicit
spellings of §6 are added beside them.

### 5.3 Read-only arrays

`use NAME readonly` declares that the program only reads a store-backed array: every
in-text writer of it declines (`NAME is read-only`), the END commit is skipped, and
the store's schema is the sole source of the value kind -- the only sound choice when
all inputs are external. A reporting pass then cannot corrupt a store.

## 6. Declaration surface

Today: `declare NAME(col str|i64, …)` / `use NAME` for stores, and `as assoc`,
`as assoc(str)`, `as array`, `as array(str)` for binds; a bare `declare NAME` is
implicitly i64. Planned, consistent with both: the value-kind slot becomes
`i64 | f64 | str` on both surfaces (`as assoc(f64)`, `declare NAME as assoc(f64)`),
`use NAME readonly`, and `BEGIN { declare c as assoc(i64) }` as the optional
assertion in a closed program.

## 7. Plan (element values)

1. Key hygiene; this document. (`a[NR]` no longer reaches clang.) The escape
   predicate (§5.1) lands with PR 2, before the first new input form.
2. Numeric element writes `a[k] = expr`; one kind-inference predicate replacing the
   three separate value-kind collectors; one IR kind check replacing the two above;
   `a[k] += $2` beside scalar work.
3. Element reads as expressions: conditions (`if (c[$1] > 1)`), arithmetic, END loops.
4. String element writes (`a[k] = "x"`, `a[k] = a[k] $2`) and `a[NR] = $0` with a
   numeric END loop (tac).
5. Declarations (§6: `as assoc(f64)`, `declare NAME as assoc(KIND)`,
   `use NAME readonly`) with reason messages; the required-declaration decline for
   escaping arrays and the contradiction error (§5.2) enforced.

## 8. Tests

- `tests/test_plawk_array_kinds.pl`: key kinds, special-variable keys, integer-literal
  keys on counted tables.
- `tests/test_plawk_posarray_keyspace.pl`, `tests/test_plawk_literal_assoc_key.pl`:
  the positional/counted key-space rule.
- `tests/test_plawk_absent_key_read.pl`, `tests/test_plawk_mixed_rules.pl`: absent
  elements.
