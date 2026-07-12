<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (@s243a)
-->

# plawk assoc for-in with per-entry logic

Design for iterating an associative array (a hash table built with
`arr[key] = ...` / `arr[key]++`) and doing **real work per entry** —
not only the print-shaped body that ships today.

## What works today

`for (k in arr)` iterates the table's keys. The body is restricted to a
single `print` whose fields are drawn from a fixed plan:

- `k` — the loop key (resolved to text, or numeric for a positional array);
- `arr[k]` — the iterated table's value at the current key (`value_self`);
- `other[k]` — another table's value at the current key (`value_lookup`);
- a string literal.

```awk
{ arr[$1]++ }
END { for (k in arr) print k, arr[k] }          # key + value, per entry
```

That covers "dump the histogram." It does **not** cover filtering,
aggregating, or decoding a value into a structured record — the loop body
is a print *plan*, not a general action sequence, and `k` / `arr[k]` exist
only as print-field plan atoms, not as readable expression operands.

## The gap, precisely

Three independent things are missing; each stage below closes one.

1. **`k` and `arr[k]` are not expression/pattern operands.** The pattern
   grammar has no production for a bare loop variable or an `arr[k]`
   read, so `if (arr[k] > 100)` and `total += arr[k]` do not parse. They
   are meaningful *only inside* the lexical scope of a `for (k in arr)`
   body, so the natural home is a for-in-scoped operand set, not the
   global pattern/expression grammar (where `k` has no binding).

2. **The for-in body is not an action sequence.** It is planned as
   `[print(Fields)]` and emitted by a bespoke per-key print loop
   (`assoc_forin_action`), separate from the scalar action-sequence
   walker (`plawk_scalar_action_sequence_pairs`) that lowers `if`,
   updates, and record binds/views everywhere else.

3. **The assoc driver has no scalar-slot state.** Assoc programs
   accumulate in *tables*, not scalar variables, so there is no
   loop-carried scalar threading of the kind `foreach` uses (head phis
   per slot). `total += arr[k]` needs a scalar `total` that persists
   across the record loop *and* mutates inside the per-record for-in
   loop.

## Architecture — reuse the `foreach` harness

`foreach` already solved the hard part: a **runtime loop with a scalar
action-sequence body and loop-carried accumulator state**. It threads one
head phi per scalar slot (`plawk_foreach_head_phi_lines`), runs the body
through `plawk_scalar_action_sequence_pairs` with those phis as the
incoming slot values, and exposes the current element's fields as `$k`
reads resolved to a fixed **staging area** in the record buffer.

The plan is to make `for (k in arr) { BODY }` a sibling of `foreach`:

- **Iteration source:** the assoc table's occupied slots
  (`wam_assoc_i64_iter_next` / `_key_at` / `_value_at`) instead of a
  repetition field's elements.
- **Per-iteration staging:** write the current `(key, value)` into a
  two-field staging area at loop-body entry — exactly as `foreach`
  memcpy's the current element into staging. Then rewrite the body's
  `k` → staging field 1 and `arr[k]` → staging field 2 (a resolve pass,
  mirroring `foreach`'s field-shift). The body becomes an ordinary scalar
  action sequence reading two staging fields; **all** existing emitters
  (updates, `if`, record binds/views) apply unchanged.
- **Loop-carried state:** reuse `plawk_foreach_head_phi_lines` for the
  accumulator slots.

Reusing the staging trick means `k`/`arr[k]` never need to become global
expression operands — they are for-in-local staging fields, resolved once
at desugar time, and the scalar machinery does the rest. This is the same
move that let record views nest in `foreach` bodies (they rode the
element staging).

The one genuinely new driver concern is #3 (scalar-slot state in the
assoc driver): the assoc rule chain must carry a scalar `Slots` plan and
thread the final slot values into END, so a `total` mutated inside a
for-in survives to the END print.

## Staged plan

Each stage is a shippable PR with tests.

- **Stage 1 — filter + print (per-iteration output, no loop-carried
  state).** `for (k in arr) { if (arr[k] CMP int) print k, arr[k] }`.
  Adds a for-in-scoped condition (`k`/`arr[k]` compared to an integer)
  and an `if` wrapper around the existing per-key print. No scalar
  accumulation, so no head-phi threading yet — the guard just gates the
  print inside the current bespoke loop. Smallest useful slice: "print
  the histogram entries above a threshold." **LANDED** (rule-body and END
  forms; `tests/test_plawk_forin_filter.pl`).

- **Stage 2 — scalar accumulation.** `for (k in arr) total += arr[k]`.
  The canonical "sum/aggregate the hash" idiom and the largest driver
  piece. Two concrete blockers scoped it out of the filter PR (both
  confirmed empirically):
  1. **The END for-in is a whole-END construct.** The driver matches
     exactly `[end([for_in(...)])]`; `END { for (k in c) ... ; print sum }`
     (a for-in *plus* a later action to read the accumulator) is a parse
     error today. Reading an accumulator therefore needs the END grammar
     + driver to allow a for-in among multiple END actions.
  2. **Loop-carried scalar threading.** The assoc driver *does* carry
     scalar slots (a record-loop `s += 1` alongside `c[$1]++` works and
     survives to END), but the for-in loop does not thread a slot as a
     head phi, so a scalar mutated per entry is not carried out. Stage 2
     brings `foreach`'s head-phi threading to the for-in loop and lets the
     mutated slot flow to the following END action.

- **Stage 3 — decode a value into a struct.** `for (k in arr) {
  (n, m) = dyncall@decode(arr[k]) as (i64 i64) ; ... }`. Once the body is
  a real scalar sequence (Stage 2), record binds/views already lower
  inside it (as they do in `foreach` bodies today); the remaining piece
  is a surface for passing `arr[k]` — the current value — as a grammar
  argument (the staging field 2 read, marshalled like any i64 arg).

## Parser notes

The for-in body already parses through `action_block`, so `if` / updates
/ record binds parse structurally. What does not parse is the **operands**
`k` and `arr[k]` in expression/pattern position. Rather than widen the
global grammar (where a bare `k` is ambiguous), the operands are accepted
in a for-in-body context and validated against the loop's key variable and
the tables in scope; the desugar-to-staging pass then rewrites them before
the generic emitters run, so no emitter needs a special "for-in operand"
case.

## Relationship to `foreach`

`foreach` covers the **list-shaped** collection (a repetition field whose
elements are tuples), and already supports the full action body including
record binds/views (see `tests/test_plawk_foreach_tuples.pl`). This design
brings the same expressive body to the **hash-shaped** collection
(`for (k in arr)`), reusing `foreach`'s loop harness. When both land, the
two collection shapes have parity: iterate, read the item (tuple fields /
key+value), and decode each into a struct.
