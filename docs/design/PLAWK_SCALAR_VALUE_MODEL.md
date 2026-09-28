<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (@s243a)
-->

# plawk scalar value model: storage kind × unset

**Status**: design specification. It describes behaviour that is implemented and
pinned by tests; changes to it are design changes, not bug fixes.

Related: `PLAWK_STRNUM_DUALITY.md` (how the storage kind is chosen, and the
string/number duality of field copies).

## 1. The awk rule

awk is dynamically typed. Every value can be read as a number or as a string, and
a variable that has never been assigned is a third state, *uninitialised*, which
reads as:

| context | value |
|---|---|
| numeric (`n + 1`, `n == 0`, `printf "%d"`) | `0` |
| string (`print n`, `n "x"`, `n == ""`, `printf "%s"`) | `""` |

`print` renders its operands as strings, so an unassigned variable prints as the
empty string:

```
$ printf 'a 5\nb 7\n' | gawk '{ print n, $1; n += $2 }'
 a                  <- n unset: ""
5 b
```

gawk 5.1.0 (`LC_ALL=C`) is the oracle for everything below.

## 2. plawk's model: a slot is `kind | unset`

plawk types every scalar **statically**, into one storage kind (chosen by
`plawk_scalar_typed_slots/3`; precedence double > strnum > string > counter):

| slot kind | storage | typical source |
|---|---|---|
| `scalar_counter(N)` | `i64` | `n++`, `n = 3` |
| `scalar_double(N)` | `double` | `t += $2`, `x = 1.5` |
| `scalar_string(N)` | interned atom id (`i64`) | `s = "a" "b"`, `sub/gsub` target |
| `scalar_strnum(N)` | interned atom id of the field bytes | `x = $1` |

The storage kind is **not** the whole type. Semantically a slot holds

    kind | unset

and `unset` is represented **outside** the storage value:

- **numeric slots** (counter, double): the register starts at the type zero
  (`0`, `0.0`) and a separate **assigned mark** records whether the slot has ever
  been written: one bit per slot in `@plawk_slot_assigned`. The mark is
  *monotonic* (false → true, never back), so it needs no SSA threading through the
  record loop's phis: a plain store wherever a write is emitted is correct, inside
  branches and loops included.
- **string / strnum slots**: the interned atom id `0` is the unset sentinel; it
  renders as `""`. No mark is needed.

The type zero is therefore *not* a second value of the type. It is how the `unset`
case **reads in a numeric context**, which is exactly awk's rule. The empty string
is not a double; it is how the `unset` case **renders in a print**.

## 3. How each context reads the `unset` case

| context | numeric slot (counter / double) | string / strnum slot |
|---|---|---|
| arithmetic, numeric comparison | the type zero (register value) | — |
| `print` (END, rule body, BEGIN prelude, END loop body) | `""` when the mark is clear; else the number | `""` for id 0 |
| `printf "%d"` / `"%g"` argument | the number (unset = `0`) | — |
| string contexts of a numeric slot: `n "x"`, `n == ""`, `printf "%s", n` | **declined** (exit 3): no string reading of a numeric slot is lowered | as a string |

Declining is deliberate: a context whose awk value would be `""` is never allowed
to silently produce `"0"`.

## 4. Where the print rule is enforced

- **END prints**: `plawk_numeric_slot_print/5` renders a tracked numeric slot from
  its mark (the original END-context fix: `{ n++ } END { print n }` on empty input
  prints an empty line).
- **Rule-body, BEGIN-prelude and END-loop prints** (the shared rule walker): a
  print of a *tracked* slot that is not **definitely assigned** at that point is
  rewritten by `plawk_unset_counter_read_field/4` to
  - `ssa_or_unset(Value, Slot)` for a counter, which lowers to the `i64_or_empty`
    print type (the same one an absent array element uses), and
  - `ssa_f64_or_unset(Value, Slot)` for a double, which lowers to the `f64_or_empty`
    print type: the mark selects `@wam_print_awk_number`'s empty format, which
    prints nothing.
- **Definitely assigned** (`plawk_walker_assigned_get/1`) is tracked as the walker
  goes: a straight-line write adds the name, and a branch or loop body may not run,
  so the set is restored after it. A value that is the direct result of an
  assignment op is also known to be assigned. So `{ n++; print n }` and
  `for (i = 0; ...) print i` emit no runtime check; the check appears only where
  the variable can really be unassigned (a read before any write, a write only
  under an `if`, END over possibly empty input).

## 5. The trackability rule (the safety invariant)

A slot renders `""`-when-unset **only if every reachable write to it stores the
mark**. `plawk_unset_tracked_slots/3` enforces this: a name is tracked only when
each of its assignments is update-shaped (it goes through the one marking emitter,
`plawk_scalar_update_operation_ir/9`), or is one of the writers that store the mark
themselves (`plawk_unset_marking_action/2`: the `n = split(...)` and `n = gsub(...)`
counts).

A name also written by a writer that does not mark (a getline capture status, a
dynrec bind) is **not tracked** and keeps printing the type zero. So the failure
mode of a missed writer is a *known divergence* (`0` where awk prints `""`), never
a wrongly empty value for an assigned variable.

## 6. Array elements follow the same rule

An absent element is the array analogue of `unset`, with existence playing the
part of the mark: `print c[k]` prints `""` for an absent key (`i64_or_empty` on
`@wam_assoc_i64_exists`), and a numeric read of it is `0`. A positional (split)
table's absent position prints `""` through `@wam_assoc_str_print`.

## 7. Tests that pin this specification

- `tests/test_plawk_unset_scalar.pl`: END rendering, the trackability rule,
  rule-body counter and double prints, definitely-assigned elision.
- `tests/test_plawk_begin_compute.pl`: BEGIN-prelude values and unset reads.
- `tests/test_plawk_subgsub_match.pl`: an unset `n = gsub(...)` count.
- `tests/test_plawk_mixed_rules.pl`: absent array elements print `""`.

## 8. Open edges

- The string readings of a numeric slot (`n "x"`, `printf "%s", n`, `n == ""`) are
  declined, not lowered. Lowering them would need a number-to-string conversion
  that is `""` for unset and CONVFMT-formatted otherwise.
- Writers that do not store the mark (getline capture status, dynrec binds) leave
  their names untracked (see §5).
