# plawk: spec-level / plan-level drift audit

Three consecutive plawk PRs each fixed a variant of the same defect: a property
of a table (is it string-valued? is it positionally keyed?) was computed
**twice**, over two different term representations of the same program, and the
two answers drifted apart. This is a short audit of every remaining place that
pattern could occur.

## Why the pattern exists

The codegen pipeline rewrites a program three times:

| level | representation | example term |
|---|---|---|
| surface | raw parser actions | `split_into(field(0), var(parts), string(","))` |
| spec | per-rule action specs | `assoc_split(parts, 0, ",")` |
| plan | planned actions (table indices resolved) | `assoc_split_action(0, parts, 0, 0, ",")` |

A property asked at more than one level needs one clause per level, with
*different functors each time*. Adding a new producer (a new way to fill a
table) means remembering every level it can be seen at. Miss one and the levels
disagree — and the failure is silent, because each level's answer is
individually plausible.

## Inventory and findings

| property | computations | verdict |
|---|---|---|
| str-valued table | `plawk_assoc_specs_str_arrays` (spec) · `plawk_assoc_plan_str_array` (plan) · `str_arrays/1` reader carrier | in sync (row-capture writers added in #3991; element-read value kinds in #3997) |
| positional table | `plawk_program_posarray_arrays` (surface) · `plawk_assoc_specs_posarray_arrays` (spec) · `plawk_assoc_plan_posarray_array` (plan) | **three defects found — fixed, see below** |
| table name / index | `plawk_action_table_name` (surface) · `plawk_assoc_spec_table_name` (spec) | asymmetric but not reachable: the surface mapper's key clauses (`split_into`, `add_assoc`, `set_row`, `print`) plus the reader-pass clauses (`pass_over` / `pass_records` / `pass_rows` / `cache_schema`) cover every table a pass can reference. Probed a rule-body for-in and an `in` guard inside a pass — both compile and match gawk. Any residual gap fails `plawk_assoc_table_index` and declines cleanly (exit 3) rather than miscompiling. |
| table set | `plawk_assoc_plan_specs_tables` · `plawk_forin_assoc_plan` | two computations but they **share** their producers (`plawk_assoc_spec_table_name` + `plawk_assoc_spec_forin_array` + `plawk_membership_array`), so no independent list to drift |

## The positional-array defects

A table filled by `split` (or an `as array` bind) is keyed by integer positions
`1..n`, never by interned atom ids. Three views of that fact disagreed:

1. **Miscompile.** The multi-pass `over TABLE` reader built its plan as
   `assoc_plan(Tables, [str_arrays(StrArrays)])` — it carried the program's
   str-valued tables (its writer lives in a *different* pass function, so it
   cannot infer them) but not its positional ones. The loop-key emitter asks
   the plan, got "not positional", and resolved position `1` as atom id `1`:

   ```
   pass { split($0, parts, ",") }
   pass over parts as k { print k, parts[k] }
   ```
   printed raw bytes for the keys where gawk prints `1 a` / `2 b` / `3 c`.

2. **Surface gap.** `plawk_program_posarray_arrays` — the raw-action walk behind
   the text-mode integer-key gate — recognised only `as array` binds, so
   `{ split($0,p,",") } END { print p[1] }` was declined (exit 3) even though
   `p`'s keys are positions. Conservative, but a real hole: the spec and plan
   levels both already counted split tables as positional.

3. **Nondeterminism, exposed by fixing (1).** The two loop-key emitters —
   binary-mode keys and positional keys — were not mutually exclusive, and the
   atom-resolving fallback had no negative guard. Once a plan could be *both*
   positional and reached with a descriptor, the emitter offered two identical
   solutions, and the multi-pass driver's one-function-per-pass check
   (`length(FnPairs, NPasses)`) rejected the whole program.

Fixing (2) also surfaced a fourth: the pure-assoc END integer-key printer
resolved the stored id with `@wam_atom_to_string` and no presence probe, so an
absent position printed `(null)` where gawk prints empty. The `lookup_int`
printer already probed; this emitter did not — the same two-emitters-for-one-read
shape.

## The fix

Producers live in **one** table, with a row per level:

```prolog
plawk_posarray_producer(surface, dynposarray_bind(var(A), _Call), A).
plawk_posarray_producer(surface, dynposarray_bind_str(var(A), _Call), A).
plawk_posarray_producer(surface, split_into(_Src, var(A), _Sep), A).
plawk_posarray_producer(spec, dynassoc(A, posarray(_Call)), A).
plawk_posarray_producer(spec, dynassoc(A, posarray_str(_Call)), A).
plawk_posarray_producer(spec, assoc_split(A, _Ki, _Sep), A).
plawk_posarray_producer(plan, assoc_dyn_action(_I, A, _Ti, posarray(_Call)), A).
plawk_posarray_producer(plan, assoc_dyn_action(_I, A, _Ti, posarray_str(_Call)), A).
plawk_posarray_producer(plan, assoc_split_action(_I, A, _Ti, _Ki, _Sep), A).
```

A new positional producer now means adding its rows here, and a missing row is
visible at a glance instead of showing up as one level's silently wrong answer.
Alongside it:

* reader plans carry `posarrays(PosArrays)` next to `str_arrays(StrArrays)`,
  threaded to the pass emitters as `array_kinds(StrArrays, PosArrays)`;
* `plawk_forin_key_numeric/3` decides "numeric key" once (binary mode *or*
  positional), so the two loop-key emitters are mutually exclusive and the
  emitter is deterministic;
* the END integer-key printer uses the presence-probing print in text mode.

Covered by `tests/test_plawk_posarray_view.pl` (10 tests), including a
determinism test that would have caught (3).

## A sibling shape: N emitters, one output contract

The same defect class shows up without any level distinction at all — several
emitters for **one output contract**, each spelling it out itself. The
whole-record print was this: FOUR emitters print a record without slicing it
into fields (the single-action driver, the prefixed emitter for a print among
statements, the gsub driver, the field-assign driver), and all four formatted it
with the `"%s\n"` global `@.plawk_surface_print_line` — a newline baked into the
format, so no `ORS` could reach it:

```
BEGIN { ORS = "|" } { print }      emitted "a 1\n"   gawk: "a 1|"
BEGIN { ORS = "|" } { print $1 }   emitted "a 1|"    correct
```

Every print that *names* its fields was already right, because those all funnel
through `plawk_print_fields_ir//4`, whose base case calls
`plawk_ors_terminator_ir/4`. The whole-record emitters bypassed that base case —
they have no field list to recurse over, so each terminated the record itself,
and "how a record is terminated" was re-derived four times. It only became
*visible* when the bare `print` desugaring made `print $0` the idiomatic
spelling of that path.

The fix is the same as everywhere else in this document: each of the four now
appends `plawk_ors_terminator_ir/4` — one emitter, four callers — so the
contract cannot be honoured in one spelling and dropped in another. Nothing was
threaded to make that possible: the ORS already rides a pointer global
(`@plawk_ors_ptr`) that every driver emits.

The tell for this shape is different from the level-drift one. There is no
functor to grep for, so look instead for a **fast path that skips the general
walker**: if the general path's base case does the finishing work, a
special-case clause that returns early must do the same finishing work, and
nothing checks that it does. Covered by
`tests/test_plawk_whole_record_ors.pl`, which asserts the four drivers against
gawk and pins the whole-record-vs-field terminator equivalence directly.

## The sharpest instance: two copies differing in ONE guard

The string-scalar comparison (`s == "text"` on a scalar slot) is written twice:

| copy | context | required a text-holding slot? |
|---|---|---|
| `plawk_resolve_scalar_cmp/4` | the bare string-scalar rule PATTERN | **yes** (`scalar_string` / `scalar_strnum`) |
| `plawk_if_cond_ir/8` | the `if` guard | **no** — matched on slot NAME alone |

Same semantics, same six operators, different temporary names — and one of them
silently mis-compared. A numeric counter against a string literal compared a
*count* against an *interned atom id*:

```
{ n++; if (n == "3") print "eq" }    printed nothing;  gawk prints eq
```

awk compares a number against a string **as strings**, so `n == "3"` with `n = 3`
is true. Nothing forced the two copies to agree, and the unguarded one had been
wrong since it was written.

This is the same shape as the whole-record print above — N emitters, one
contract — but it shows the failure mode at its sharpest: the copies were not
*missing* a feature relative to each other, they differed in a **single
precondition**, which is exactly the kind of divergence that survives review.
Both `if` clauses now require `plawk_slot_holds_text/1`, so the shape declines
instead of mis-comparing, in the rule body and in `END` alike.

Note what did NOT fix it: adding a third caller. Routing `END` through the `if`
emitter (so a string guard works there at all) *propagated* the missing guard to
a new context before it was closed. Reusing an emitter inherits its bugs as well
as its behaviour — worth checking the emitter's preconditions when you add a
caller, not only its output.

## Suggested habit

When adding a way to populate a table, grep the producer tables
(`plawk_posarray_producer/3`, `plawk_assoc_specs_str_arrays` /
`plawk_assoc_plan_str_array`) and add the new shape at *every* level it can
appear at. When adding a second emitter for a read that already has one, check
whether the existing one probes for presence first — awk's uninitialised
element is empty in string context, and only one of the two emitters used to
know that.

When adding a fast path that short-circuits a general walker, check what the
walker's **base case** does after the last element and do that too. `ORS`
termination, table frees and separator emission all live there.
