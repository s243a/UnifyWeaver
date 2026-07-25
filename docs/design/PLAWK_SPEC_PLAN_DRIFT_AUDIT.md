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

## Suggested habit

When adding a way to populate a table, grep the producer tables
(`plawk_posarray_producer/3`, `plawk_assoc_specs_str_arrays` /
`plawk_assoc_plan_str_array`) and add the new shape at *every* level it can
appear at. When adding a second emitter for a read that already has one, check
whether the existing one probes for presence first — awk's uninitialised
element is empty in string context, and only one of the two emitters used to
know that.
