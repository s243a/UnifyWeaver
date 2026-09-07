<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# GP9SCHEMA Integration Patch

Adds **JSON `schema` mode** to the TypeScript/Node data-source consumer (G-P9).
When a JSON source predicate is declared with `schema([field(Name, Path, Type),
...])`, the emitted self-contained Node script (fs + `JSON.parse`, no npm deps)
parses each JSON record into a **typed object** with exactly the schema's field
keys, in declared order, coercing each field to its declared type, and emits each
record as a `JSON.stringify` line — instead of the flat `columns` `:`-joined
projection. This mirrors what `csharp_target.pl` does with `schema_records` /
`schema_fields`, but ported to Node.

This work is **self-contained in its worktree** — it edits no shared hot files
and requires no central INT-0 application. It branched from `main`.

## Files changed

- `src/unifyweaver/sources/json_source.pl`
  - New helpers `json_schema_projection_js/2` and `json_schema_field_js/1` (build
    the JS field-descriptor literal from the normalized schema field dicts).
  - Two new hardcoded templates: `json_file_source_typescript_schema` and
    `json_stdin_source_typescript_schema`.
  - `generate_json_bash/12` now also computes `SchemaProjection` and binds it as
    the `schema_projection` render variable. **The only non-additive edit is
    appending `schema_projection=SchemaProjection` to that one render-vars list
    line;** every existing template body (bash, pure-PowerShell, and the flat
    `_typescript` variants) is byte-identical.
- `src/unifyweaver/targets/typescript_source_compiler.pl`
  - `ts_source_dispatch(json, ...)` now checks `json_schema_fields/2`; when the
    predicate's `dynamic_source_metadata` carries a non-empty `schema_fields`, it
    routes to the schema template (suffix `_typescript_schema`) and threads the
    field descriptors through as a `schema_fields(Fields)` option. Otherwise it
    falls through unchanged to the existing flat-`columns` projection path.
- `tests/core/test_typescript_json_schema.pl` — **new** plunit suite (module
  `test_typescript_json_schema`, entry `test_typescript_json_schema/0`, running
  `run_tests([typescript_json_schema])`). Isolated from
  `tests/core/test_typescript_source.pl` (which has changes on another branch),
  which was **not** edited.

No edits to `sources.pl`, `dynamic_source_compiler.pl`, `csharp_target.pl`,
`csv_source.pl`, `core/target_registry.pl`, `docs/BINDING_MATRIX.md`,
`test_advanced.pl`, `glue/*`, or any `*wam*` template/target.

## Metadata dict keys consumed (read-only)

Schema metadata is populated **automatically** for a plain
`source(json, Name, [schema(...)])` declaration — no extra wiring was needed.
`sources.pl:augment_json_options` sets `return_object(true)` and
`determine_arity/2` forces arity 1 when `schema(_)` is present;
`dynamic_source_compiler:extract_io_metadata/3` (via `register_dynamic_source/3`)
then stores the normalized dict in `dynamic_source_metadata(Pred/1, Meta)`.

The Node path reads exactly these keys (the same ones `csharp_target` reads):

- `Meta.schema_fields` — the flat root field list (a list of `schema_field{...}`
  dicts). The wrapper's `json_schema_fields/2` reads this; empty ⇒ not schema
  mode.
- Per field dict, the keys read are:
  - `name` (atom) — becomes the emitted object key.
  - `column_type` (atom) — the declared type, lower-cased
    (`string`/`integer`/`long`/`float`/`double`/`number`/`boolean`/`json`).
    `record` (nested) fields carry `column_type: json`, so they pass through.
  - `path` (string) — the source key path; a leading `$.` / `$` (jsonpath) is
    stripped via the existing `json_column_key/2` so a plain object lookup works.

`Meta.schema_records` and `Meta.schema_type` are **not** consumed by the Node
path — the flat `schema_fields` list already gives the root record's fields in
declared order, which is all the per-record object needs. (C# uses
`schema_records` to emit nested record *classes*; Node has no type declarations to
emit, so nested records simply pass through as JSON sub-objects.)

## Per-record output shape (and why)

The existing flat `_typescript` templates print each record as its projected
values joined with `:` (arity ≥ 2) or the single value (arity 1). For schema mode
each record is instead emitted as **`JSON.stringify` of an object** with the
schema's field keys in declared order:

```
{"id":"P001","name":"Laptop","price":999,"active":true}
{"id":"P002","name":"Mouse","price":25,"active":false}
```

Rationale: a schema declares *typed, named* fields, so the caller wants the full
typed record — not a lossy `:`-joined string that discards field names and types.
One JSON object per line is the natural Node analogue of `csharp_target`'s typed
record, is trivially re-parsable, and preserves the coerced JS types (numbers stay
numbers, booleans stay booleans). A schema JSON source is always arity 1
(enforced by `sources.pl`), so there is no `:`-join to reconcile.

Type coercion emitted (in the `_coerce` switch):

| declared type              | JS coercion                                    |
|----------------------------|------------------------------------------------|
| `string`                   | `String(v)`                                    |
| `integer`, `long`          | `Math.trunc(Number(v))` (NaN ⇒ `null`)         |
| `float`, `double`, `number`| `Number(v)` (NaN ⇒ `null`)                     |
| `boolean`                  | native bool as-is; string `"true"`/`"false"`; else `Boolean(v)` |
| `json` / unknown (default) | pass through unchanged                          |
| `null` / `undefined` value | `null`                                         |

## Additive / byte-identical guarantee

- **Bash and pure-PowerShell** JSON templates are untouched — they never
  reference `{{schema_projection}}`, and the `git diff` of `json_source.pl`
  removes exactly one line (the render-vars list, to append the new binding) and
  otherwise only adds new predicates and two new template clauses.
- **Non-schema TypeScript** (flat `columns`) is unchanged: when
  `schema_fields` is absent/empty, `ts_source_dispatch` falls through to the
  original `ts_source_options` + `_typescript` path verbatim. The control tests
  `control_flat_columns_unchanged_code` /
  `control_flat_columns_unchanged_execution` prove the old flat
  `["id","name","price"]` projection + `:`-join output is byte-for-byte preserved.

## Suite results (verbatim)

1. `swipl -q -g test_typescript_json_schema -t halt tests/core/test_typescript_json_schema.pl`
   → **exit 0**, 6/6 tests pass (2 node-execution tests run under node v22.22.2;
   `condition(node_available)`-gated so they skip cleanly when node is absent).
2. `swipl -q -t halt tests/core/test_typescript_source.pl` → **exit 0**, green
   (file untouched).
3. `swipl -q -t halt tests/core/test_json_source_validation.pl` → **exit 0**,
   13/13 pass.
4. `swipl -q -g test_csv_source -t halt tests/core/test_csv_source.pl` → **exit
   0**, "All CSV source tests passed".
