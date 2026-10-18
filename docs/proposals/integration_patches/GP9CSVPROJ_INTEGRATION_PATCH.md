<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# CSV subset/reorder projection Integration Patch (D33)

**Task:** G-P9 residual — the CSV subset/reorder projection that D31
(GP9POLISH) explicitly deferred as "blocked-upstream". Unblocked here with an
**additive** option that keeps bash/PowerShell output byte-identical.

**Why D31 called it blocked:** the existing `columns([...])` option only *names*
columns (its length must equal the arity) and the awk output is always positional
`$1..$Arity` — bash/PS do no projection. Making `columns` also do subset/reorder
would have required changing `sources.pl:determine_arity/2` and
`csv_source:compile_source/4` in ways that alter bash/PS output for the
(currently degenerate) header-subset combination. D31 recommended a **distinct
new option** instead; this is that option.

## The option

```prolog
:- source(csv, person, [ csv_file('people.csv'),
                         has_header(true),
                         project_columns([city, name]),  % subset + reorder
                         arity(2) ]).
```

- `project_columns([Name, ...])` is **distinct from** `columns([...])`. It selects
  and **reorders** exactly the named columns by matching the file's detected
  **header** row (so the file must have a header; combine with `has_header(true)`
  so the header line is skipped at runtime).
- Column names are resolved to **0-based indices at compile time**
  (`resolve_projection/5`, via the existing `detect_csv_headers/3`), so the
  emitted Node needs no header logic — it just picks fields by index.
- `arity(N)` is the projected column count (explicit `arity` wins in
  `determine_arity/2`, so no upstream change is needed).
- **TypeScript/Node only.** bash and pure-PowerShell templates never reference the
  projection vars, so their output is unchanged (documented, matching the
  delimiter/JSON-projection precedent from D31).

## Files changed (all inside the allowed set — no central wiring)

| File | Change |
|------|--------|
| `src/unifyweaver/sources/csv_source.pl` | Additive. `validate_config/1` accepts `project_columns`; `compile_source/4`'s `has_header` branch tolerates projected-arity ≠ header-width (uses projected names as the column comment, no warning); `generate_csv_bash/11` calls new `resolve_projection/5` + `projection_ts_vars/3` and appends the TS-only vars to both render dicts; the two `_typescript` templates use `{{min_fields}}`/`{{output_expr}}` (binary) and `{{u_min}}`/`{{u_field}}` (unary). |
| `tests/core/test_typescript_source.pl` | 4 new tests: projected-code shape, node reorder execution (`nyc:alice …`), single-column subset (arity 1), and unknown-name compile failure. |

## Byte-identical guarantee (no projection)

`projection_ts_vars([], _, Vars)` returns pass-through defaults that reproduce the
pre-projection template text exactly:

| var | no-projection default | projection example (`[city,name]` on `name,age,city`) |
|-----|----------------------|--------------------------------------------------------|
| `min_fields` | `arity` | `3` (max index + 1) |
| `output_expr` | `fields.slice(0, arity).join(":")` | `[fields[2], fields[0]].join(":")` |
| `u_min` | `1` | `3` |
| `u_field` | `fields[0]` | `fields[2]` |

Verified: a no-projection binary render still contains
`if (fields.length < arity) { continue; }` and
`out.push(fields.slice(0, arity).join(":"));` verbatim.

## Errors

- unknown projected name → `Error: project_columns name <N> not found in header …`, compile fails.
- `length(project_columns) =\= arity` → `Error: project_columns length (…) does not match arity (…)`, compile fails.
- missing/unreadable header → `Error: project_columns requires a readable header row in …`, compile fails.

## Acceptance (verbatim)

```
# people.csv:  name,age,city / alice,25,nyc / bob,30,sf / charlie,35,la
# project_columns([city, name]), arity(2)  -> node:
nyc:alice sf:bob la:charlie
# project_columns([city]), arity(1)        -> node:
nyc sf la
```

Suites green: `test_typescript_source` (15 tests, incl. 4 new projection),
`test_csv_source`, `test_json_source_validation`, `test_typescript_target`
(core), `test_annotated_js_target`, `test_vanilla_js_target`.

## Not done (still deferred, documented)

- **JSON `schema` mode** — the Option-B query-plan model (typed record objects,
  `schema_fields`/`schema_records` reaching only `csharp_target`); a larger change
  than a template polish, explicitly deferred by G-P9.
