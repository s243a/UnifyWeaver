% JSON Dynamic Source Reader Plan

## Overview

- Goal: allow UnifyWeaver dynamic sources to ingest JSON streams directly in the C# query runtime.
- Approach: add metadata + runtime support so `record_format=json` triggers a JSON reader instead of `DelimitedTextReader`.

## Current Capabilities

- Metadata emitted by `dynamic_source_compiler` carries `record_format=json`, column definitions, and (new) schema descriptors.
- `JsonStreamReader` inside `src/unifyweaver/targets/csharp_query_runtime/QueryRuntime.cs` consumes newline/NUL-delimited streams or JSON arrays via `System.Text.Json`.
- `csharp_query_target` inspects the metadata and emits either column-oriented projections or schema-generated POCO records (see `schema/1` + `record_type/1` in `src/unifyweaver/sources.pl`).
- Validation rules for JSON sources live in `src/unifyweaver/sources.pl`, with coverage in `tests/core/test_json_source_validation.pl`.
- Column and schema selectors now accept `jsonpath/1` expressions (root, wildcards, recursive descent); generators emit `JsonColumnSelectorConfig` so the runtime evaluates JSONPath selectors directly.
- Nested records are available via `field(Name, Path, record(Type, Fields))`; code generation emits every record type, and the runtime instantiates sub-records recursively (see `tests/core/test_csharp_query_target.pl:verify_json_nested_schema_record_plan/0`).
- JSON Lines streams are supported via `record_format(jsonl)`, which disables array parsing and reads one JSON object per line.
- Null-handling policies (`null_policy(fail|skip|default(Value))`) apply to column projections so runtime callers can reject, drop, or replace incomplete rows.

## Requirements (Completed)

- **Metadata**: `record_format=json`, `record_separator(line_feed|nul)`, `columns([field.path,...])`, optional `input(file(Path))`.
- **Reader**: `JsonStreamReader` that:
  - uses `System.Text.Json` for forward-only parsing.
  - supports newline, NUL, or array-wrapped streams.
  - projects specified column paths (dot notation) into `object[]` rows; defaults to synthetic column names.
- **Generator**:
  - when metadata says `record_format=json`, emit `new JsonStreamReader(new JsonSourceConfig { ... })`.
  - carry over skip rows, record separator, expected width, and columns array.
- **Tests**: add JSON fixture (e.g., `test_products.json`) and extend `test_csharp_query_target.pl` with `verify_json_dynamic_source_plan/0` so dotnet exercises the new reader.

## Future Options

- Allow schema hints (`dotnet_type/1`) to deserialize into POCOs before projection.
- Support simple JSONPath-style selectors for deeply nested fields.
- Provide streaming PowerShell example where the generated assembly is loaded and queried interactively.

## Roadmap

1. **Reader-level resilience**
   - String-to-number coercion toggles.
   - Better diagnostics surfaced through `test_json_source_validation.pl` for malformed metadata.
   - Provide per-column null overrides (e.g., allow nulls for selected selectors only).

2. **Documentation & skills**
   - Extend playbooks with full JSON-to-target walkthroughs (PowerShell, LiteDB, etc.).
   - Keep this proposal in sync with implementation details so Claude/Codex agents can reason about capabilities quickly.
