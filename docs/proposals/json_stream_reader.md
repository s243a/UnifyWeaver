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

1. **Nested POCO schemas**
   - Extend `schema/1` to allow references to sub-records, e.g., `field(address, '$.address', record('AddressRecord'))`.
   - `csharp_query_target` should emit multiple record declarations plus factory helpers, wiring parent/child construction logic.
   - Add `record_namespace/1` so playbooks can isolate generated types.
   - Tests + docs demonstrating multi-level objects flowing through dotnet execution.

2. **Reader-level resilience**
   - Null-handling policy (`null_policy(fail|skip|default(Value))`) and string-to-number coercion toggles.
   - Support JSON Lines streams explicitly via `record_format=jsonl` with configurable separators.
   - Better diagnostics surfaced through `test_json_source_validation.pl` for malformed metadata.

3. **Documentation & skills**
   - Update `skills/skill_json_sources.md` once JSONPath/nested schemas land.
   - Add agent guidance for selecting temp locations + schema best practices.
   - Keep this proposal in sync with implementation details so Claude/Codex agents can reason about capabilities quickly.
