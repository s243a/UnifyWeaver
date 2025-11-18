% JSON Dynamic Source Reader Plan

## Overview

- Goal: allow UnifyWeaver dynamic sources to ingest JSON streams directly in the C# query runtime.
- Approach: add metadata + runtime support so `record_format=json` triggers a JSON reader instead of `DelimitedTextReader`.

## Requirements

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

