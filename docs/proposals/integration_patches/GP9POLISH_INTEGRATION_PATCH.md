<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# G-P9 Polish Integration Patch (for INT-0)

**Task:** G-P9 follow-up — polish the TypeScript/Node data-source consumer:
(1) non-comma CSV delimiters (tab / pipe / semicolon), and (2) `columns` /
`schema` projection. Builds directly on the D26 base consumer (Option A,
self-contained Node, no npm).

**Worktree:** `agent-ad284a7145f977688`

**Shared-file rule:** this agent did **NOT** edit any forbidden/shared file —
no `typescript_target.pl` routing, no `wam_*`, `clojure*`,
`powershell_compiler.pl`, `core/target_registry.pl`, `docs/BINDING_MATRIX.md`,
`tests/test_advanced.pl`, or `glue/js_glue.pl`. It also did not need to touch
`sources.pl`, `core/dynamic_source_compiler.pl`, or `csharp_target.pl`.

## Files changed (all inside the allowed set — no central wiring required)

| File | Change |
|------|--------|
| `src/unifyweaver/sources/csv_source.pl` | **Templates + one generator helper, additive.** Added `js_escape_delimiter/2`; `generate_csv_bash/11` now also passes `js_delimiter` to the render dict; the two `_typescript` templates split on `{{js_delimiter}}` instead of `{{delimiter}}`. Bash and `_powershell_pure` still use `{{delimiter}}` (awk-escaped) — byte-for-byte unchanged. |
| `src/unifyweaver/sources/json_source.pl` | **Templates + generator helpers, additive.** Added `json_projection_js/2`, `json_column_key/2`, `js_string_literal/2`; `generate_json_bash/12` now also passes `projection` to the render dict; the two `_typescript` templates select/order keys from `{{projection}}`. Bash and `_powershell_pure` ignore `{{projection}}` (jq does projection via `jq_filter`) — unchanged. |
| `tests/core/test_typescript_source.pl` | New delimiter tests (tab + pipe, compile-shape and node execution vs expected rows) and JSON `columns()` projection tests (reorder + single-key subset). |

**No central wiring required.** All edits are additive template variables that
the bash/PS templates simply do not reference, plus new generator helpers.
`annotated_js` / `vanilla_js` inherit the behaviour automatically via
`typescript_target:compile_predicate_to_typescript/3`.

## 1. Delimiter escaping (DONE)

The emitted Node scripts use `String.prototype.split(<string>)`, **not** a
`RegExp`, so delimiters need only *string-literal* escaping, never regex
escaping. `js_escape_delimiter/2` maps the raw delimiter:

| delimiter | JS literal emitted | split call |
|-----------|--------------------|------------|
| `,` (default) | `,` | `line.split(",")` |
| tab (`\t`) | `\t` | `line.split("\t")` |
| `\|` | `\|` (no escaping — `\|` is a normal char in a string) | `line.split("\|")` |
| `;` | `;` | `line.split(";")` |
| `\\` | `\\\\` | `line.split("\\\\")` |
| `"` | `\\"` | `line.split("\\"")` |

This is deliberately independent of the awk `escape_delimiter/2` (which escapes
`\|` → `\\\|` for awk's regex `-F`), so the two targets never interfere.

## 2. `columns` projection — JSON (DONE), CSV (blocked-upstream)

### JSON: implemented

For JSON, `sources.pl` already **requires** `columns([...])` (count == arity)
and `csharp_target` already treats it as `JsonStreamReader.ColumnSelectors`.
The polish makes the pure-Node template honour the same list:
`columns([price, name, id])` emits `["price","name","id"]` and the script reads
exactly those keys in that order (dotted paths and `jsonpath($.a.b)` wrappers
are traversed at runtime; the leading `$`/`$.` is stripped). With no columns the
literal is `[]` and it falls back to `Object.values` (previous behaviour).

### CSV projection & JSON `schema`: blocked-upstream (NOT done)

These need shared-file changes that were **out of scope** for this worktree:

1. **CSV subset/reorder projection.** In the `csv_source:compile_source/4`
   (bash / PS / TS) path, `columns` only *names* columns and its length must
   equal the arity; the awk output is always positional `$1..$Arity` (bash/PS do
   **no** projection). `has_header(true)` and `columns([...])` are mutually
   exclusive branches, and `determine_arity/2` in `sources.pl` sets
   `arity = length(columns)`, so a header-name subset (e.g. pick 2 of 3 headers)
   currently hits the "detected N ≠ arity M" warning branch and degrades to
   `col1..colN`. Real CSV projection therefore requires editing
   **`sources.pl`** (`determine_arity/2`) and **`csv_source.pl`**
   (`compile_source/4` to allow `has_header(true)` + `columns([...])` and thread
   header→index mapping) — and doing so would change bash/PS output for that
   (currently degenerate) combination. Recommended follow-up: add a
   `project_columns([...])` option distinct from the arity-defining `columns`,
   resolve it to 0-based indices against the detected header at compile time, and
   thread the index list to all three templates.

2. **JSON `schema` mode.** `schema(...)` drives `return_object(true)`, arity 1,
   and the rich `schema_fields` / `schema_records` metadata, which reaches only
   `csharp_target` via `dynamic_source_metadata/2` (`JsonStreamReader.SchemaFields`
   + generated record classes). It is **not** threaded into
   `generate_json_bash/12`, and emitting typed record objects is the Option-B
   query-plan model that G-P9 explicitly deferred. Supporting it in Node would
   mean either wiring the TS path to `dynamic_source_metadata/2` or extending
   `generate_json_bash/12` to receive `schema_fields` — a larger change than a
   template polish.

## Acceptance (verbatim)

Node output matches `awk` for both non-comma delimiters, and `columns()`
reordering reorders the emitted rows:

```
# tab.js (node)                # awk -F'\t' 'NR>1{print $1":"$2":"$3}'
alice:25:nyc                   alice:25:nyc
bob:30:sf                      bob:30:sf
charlie:35:la                  charlie:35:la
# node tab.js bob  ->  bob:30:sf

# pipe.js (node)               # awk -F'|' 'NR>1{print $1":"$2":"$3}'
alice:25:nyc                   alice:25:nyc
bob:30:sf                      bob:30:sf
charlie:35:la                  charlie:35:la

# JSON before  columns([id,name,price]) -> ["id","name","price"]
P001:Laptop:999 / P002:Mouse:25 / P003:Keyboard:75
# JSON after   columns([price,name,id]) -> ["price","name","id"]
999:Laptop:P001 / 25:Mouse:P002 / 75:Keyboard:P003
```

Suites green: `test_typescript_source` (11 tests, incl. new delimiter +
projection), `test_typescript_target_core`, `test_annotated_js_target`,
`test_vanilla_js_target`, `test_csv_source`, `test_json_source_validation`,
`test_powershell_native_lowering`.
