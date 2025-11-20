# Streaming JSON into LiteDB via UnifyWeaver

This note is for Claude (and other coding agents) who are building playbooks that load JSON documents into LiteDB using inline .NET inside a PowerShell pipeline. It explains how to leverage the existing JSON dynamic-source reader so UnifyWeaver handles the JSON parsing instead of ad-hoc scripts.

## When to use this flow

- You have JSON files (or streams) that must be normalized before insertion into LiteDB.
- You want UnifyWeaver/Prolog to drive the schema and target generation instead of writing raw PowerShell glue.
- You plan to emit PowerShell (or C#) targets that can call LiteDB APIs, possibly alongside other sources (CSV, XML, etc.).

## Key modules

| Purpose | File |
| --- | --- |
| JSON source validation + schema options | `src/unifyweaver/sources.pl` (`source(json, Pred, Options)`) |
| Metadata compiler for dynamic sources | `src/unifyweaver/core/dynamic_source_compiler.pl` |
| C# runtime that streams JSON | `src/unifyweaver/targets/csharp_query_runtime/QueryRuntime.cs` |
| C# query target generator | `src/unifyweaver/targets/csharp_query_target.pl` |
| Validation tests/examples | `tests/core/test_json_source_validation.pl`, `tests/core/test_csharp_query_target.pl` |

Claude can read those files (or their documentation in `skills/skill_json_sources.md`) to understand the exact predicates and options.

## Prolog API recap

```prolog
:- source(json, product_rows, [
    json_file('test_data/products.json'),
    schema([
        field(id, 'id', string),
        field(name, 'name', string),
        field(price, 'price', double)
    ]),
    record_type('ProductRecord')
]).
```

- `schema/1` produces a typed record, so downstream C# already has strongly typed fields.
- For column-based projections, use `columns/1` instead of `schema/1`.
- Complex selectors are available via `jsonpath('$.orders[*].total')`; strings beginning with `$` are treated as JSONPath automatically.
- Additional options: `record_separator/1`, `field_separator/1`, `return_object(true)` if you want raw `JsonObject` rows.

## Streaming into LiteDB (conceptual pipeline)

1. **Define the JSON source** (as above). Predicate arity must be 1 when using `schema/1`. This lives in any Prolog module you load before compilation.
2. **Write a predicate that transforms rows into LiteDB inserts**. Example skeleton:

   ```prolog
   :- use_module(library(unifyweaver/targets)).

   insert_products_to_litedb :-
       product_rows(Row),
       format('~w|~w|~2f~n', [Row.Id, Row.Name, Row.Price]).
   ```

   The printed format can later be consumed by inline PowerShell or a dedicated target.

3. **Generate a PowerShell target that references LiteDB**:

   ```prolog
   :- csharp_query_target:compile(
          product_rows/1,
          [ target(powershell),
            references(['LiteDB']),
            postprocess_script('scripts/powershell/litedb_insert.ps1')
          ]).
   ```

   The PowerShell script can load the generated assembly plus `LiteDB.dll`, iterate over emitted rows, and call LiteDB’s `GetCollection().Insert(...)`.

4. **Inline PowerShell handler** (simplified excerpt Claude can adapt):

   ```powershell
   # Assume generated DLL outputs ProductRecord rows
   Add-Type -Path "tmp/GeneratedQuery.dll"
   Add-Type -Path "tmp/LiteDB.dll"

   $db = [LiteDB.LiteDatabase]::new("tmp/users.db")
   $col = $db.GetCollection("users")

   foreach ($row in [UnifyWeaver.Generated.ProductQuery]::Run()) {
       $col.Insert([LiteDB.BsonDocument]@{
           _id   = $row.Id
           name  = $row.Name
           price = $row.Price
       })
   }
   $db.Dispose()
   ```

   Replace `Run()` with whatever static entry point the target exposes (check the generated PowerShell/C# output).

## Tips for Claude

- Reuse the JSON reader rather than reinventing parsing logic; this ensures consistent schema enforcement and lets you later switch targets (Bash, C#, etc.) with minimal changes.
- If LiteDB import needs the raw JSON object, use `return_object(true)` with a `type_hint('System.Text.Json.Nodes.JsonObject')` and deserialize inside PowerShell.
- When referencing LiteDB from generated C#, add `references(['LiteDB'])` so the build pulls the NuGet package/dll.
- For large imports, consider generating a C# query target, compiling it once, and then calling it from PowerShell to push documents into LiteDB.

Documenting this behavior here keeps Claude aligned with the Prolog APIs and file locations, so future playbooks can cite this guide instead of duplicating instructions inline.
