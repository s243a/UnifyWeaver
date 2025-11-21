# Playbook: Streaming JSON into LiteDB via PowerShell

## Audience
This playbook guides coding agents in using UnifyWeaver to stream JSON data into a LiteDB NoSQL database using inline .NET code within PowerShell scripts.

## Overview
This playbook demonstrates:
1. Reading JSON files using UnifyWeaver's JSON dynamic source
2. Defining typed schemas for strong typing
3. Generating PowerShell scripts with inline .NET code
4. Streaming data into LiteDB NoSQL database
5. Querying the data back from LiteDB

## Prerequisites

- .NET SDK 6.0+ installed
- LiteDB library available (use `scripts/setup/setup_litedb.sh` or `.ps1`)
- PowerShell 7+ (for cross-platform support)

## Workflow

### Step 1: Define JSON Source with Schema

UnifyWeaver's JSON dynamic source automatically handles parsing and type conversion:

```prolog
:- source(json, product/1, [
    json_file('test_data/test_products.json'),
    schema([
        field(id, '$.id', string),
        field(name, '$.name', string),
        field(price, '$.price', double),
        field(category, '$.category', string)
    ]),
    record_type('ProductRecord')
]).
```

**Key Options**:
- `schema/1` - Defines typed fields with JSONPath selectors
- `record_type/1` - Names the generated C# record type
- `json_file/1` - Source JSON file path

### Step 2: Create LiteDB Insertion Predicate

Define a predicate that processes JSON rows and inserts them into LiteDB:

```prolog
:- dynamic_source(
    insert_product/1,
    [source_type(dotnet), target(powershell)],
    [csharp_inline('
using LiteDB;
using System;
using System.Text.Json.Nodes;

namespace UnifyWeaver.Generated.InsertProduct {
    public class InsertProductHandler {
        private static LiteDatabase _db = null;

        public string ProcessInsertProduct(string jsonRecord) {
            // Initialize database connection (singleton pattern)
            if (_db == null) {
                _db = new LiteDatabase("products.db");
            }

            // Parse JSON record
            var obj = JsonNode.Parse(jsonRecord);

            // Get collection
            var products = _db.GetCollection("products");

            // Create BsonDocument
            var doc = new BsonDocument {
                ["_id"] = obj["id"]?.ToString(),
                ["name"] = obj["name"]?.ToString(),
                ["price"] = obj["price"]?.GetValue<double>(),
                ["category"] = obj["category"]?.ToString()
            };

            // Insert
            products.Insert(doc);

            return $"Inserted: {obj["name"]}";
        }

        public static void Cleanup() {
            _db?.Dispose();
        }
    }
}
'),
    pre_compile(true),  // Enable DLL caching for performance
    references(['lib/LiteDB.dll'])
    ]
).
```

### Step 3: Create Query Predicate

Query data back from LiteDB:

```prolog
:- dynamic_source(
    query_products_by_category/2,
    [source_type(dotnet), target(powershell)],
    [csharp_inline('
using LiteDB;
using System;
using System.Linq;

namespace UnifyWeaver.Generated.QueryProductsByCategory {
    public class QueryProductsByCategoryHandler {
        public string ProcessQueryProductsByCategory(string category) {
            using (var db = new LiteDatabase("products.db")) {
                var products = db.GetCollection("products");
                var results = products.Find(x => x["category"] == category);

                return string.Join("\\0",
                    results.Select(doc =>
                        $"{doc["name"]}:{doc["price"]:F2}"));
            }
        }
    }
}
'),
    pre_compile(true),
    dll_references(['lib/LiteDB.dll'])
    ]
).
```

## Executable Example

### Bash Script

See `playbooks/examples_library/json_litedb_examples.md` record `unifyweaver.execution.json_to_litedb_bash`

### PowerShell Script

See `playbooks/examples_library/json_litedb_examples.md` record `unifyweaver.execution.json_to_litedb_ps`

## Expected Output

```
Loading JSON data into LiteDB...
Inserted: Widget Pro
Inserted: Gadget X
Inserted: Tool Master
Inserted: Device Alpha
✅ 4 products loaded

Querying products by category 'Electronics'...
Widget Pro:$29.99
Gadget X:$49.99

Success: JSON data streamed into LiteDB
```

## Architecture

```
JSON File → UnifyWeaver JSON Source → Typed Records
                                            ↓
                              PowerShell + Inline .NET
                                            ↓
                              LiteDB Insert/Query
                                            ↓
                              Results
```

## Performance Notes

- **First Run**: ~500ms (compiles .NET code + processes data)
- **Cached Runs**: ~5ms (pre-compiled DLL)
- **DLL Caching**: Enabled with `pre_compile(true)`

## Comparison with Other Approaches

| Approach | Use Case | Performance |
|----------|----------|-------------|
| **This (JSON → LiteDB)** | NoSQL storage, flexible schema | Fast (cached) |
| **CSV → Bash** | Simple text processing | Very fast |
| **C# Codegen** | Standalone executables | Fast (compiled) |
| **XML → PowerShell** | Hierarchical data | Medium |

## Tips

1. **Batch Inserts**: For large datasets, collect records and use `InsertBulk()`
2. **Indexing**: Create indexes on frequently queried fields: `collection.EnsureIndex(x => x["category"])`
3. **Transactions**: Wrap multiple inserts in `db.BeginTrans()` / `db.Commit()`
4. **Schema Evolution**: LiteDB is schema-less, but using typed schemas in Prolog ensures consistency

## Troubleshooting

### "LiteDB.dll not found"
Run the setup script first:
```bash
bash scripts/setup/setup_litedb.sh
# Select option 2 (stable) for local installation
```

### "Database is locked"
Ensure previous database connections are disposed:
```powershell
[System.GC]::Collect()
[System.GC]::WaitForPendingFinalizers()
```

### Performance Issues
Enable pre-compilation:
```prolog
pre_compile(true)  % Caches compiled .NET DLL
```

## References

- **Codex's Guide**: `docs/guides/json_to_litedb_streaming.md`
- **JSON Source Skill**: `skills/skill_json_sources.md`
- **LiteDB Documentation**: https://www.litedb.org/
- **Dynamic Source Examples**: `examples/powershell_dotnet_example.pl`
