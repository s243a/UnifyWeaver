# JSON to LiteDB Streaming Examples

## `unifyweaver.execution.json_to_litedb_bash`

> [!example-record]
> id: unifyweaver.execution.json_to_litedb_bash
> name: JSON to LiteDB Streaming Example (Bash)
> platform: bash

This record demonstrates streaming JSON data into LiteDB using UnifyWeaver's JSON source and inline .NET.

```bash
#!/bin/bash
set -e

echo "=========================================="
echo "JSON to LiteDB Streaming Demo"
echo "=========================================="

# Ensure LiteDB is installed
if [ ! -f "lib/LiteDB.dll" ]; then
    echo "❌ Error: LiteDB.dll not found in lib/"
    echo "Run: bash scripts/setup/setup_litedb.sh"
    exit 1
fi

# Create sample JSON data
mkdir -p tmp
cat > tmp/products.json <<'EOF'
[
    {"id": "P001", "name": "Widget Pro", "price": 29.99, "category": "Electronics"},
    {"id": "P002", "name": "Gadget X", "price": 49.99, "category": "Electronics"},
    {"id": "P003", "name": "Tool Master", "price": 19.99, "category": "Tools"},
    {"id": "P004", "name": "Device Alpha", "price": 99.99, "category": "Electronics"}
]
EOF

# Create Prolog program
cat > tmp/json_to_litedb.pl <<'PROLOG'
:- use_module('src/unifyweaver/sources').
:- use_module('src/unifyweaver/core/dynamic_source_compiler').
:- use_module('src/unifyweaver/sources/dotnet_source').

% Define LiteDB inserter as dynamic source
:- source(dotnet, load_products, [
    arity(0),
    target(powershell),
    csharp_inline('
using LiteDB;
using System;
using System.IO;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace UnifyWeaver.Generated.LoadProducts {
    public class LoadProductsHandler {
        public string ProcessLoadProducts() {
            // Read JSON file
            var jsonText = File.ReadAllText("tmp/products.json");
            var jsonArray = JsonNode.Parse(jsonText).AsArray();

            // Open database
            using (var db = new LiteDatabase("tmp/products.db")) {
                var products = db.GetCollection("products");

                // Clear existing data
                products.DeleteAll();

                int count = 0;
                foreach (var item in jsonArray) {
                    var doc = new BsonDocument {
                        ["_id"] = item["id"].ToString(),
                        ["name"] = item["name"].ToString(),
                        ["price"] = item["price"].GetValue<double>(),
                        ["category"] = item["category"].ToString()
                    };
                    products.Insert(doc);
                    count++;
                }

                return $"Loaded {count} products into LiteDB";
            }
        }
    }
}
'),
    % Auto-selects external_compile if dotnet SDK available, otherwise pre_compile
    references(['System.Text.Json', 'lib/LiteDB.dll'])
]).

% Define query predicate
:- source(dotnet, query_by_category, [
    arity(1),
    target(powershell),
    csharp_inline('
using LiteDB;
using System;
using System.Linq;

namespace UnifyWeaver.Generated.QueryByCategory {
    public class QueryByCategoryHandler {
        public string ProcessQueryByCategory(string category) {
            using (var db = new LiteDatabase("tmp/products.db")) {
                var products = db.GetCollection("products");
                var results = products.Find(x => x["category"] == category);

                var output = string.Join("\\n",
                    results.Select(doc =>
                        $"  {doc["name"],-20} ${doc["price"]:F2}"));

                return output;
            }
        }
    }
}
'),
    % Auto-selects external_compile if dotnet SDK available, otherwise pre_compile
    references(['lib/LiteDB.dll'])
]).

% Compile and write PowerShell scripts
:- compile_dynamic_source(load_products/0, [], Code1),
   open('tmp/load_products.ps1', write, Stream1),
   write(Stream1, Code1),
   close(Stream1).

:- compile_dynamic_source(query_by_category/1, [], Code2),
   open('tmp/query_products.ps1', write, Stream2),
   write(Stream2, Code2),
   close(Stream2).
PROLOG

# Compile Prolog to PowerShell
echo ""
echo "Compiling Prolog to PowerShell..."
swipl -g "asserta(file_search_path(unifyweaver, 'src/unifyweaver')), \
          [tmp/json_to_litedb], halt" -t halt

if [ ! -f tmp/load_products.ps1 ] || [ ! -f tmp/query_products.ps1 ]; then
    echo "❌ Error: PowerShell scripts not generated"
    exit 1
fi

echo "✅ PowerShell scripts generated"

# Execute: Load data
echo ""
echo "Loading JSON data into LiteDB..."
pwsh tmp/load_products.ps1

# Execute: Query data
echo ""
echo "Querying products by category 'Electronics'..."
pwsh tmp/query_products.ps1 "Electronics"

echo ""
echo "=========================================="
echo "✅ Success: JSON streamed into LiteDB"
echo "=========================================="

# Show database info
if [ -f tmp/products.db ]; then
    DB_SIZE=$(du -h tmp/products.db | cut -f1)
    echo "Database: tmp/products.db ($DB_SIZE)"
fi
```

## `unifyweaver.execution.json_to_litedb_ps`

> [!example-record]
> id: unifyweaver.execution.json_to_litedb_ps
> name: JSON to LiteDB Streaming Example (PowerShell)
> platform: powershell

This record demonstrates streaming JSON data into LiteDB using PowerShell-native approach.

```powershell
$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "JSON to LiteDB Streaming Demo" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Ensure LiteDB is installed
if (-not (Test-Path "lib/LiteDB.dll")) {
    Write-Host "❌ Error: LiteDB.dll not found in lib/" -ForegroundColor Red
    Write-Host "Run: .\scripts\setup\setup_litedb.ps1"
    exit 1
}

# Create sample JSON data
New-Item -ItemType Directory -Force -Path "tmp" | Out-Null

$sampleData = @'
[
    {"id": "P001", "name": "Widget Pro", "price": 29.99, "category": "Electronics"},
    {"id": "P002", "name": "Gadget X", "price": 49.99, "category": "Electronics"},
    {"id": "P003", "name": "Tool Master", "price": 19.99, "category": "Tools"},
    {"id": "P004", "name": "Device Alpha", "price": 99.99, "category": "Electronics"}
]
'@

Set-Content -Path "tmp/products.json" -Value $sampleData

# Create Prolog program (same as bash version)
$prologCode = @'
:- use_module('src/unifyweaver/sources').
:- use_module('src/unifyweaver/core/dynamic_source_compiler').
:- use_module('src/unifyweaver/sources/dotnet_source').

% Define LiteDB inserter as dynamic source
:- source(dotnet, load_products, [
    arity(0),
    target(powershell),
    % Auto-selects external_compile if dotnet SDK available, otherwise pre_compile
    csharp_inline('
using LiteDB;
using System;
using System.IO;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace UnifyWeaver.Generated.LoadProducts {
    public class LoadProductsHandler {
        public string ProcessLoadProducts() {
            var jsonText = File.ReadAllText("tmp/products.json");
            var jsonArray = JsonNode.Parse(jsonText).AsArray();

            using (var db = new LiteDatabase("tmp/products.db")) {
                var products = db.GetCollection("products");
                products.DeleteAll();

                int count = 0;
                foreach (var item in jsonArray) {
                    var doc = new BsonDocument {
                        ["_id"] = item["id"].ToString(),
                        ["name"] = item["name"].ToString(),
                        ["price"] = item["price"].GetValue<double>(),
                        ["category"] = item["category"].ToString()
                    };
                    products.Insert(doc);
                    count++;
                }

                return $"Loaded {count} products into LiteDB";
            }
        }
    }
}
'),
    references(['System.Text.Json', 'lib/LiteDB.dll'])
]).

% Define query predicate
:- source(dotnet, query_by_category, [
    arity(1),
    target(powershell),
    % Auto-selects external_compile if dotnet SDK available, otherwise pre_compile
    csharp_inline('
using LiteDB;
using System;
using System.Linq;

namespace UnifyWeaver.Generated.QueryByCategory {
    public class QueryByCategoryHandler {
        public string ProcessQueryByCategory(string category) {
            using (var db = new LiteDatabase("tmp/products.db")) {
                var products = db.GetCollection("products");
                var results = products.Find(x => x["category"] == category);

                var output = string.Join("\\n",
                    results.Select(doc =>
                        $"  {doc["name"],-20} ${doc["price"]:F2}"));

                return output;
            }
        }
    }
}
'),
    references(['lib/LiteDB.dll'])
]).

:- compile_dynamic_source(load_products/0, [], Code1),
   open('tmp/load_products.ps1', write, Stream1),
   write(Stream1, Code1),
   close(Stream1).

:- compile_dynamic_source(query_by_category/1, [], Code2),
   open('tmp/query_products.ps1', write, Stream2),
   write(Stream2, Code2),
   close(Stream2).
'@

Set-Content -Path "tmp/json_to_litedb.pl" -Value $prologCode

# Compile Prolog to PowerShell
Write-Host ""
Write-Host "Compiling Prolog to PowerShell..." -ForegroundColor Yellow

$goal = "asserta(file_search_path(unifyweaver, 'src/unifyweaver')), " +
        "[tmp/json_to_litedb], halt"

swipl -g $goal -t halt

if (-not (Test-Path "tmp/load_products.ps1") -or -not (Test-Path "tmp/query_products.ps1")) {
    Write-Host "❌ Error: PowerShell scripts not generated" -ForegroundColor Red
    exit 1
}

Write-Host "✅ PowerShell scripts generated" -ForegroundColor Green

# Execute: Load data
Write-Host ""
Write-Host "Loading JSON data into LiteDB..." -ForegroundColor Yellow
& "tmp/load_products.ps1"

# Execute: Query data
Write-Host ""
Write-Host "Querying products by category 'Electronics'..." -ForegroundColor Yellow
& "tmp/query_products.ps1" "Electronics"

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "✅ Success: JSON streamed into LiteDB" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan

# Show database info
if (Test-Path "tmp/products.db") {
    $dbSize = (Get-Item "tmp/products.db").Length
    $dbSizeKB = [math]::Round($dbSize / 1KB, 2)
    Write-Host "Database: tmp/products.db ($dbSizeKB KB)" -ForegroundColor Gray
}
```
