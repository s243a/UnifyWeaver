# C# Compilation & .NET Integration Guide

UnifyWeaver provides robust support for integrating C# code into your Prolog-to-Bash/PowerShell pipelines. This guide covers the compilation strategies, architecture, and integration with external libraries like LiteDB.

## Compilation Strategies

UnifyWeaver supports two primary strategies for compiling C# code, automatically selected based on your environment and needs.

### 1. External Compilation (`external_compile`)

This is the **recommended** strategy for robust development. It uses the .NET SDK (`dotnet build`) to compile C# code into DLLs, which are then loaded by PowerShell.

**Benefits:**
- **Dependency Management**: Supports NuGet packages (`<PackageReference>`) and local DLL references.
- **Isolation**: Compilation happens in a separate process, preventing "assembly already loaded" errors.
- **File Locking Solution**: Uses unique, timestamped build directories to ensure that recompiling code doesn't fail due to file locks held by the PowerShell process.

**Requirements:**
- .NET SDK installed (`dotnet` command available).

**How it works:**
1. UnifyWeaver generates a unique build directory in your OS temp folder (for example `%TEMP%/unifyweaver_dotnet_build/Build_MySource_123456`).
2. It generates a `.csproj` file with necessary references.
3. It runs `dotnet build` to produce a DLL.
4. The generated PowerShell script loads this DLL using `Add-Type -Path ...`.

### 2. Pre-Compilation (`pre_compile`)

This is a **fallback** strategy that uses PowerShell's built-in `Add-Type` cmdlet to compile C# code on the fly.

**Benefits:**
- **No SDK Required**: Works on systems with just the .NET Runtime (standard on Windows).
- **Simplicity**: No intermediate build artifacts or project files.

**Limitations:**
- **File Locking**: Once an assembly is loaded, it cannot be unloaded. Recompiling the same class in the same session will fail.
- **Dependencies**: Harder to manage complex NuGet dependencies.

**When to use:**
- Simple scripts without external dependencies.
- Environments where the .NET SDK is not installed.

## Automatic Selection

UnifyWeaver's `dotnet_source` module automatically selects the best strategy:

1. Checks if `dotnet` is available in the system PATH.
2. If available, defaults to `external_compile`.
3. If not available, falls back to `pre_compile`.

You can override the default selection when necessary:

```prolog
:- source(dotnet, my_predicate, [
    external_compile(false),  % Force Add-Type even if dotnet is available
    pre_compile(true)
]).

:- source(dotnet, other_predicate, [
    external_compile(true)     % Force dotnet build even if auto-detect fails
]).
```

## LiteDB Integration

UnifyWeaver includes built-in support for LiteDB, a lightweight, serverless NoSQL document store. This allows you to stream data into a structured database and query it efficiently.

### Setup

Use the included setup scripts to download the LiteDB DLL:

**Bash/WSL:**
```bash
bash scripts/setup/setup_litedb.sh
```

**PowerShell:**
```powershell
.\scripts\setup\setup_litedb.ps1
```

This places `LiteDB.dll` in the `lib/` directory.

### Usage Example

Here is how to define a source that loads data into LiteDB:

```prolog
:- source(dotnet, load_products, [
    arity(0),
    target(powershell),
    csharp_inline('
using LiteDB;
using System.IO;

namespace UnifyWeaver.Generated {
    public class Handler {
        public string Process() {
            using (var db = new LiteDatabase("data.db")) {
                var col = db.GetCollection("products");
                col.Insert(new BsonDocument { ["name"] = "Widget", ["price"] = 9.99 });
                return "Inserted";
            }
        }
    }
}
'),
    references(['lib/LiteDB.dll'])  % Reference the DLL
]).
```

## Solving DLL File Locking

A common issue in PowerShell development is that once a DLL is loaded, the file is locked by the process and cannot be overwritten. This makes iterative development (compile -> run -> fix -> compile) difficult.

**UnifyWeaver's Solution:**
When using `external_compile`, UnifyWeaver generates a **unique build directory** for every compilation (e.g., using a timestamp).

- **Build 1:** `tmp/Build_Source_A_1001/bin/Debug/net8.0/Source_A.dll`
- **Build 2:** `tmp/Build_Source_A_1002/bin/Debug/net8.0/Source_A.dll`

Because the file path changes, PowerShell treats it as a new assembly, avoiding the lock on the previous file. The template automatically cleans up old build artifacts to prevent disk clutter.
