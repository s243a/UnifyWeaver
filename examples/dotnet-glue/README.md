# .NET Glue Examples

Examples demonstrating UnifyWeaver's .NET cross-target communication capabilities.

## Data Processing Pipeline

A four-stage pipeline that processes user data using multiple .NET-hosted runtimes:

```
C# (validate) → PowerShell (filter) → IronPython (enrich) → CPython (score)
```

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    .NET Host Process                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────┐   ┌─────────────────┐   ┌────────────────┐     │
│  │   C#    │──▶│   PowerShell    │──▶│   IronPython   │──┐  │
│  │ Validate│   │    Filter       │   │    Enrich      │  │  │
│  └─────────┘   │  (in-process)   │   │  (in-process)  │  │  │
│                └─────────────────┘   └────────────────┘  │  │
│                                                          │  │
│                                            ┌─────────────▼──┤
│                                            │   Pipe (JSON)  │
│                                            └─────────────┬──┤
│                                                          │  │
└──────────────────────────────────────────────────────────┼──┘
                                                           │
                                            ┌──────────────▼──┐
                                            │     CPython     │
                                            │   Score (numpy) │
                                            │  (subprocess)   │
                                            └─────────────────┘
```

### Files

| File | Description |
|------|-------------|
| `data_pipeline.pl` | Pipeline definition and code generator |
| Generated outputs: |
| `PowerShellBridge.cs` | C# ↔ PowerShell in-process bridge |
| `IronPythonBridge.cs` | C# ↔ IronPython in-process bridge |
| `CPythonBridge.cs` | C# ↔ CPython pipe-based bridge |
| `UserScorerPipeline.cs` | Complete pipeline implementation |
| `DataPipeline.csproj` | .NET project file |
| `sample_users.json` | Sample input data |

### Usage

```bash
# Generate the pipeline code
swipl data_pipeline.pl

# Build the project (requires .NET 8 SDK)
dotnet build

# Run the pipeline
dotnet run
```

### Pipeline Stages

**Stage 1: Validate (C#)**
- Native C# validation
- Checks required fields exist
- Validates data types
- Filters out invalid records

**Stage 2: Filter (PowerShell)**
- Runs in-process via PowerShell SDK
- Filters: Only active users aged 18+
- Transforms: Adds computed `adult` field

**Stage 3: Enrich (IronPython)**
- Runs in-process via IronPython
- Uses only IronPython-compatible modules (`json`, `collections`)
- Adds `category` and `risk_tier` fields

**Stage 4: Score (CPython)**
- Runs via subprocess pipe (fallback)
- Uses `numpy` (not IronPython compatible)
- Applies ML-based risk scoring
- Adds `ml_score` and `recommendation`

### Runtime Selection

The system automatically chooses the appropriate Python runtime:

| Imports | Runtime |
|---------|---------|
| `[json, collections, re]` | IronPython (in-process) |
| `[numpy, json]` | CPython (subprocess) |
| `[pandas, sys, os]` | CPython (subprocess) |
| `[sys, os, math, datetime]` | IronPython (in-process) |

IronPython is preferred for performance (no process spawn), but CPython is used when C-extension modules are required.

### Sample Input

```json
[
  {"id": "001", "name": "Alice", "age": "28", "status": "active"},
  {"id": "002", "name": "Bob", "age": "45", "status": "active"},
  {"id": "003", "name": "Charlie", "age": "17", "status": "active"},
  {"id": "004", "name": "Diana", "age": "62", "status": "inactive"},
  {"id": "005", "name": "Eve", "age": "35", "status": "active"}
]
```

### Expected Output

```json
[
  {
    "id": "001",
    "name": "Alice",
    "age": "28",
    "status": "active",
    "adult": true,
    "category": "young_adult",
    "risk_tier": "low",
    "ml_score": 0.1284,
    "recommendation": "review"
  },
  {
    "id": "002",
    "name": "Bob",
    "age": "45",
    "status": "active",
    "adult": true,
    "category": "middle_adult",
    "risk_tier": "low",
    "ml_score": 0.1550,
    "recommendation": "review"
  }
]
```

Records filtered out:
- Charlie (age < 18)
- Diana (status != "active")
- Invalid record (empty id)
- Henry (invalid age format)

## How It Works

### In-Process Bridges

PowerShell and IronPython run **in the same .NET process** as the host C# application:

```csharp
// PowerShell: Uses System.Management.Automation
using var ps = PowerShell.Create();
ps.AddScript(script);
var results = ps.Invoke<T>();

// IronPython: Uses IronPython.Hosting
var engine = Python.CreateEngine();
var result = engine.Execute(script);
```

Benefits:
- No process spawn overhead
- Shared memory access
- Direct object passing

### Pipe-Based Fallback

When IronPython can't handle required modules (numpy, pandas, etc.), the system falls back to CPython via subprocess pipes:

```csharp
// CPython: Subprocess with JSON pipes
var psi = new ProcessStartInfo("python3");
psi.RedirectStandardInput = true;
psi.RedirectStandardOutput = true;

// Send input as JSON, receive output as JSON
process.StandardInput.WriteLine(JsonSerializer.Serialize(input));
var output = process.StandardOutput.ReadLine();
return JsonSerializer.Deserialize<T>(output);
```

This maintains compatibility with the full Python ecosystem.

## Requirements

- .NET 8.0 SDK
- PowerShell 7.x (for PowerShell SDK)
- Python 3.x with numpy (for CPython fallback)
- IronPython 3.4+ (optional, for in-process Python)

### NuGet Packages

```xml
<PackageReference Include="Microsoft.PowerShell.SDK" Version="7.4.0" />
<PackageReference Include="IronPython" Version="3.4.1" />
```
