# CSnakes + RPyC Example

Use CSnakes to embed CPython in .NET 9+ and access RPyC.

## Status

| Feature | Status |
|---------|--------|
| Source generation | ✅ Works - generates typed wrappers |
| Build | ✅ Works |
| Runtime (Windows) | ✅ Works |
| Runtime (Linux) | ✅ Works (via redistributable Python) |

**Tested:** CSnakes 1.2.1 with `FromRedistributable("3.12")` works on Linux.
The redistributable auto-downloads Python 3.12 (~60MB first run) and includes
all necessary native libraries.

## Important: Source Generator Approach

CSnakes uses a **source generator** approach - it generates C# wrapper classes from Python files at compile time. This is different from Python.NET which allows dynamic Python execution.

## Prerequisites

### 1. .NET SDK 9.0

```bash
# Check version
dotnet --version  # Should be 9.0.x
```

### 2. RPyC Server Running

```bash
# From project root
python examples/rpyc-integration/rpyc_server.py
```

**Note:** Python for CSnakes itself is auto-downloaded via redistributable.
Only the RPyC server requires a local Python installation.

## Running the Example

```bash
cd examples/python-bridges/csnakes

# Build and run (first run downloads Python ~60MB)
dotnet run
```

### Using System Python Instead

To use your system Python instead of the redistributable:

```bash
export CSNAKES_USE_REDIST=0
export PYTHON_HOME=/usr
pip install rpyc plumbum  # Install in system Python
dotnet run
```

## Expected Output

```
CSnakes + RPyC Integration
==========================

Python initialized

Testing RPyC connection...
  connect_rpyc: OK
  numpy.mean([1,2,3,4,5]) = 3.0
  Server Python: 3.8.10

All tests passed!
```

## How It Works

### 1. Python Wrapper File (`rpyc_wrapper.py`)

Type-annotated functions that CSnakes will wrap:

```python
def connect_rpyc(host: str, port: int) -> bool:
    """Connect to RPyC server and verify connection."""
    import rpyc
    conn = rpyc.classic.connect(host, port)
    try:
        return conn.modules.math.sqrt(16) == 4.0
    finally:
        conn.close()
```

### 2. CSnakes Source Generation

At compile time, CSnakes generates:

```csharp
// Auto-generated
public interface IRpycWrapper {
    bool ConnectRpyc(string host, long port);
    double GetNumpyMean(string host, long port, IReadOnlyList<double> values);
    string GetServerPythonVersion(string host, long port);
}
```

### 3. C# Usage

```csharp
var wrapper = env.RpycWrapper();
bool ok = wrapper.ConnectRpyc("localhost", 18812);
```

## Configuration Options

### Using System Python

```csharp
builder.Services
    .WithPython()
    .WithHome(AppContext.BaseDirectory)
    .FromEnvironmentVariable("PYTHON_HOME", "3.8")
    .WithPipInstaller();  // Installs from requirements.txt
```

### Using Redistributable Python

```csharp
builder.Services
    .WithPython()
    .WithHome(AppContext.BaseDirectory)
    .FromRedistributable("3.12")  // Downloads Python automatically
    .WithPipInstaller();
```

## CSnakes vs Python.NET

| Feature | CSnakes | Python.NET |
|---------|---------|------------|
| .NET Version | 9.0+ | 6.0+ |
| Execution | Compile-time generated | Runtime dynamic |
| Type Safety | Strong (source gen) | Dynamic |
| API Style | Generated wrappers | Direct Python calls |
| Setup | Requires PYTHON_HOME | Auto-detects |

**For RPyC specifically:**
- **Python.NET** is simpler - directly import and call rpyc
- **CSnakes** requires wrapper functions but provides type safety

## Troubleshooting

### "PYTHON_HOME not set"

Set the environment variable to your Python installation:

```bash
# Linux - Python in /usr/bin/python3
export PYTHON_HOME=/usr

# Linux - Python in /usr/local
export PYTHON_HOME=/usr/local

# Windows
set PYTHON_HOME=C:\Python312
```

### "ModuleNotFoundError: No module named 'rpyc'"

Install rpyc in your Python environment:

```bash
pip install rpyc plumbum
```

### "Connection refused"

Start the RPyC server:

```bash
python examples/rpyc-integration/rpyc_server.py
```

### "Unable to load shared library 'csnakes_python'" (Linux)

This error occurs with old CSnakes versions (< 1.2.x) or when using
`FromEnvironmentVariable` with system Python. The solution is to use
`FromRedistributable("3.12")` which includes all native libraries.

If you must use system Python:
1. Upgrade to CSnakes 1.2.1+
2. Use `FromRedistributable` instead of `FromEnvironmentVariable`
3. The redistributable downloads Python with all required libraries

## Architecture

```
┌────────────────────────────────────────────┐
│ .NET 9 Process                             │
│ ┌────────────────────────────────────────┐ │
│ │ C# Application                         │ │
│ │   var wrapper = env.RpycWrapper();     │ │
│ │   wrapper.ConnectRpyc("localhost");    │ │
│ └────────────────────────────────────────┘ │
│               │                            │
│               ▼                            │
│ ┌────────────────────────────────────────┐ │
│ │ CSnakes Generated Wrapper              │ │
│ │   (calls rpyc_wrapper.py functions)    │ │
│ └────────────────────────────────────────┘ │
│               │                            │
│               ▼                            │
│ ┌────────────────────────────────────────┐ │
│ │ Embedded CPython (from PYTHON_HOME)    │ │
│ │   rpyc_wrapper.connect_rpyc(...)       │ │
│ └────────────────────────────────────────┘ │
└────────────────────────────────────────────┘
                │
                │ TCP/IP
                ▼
┌────────────────────────────────────────────┐
│ RPyC Server (Python)                       │
│   - NumPy, SciPy, pandas                   │
│   - Custom services                        │
└────────────────────────────────────────────┘
```

## Files

- `rpyc_wrapper.py` - Python functions with type annotations
- `requirements.txt` - Python dependencies (rpyc, plumbum)
- `RPyCClient.cs` - C# main program
- `csnakes-rpyc.csproj` - Project file with CSnakes reference
