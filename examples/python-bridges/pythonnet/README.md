# Python.NET + RPyC Example

Use Python.NET to embed CPython in .NET and access RPyC.

## Prerequisites

### .NET SDK (Required)

Install .NET 8.0 SDK (recommended):

```bash
# Ubuntu/Debian
sudo apt install dotnet-sdk-8.0

# Or download from https://dotnet.microsoft.com/download
```

### Python Packages

```bash
pip install pythonnet rpyc
```

## Running the Examples

### Python Example (Testing the Pattern)

1. Start RPyC server:
   ```bash
   python examples/rpyc-integration/rpyc_server.py
   ```

2. Run the Python client:
   ```bash
   python examples/python-bridges/pythonnet/rpyc_client.py
   ```

### C# Example

1. Start RPyC server (in another terminal):
   ```bash
   python examples/rpyc-integration/rpyc_server.py
   ```

2. Build and run:
   ```bash
   cd examples/python-bridges/pythonnet
   dotnet run
   ```

## Runtime Configuration

Python.NET supports multiple .NET runtimes:

| Runtime | Environment Variable | Notes |
|---------|---------------------|-------|
| **.NET Core** | `PYTHONNET_RUNTIME=coreclr` | **Default, recommended** |
| **Mono** | `PYTHONNET_RUNTIME=mono` | Legacy, avoid unless needed |

### Setting .NET Core Explicitly

```python
import os
os.environ["PYTHONNET_RUNTIME"] = "coreclr"
import clr  # Now uses .NET Core
```

### Specifying .NET Version

```python
os.environ["PYTHONNET_DOTNET_VERSION"] = "8.0.0"
```

## Architecture

```
┌─────────────────────────────────────┐
│ .NET Process                        │
│ ┌─────────────────────────────────┐ │
│ │ C# / F# Application             │ │
│ │                                 │ │
│ │   using Python.Runtime;         │ │
│ │   Py.Import("rpyc")             │ │
│ └─────────────────────────────────┘ │
│               │                     │
│               ▼                     │
│ ┌─────────────────────────────────┐ │
│ │ Embedded CPython                │ │
│ │   import rpyc                   │ │
│ │   conn = rpyc.connect(...)      │ │
│ └─────────────────────────────────┘ │
└─────────────────────────────────────┘
                │
                │ TCP/IP
                ▼
┌─────────────────────────────────────┐
│ RPyC Server (Python)                │
│   - math, numpy, pandas             │
│   - Custom services                 │
└─────────────────────────────────────┘
```

## Note on Mono

Mono is the legacy cross-platform .NET runtime. While Python.NET still supports it, we recommend .NET Core/.NET 8 for:

- Better performance
- Active development
- Modern features
- Official Microsoft support

Only use Mono if you have specific legacy requirements.
