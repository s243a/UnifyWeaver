# CSnakes + RPyC Example

Use CSnakes to embed CPython in .NET 8+ and access RPyC.

## What is CSnakes?

CSnakes is a modern .NET 8+ library for embedding Python. It offers:
- Simpler API than Python.NET
- Better async support
- Source generators for type safety
- First-class dependency injection support

## Prerequisites

### .NET SDK 8.0+ (Required)

```bash
# Ubuntu/Debian
sudo apt install dotnet-sdk-8.0

# Or download from https://dotnet.microsoft.com/download
```

### Python Packages

```bash
pip install rpyc
```

## Running the Example

1. Start RPyC server (in another terminal):
   ```bash
   python examples/rpyc-integration/rpyc_server.py
   ```

2. Build and run:
   ```bash
   cd examples/python-bridges/csnakes
   dotnet run
   ```

## CSnakes vs Python.NET

| Feature | CSnakes | Python.NET |
|---------|---------|------------|
| .NET Version | 8.0+ only | 6.0+ |
| API Style | Modern, DI-based | Classic embedding |
| Async Support | First-class | Limited |
| Source Generators | Yes | No |
| Maturity | Newer (2024) | Mature (2003) |

Choose **CSnakes** for:
- New .NET 8+ projects
- Simpler, more modern API
- Async Python calls

Choose **Python.NET** for:
- .NET 6/7 compatibility
- Existing Python.NET codebases
- Maximum control over Python runtime

## Architecture

```
┌─────────────────────────────────────┐
│ .NET 8+ Process                     │
│ ┌─────────────────────────────────┐ │
│ │ C# Application                  │ │
│ │   using CSnakes.Runtime;        │ │
│ │   env.Execute("import rpyc")    │ │
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
│   - NumPy, SciPy, pandas            │
│   - Custom services                 │
└─────────────────────────────────────┘
```
