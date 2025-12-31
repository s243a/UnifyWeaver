# JPype + RPyC Example

Use JPype to embed CPython in JVM and access RPyC.

## Prerequisites

```bash
pip install jpype1 rpyc
```

## Running the Example

1. Start RPyC server:
   ```bash
   python examples/rpyc-integration/rpyc_server.py
   ```

2. Run the Python client (demonstrates the pattern):
   ```bash
   python examples/python-bridges/jpype/rpyc_client.py
   ```

## How It Works

JPype embeds CPython in the JVM, allowing Java code to call Python functions.
The Python code then uses RPyC to connect to a remote Python server.

```
┌─────────────────────────────────────┐
│ JVM Process                         │
│ ┌─────────────────────────────────┐ │
│ │ Java Application                │ │
│ │                                 │ │
│ │   JPype.startJVM()              │ │
│ │   PyModule.import_("rpyc")      │ │
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
│   - math, numpy, etc.               │
│   - Custom services                 │
└─────────────────────────────────────┘
```

## Key Features

- **Shared memory** for NumPy arrays (fast!)
- **Live object proxies** via RPyC
- **Full Python ecosystem** available in Java
