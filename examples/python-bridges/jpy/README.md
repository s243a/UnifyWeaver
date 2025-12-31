# jpy + RPyC Example

Use jpy for bi-directional Java↔Python communication and RPyC access.

## What is jpy?

jpy provides **true bi-directional** calling between Java and Python:
- Java can call Python functions
- Python can call Java classes
- Used by JetBrains tools and scientific computing projects

## Prerequisites

### Java JDK 11+ (Required)

```bash
# Ubuntu/Debian
sudo apt install openjdk-11-jdk

# Set JAVA_HOME
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
```

### Maven (Required for jpy build)

```bash
# Ubuntu/Debian
sudo apt install maven
```

### Python Packages

```bash
# Ensure JAVA_HOME is set first
pip install jpy rpyc
```

## Running the Example

1. Start RPyC server (in another terminal):
   ```bash
   python examples/rpyc-integration/rpyc_server.py
   ```

2. Run the Python client:
   ```bash
   python examples/python-bridges/jpy/rpyc_client.py
   ```

## jpy vs JPype

| Feature | jpy | JPype |
|---------|-----|-------|
| Direction | Bi-directional | Python → Java |
| NumPy Support | Yes | Yes (shared memory) |
| Build Requires | Maven | None |
| API Style | get_type() | Pythonic imports |
| Active Development | Yes | Yes |

Choose **jpy** for:
- Java code that needs to call Python
- Mixed Java/Python codebases
- JetBrains tools compatibility

Choose **JPype** for:
- Python-first projects
- Simpler installation (no Maven)
- Shared memory NumPy performance

## Bi-Directional Example

### Python Calling Java

```python
import jpy
jpy.create_jvm(["-Xmx512m"])

ArrayList = jpy.get_type("java.util.ArrayList")
list = ArrayList()
list.add("hello")
```

### Java Calling Python (via jpy)

```java
import org.jpy.PyLib;
import org.jpy.PyObject;

PyLib.startPython();
PyObject result = PyLib.execCode("2 + 2");
System.out.println(result);  // 4
```

## Architecture

```
┌─────────────────────────────────────┐
│ Process                             │
│ ┌─────────────────────────────────┐ │
│ │ Java Code                       │ │
│ │   PyLib.execCode("...")         │ │
│ │          ↕                      │ │
│ │ Python Code                     │ │
│ │   jpy.get_type("...")           │ │
│ └─────────────────────────────────┘ │
│               │                     │
│               ▼                     │
│ ┌─────────────────────────────────┐ │
│ │ RPyC Client                     │ │
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

## Note

jpy requires Maven to build its Java components during pip install. If you don't have Maven and don't need bi-directional calling, consider using **JPype** instead.
