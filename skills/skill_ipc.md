# Skill: Inter-Process Communication (Sub-Master)

Patterns for communication between Prolog and other languages/processes via pipes, network RPC, and cross-runtime bridges.

## When to Use

- User asks "how do I connect Prolog to Python?"
- User needs Unix pipe-based data flow
- User wants network-based remote procedure calls
- User needs to embed Python in .NET, JVM, Rust, or other runtimes

## Skill Hierarchy

```
skill_server_tools.md (parent)
└── skill_ipc.md (this file)
    ├── skill_pipe_communication.md - TSV/JSON pipe protocols
    ├── skill_rpyc.md - Remote Python Call (network RPC)
    └── skill_python_bridges.md - Cross-runtime Python embedding
```

## Quick Start

### Pipe Communication (Unix Pipelines)

```prolog
:- use_module('src/unifyweaver/glue/pipe_glue').

% Generate TSV writer for Python
generate_pipe_writer(python, [name, age, score], [], WriterCode).

% Generate TSV reader for AWK
generate_pipe_reader(awk, [name, age, score], [], ReaderCode).

% Generate pipeline orchestrator
generate_pipeline_script([
    step(extract, local, 'cat data.tsv'),
    step(transform, local, 'awk processing'),
    step(load, remote, 'http://api/load')
], [], PipelineScript).
```

### RPyC (Network-Based Python RPC)

```prolog
:- use_module('src/unifyweaver/glue/rpyc_glue').

% Connect to Python service
rpyc_connect('localhost', [
    security(unsecured),
    acknowledge_risk(true),
    remote_port(18812)
], Proxy).

% Call remote function
rpyc_call(Proxy, numpy, mean, [[1,2,3,4,5]], Result).

% Async call
rpyc_async_call(Proxy, scipy, optimize, [Func, X0], AsyncResult).
rpyc_await(AsyncResult, Result).

% Cleanup
rpyc_disconnect(Proxy).
```

### Python Bridges (Cross-Runtime Embedding)

```prolog
:- use_module('src/unifyweaver/glue/python_bridges_glue').

% Detect available bridges
detect_all_bridges(Bridges).

% Auto-select best bridge for platform
auto_select_bridge(RuntimeEnv, Preferences, Bridge).

% Generate bridge client code
generate_pythonnet_rpyc_client(Predicates, CSharpCode).
generate_jpype_rpyc_client(Predicates, JavaCode).
generate_pyo3_rpyc_client(Predicates, RustCode).
```

## IPC Pattern Comparison

| Pattern | Use Case | Latency | Complexity | Targets |
|---------|----------|---------|------------|---------|
| Pipes | Unix data flow | Low | Low | awk, python, bash, go, rust |
| RPyC | Network RPC | Medium | Medium | Python (any client) |
| Bridges | Embedded Python | Very Low | High | .NET, JVM, Rust, Ruby, FFI |

## When to Use Each Pattern

### Pipes (skill_pipe_communication.md)
- **Best for:** ETL pipelines, streaming data processing
- **Data format:** TSV (simple) or JSON Lines (structured)
- **Examples:** `cat data.csv | python transform.py | awk '{print $1}'`

### RPyC (skill_rpyc.md)
- **Best for:** Remote Python services, process isolation
- **Security:** SSH, SSL, or unsecured with explicit acknowledgment
- **Features:** Live object proxies, bidirectional calls, async execution

### Python Bridges (skill_python_bridges.md)
- **Best for:** Native integration, lowest latency
- **Platforms:**
  - .NET: Python.NET, CSnakes
  - JVM: JPype, jpy
  - Rust: PyO3
  - Ruby: PyCall.rb
  - FFI: Go, Node.js via koffi/ffi-napi

## Child Skills

- `skill_pipe_communication.md` - TSV/JSON pipe readers and writers
- `skill_rpyc.md` - Remote Python Call with proxy layers
- `skill_python_bridges.md` - Cross-runtime Python embedding

## Related

**Parent Skill:**
- `skill_server_tools.md` - Backend services master

**Sibling Skills:**
- `skill_web_frameworks.md` - HTTP-based APIs
- `skill_infrastructure.md` - Deployment and ops

**Code:**
- `src/unifyweaver/glue/pipe_glue.pl`
- `src/unifyweaver/glue/rpyc_glue.pl`
- `src/unifyweaver/glue/python_bridges_glue.pl`
