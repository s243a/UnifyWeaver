# Cross-Target Glue: Specification

## 1. Target Location Model

### 1.1 Location Types

```prolog
% Location type hierarchy
location_type(in_process).      % Same process, direct calls
location_type(local_process).   % Separate process, same machine
location_type(remote).          % Different machine, network

% Sub-types
location_type(pipe) :- parent(local_process).
location_type(shared_memory) :- parent(local_process).
location_type(socket) :- parent(remote).
location_type(http) :- parent(remote).
location_type(grpc) :- parent(remote).
```

### 1.2 Location Specification

```prolog
% Full location specification
:- target_location(Predicate/Arity, Target, Options).

% Options:
%   process(same | separate)     - Process boundary
%   host(Hostname)               - Machine location
%   port(Port)                   - Network port
%   transport(pipe | socket | http | grpc)
%   format(tsv | json | binary | native)
%   buffer(none | line | block(Size))
%   timeout(Seconds)
%   retry(Count)
```

### 1.3 Default Resolution

```prolog
% Default location resolution rules
default_location(Target1, Target2, in_process) :-
    same_runtime_family(Target1, Target2).

default_location(Target1, Target2, local_process(pipe)) :-
    \+ same_runtime_family(Target1, Target2).

% Runtime families
runtime_family(csharp, dotnet).
runtime_family(fsharp, dotnet).
runtime_family(powershell, dotnet).
runtime_family(ironpython, dotnet).

runtime_family(java, jvm).
runtime_family(scala, jvm).
runtime_family(clojure, jvm).
runtime_family(jython, jvm).

runtime_family(bash, shell).
runtime_family(awk, shell).
runtime_family(perl, shell).

runtime_family(python, python).   % CPython standalone
runtime_family(go, native).
runtime_family(rust, native).
runtime_family(c, native).
```

## 2. Communication Protocols

### 2.1 Pipe Protocol (Default for local_process)

```
┌─────────────────────────────────────────────────────┐
│                   PIPE PROTOCOL                      │
├─────────────────────────────────────────────────────┤
│ Format: TSV (Tab-Separated Values)                  │
│ Encoding: UTF-8                                      │
│ Record delimiter: Newline (\n)                       │
│ Field delimiter: Tab (\t)                            │
│ Escape: Backslash for \t, \n, \\                    │
│ NULL: Empty field                                    │
│ EOF: Close pipe                                      │
└─────────────────────────────────────────────────────┘
```

**Header negotiation (optional):**
```
#UNIFYWEAVER:v1:tsv
#FIELDS:name:string,age:int,salary:float
Alice	30	75000.00
Bob	25	65000.00
```

### 2.2 JSON Protocol

```
┌─────────────────────────────────────────────────────┐
│                  JSON PROTOCOL                       │
├─────────────────────────────────────────────────────┤
│ Format: JSON Lines (one JSON object per line)        │
│ Encoding: UTF-8                                      │
│ Record delimiter: Newline (\n)                       │
│ NULL: JSON null                                      │
│ EOF: Close connection                                │
└─────────────────────────────────────────────────────┘
```

**Example:**
```json
{"name": "Alice", "age": 30, "salary": 75000.00}
{"name": "Bob", "age": 25, "salary": 65000.00}
```

### 2.3 In-Process Protocol (Same Runtime)

```
┌─────────────────────────────────────────────────────┐
│               IN-PROCESS PROTOCOL                    │
├─────────────────────────────────────────────────────┤
│ Format: Native objects                               │
│ Transfer: Direct memory reference                    │
│ Streaming: Iterator/IEnumerable/Channel             │
│ No serialization overhead                            │
└─────────────────────────────────────────────────────┘
```

**Examples by runtime:**
- .NET: `IEnumerable<T>` or `IAsyncEnumerable<T>`
- JVM: `Stream<T>` or `Iterator<T>`
- Native: Channels or callbacks

## 3. Target Declarations

### 3.1 Declaring Target Affinity

```prolog
% Declare which target compiles a predicate
:- target(filter_logs/2, awk).
:- target(analyze_data/2, python).
:- target(store_results/2, sql).

% With options
:- target(process_stream/2, go, [
    optimize(speed),
    streaming(true)
]).
```

### 3.2 Declaring Location

```prolog
% Explicit location override
:- location(analyze_data/2, [
    host('ml-worker.local'),
    port(8080),
    transport(http)
]).

% Force separate process even for same runtime
:- location(untrusted_code/2, [
    process(separate),
    sandbox(true)
]).
```

### 3.3 Declaring Data Format

```prolog
% Override default format
:- format(heavy_data/2, binary).
:- format(config_data/1, json).

% Schema for typed protocols
:- schema(employee/3, [
    field(name, string),
    field(age, integer),
    field(salary, float)
]).
```

## 4. Glue Code Generation

### 4.1 Producer Side (Writer)

Generated code must:
1. Serialize output to agreed format
2. Write to agreed transport
3. Handle backpressure (if supported)
4. Signal completion (EOF/close)

**Example - AWK producer for pipe:**
```awk
# Generated glue code
{
    # ... predicate logic ...

    # Output in TSV format
    print result_field1 "\t" result_field2
}

END {
    # Pipe closed automatically
}
```

### 4.2 Consumer Side (Reader)

Generated code must:
1. Read from transport
2. Deserialize from agreed format
3. Handle EOF/errors
4. Provide streaming interface to logic

**Example - Python consumer from pipe:**
```python
# Generated glue code
import sys

def read_input():
    for line in sys.stdin:
        fields = line.rstrip('\n').split('\t')
        yield {
            'field1': fields[0],
            'field2': int(fields[1])
        }

# Predicate logic uses read_input()
```

### 4.3 Orchestrator

When multiple targets compose, generate orchestration:

**Bash orchestrator example:**
```bash
#!/bin/bash
# Generated pipeline orchestrator

# AWK filter → Python analyze → SQL store
awk -f filter.awk input.tsv \
    | python3 analyze.py \
    | sqlite3 results.db ".import /dev/stdin results"
```

## 5. Runtime Family Details

### 5.1 .NET Family

**In-process communication:**
```csharp
// C# calling PowerShell in same process
using System.Management.Automation;

var ps = PowerShell.Create();
ps.AddScript(scriptCode);
var results = ps.Invoke<OutputType>();
```

**IronPython consideration:**
- Use IronPython when: All required libraries available
- Fall back to CPython when: NumPy, pandas, etc. needed
- Detection: Check imports at compile time

```prolog
% Automatic runtime selection
python_runtime(ironpython) :-
    required_modules(Modules),
    forall(member(M, Modules), ironpython_supports(M)).

python_runtime(cpython) :-
    \+ python_runtime(ironpython).
```

### 5.2 Shell Family (Bash, AWK)

**Always separate processes, pipe communication:**
```bash
# Bash → AWK
bash_output | awk -f script.awk

# AWK → Bash
awk -f script.awk | while read line; do
    # process
done
```

### 5.3 Native Family (Go, Rust)

**Compiled to binaries, pipe or socket:**
```bash
# Process pipe
./go_processor | ./rust_analyzer

# Or socket for bidirectional
./go_server &
./rust_client --connect localhost:8080
```

## 6. Error Handling

### 6.1 Error Propagation

```prolog
% Errors cross target boundaries
error_handling(propagate).    % Default: propagate to caller
error_handling(log_continue). % Log and continue processing
error_handling(retry(N)).     % Retry N times before failing
error_handling(fallback(Pred)). % Call fallback predicate
```

### 6.2 Error Format

```json
{
    "error": true,
    "type": "runtime_error",
    "target": "python",
    "predicate": "analyze/2",
    "message": "Division by zero",
    "trace": ["analyze.py:42", "main.py:15"]
}
```

## 7. Configuration

### 7.1 Project-Level Config

```prolog
% unifyweaver.config.pl

% Default transport for cross-process
:- default_transport(pipe).

% Default format
:- default_format(tsv).

% Timeouts
:- default_timeout(30). % seconds

% .NET runtime preference
:- dotnet_python(ironpython). % or cpython
```

### 7.2 Environment-Specific

```prolog
% production.config.pl
:- location(ml_model/2, [host('ml-cluster.prod'), port(443), transport(https)]).

% development.config.pl
:- location(ml_model/2, [host('localhost'), port(8080), transport(http)]).
```

## 8. Introspection API

```prolog
% Query target information
?- target_of(analyze/2, Target).
Target = python.

?- location_of(analyze/2, Location).
Location = local_process(pipe).

?- communication_path(fetch/1, store/1, Path).
Path = [fetch/1:bash, pipe, transform/2:python, pipe, store/1:sql].
```
