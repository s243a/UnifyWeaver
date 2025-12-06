# Cross-Target Glue: Implementation Plan

## Phase Overview

```
Phase 1: Foundation        → Target registry, basic pipe glue
Phase 2: Shell Integration → Bash ↔ AWK ↔ Python pipes
Phase 3: .NET Integration  → C# ↔ PowerShell ↔ IronPython in-process
Phase 4: Native Targets    → Go/Rust binary orchestration
Phase 5: Network Layer     → Remote target communication
Phase 6: Advanced Features → Error handling, monitoring, optimization
```

## Phase 1: Foundation

**Goal:** Establish core infrastructure for target management and basic communication.

### 1.1 Target Registry

Create a central registry for target metadata:

```prolog
% src/unifyweaver/core/target_registry.pl

:- module(target_registry, [
    register_target/3,        % register_target(Name, Family, Capabilities)
    target_family/2,          % target_family(Target, Family)
    targets_same_family/2,    % targets_same_family(T1, T2)
    target_capabilities/2,    % target_capabilities(Target, Caps)
    list_targets/1            % list_targets(Targets)
]).

% Built-in target definitions
:- register_target(bash, shell, [streaming, pipes, process_control]).
:- register_target(awk, shell, [streaming, pipes, regex, aggregation]).
:- register_target(python, python, [streaming, pipes, libraries, ml]).
:- register_target(go, native, [compiled, streaming, concurrency]).
:- register_target(rust, native, [compiled, streaming, memory_safe]).
:- register_target(csharp, dotnet, [compiled, streaming, linq]).
:- register_target(powershell, dotnet, [scripting, streaming, system_admin]).
:- register_target(sql, database, [queries, transactions]).
```

### 1.2 Predicate-Target Mapping

```prolog
% src/unifyweaver/core/target_mapping.pl

:- module(target_mapping, [
    declare_target/2,         % declare_target(Pred/Arity, Target)
    declare_target/3,         % declare_target(Pred/Arity, Target, Options)
    predicate_target/2,       % predicate_target(Pred/Arity, Target)
    predicate_location/2      % predicate_location(Pred/Arity, Location)
]).

% User declarations stored as dynamic facts
:- dynamic user_target/3.

declare_target(Pred/Arity, Target) :-
    declare_target(Pred/Arity, Target, []).

declare_target(Pred/Arity, Target, Options) :-
    assertz(user_target(Pred/Arity, Target, Options)).
```

### 1.3 Basic Pipe Glue Generator

```prolog
% src/unifyweaver/glue/pipe_glue.pl

:- module(pipe_glue, [
    generate_pipe_writer/3,   % generate_pipe_writer(Target, Schema, Code)
    generate_pipe_reader/3,   % generate_pipe_reader(Target, Schema, Code)
    generate_orchestrator/3   % generate_orchestrator(Pipeline, Options, Code)
]).
```

### Deliverables

- [ ] `target_registry.pl` - Target metadata management
- [ ] `target_mapping.pl` - Predicate-to-target declarations
- [ ] `pipe_glue.pl` - Basic TSV pipe code generation
- [ ] Unit tests for registry and mapping
- [ ] Documentation updates

### Estimated Effort
2-3 weeks

---

## Phase 2: Shell Integration

**Goal:** Enable seamless Bash ↔ AWK ↔ Python pipelines.

### 2.1 Shell Pipe Writer Templates

```prolog
% AWK output template
awk_pipe_writer(Fields, Code) :-
    format(atom(Code), '{ print ~w }', [FieldsFormatted]).

% Python output template
python_pipe_writer(Fields, Code) :-
    format(atom(Code), 'print("\\t".join([str(~w)]))', [Fields]).

% Bash output template
bash_pipe_writer(Fields, Code) :-
    format(atom(Code), 'echo -e "~w"', [FieldsTabSeparated]).
```

### 2.2 Shell Pipe Reader Templates

```prolog
% AWK input (automatic via $1, $2, etc.)
awk_pipe_reader(Fields, Code) :-
    % AWK reads TSV natively
    Code = ''.

% Python input template
python_pipe_reader(Fields, Code) :-
    Code = 'import sys\nfor line in sys.stdin:\n    fields = line.rstrip().split("\\t")'.

% Bash input template
bash_pipe_reader(Code) :-
    Code = 'while IFS=$\'\\t\' read -r field1 field2; do'.
```

### 2.3 Pipeline Orchestrator

```prolog
% Generate bash script to orchestrate pipeline
generate_shell_pipeline(Steps, Script) :-
    % Steps = [step(Pred, Target, InFile, OutFile), ...]
    maplist(step_to_command, Steps, Commands),
    join_with_pipes(Commands, PipelineCmd),
    format(atom(Script), '#!/bin/bash\n~w\n', [PipelineCmd]).
```

### Example Output

```bash
#!/bin/bash
# Generated pipeline: filter → analyze → summarize

awk -f filter.awk input.tsv \
    | python3 analyze.py \
    | awk -f summarize.awk \
    > output.tsv
```

### Deliverables

- [ ] AWK glue templates (reader/writer)
- [ ] Python glue templates (reader/writer)
- [ ] Bash glue templates (reader/writer)
- [ ] Shell pipeline orchestrator
- [ ] Integration tests: AWK→Python, Python→AWK, Bash→AWK→Python
- [ ] Example: Log analysis pipeline

### Estimated Effort
2-3 weeks

---

## Phase 3: .NET Integration

**Goal:** Enable in-process communication between C#, PowerShell, and IronPython.

### 3.1 .NET Runtime Detection

```prolog
% Detect available .NET runtimes
detect_dotnet_runtime(Runtime) :-
    % Check for dotnet CLI
    shell('dotnet --version', _) -> Runtime = dotnet_core
    ; Runtime = none.

detect_ironpython(Available) :-
    % Check for IronPython
    shell('ipy --version', _) -> Available = true
    ; Available = false.
```

### 3.2 In-Process Communication

```csharp
// Generated C# hosting code for PowerShell
using System.Management.Automation;

public static class PowerShellBridge
{
    public static IEnumerable<T> InvokePowerShell<T>(string script, object input)
    {
        using var ps = PowerShell.Create();
        ps.AddScript(script);
        ps.AddParameter("Input", input);
        return ps.Invoke<T>();
    }
}
```

```csharp
// Generated C# hosting code for IronPython
using IronPython.Hosting;
using Microsoft.Scripting.Hosting;

public static class PythonBridge
{
    private static readonly ScriptEngine Engine = Python.CreateEngine();

    public static dynamic InvokePython(string script, dynamic input)
    {
        var scope = Engine.CreateScope();
        scope.SetVariable("input", input);
        return Engine.Execute(script, scope);
    }
}
```

### 3.3 IronPython Compatibility Check

```prolog
% Check if predicate can use IronPython
can_use_ironpython(Pred/Arity) :-
    predicate_imports(Pred/Arity, Imports),
    forall(member(Import, Imports), ironpython_compatible(Import)).

% Known compatible modules
ironpython_compatible(sys).
ironpython_compatible(os).
ironpython_compatible(json).
ironpython_compatible(re).
ironpython_compatible(collections).

% Known incompatible (need CPython)
\+ ironpython_compatible(numpy).
\+ ironpython_compatible(pandas).
\+ ironpython_compatible(tensorflow).
```

### 3.4 Fallback to CPython

```prolog
% If IronPython incompatible, use CPython via pipes
python_runtime(Pred/Arity, ironpython) :-
    can_use_ironpython(Pred/Arity).

python_runtime(Pred/Arity, cpython_pipe) :-
    \+ can_use_ironpython(Pred/Arity).
```

### Deliverables

- [ ] .NET runtime detection
- [ ] C# ↔ PowerShell in-process bridge
- [ ] C# ↔ IronPython in-process bridge
- [ ] IronPython compatibility checker
- [ ] CPython fallback via pipes
- [ ] Integration tests: C#→PowerShell, C#→IronPython, fallback scenarios
- [ ] Example: .NET data processing with Python ML

### Estimated Effort
3-4 weeks

---

## Phase 4: Native Targets

**Goal:** Orchestrate Go and Rust compiled binaries.

### 4.1 Binary Management

```prolog
% Track compiled binaries
:- dynamic compiled_binary/3.  % compiled_binary(Pred/Arity, Target, Path)

compile_if_needed(Pred/Arity, Target, BinaryPath) :-
    compiled_binary(Pred/Arity, Target, BinaryPath),
    file_exists(BinaryPath),
    !.

compile_if_needed(Pred/Arity, Target, BinaryPath) :-
    compile_to_target(Pred/Arity, Target, SourcePath),
    build_binary(Target, SourcePath, BinaryPath),
    assertz(compiled_binary(Pred/Arity, Target, BinaryPath)).
```

### 4.2 Native Pipe Integration

```prolog
% Generate pipe-compatible main() for Go
go_pipe_main(Pred/Arity, Code) :-
    Code = '
func main() {
    scanner := bufio.NewScanner(os.Stdin)
    for scanner.Scan() {
        fields := strings.Split(scanner.Text(), "\\t")
        result := ~w(fields)
        fmt.Println(strings.Join(result, "\\t"))
    }
}'.

% Generate pipe-compatible main() for Rust
rust_pipe_main(Pred/Arity, Code) :-
    Code = '
fn main() {
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let fields: Vec<&str> = line.unwrap().split(\'\\t\').collect();
        let result = ~w(&fields);
        println!("{}", result.join("\\t"));
    }
}'.
```

### Deliverables

- [ ] Binary compilation management
- [ ] Go pipe-compatible wrapper generation
- [ ] Rust pipe-compatible wrapper generation
- [ ] Cross-compilation support
- [ ] Integration tests: Shell→Go, Go→Rust, mixed pipelines
- [ ] Example: High-performance data pipeline

### Estimated Effort
2-3 weeks

---

## Phase 5: Network Layer

**Goal:** Enable remote target communication via sockets/HTTP.

### 5.1 Network Protocol

```prolog
% Network location specification
:- location(remote_pred/2, [
    host('worker.example.com'),
    port(8080),
    transport(http),        % or socket, grpc
    format(json)
]).
```

### 5.2 HTTP Client/Server Generation

```prolog
% Generate HTTP server wrapper
generate_http_server(Pred/Arity, Target, Port, Code) :-
    % Generate server that accepts POST, returns JSON
    ...

% Generate HTTP client wrapper
generate_http_client(Pred/Arity, Host, Port, Code) :-
    % Generate client that POSTs input, parses JSON response
    ...
```

### 5.3 Service Discovery (Simple)

```prolog
% Static service registry
:- service(ml_model/2, 'http://ml-service:8080/predict').
:- service(db_query/2, 'postgresql://db-host:5432/mydb').
```

### Deliverables

- [ ] HTTP server wrapper generation (Go, Python)
- [ ] HTTP client wrapper generation (all targets)
- [ ] Socket-based communication option
- [ ] Service registry
- [ ] Integration tests: local→remote calls
- [ ] Example: Distributed processing

### Estimated Effort
3-4 weeks

---

## Phase 6: Advanced Features

**Goal:** Production-ready error handling, monitoring, and optimization.

### 6.1 Error Handling

```prolog
% Error propagation configuration
:- error_handling(analyze/2, [
    on_error(retry(3)),
    on_failure(fallback(analyze_simple/2)),
    timeout(30)
]).
```

### 6.2 Monitoring

```prolog
% Generate monitoring hooks
:- monitoring(pipeline, [
    metrics(throughput, latency, errors),
    output(prometheus)
]).
```

### 6.3 Optimization

```prolog
% Automatic batching for network calls
:- optimize(remote_pred/2, [batch(100), parallel(4)]).

% Buffer tuning for pipes
:- optimize(heavy_transform/2, [buffer(block(65536))]).
```

### Deliverables

- [ ] Error propagation framework
- [ ] Retry/fallback mechanisms
- [ ] Timeout handling
- [ ] Metrics collection
- [ ] Performance profiling hooks
- [ ] Batching optimization
- [ ] Documentation: Production deployment guide

### Estimated Effort
4-5 weeks

---

## Summary Timeline

| Phase | Description | Duration | Dependencies |
|-------|-------------|----------|--------------|
| 1 | Foundation | 2-3 weeks | None |
| 2 | Shell Integration | 2-3 weeks | Phase 1 |
| 3 | .NET Integration | 3-4 weeks | Phase 1 |
| 4 | Native Targets | 2-3 weeks | Phase 2 |
| 5 | Network Layer | 3-4 weeks | Phases 2, 4 |
| 6 | Advanced Features | 4-5 weeks | All previous |

**Total estimated: 16-22 weeks**

Phases 2, 3, 4 can partially overlap after Phase 1 completes.

## Quick Wins (Can Start Immediately)

1. **Target registry** - Simple metadata, high value
2. **AWK ↔ Python pipe glue** - Most common use case
3. **Pipeline orchestrator** - Immediate usability

## Success Metrics

- [ ] Can compose 3+ targets in single pipeline
- [ ] In-process .NET communication working
- [ ] Remote target calls functional
- [ ] Error handling doesn't lose data
- [ ] Performance overhead < 5% for in-process, < 10ms for pipes
