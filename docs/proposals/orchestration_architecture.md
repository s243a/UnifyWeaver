# Proposal: Orchestration Architecture

**Status:** Draft
**Author:** John William Creighton (@s243a)
**Co-Author:** Claude Code (Sonnet 4.5)
**Date:** 2025-11-21
**Version:** 1.0

---

## Executive Summary

UnifyWeaver's orchestration layer enables intelligent coordination of multiple target languages (Bash, C#, Python, Prolog) within data processing pipelines. The orchestration system uses a **location-aware execution model** that prefers in-process communication (Janus) over inter-process (pipes) over network execution, optimizing for performance while maintaining flexibility.

**Key Innovation:** A Prolog-defined location model that makes intelligent decisions about where and how to execute code, automatically choosing the most efficient communication pattern based on runtime capabilities and deployment constraints.

---

## Motivation

### The Multi-Target Reality

Modern data pipelines benefit from different languages' strengths:
- **Bash**: Orchestration, text processing, system integration
- **C#**: LINQ-style queries, enterprise integration, Windows environments
- **Python**: Data science, ML, scientific computing
- **Prolog**: Logic programming, complex rule systems

### Current Limitations

Without orchestration:
- ❌ Each target operates in isolation
- ❌ Manual coordination required (shell scripts, makefiles)
- ❌ No awareness of execution locality
- ❌ Inefficient communication (always uses pipes/files)
- ❌ No automatic optimization

### Vision: Intelligent Orchestration

UnifyWeaver as orchestration layer:
- ✅ Automatic target selection based on task
- ✅ Location-aware execution (same process preferred)
- ✅ Seamless communication across targets
- ✅ Optimization based on deployment context
- ✅ Platform-aware (Termux, WSL, native)

---

## Design Goals

1. **Location Awareness**: System knows where code executes (process, machine)
2. **Preference Hierarchy**: Optimize for in-process > inter-process > network
3. **Transparent Communication**: Developers don't manage pipes/sockets
4. **Platform Agnostic**: Works on Linux, Windows, macOS, Android (Termux)
5. **Fallback Gracefully**: Degrades when optimal path unavailable
6. **Declarative Specification**: Prolog rules define orchestration logic

---

## Location Model

### Location Concept (Prolog Definition)

```prolog
% Location describes where code executes
:- dynamic execution_location/2.

% Locations with preference ordering (lower number = preferred)
location_preference(same_process, 1).      % Janus, in-memory
location_preference(same_machine, 2).      % Pipes, localhost
location_preference(same_network, 3).      % Network, low latency
location_preference(remote, 4).            % Remote network

% Component location facts
execution_location(prolog_runtime, same_process).
execution_location(python_via_janus, same_process).    % When Janus available
execution_location(python_subprocess, same_machine).   % Fallback
execution_location(csharp_local, same_machine).
execution_location(csharp_remote, remote).
```

### Location Detection

```prolog
% Detect if Janus is available
can_use_janus :-
    catch(
        (current_prolog_flag(janus, true),
         py_call(sys:version, _)),
        _,
        fail
    ).

% Determine actual execution location for a target
actual_location(python, same_process) :-
    can_use_janus, !.
actual_location(python, same_machine).

actual_location(csharp, same_machine) :-
    current_prolog_flag(windows, true), !.
actual_location(csharp, same_machine).

actual_location(bash, same_process) :-
    % Can execute bash via process_create in same OS process
    current_prolog_flag(unix, true), !.
actual_location(bash, same_machine).
```

### Preference-Based Selection

```prolog
% Choose best target for a task based on location
choose_target(Task, Target) :-
    findall(
        Pref-T,
        (
            capable_target(Task, T),
            actual_location(T, Loc),
            location_preference(Loc, Pref)
        ),
        Candidates
    ),
    keysort(Candidates, Sorted),
    Sorted = [_-Target|_].

% Example: For Python code, prefer Janus over subprocess
capable_target(execute_python(_Code), python).

% Usage:
?- choose_target(execute_python("print('hello')"), Target).
Target = python.  % Uses Janus if available, else subprocess
```

---

## Communication Patterns

### 1. Same Process (Janus)

**Best for**: Prolog ↔ Python integration

```prolog
% Call Python from Prolog (same process via Janus)
execute_python_janus(Code, Result) :-
    py_call(exec(Code), Result).

% Example: Data science in Python, orchestration in Prolog
analyze_data(Data, Stats) :-
    py_call(numpy:array(Data), Array),
    py_call(numpy:mean(Array), Mean),
    py_call(numpy:std(Array), Std),
    Stats = stats(Mean, Std).
```

**Advantages:**
- ✅ No serialization overhead
- ✅ Shared memory (fast data transfer)
- ✅ Direct exception propagation
- ✅ No subprocess management

**Limitations:**
- ❌ Requires Janus support in SWI-Prolog
- ❌ Python and Prolog must be compatible versions
- ❌ Single-threaded (GIL limitations)

### 2. Inter-Process (Pipes)

**Best for**: Streaming data between different targets

```prolog
% Pipe data from bash to Python to C#
orchestrate_pipeline(InputFile, Results) :-
    % Bash extracts data
    bash_target:compile(extract(InputFile), ExtractScript),

    % Python analyzes (null-delimited JSON)
    python_target:compile(analyze, AnalyzeScript),

    % C# aggregates
    csharp_target:compile(aggregate, AggregateScript),

    % Execute pipeline
    pipeline_execute([
        bash(ExtractScript),
        python(AnalyzeScript),
        csharp(AggregateScript)
    ], Results).

% Pipeline executor handles piping
pipeline_execute([Stage|Rest], Results) :-
    execute_stage(Stage, Output),
    (   Rest = []
    ->  Results = Output
    ;   pipe_to_next(Output, Rest, Results)
    ).
```

**Data Format** (Null-delimited JSON):
```
{"id":1,"value":42}\0
{"id":2,"value":17}\0
{"id":3,"value":99}\0
```

**Advantages:**
- ✅ Language-agnostic streaming
- ✅ Can handle large datasets
- ✅ Targets can be on same or different machines
- ✅ Standard Unix pipe semantics

**Limitations:**
- ❌ Serialization/deserialization overhead
- ❌ Buffering can cause latency
- ❌ Error handling more complex

### 3. Network (Remote Execution)

**Best for**: Distributed computing, cloud functions

```prolog
% Execute on remote machine
execute_remote(Target, Code, Host, Results) :-
    % Compile code for target
    compile_for_target(Target, Code, Script),

    % Transfer and execute
    ssh_execute(Host, Script, RawOutput),

    % Deserialize results
    parse_results(RawOutput, Results).

% Example: Heavy computation on cloud VM
heavy_computation(Data, Results) :-
    execution_location(gpu_cluster, remote),
    execute_remote(python, ml_training(Data), 'gpu-1.cloud', Results).
```

**Advantages:**
- ✅ Access to specialized hardware (GPUs, clusters)
- ✅ Scale beyond local machine
- ✅ Parallel execution across machines

**Limitations:**
- ❌ Network latency
- ❌ Security/authentication concerns
- ❌ Error handling complexity

---

## Target Coordination

### Bash/PowerShell as Orchestration Layer

```bash
#!/bin/bash
# Generated orchestration script

# Stage 1: Extract (bash)
bash extract_data.sh | \

# Stage 2: Analyze (Python via Janus if available, else subprocess)
if command -v swipl &> /dev/null && swipl -g "current_prolog_flag(janus, true)" -t halt; then
    swipl -g "use_module(library(janus)), py_call(analyze:process(stdin), Result)" -t halt
else
    python3 analyze.py
fi | \

# Stage 3: Aggregate (C#)
dotnet run --project aggregate.csproj | \

# Stage 4: Report (bash)
bash generate_report.sh
```

### Prolog as Orchestration Runtime

```prolog
% Orchestration rules
orchestrate_analysis_pipeline(Input, Results) :-
    % Determine optimal execution plan
    choose_target(extract(Input), ExtractTarget),
    choose_target(analyze, AnalyzeTarget),
    choose_target(aggregate, AggregateTarget),

    % Execute with best communication pattern
    execute_with_coordination([
        stage(extract, ExtractTarget, Input),
        stage(analyze, AnalyzeTarget),
        stage(aggregate, AggregateTarget)
    ], Results).

% Coordination logic chooses communication pattern
execute_with_coordination(Stages, Results) :-
    analyze_stages(Stages, Plan),
    execute_plan(Plan, Results).

analyze_stages(Stages, Plan) :-
    % If consecutive stages can use Janus, use it
    % Otherwise fall back to pipes
    partition_by_location(Stages, Groups),
    optimize_communication(Groups, Plan).
```

---

## Platform Awareness

### Termux/Android Optimization

```prolog
% Prefer Python (works well on Termux) over C# (difficult to test)
platform_target_preference(termux, python, 10).
platform_target_preference(termux, bash, 9).
platform_target_preference(termux, csharp, 1).  % Low preference

% Detect platform
detect_platform(termux) :-
    getenv('PREFIX', Prefix),
    sub_string(Prefix, _, _, _, 'com.termux').

detect_platform(wsl) :-
    file_exists('/proc/version'),
    read_file_to_string('/proc/version', Content, []),
    sub_string(Content, _, _, _, 'microsoft').

detect_platform(windows) :-
    current_prolog_flag(windows, true).

% Adjust target selection based on platform
choose_target_for_platform(Task, Target) :-
    detect_platform(Platform),
    findall(
        Score-T,
        (
            capable_target(Task, T),
            platform_target_preference(Platform, T, Score)
        ),
        Scored
    ),
    keysort(Scored, Sorted),
    reverse(Sorted, [_-Target|_]).
```

---

## Implementation Phases

### Phase 1: Foundation (Current)
- ✅ Multiple target languages exist (Bash, C#, Prolog)
- ✅ Janus integration for C# testing
- 🚧 Location model definition
- 🚧 Basic orchestration predicates

### Phase 2: Location Detection (v0.2)
- [ ] Implement location detection predicates
- [ ] Platform detection (Termux, WSL, Windows)
- [ ] Janus availability checking
- [ ] Target capability registry

### Phase 3: Communication Patterns (v0.3)
- [ ] Janus-based Python execution
- [ ] Null-delimited JSON streaming
- [ ] Pipe coordination for multi-stage pipelines
- [ ] Error handling across boundaries

### Phase 4: Python Target (v0.3)
- [ ] Python code generation (see python_target_language.md)
- [ ] Dual mode: Janus vs subprocess
- [ ] Integration with orchestration layer

### Phase 5: Intelligent Orchestration (v0.4)
- [ ] Preference-based target selection
- [ ] Automatic communication pattern selection
- [ ] Platform-aware optimization
- [ ] Fallback handling

### Phase 6: Distributed Execution (v0.5+)
- [ ] Remote execution over SSH
- [ ] Network protocol design
- [ ] Security/authentication
- [ ] Load balancing

---

## Examples

### Example 1: Local Pipeline (Janus Preferred)

```prolog
% User code: ETL pipeline
process_sales_data(CSVFile, Report) :-
    % Extract: Read CSV (Bash)
    extract_csv(CSVFile, RawData),

    % Transform: Clean and normalize (Python via Janus)
    transform_data(RawData, CleanData),

    % Load: Query and aggregate (C# LINQ)
    aggregate_sales(CleanData, Summary),

    % Report: Generate markdown (Bash)
    generate_report(Summary, Report).

% UnifyWeaver orchestration layer chooses:
% - Bash for extract (good at text/files)
% - Python via Janus for transform (same process, no overhead)
% - C# as subprocess for aggregate (LINQ query runtime)
% - Bash for report (text generation)
```

### Example 2: Distributed ML Pipeline

```prolog
% Train ML model on remote GPU cluster
train_model(TrainingData, Model) :-
    % Preprocess locally (Bash)
    preprocess(TrainingData, Processed),

    % Train remotely (Python on GPU cluster)
    execution_location(gpu_cluster, remote),
    execute_remote(
        python,
        train_neural_network(Processed),
        'gpu-cluster.example.com',
        TrainedModel
    ),

    % Validate locally (Python via Janus)
    validate_model(TrainedModel, Metrics),

    Model = model(TrainedModel, Metrics).
```

### Example 3: Fallback Handling

```prolog
% Try Janus, fall back to subprocess
execute_python_safe(Code, Result) :-
    (   can_use_janus
    ->  execute_python_janus(Code, Result)
    ;   format(user_error, 'Janus unavailable, using subprocess~n', []),
        execute_python_subprocess(Code, Result)
    ).

% Subprocess fallback
execute_python_subprocess(Code, Result) :-
    tmp_file_stream(text, ScriptFile, Stream),
    write(Stream, Code),
    close(Stream),
    process_create(path(python3), [ScriptFile], [stdout(pipe(Out))]),
    read_string(Out, _, Result),
    close(Out).
```

---

## Open Questions

1. **Janus Stability**: How reliable is Janus for production use?
2. **Error Propagation**: How to handle errors across Janus boundary?
3. **Performance Overhead**: Measure Janus vs subprocess performance
4. **Version Compatibility**: What Python versions work with Janus?
5. **Threading**: Can we use Python threads via Janus?
6. **State Management**: How to manage state across pipeline stages?
7. **Debugging**: How to debug multi-target pipelines?
8. **Security**: Sandboxing for remote execution?

---

## Related Work

**Similar Systems:**
- **Apache Airflow**: Workflow orchestration (Python-centric)
- **Luigi**: Data pipeline management (Python)
- **Nextflow**: Scientific workflow engine (Groovy DSL)
- **Snakemake**: Workflow management (Python/DSL)

**UnifyWeaver's Niche:**
- **Logic-based orchestration**: Prolog rules define execution
- **Location-aware**: Prefers in-process over IPC over network
- **Multi-paradigm**: Bash, C#, Python, Prolog in one pipeline
- **Platform-adaptive**: Termux, WSL, Windows, Linux

---

## Success Criteria

**Phase 2 (Location Model):**
- ✅ Can detect Janus availability
- ✅ Can detect platform (Termux, WSL, Windows)
- ✅ Location predicates work correctly

**Phase 3 (Communication):**
- ✅ Janus executes Python from Prolog
- ✅ Pipes work for Bash → Python → C#
- ✅ Error handling across boundaries

**Phase 4 (Python Target):**
- ✅ Python code generation working
- ✅ Dual mode (Janus + subprocess) implemented
- ✅ Can run Python in orchestrated pipeline

**Phase 5 (Orchestration):**
- ✅ System chooses Janus over subprocess automatically
- ✅ Platform-aware target selection works
- ✅ Fallback handling robust

---

## References

- SWI-Prolog Janus documentation: https://www.swi-prolog.org/pldoc/man?section=janus
- `tests/core/test_csharp_janus.pl` - Existing Janus integration
- `docs/proposals/python_target_language.md` - Python target design
- `docs/proposals/prolog_as_target_language.md` - Prolog target design

---

**Status:** Draft - Awaiting review and approval
**Next Steps:** Create Python target proposal, implement Phase 2
