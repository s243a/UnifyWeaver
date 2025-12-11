# Proposal: Python Object Pipeline Support

**Status:** Draft
**Date:** 2025-12-10
**Author:** Claude Code (Opus 4.5)

---

## Overview

This proposal outlines the design for Python object pipeline support, including runtime selection (CPython, IronPython, Jython, PyPy) based on integration context. The design must integrate with UnifyWeaver's cross-target glue system.

## Problem Statement

Currently, Python target compilation generates standalone scripts. However:

1. **No streaming pipeline support** - Unlike PowerShell's `ValueFromPipeline`, Python predicates can't easily chain
2. **No runtime selection** - Always assumes CPython, but IronPython is better for .NET integration
3. **No structured output** - Results are text-based, not typed objects
4. **Cross-target friction** - Python ↔ C# ↔ PowerShell requires manual glue code

## Goals

1. Add pipeline input/output options mirroring PowerShell target
2. Implement automatic runtime selection based on integration context
3. Enable seamless cross-target glue with .NET targets
4. Maintain backward compatibility with existing Python compilation

---

## Design

### 1. Pipeline Options (Phase 1)

Mirror PowerShell's pipeline options:

```prolog
compile_predicate_to_python(user_info/2, [
    pipeline_input(true),           % Enable streaming input
    output_format(object),          % Yield typed dicts instead of text
    arg_names(['UserId', 'Email']), % Property names for output
    runtime(auto)                   % Auto-select runtime
], Code).
```

#### Generated Python (CPython mode)

```python
from typing import Iterator, Dict, Any, Generator

def user_info(stream: Iterator[Dict]) -> Generator[Dict, None, None]:
    """Pipeline-enabled predicate with structured output."""
    for record in stream:
        user_id = record.get('UserId')
        if user_id:
            # Predicate logic here
            yield {'UserId': user_id, 'Email': f'{user_id}@example.com'}
```

### 2. Runtime Selection (Phase 2)

#### Runtime Options

| Runtime | Use Case | Integration |
|---------|----------|-------------|
| `cpython` | Default, standalone scripts | Shell pipes, JSONL |
| `ironpython` | .NET integration | In-process with C#/PowerShell |
| `jython` | JVM integration | In-process with Java |
| `pypy` | Performance-critical | Shell pipes, faster execution |
| `auto` | Context-dependent | Selected by cross-target glue |

#### Selection Logic

```prolog
%% select_python_runtime(+Context, -Runtime)
%  Select optimal Python runtime based on integration context

select_python_runtime(Context, Runtime) :-
    (   member(target(csharp), Context)
    ->  Runtime = ironpython
    ;   member(target(powershell), Context)
    ->  Runtime = ironpython  % PowerShell can host IronPython in-process
    ;   member(target(java), Context)
    ->  Runtime = jython
    ;   member(performance(critical), Context)
    ->  Runtime = pypy
    ;   Runtime = cpython
    ).
```

### 3. Cross-Target Glue Integration (Phase 3)

#### Scenario: Python Predicate Called from C#

```prolog
% User defines predicate
user_info(Id, Email) :-
    lookup_user(Id, Email).

% Compilation with .NET context
compile_predicate_to_python(user_info/2, [
    runtime(auto),
    glue_context([target(csharp)])
], Code).
```

#### Generated IronPython Code

```python
# Generated for IronPython (.NET integration)
import clr
clr.AddReference('System.Collections')
from System.Collections.Generic import Dictionary

def user_info(user_id):
    """IronPython predicate callable from C#."""
    # Returns .NET Dictionary instead of Python dict
    result = Dictionary[str, object]()
    result['UserId'] = user_id
    result['Email'] = f'{user_id}@example.com'
    return result
```

#### C# Hosting Code

```csharp
// Generated glue code for hosting IronPython
using IronPython.Hosting;
using Microsoft.Scripting.Hosting;

public class UserInfoPredicate
{
    private readonly ScriptEngine _engine;
    private readonly ScriptScope _scope;
    private readonly dynamic _userInfo;

    public UserInfoPredicate()
    {
        _engine = Python.CreateEngine();
        _scope = _engine.CreateScope();
        _engine.ExecuteFile("user_info.py", _scope);
        _userInfo = _scope.GetVariable("user_info");
    }

    public IDictionary<string, object> Invoke(string userId)
    {
        return _userInfo(userId);
    }
}
```

### 4. Pipeline Chaining

#### Same-Runtime Chaining (Efficient)

```prolog
% Both predicates compiled for same runtime
get_users(Users) :- ...
filter_active(User, Active) :- ...

% Pipeline: get_users | filter_active
pipeline([get_users/1, filter_active/2], [runtime(cpython)], Code).
```

**Generated:**
```python
def pipeline(input_stream):
    users = get_users(input_stream)
    active = (filter_active(u) for u in users if filter_active(u))
    yield from active
```

#### Cross-Runtime Chaining (Via Glue)

```prolog
% Python predicate -> C# predicate -> PowerShell predicate
pipeline([
    python:get_users/1,
    csharp:validate_user/1,
    powershell:send_notification/1
], [glue(auto)], Code).
```

**Generated glue uses:**
- IronPython for Python ↔ C# (in-process)
- PowerShell hosting from C# (in-process)
- Or JSONL serialization if runtimes can't share process

---

## Runtime-Specific Considerations

### CPython

**Pros:**
- Most libraries available (numpy, pandas, etc.)
- Best tooling and debugging
- Default choice for standalone scripts

**Cons:**
- Cross-process only for .NET integration
- Serialization overhead for structured data

**Generated Code Pattern:**
```python
#!/usr/bin/env python3
import sys
import json

def process_stream(stream):
    for line in stream:
        record = json.loads(line)
        # Process...
        yield result

if __name__ == '__main__':
    for result in process_stream(sys.stdin):
        print(json.dumps(result))
```

### IronPython

**Pros:**
- In-process .NET integration
- Direct access to .NET types
- No serialization for C#/PowerShell calls

**Cons:**
- Python 2.7 compatible only (IronPython 2.x)
- IronPython 3 is in development but not stable
- Limited library support (no numpy, etc.)

**Generated Code Pattern:**
```python
# IronPython - .NET integration
import clr
clr.AddReference('System')
clr.AddReference('System.Core')
from System import *
from System.Collections.Generic import *

def process(input_dict):
    # Works with .NET Dictionary directly
    result = Dictionary[String, Object]()
    # Process...
    return result
```

### Jython

**Pros:**
- In-process JVM integration
- Direct access to Java classes

**Cons:**
- Python 2.7 only
- Limited library support

### PyPy

**Pros:**
- JIT compilation for speed
- Compatible with most pure-Python code

**Cons:**
- Some C extensions don't work
- Larger memory footprint

---

## Implementation Plan

### Phase 1: Pipeline Options (Standalone)
- Add `pipeline_input/1`, `output_format/1`, `arg_names/1` options
- Generate streaming generator code
- Works with CPython only
- ~2-3 days implementation

### Phase 2: Runtime Selection
- Add `runtime/1` option
- Implement `select_python_runtime/2`
- Generate runtime-specific headers/imports
- ~2-3 days implementation

### Phase 3: IronPython Support
- Generate IronPython-compatible code
- .NET type integration (Dictionary, List)
- CLR reference management
- ~3-4 days implementation

### Phase 4: Cross-Target Glue
- Integrate with existing glue system
- Generate hosting code for C#/PowerShell
- Automatic glue selection
- ~4-5 days implementation

### Phase 5: Pipeline Chaining
- Same-runtime pipeline optimization
- Cross-runtime glue generation
- ~3-4 days implementation

---

## Open Questions

### 1. IronPython 2 vs 3?

IronPython 2.7 is stable but Python 2 only. IronPython 3 supports Python 3 but is still in development. Options:

- **Option A:** Support IronPython 2.7 only (stable, but Python 2 syntax)
- **Option B:** Support IronPython 3 (Python 3 syntax, but less stable)
- **Option C:** Generate compatible subset that works with both

**Recommendation:** Option A initially, with Option C as target.

### 2. Binding Compatibility?

Current bindings assume CPython. For IronPython:

- Math bindings: Use `System.Math` instead of `math` module?
- I/O bindings: Use `System.IO` instead of Python `open()`?
- Regex bindings: Use `System.Text.RegularExpressions`?

**Options:**
- **Option A:** Separate binding sets per runtime
- **Option B:** Binding options specify runtime-specific targets
- **Option C:** Runtime-agnostic bindings with translation layer

**Recommendation:** Option B - extend binding/6 with runtime-specific options:

```prolog
declare_binding(python, sqrt/2, 'math.sqrt',
    [number], [float],
    [
        pure, import('math'),
        ironpython_target('[Math]::Sqrt'),  % .NET fallback
        jython_target('java.lang.Math.sqrt') % JVM fallback
    ]).
```

### 3. Glue Protocol?

When runtimes can't share process, what serialization?

- **JSONL:** Universal, but verbose
- **MessagePack:** Binary, efficient, but needs library
- **Protocol Buffers:** Typed, efficient, but complex setup
- **Pickle:** Python-native, but security concerns

**Recommendation:** JSONL as default (universal), MessagePack as option.

### 4. Error Handling Across Runtimes?

How do exceptions propagate across runtime boundaries?

**Options:**
- **Option A:** Convert all to JSON error objects
- **Option B:** Runtime-specific exception translation
- **Option C:** Wrapper types that serialize exception info

**Recommendation:** Option A for simplicity.

### 5. Integration with Existing Firewall/Preference System

UnifyWeaver already has a Control Plane with **Firewall** (hard security boundaries) and **Preferences** (flexible implementation choices). See `docs/CONTROL_PLANE.md` and `docs/ARCHITECTURE.md`.

**The Problem:** Runtime selection involves both:
- **Firewall concerns** - "IronPython is denied" (security)
- **Preference concerns** - "Prefer in-process over cross-process" (optimization)

And conflicting preferences, e.g., "prefer CPython" vs "prefer in-process communication".

**Solution: Integrate with Existing System**

#### Firewall Layer (Hard Constraints)

Use existing `firewall_implies/2` for runtime constraints:

```prolog
%% New firewall implications for Python runtime
firewall_implies_default(ironpython_unavailable,
                        denied(python_runtime(ironpython))).

firewall_implies_default(require_numpy,
                        denied(python_runtime(ironpython))).  % numpy needs CPython

firewall_implies_default(require_numpy,
                        denied(python_runtime(pypy))).        % numpy C extension issues

firewall_implies_default(environment(no_dotnet),
                        denied(python_runtime(ironpython))).

firewall_implies_default(environment(no_jvm),
                        denied(python_runtime(jython))).

%% Predicate-specific firewall rules
:- assertz(firewall:rule_firewall(data_science_pred/2, [
    denied([python_runtime(ironpython), python_runtime(jython)]),  % needs numpy
    services([python3])
])).
```

#### Preference Layer (Soft Preferences)

Extend `rule_preferences/2` with new preference terms:

```prolog
%% New preference terms for Python runtime
%%
%% - prefer_runtime([cpython, ironpython, pypy]) - Runtime preference order
%% - prefer_communication(in_process | cross_process) - IPC preference
%% - optimization(latency | throughput | memory) - Performance hint
%% - fallback_runtime([...]) - Order to try if preferred fails

%% Global default: prefer in-process when available
:- assertz(preferences:preferences_default([
    prefer_communication(in_process),
    prefer_runtime([ironpython, cpython, pypy]),  % in-process first
    fallback_runtime([cpython])                   % always have fallback
])).

%% Predicate-specific: this one needs CPython for library support
:- assertz(preferences:rule_preferences(ml_predict/2, [
    prefer_runtime([cpython]),           % needs full library support
    optimization(throughput)
])).

%% Predicate-specific: this one is latency-sensitive
:- assertz(preferences:rule_preferences(realtime_filter/2, [
    prefer_communication(in_process),    % minimize latency
    prefer_runtime([ironpython, pypy]),  % fast runtimes
    optimization(latency)
])).
```

#### Selection Algorithm

The selection algorithm respects the existing merge precedence (Runtime > Rule > Default):

```prolog
%% select_python_runtime(+PredIndicator, +Imports, +Context, -Runtime)
%
% Selects Python runtime respecting firewall and preferences.
%
select_python_runtime(PredIndicator, Imports, Context, Runtime) :-
    % 1. Get merged preferences (uses existing preferences:get_final_options/3)
    preferences:get_final_options(PredIndicator, Context, FinalOptions),

    % 2. Get firewall policy (uses existing firewall:get_firewall_policy/2)
    firewall:get_firewall_policy(PredIndicator, Firewall),

    % 3. Determine candidate runtimes (filtered by hard constraints)
    findall(R, valid_runtime_candidate(R, Imports, Firewall), Candidates),
    Candidates \= [],

    % 4. Score candidates against preferences
    score_runtime_candidates(Candidates, FinalOptions, ScoredCandidates),

    % 5. Select best (or use fallback order)
    select_best_runtime(ScoredCandidates, FinalOptions, Runtime).

%% valid_runtime_candidate(+Runtime, +Imports, +Firewall) is semidet.
%
% Check if runtime passes all hard constraints.
%
valid_runtime_candidate(Runtime, Imports, Firewall) :-
    member(Runtime, [cpython, ironpython, pypy, jython]),
    % Not explicitly denied in firewall
    \+ member(denied(python_runtime(Runtime)), Firewall),
    % Runtime is available on system
    runtime_available(Runtime),
    % Compatible with required imports
    (   Runtime = ironpython
    ->  dotnet_glue:can_use_ironpython(Imports)
    ;   Runtime = jython
    ->  can_use_jython(Imports)  % Similar check for JVM
    ;   true  % CPython/PyPy support everything
    ).

%% score_runtime_candidates(+Candidates, +Options, -Scored)
%
% Score each candidate against preference dimensions.
%
score_runtime_candidates(Candidates, Options, Scored) :-
    maplist(score_single_runtime(Options), Candidates, Scored).

score_single_runtime(Options, Runtime, Runtime-Score) :-
    % Base score from preference order
    (   member(prefer_runtime(Order), Options),
        nth0(Idx, Order, Runtime)
    ->  OrderScore is 10 - Idx  % Higher = better
    ;   OrderScore = 0
    ),

    % Communication preference bonus
    (   member(prefer_communication(Comm), Options),
        runtime_communication(Runtime, Comm)
    ->  CommScore = 5
    ;   CommScore = 0
    ),

    % Optimization hint bonus
    (   member(optimization(Opt), Options),
        runtime_optimization(Runtime, Opt)
    ->  OptScore = 3
    ;   OptScore = 0
    ),

    % Metrics bonus (if available)
    metrics_bonus(Runtime, MetricsBonus),

    Score is OrderScore + CommScore + OptScore + MetricsBonus.

%% Runtime characteristics (static facts)
runtime_communication(ironpython, in_process).   % When called from .NET
runtime_communication(cpython, cross_process).   % Via pipes/JSONL
runtime_communication(pypy, cross_process).
runtime_communication(jython, in_process).       % When called from JVM

runtime_optimization(pypy, throughput).          % JIT for long-running
runtime_optimization(pypy, latency).             % Fast after warmup
runtime_optimization(ironpython, latency).       % No serialization
runtime_optimization(cpython, memory).           % Most efficient
```

#### Metrics-Driven Adaptation (Future)

Extend the system with runtime metrics that feed back into selection:

```prolog
%% runtime_metric(+Runtime, +Metric, -Value) is nondet.
%
% Dynamic metrics collected from execution history.
%
:- dynamic runtime_metric/3.

%% Record execution outcome
record_runtime_execution(Runtime, DurationMs, Success) :-
    % Update success rate
    update_success_rate(Runtime, Success),
    % Update latency moving average
    update_latency_average(Runtime, DurationMs).

%% metrics_bonus(+Runtime, -Bonus)
%
% Calculate bonus score from historical metrics.
%
metrics_bonus(Runtime, Bonus) :-
    (   runtime_metric(Runtime, success_rate, SuccessRate),
        runtime_metric(Runtime, avg_latency_ms, AvgLatency)
    ->  % Higher success rate = higher bonus (0-2 points)
        SuccessBonus is SuccessRate * 2,
        % Lower latency = higher bonus (0-2 points, inverse scale)
        LatencyBonus is max(0, 2 - (AvgLatency / 100)),
        Bonus is SuccessBonus + LatencyBonus
    ;   Bonus = 0  % No metrics yet
    ).
```

#### Example Usage

```prolog
%% Configuration
:- assertz(firewall:firewall_default([
    execution([python, bash]),
    denied([python_runtime(jython)])  % No JVM in this environment
])).

:- assertz(preferences:preferences_default([
    prefer_communication(in_process),
    prefer_runtime([ironpython, cpython]),
    optimization(latency)
])).

%% Compilation
?- compile_predicate_to_python(my_filter/2, [runtime(auto)], Code).
% System:
%   1. Checks firewall: jython denied, others allowed
%   2. Checks preferences: prefer ironpython, then cpython
%   3. Checks imports: my_filter uses 'json', 're' (IronPython compatible)
%   4. Scores: ironpython=15, cpython=10
%   5. Selects: ironpython
%   6. Generates: IronPython-compatible code + C# hosting glue

%% Override at runtime
?- compile_predicate_to_python(my_filter/2, [prefer_runtime([cpython])], Code).
% Runtime option overrides default, selects cpython
```

This design:
1. **Reuses existing infrastructure** - No new preference/firewall systems
2. **Maintains separation** - Firewall for security, Preferences for optimization
3. **Follows merge precedence** - Runtime > Rule > Default
4. **Supports metrics** - Future adaptation based on execution history
5. **Is extensible** - New dimensions can be added to preferences

---

## Already Implemented (dotnet_glue.pl)

Analysis of `src/unifyweaver/glue/dotnet_glue.pl` reveals significant infrastructure already exists:

### Runtime Detection

| Predicate | Purpose |
|-----------|---------|
| `detect_dotnet_runtime/1` | Returns `dotnet_modern`, `dotnet_core`, `mono`, or `none` |
| `detect_ironpython/1` | Checks for `ipy` or `ipy64` executables |
| `detect_powershell/1` | Returns `core(Version)`, `windows(Version)`, or `none` |

### IronPython Compatibility

| Predicate | Purpose |
|-----------|---------|
| `ironpython_compatible/1` | 30+ modules declared (sys, os, json, re, collections, clr...) |
| `can_use_ironpython/1` | Check if all imports are IronPython-compatible |
| `python_runtime_choice/2` | Returns `ironpython` or `cpython_pipe` based on imports |

**Incompatible modules listed:** numpy, scipy, pandas, matplotlib, PIL/pillow, cv2, h5py, tensorflow, torch, sklearn

### Bridge Generation

| Predicate | Generated Code |
|-----------|----------------|
| `generate_ironpython_bridge/2` | C# class with `ExecuteStream<TInput>` (uses `yield return`) |
| `generate_cpython_bridge/2` | C# class with `ExecuteStream<TInput, TOutput>` (uses pipes + JSONL) |
| `generate_powershell_bridge/2` | C# class with `InvokeStream<TInput, TOutput>` |
| `generate_csharp_host/3` | Multi-target host combining all bridges |
| `generate_dotnet_pipeline/3` | Pipeline with `Step1`, `Step2`, ... methods |

### What This Means for Implementation

**Already done (can reuse):**
- Runtime detection (Phase 2 partial)
- IronPython compatibility checking (Phase 2)
- C# hosting code generation (Phase 3/4 partial)
- Pipeline chaining across targets (Phase 5 partial)

**Still needed:**
- Pipeline options in `python_target.pl` (Phase 1) - `pipeline_input`, `output_format`, `arg_names`
- Python generator code generation (Phase 1) - the `def foo(stream): yield from ...` pattern
- Integration of `python_runtime_choice/2` into `compile_predicate_to_python/3` (Phase 2)
- Binding registry extension for runtime-specific targets (Phase 3)

---

## Relationship to Existing Systems

### Cross-Target Glue (docs/guides/cross-target-glue.md)

The Python runtime selection integrates with:

- **Phase 1:** Shell integration (CPython with JSONL)
- **Phase 4:** .NET bridges (IronPython in-process)
- **Phase 6:** Service registry (runtime as service metadata)

### PowerShell Target (v2.5)

Mirror the options but adapt for Python:

| PowerShell Option | Python Equivalent |
|-------------------|-------------------|
| `pipeline_input(true)` | `pipeline_input(true)` |
| `output_format(object)` | `output_format(object)` |
| `arg_names([...])` | `arg_names([...])` |
| `powershell_mode(pure\|baas)` | `runtime(cpython\|ironpython\|auto)` |

### Binding Registry

Extend bindings for runtime awareness:

```prolog
% Current
binding(python, Pred, Target, Inputs, Outputs, Options).

% Extended (backward compatible)
binding(python, Pred, Target, Inputs, Outputs, Options) :-
    % Options may include:
    % - runtime_target(ironpython, DotNetTarget)
    % - runtime_target(jython, JvmTarget)
```

---

## Success Criteria

1. **Pipeline options work** - `pipeline_input`, `output_format`, `arg_names` generate correct code
2. **Runtime selection works** - `runtime(auto)` selects appropriate runtime
3. **IronPython integration** - Python predicates callable from C# in-process
4. **Cross-target glue** - Automatic glue generation for Python ↔ .NET
5. **Backward compatible** - Existing Python compilation unchanged
6. **Tests pass** - Comprehensive test coverage for all patterns

---

## Appendix: Example Workflows

### Workflow 1: Standalone Python Pipeline

```prolog
% Define predicates
user_data(Id, Name, Email) :- ...
filter_domain(Email, Domain, Filtered) :- ...

% Compile with pipeline support
compile_predicate_to_python(user_data/3, [
    pipeline_input(true),
    output_format(object),
    arg_names(['Id', 'Name', 'Email'])
], Code1).

compile_predicate_to_python(filter_domain/3, [
    pipeline_input(true),
    output_format(object)
], Code2).
```

**Usage:**
```bash
cat users.jsonl | python user_data.py | python filter_domain.py --domain=example.com
```

### Workflow 2: Python in .NET Application

```prolog
% Compile for .NET integration
compile_predicate_to_python(process_order/2, [
    runtime(ironpython),
    output_format(object),
    glue_target(csharp)
], PythonCode, GlueCode).
```

**C# Usage:**
```csharp
var processor = new ProcessOrderPredicate();
var result = processor.Invoke(orderId);
Console.WriteLine(result["Status"]);
```

### Workflow 3: Mixed Pipeline (Python + C# + PowerShell)

```prolog
% Define cross-target pipeline
compile_pipeline([
    {python, extract_data/1, [runtime(ironpython)]},
    {csharp, validate_data/1, []},
    {powershell, notify_admin/1, []}
], [glue(in_process)], PipelineCode).
```

**Generated:** Single .NET assembly with IronPython, C#, and PowerShell hosted together.

---

## Next Steps

1. **Review proposal** - Get feedback on design decisions
2. **Resolve open questions** - Decide on IronPython version, binding strategy
3. **Phase 1 implementation** - Pipeline options for CPython
4. **Iterate** - Add runtime selection and cross-target features

---

## References

- [PowerShell Target Guide](../POWERSHELL_TARGET.md)
- [Cross-Target Glue Guide](../guides/cross-target-glue.md)
- [Python Target Guide](../PYTHON_TARGET.md)
- [Binding Registry](../proposals/BINDING_PREDICATE_PROPOSAL.md)
- [IronPython Documentation](https://ironpython.net/)
