# Janus Integration Design

**Status:** Draft
**Date:** 2025-11-21
**Version:** 1.0
**Related:** python_target_language.md, orchestration_architecture.md

---

## Executive Summary

Janus is SWI-Prolog's Python bridge that enables **in-process** execution of Python code from Prolog. This document defines how UnifyWeaver leverages Janus for the Python target, achieving zero-overhead Python execution while maintaining fallback to subprocess mode when Janus is unavailable.

**Key Benefits:**
- ✅ Near-zero startup overhead (< 1ms vs ~50ms subprocess)
- ✅ No serialization cost (shared memory)
- ✅ Direct exception propagation
- ✅ Seamless Prolog ↔ Python data transfer

---

## Janus Basics

### What is Janus?

Janus embeds Python interpreter within SWI-Prolog process:

```prolog
% Check if Janus is available
?- current_prolog_flag(janus, true).
true.

% Execute Python code
?- py_call(print('Hello from Python')).
Hello from Python
true.

% Call Python functions
?- py_call(math:sqrt(16), Result).
Result = 4.0.

% Import modules
?- py_call(importlib:import_module('numpy'), Numpy).
Numpy = <module 'numpy' from '/usr/lib/python3.11/...>.
```

### Data Type Mapping

| Prolog | Python | Notes |
|--------|--------|-------|
| `42` | `42` | Integers map directly |
| `3.14` | `3.14` | Floats map directly |
| `atom` | `'atom'` | Atoms become strings |
| `"string"` | `'string'` | Strings map directly |
| `[1,2,3]` | `[1, 2, 3]` | Lists map to Python lists |
| `_{a:1, b:2}` | `{'a': 1, 'b': 2}` | Dicts map bidirectionally |
| `true`/`false` | `True`/`False` | Booleans map directly |
| `@(none)` | `None` | Special notation for None |

---

## Architecture

### Execution Flow

```
┌──────────────────────────────────────────────────────┐
│ UnifyWeaver Orchestration Layer                      │
│ ┌──────────────────────────────────────────────────┐ │
│ │ choose_python_mode(DataSize, Mode)               │ │
│ │   ├─ can_use_janus? ──→ Yes ──→ janus           │ │
│ │   └─ can_use_janus? ──→ No  ──→ subprocess      │ │
│ └──────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
           │                             │
           ▼                             ▼
    ┌─────────────┐            ┌──────────────────┐
    │ Janus Mode  │            │ Subprocess Mode  │
    │ (in-process)│            │ (pipes/JSON)     │
    └─────────────┘            └──────────────────┘
           │                             │
           ▼                             ▼
    ┌─────────────┐            ┌──────────────────┐
    │   Python    │            │  Python Process  │
    │ Interpreter │            │ (separate PID)   │
    │ (embedded)  │            └──────────────────┘
    └─────────────┘
```

### Mode Selection Logic

```prolog
:- module(python_executor, [
    execute_python/3,
    can_use_janus/0,
    choose_python_mode/2
]).

% Check Janus availability
can_use_janus :-
    catch(
        (   current_prolog_flag(janus, true),
            py_call(sys:version, _)  % Verify Python accessible
        ),
        _Error,
        fail
    ).

% Choose execution mode based on data size and Janus availability
choose_python_mode(DataSize, Mode) :-
    (   can_use_janus,
        DataSize < 10000  % Threshold for in-process (configurable)
    ->  Mode = janus
    ;   Mode = subprocess
    ).

% Unified interface (automatically selects mode)
execute_python(Code, Input, Output) :-
    length(Input, DataSize),
    choose_python_mode(DataSize, Mode),
    execute_python_mode(Mode, Code, Input, Output).

% Dispatch to appropriate executor
execute_python_mode(janus, Code, Input, Output) :-
    execute_python_janus(Code, Input, Output).
execute_python_mode(subprocess, Code, Input, Output) :-
    execute_python_subprocess(Code, Input, Output).
```

---

## Janus Execution Implementation

### Basic Pattern

```prolog
% Execute Python code via Janus
execute_python_janus(Code, Input, Output) :-
    % Define Python function in embedded interpreter
    py_call(exec(Code), _),

    % Call the function with Prolog data
    py_call(process(Input), Output).
```

### Example: Simple Transformation

```prolog
% Prolog side
double_via_janus(Numbers, Doubled) :-
    PythonCode = "
def process(numbers):
    return [x * 2 for x in numbers]
",
    execute_python_janus(PythonCode, Numbers, Doubled).

% Usage
?- double_via_janus([1, 2, 3], Result).
Result = [2, 4, 6].
```

### Example: NumPy Integration

```prolog
% Call NumPy for statistical analysis
analyze_with_numpy(Data, Stats) :-
    % Python code using NumPy
    Code = "
import numpy as np

def process(data):
    arr = np.array(data)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'median': float(np.median(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr))
    }
",
    execute_python_janus(Code, Data, StatsDict),
    % Convert Python dict to Prolog terms
    Stats = stats{
        mean: StatsDict.mean,
        std: StatsDict.std,
        median: StatsDict.median,
        min: StatsDict.min,
        max: StatsDict.max
    }.

% Usage
?- analyze_with_numpy([1,2,3,4,5,6,7,8,9,10], Stats).
Stats = stats{mean:5.5, std:2.872, median:5.5, min:1, max:10}.
```

---

## Advanced Patterns

### Pattern 1: Module Management

```prolog
% Pre-import modules for reuse
:- initialization(setup_python_modules, now).

setup_python_modules :-
    catch(
        (   py_call(importlib:import_module('numpy'), _),
            py_call(importlib:import_module('pandas'), _),
            py_call(importlib:import_module('sklearn'), _),
            format('[Janus] Python modules imported~n', [])
        ),
        Error,
        format(user_error, '[Janus] Warning: ~w~n', [Error])
    ).
```

### Pattern 2: Persistent Python State

```prolog
% Load ML model once, reuse for predictions
:- dynamic ml_model_loaded/1.

load_ml_model(ModelPath) :-
    \+ ml_model_loaded(_),
    Code = "
import pickle

def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

global model
model = None
",
    py_call(exec(Code), _),
    py_call('load_model'(ModelPath), Model),
    py_call(globals().__setitem__('model', Model)),
    assertz(ml_model_loaded(Model)).

% Predict using loaded model
predict_janus(Features, Prediction) :-
    ml_model_loaded(_),
    py_call(model:predict([Features]), [Prediction]).

% Usage
?- load_ml_model('trained_model.pkl').
true.

?- predict_janus([5.1, 3.5, 1.4, 0.2], Class).
Class = 0.  % Iris setosa
```

### Pattern 3: Exception Handling

```prolog
% Safe execution with error handling
safe_execute_python_janus(Code, Input, Output) :-
    catch(
        execute_python_janus(Code, Input, Output),
        error(python_error(Type, Value, _Trace), _),
        (   format(user_error, 'Python error: ~w: ~w~n', [Type, Value]),
            fail
        )
    ).

% Example: Division by zero
?- safe_execute_python_janus("
def process(x):
    return 10 / x
", 0, _).
Python error: 'ZeroDivisionError': division by zero
false.
```

---

## Code Generation Integration

### Generating Janus-Compatible Python

```prolog
% Generate Python code that works in both modes
generate_python_function(PrologPred, PythonCode) :-
    predicate_to_python(PrologPred, FunctionBody),
    format(atom(PythonCode), '~w~n~w', [
        "def process(input_data):",
        FunctionBody
    ]).

% Example: Factorial
predicate_to_python(factorial/2, Code) :-
    Code = "
    if input_data == 0:
        return 1
    else:
        return input_data * process(input_data - 1)
".

% Compile and execute via Janus
compile_and_run_janus(Predicate, Input, Output) :-
    generate_python_function(Predicate, Code),
    execute_python_janus(Code, Input, Output).

% Usage
?- compile_and_run_janus(factorial/2, 5, Result).
Result = 120.
```

### Dual-Mode Code Generation

```prolog
% Generate code that works in both Janus and subprocess modes
generate_python_dual_mode(Predicate, Code) :-
    predicate_to_python(Predicate, FunctionBody),

    % For Janus: Just the function
    JanusCode = FunctionBody,

    % For subprocess: Add stdin/stdout handling
    format(atom(SubprocessCode), '~w

if __name__ == "__main__":
    import sys, json
    # Read from stdin
    input_data = json.load(sys.stdin)
    # Process
    output = process(input_data)
    # Write to stdout
    json.dump(output, sys.stdout)
    sys.stdout.flush()
', [JanusCode]),

    Code = dual_mode{
        janus: JanusCode,
        subprocess: SubprocessCode
    }.
```

---

## Performance Considerations

### When Janus is Faster

✅ **Small datasets** (< 10K records)
- Subprocess overhead dominates
- Serialization cost significant
- Janus: < 1ms startup
- Subprocess: ~50ms startup

✅ **Frequent calls**
- Amortize Python interpreter initialization
- No process creation per call
- State can be cached in Python globals

✅ **Complex Python objects**
- No serialization needed
- Direct memory access
- Rich data structures preserved

### When Subprocess is Better

✅ **Large datasets** (> 100K records)
- Memory pressure in single process
- Streaming via pipes more efficient
- Can parallelize with multiprocessing

✅ **Long-running tasks**
- Avoid blocking Prolog process
- Can monitor progress
- Can kill subprocess if needed

✅ **Janus unavailable**
- Fallback path always works
- Cross-platform compatibility
- No special SWI-Prolog build needed

### Benchmark Example

```prolog
% Benchmark Janus vs Subprocess
benchmark_modes(DataSize) :-
    % Generate test data
    length(Data, DataSize),
    maplist(=(42), Data),

    % Test Janus
    get_time(Start1),
    execute_python_janus("def process(d): return sum(d)", Data, Sum1),
    get_time(End1),
    JanusTime is End1 - Start1,

    % Test Subprocess
    get_time(Start2),
    execute_python_subprocess("
import sys, json
data = json.load(sys.stdin)
print(sum(data))
", Data, Sum2),
    get_time(End2),
    SubprocessTime is End2 - Start2,

    % Report
    format('Data size: ~w~n', [DataSize]),
    format('Janus: ~3f s~n', [JanusTime]),
    format('Subprocess: ~3f s~n', [SubprocessTime]),
    format('Speedup: ~2fx~n', [SubprocessTime / JanusTime]).

% Example results:
% ?- benchmark_modes(1000).
% Data size: 1000
% Janus: 0.001 s
% Subprocess: 0.052 s
% Speedup: 52x

% ?- benchmark_modes(100000).
% Data size: 100000
% Janus: 0.234 s
% Subprocess: 0.189 s
% Speedup: 0.8x  (subprocess wins for large data)
```

---

## Limitations & Workarounds

### Limitation 1: No Stdin/Stdout in Janus

**Problem:** Janus executes in-process, can't pipe stdin/stdout

**Workaround:** Pass data directly as Prolog terms

```prolog
% ❌ Won't work - Janus can't read stdin
execute_python_janus("
import sys
data = sys.stdin.read()
", _, _).

% ✅ Works - Pass data as argument
execute_python_janus("
def process(data):
    # data is already available
    return len(data)
", "hello", Length).
```

### Limitation 2: Python GIL (Global Interpreter Lock)

**Problem:** Can't use Python threads for parallelism via Janus

**Workaround:** Use subprocess mode for parallel tasks

```prolog
% For parallel processing, use subprocess mode
parallel_python_tasks(Tasks, Results) :-
    maplist(execute_python_subprocess, Tasks, Results).
```

### Limitation 3: Module Import Paths

**Problem:** Janus Python may not see all modules

**Workaround:** Add directories to sys.path

```prolog
% Add custom module directory
add_python_path(Dir) :-
    py_call(sys:path:append(Dir), _).

% Usage
?- add_python_path('/path/to/my/modules').
?- py_call(my_module:my_function(), Result).
```

### Limitation 4: State Persistence

**Problem:** Python state persists across Janus calls (can be good or bad)

**Workaround:** Explicitly reset state when needed

```prolog
% Reset Python state
reset_python_state :-
    py_call(globals():clear(), _),
    setup_python_modules.  % Re-import needed modules
```

---

## Testing Strategy

### Unit Tests for Janus

```prolog
:- begin_tests(janus_integration).

test(janus_available) :-
    can_use_janus.

test(simple_call) :-
    py_call(sum([1,2,3]), Sum),
    Sum == 6.

test(data_types) :-
    py_call(type([1,2,3]), Type),
    Type == "<class 'list'>".

test(exception_handling) :-
    catch(
        py_call(undefined_function(), _),
        error(python_error(_, _, _), _),
        true
    ).

test(numpy_integration) :-
    py_call(importlib:import_module('numpy'), _),
    py_call(numpy:array([1,2,3]), Arr),
    py_call(numpy:mean(Arr), Mean),
    Mean =:= 2.0.

:- end_tests(janus_integration).
```

### Integration Tests

```prolog
:- begin_tests(janus_vs_subprocess).

test(mode_selection_small_data) :-
    Data = [1,2,3],
    choose_python_mode(3, Mode),
    Mode == janus.

test(mode_selection_large_data) :-
    length(Data, 100000),
    choose_python_mode(100000, Mode),
    Mode == subprocess.

test(same_results_both_modes) :-
    Data = [1,2,3,4,5],
    execute_python_janus("def process(d): return sum(d)", Data, Sum1),
    execute_python_subprocess("
import sys, json
print(sum(json.load(sys.stdin)))
", Data, Sum2),
    Sum1 == Sum2.

:- end_tests(janus_vs_subprocess).
```

---

## Integration with Python Target

### Code Generation Strategy

```prolog
% Generate Python code with mode awareness
compile_python_target(Predicate, Mode, Code) :-
    (   Mode = janus
    ->  compile_python_janus_mode(Predicate, Code)
    ;   compile_python_subprocess_mode(Predicate, Code)
    ).

% Janus mode: Just function definition
compile_python_janus_mode(Predicate, Code) :-
    generate_python_function(Predicate, FunctionDef),
    Code = FunctionDef.

% Subprocess mode: Function + stdin/stdout wrapper
compile_python_subprocess_mode(Predicate, Code) :-
    generate_python_function(Predicate, FunctionDef),
    format(atom(Code), '~w

if __name__ == "__main__":
    import sys, json
    buffer = ""
    for chunk in iter(lambda: sys.stdin.read(1), ""):
        if chunk == "\\0":
            if buffer:
                record = json.loads(buffer)
                result = process(record)
                if result is not None:
                    print(json.dumps(result), end="\\0", flush=True)
                buffer = ""
        else:
            buffer += chunk
', [FunctionDef]).
```

---

## Deployment Considerations

### Development Environment

```prolog
% Check Janus availability at startup
:- initialization(check_janus_on_startup, now).

check_janus_on_startup :-
    (   can_use_janus
    ->  format('[UnifyWeaver] Janus available - Python in-process mode enabled~n', [])
    ;   format('[UnifyWeaver] Janus unavailable - Python subprocess mode only~n', [])
    ).
```

### Production Environment

```prolog
% Production: Prefer Janus but don't fail if unavailable
:- dynamic python_mode_preference/1.

set_python_preference(Preference) :-
    retractall(python_mode_preference(_)),
    assertz(python_mode_preference(Preference)).

% Default: Auto (Janus if available, else subprocess)
:- set_python_preference(auto).

% Force subprocess (for testing or when Janus unstable)
% ?- set_python_preference(subprocess_only).

% Force Janus (fail if unavailable - for debugging)
% ?- set_python_preference(janus_only).
```

---

## Future Enhancements

### 1. Janus Pool for Parallelism

```prolog
% Idea: Multiple Python interpreters via Janus
% (Not yet supported by Janus, but possible future feature)

:- dynamic janus_pool/2.

init_janus_pool(PoolSize) :-
    % Create multiple Python interpreter instances
    % Distribute work across pool
    % Bypass GIL limitation
    ...
```

### 2. Hybrid Mode (Janus + Subprocess)

```prolog
% Use Janus for control flow, subprocess for heavy lifting
hybrid_execution(Code, Data, Result) :-
    % Janus: Decide what to do
    py_call(analyze_task(Data), TaskType),

    % Dispatch based on task type
    (   TaskType = lightweight
    ->  execute_python_janus(Code, Data, Result)
    ;   execute_python_subprocess(Code, Data, Result)
    ).
```

### 3. Async Janus Calls

```prolog
% Non-blocking Python execution via Janus
% (Future work - requires Janus async support)
async_py_call(Goal, Promise) :-
    % Start Python execution in background
    % Return promise for future result
    ...
```

---

## References

- SWI-Prolog Janus documentation: https://www.swi-prolog.org/pldoc/man?section=janus
- `tests/core/test_csharp_janus.pl` - Existing Janus usage for C# testing
- Python C API: https://docs.python.org/3/c-api/
- `docs/proposals/python_target_language.md` - Python target design

---

**Status:** Draft - Ready for implementation
**Next Steps:** Implement Janus mode for Python target (Phase 2)
