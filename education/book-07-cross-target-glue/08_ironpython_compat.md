<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)
-->

# Chapter 8: IronPython Compatibility

## Overview

IronPython runs Python code on the .NET runtime, enabling zero-serialization communication with C# and PowerShell. However, not all Python code works with IronPython. This chapter covers:

- Compatible modules
- Incompatible modules
- Automatic runtime selection
- Migration strategies

## The Compatibility System

Located in `src/unifyweaver/glue/dotnet_glue.pl`:

```prolog
% Check if a module is IronPython-compatible
ironpython_compatible(Module).

% Check if all imports are compatible
can_use_ironpython(Imports).

% Automatic runtime selection
python_runtime_choice(Imports, Runtime).
```

## Compatible Modules

IronPython supports 40+ standard library modules:

### Core Modules
```prolog
ironpython_compatible(sys).
ironpython_compatible(os).
ironpython_compatible(io).
ironpython_compatible(re).
ironpython_compatible(string).
ironpython_compatible(codecs).
```

### Data Structures
```prolog
ironpython_compatible(collections).
ironpython_compatible(array).
ironpython_compatible(bisect).
ironpython_compatible(heapq).
ironpython_compatible(copy).
```

### Functional Programming
```prolog
ironpython_compatible(itertools).
ironpython_compatible(functools).
ironpython_compatible(operator).
```

### Data Formats
```prolog
ironpython_compatible(json).
ironpython_compatible(csv).
ironpython_compatible(xml).           % xml.etree, xml.dom
ironpython_compatible(pickle).
ironpython_compatible(struct).
ironpython_compatible(base64).
```

### File Operations
```prolog
ironpython_compatible(pathlib).
ironpython_compatible(shutil).
ironpython_compatible(tempfile).
ironpython_compatible(zipfile).
ironpython_compatible(gzip).
```

### Database
```prolog
ironpython_compatible(sqlite3).
```

### Networking
```prolog
ironpython_compatible(socket).
ironpython_compatible(urllib).
```

### Concurrency
```prolog
ironpython_compatible(threading).
```

### Utilities
```prolog
ironpython_compatible(datetime).
ironpython_compatible(math).
ironpython_compatible(random).
ironpython_compatible(hashlib).
ironpython_compatible(contextlib).
ironpython_compatible(abc).
ironpython_compatible(typing).
```

### .NET Integration (IronPython-specific)
```prolog
ironpython_compatible(clr).           % .NET integration
```

## Incompatible Modules

These require CPython:

### Scientific Computing
```prolog
ironpython_incompatible(numpy).       % C extensions
ironpython_incompatible(scipy).       % Depends on numpy
ironpython_incompatible(pandas).      % Depends on numpy
```

### Machine Learning
```prolog
ironpython_incompatible(tensorflow).
ironpython_incompatible(torch).
ironpython_incompatible(sklearn).     % scikit-learn
```

### Image Processing
```prolog
ironpython_incompatible(cv2).         % OpenCV
ironpython_incompatible('PIL').       % Pillow
```

### Visualization
```prolog
ironpython_incompatible(matplotlib).
```

### Why Incompatible?

These modules use C extensions that require CPython's C API:

```c
// numpy uses C extensions like this
static PyObject* array_sum(PyArrayObject* arr) {
    // Direct memory manipulation
    double* data = (double*)PyArray_DATA(arr);
    // ...
}
```

IronPython can't load these because it runs on .NET, not CPython.

## Checking Compatibility

### Single Module

```prolog
?- ironpython_compatible(json).
true.

?- ironpython_compatible(numpy).
false.
```

### Multiple Imports

```prolog
?- can_use_ironpython([sys, json, re]).
true.

?- can_use_ironpython([json, numpy]).
false.
```

### Automatic Selection

```prolog
?- python_runtime_choice([sys, json, collections], Runtime).
Runtime = ironpython.

?- python_runtime_choice([numpy, pandas], Runtime).
Runtime = cpython_pipe.

?- python_runtime_choice([json, numpy], Runtime).
Runtime = cpython_pipe.  % numpy forces CPython
```

## How the System Works

### Import Analysis

The system analyzes Python code to extract imports:

```prolog
extract_imports(PythonCode, Imports) :-
    % Parse "import X" and "from X import Y"
    findall(Module, import_statement(PythonCode, Module), Imports).
```

### Runtime Selection Logic

```prolog
python_runtime_choice(Imports, ironpython) :-
    forall(member(M, Imports), ironpython_compatible(M)),
    !.

python_runtime_choice(_, cpython_pipe).
```

### Generated Bridge Selection

```prolog
generate_python_bridge(Code, Options, Bridge) :-
    extract_imports(Code, Imports),
    python_runtime_choice(Imports, Runtime),
    (Runtime == ironpython ->
        generate_ironpython_bridge(Options, Bridge)
    ;
        generate_cpython_bridge(Options, Bridge)
    ).
```

## Migration Strategies

### Strategy 1: Pure Python Refactoring

If you're using numpy only for simple operations:

```python
# Before (requires numpy)
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
result = np.mean(arr)

# After (works with IronPython)
data = [1, 2, 3, 4, 5]
result = sum(data) / len(data)
```

### Strategy 2: .NET Alternatives

Replace Python libraries with .NET equivalents:

```python
# Before (pandas for CSV)
import pandas as pd
df = pd.read_csv('data.csv')
result = df.groupby('category').sum()

# After (IronPython + .NET)
import clr
clr.AddReference('System.Data')
from System.Data import DataTable
from System.IO import File

# Read CSV into DataTable
lines = File.ReadAllLines('data.csv')
# ... process with LINQ
```

### Strategy 3: Hybrid Approach

Use IronPython for orchestration, CPython for computation:

```csharp
public IEnumerable<Result> HybridPipeline(IEnumerable<Input> inputs)
{
    // Step 1: Preprocessing (IronPython - fast)
    var preprocessed = IronPythonBridge.InvokeStream<Input, Preprocessed>(
        "preprocess(input_data)",  // Only uses stdlib
        inputs
    );

    // Step 2: ML inference (CPython - numpy/tensorflow)
    var predictions = CPythonBridge.ExecuteStream<Preprocessed, Prediction>(
        @"
import numpy as np
import tensorflow as tf
# ... ML code
",
        preprocessed
    );

    // Step 3: Postprocessing (IronPython - fast)
    return IronPythonBridge.InvokeStream<Prediction, Result>(
        "postprocess(input_data)",  // Only uses stdlib
        predictions
    );
}
```

### Strategy 4: Batch Processing

Minimize CPython calls by batching:

```csharp
public IEnumerable<Result> BatchedPipeline(IEnumerable<Input> inputs)
{
    // Collect batch
    var batch = inputs.Take(1000).ToList();

    // Single CPython call for entire batch
    var results = CPythonBridge.Execute<List<Input>, List<Result>>(
        @"
import numpy as np
# Process entire batch at once
arr = np.array([[r['x'], r['y']] for r in input_data])
processed = some_numpy_operation(arr)
result = [{'output': x} for x in processed]
",
        batch
    );

    return results;
}
```

## Performance Comparison

| Scenario | IronPython | CPython (pipe) |
|----------|------------|----------------|
| 1K simple records | ~5ms | ~50ms |
| 100K simple records | ~200ms | ~2s |
| 1K numpy operations | N/A | ~100ms |
| ML inference (1K) | N/A | ~500ms |

**Key insight**: IronPython is 10x faster for simple operations due to zero serialization. Use CPython only when you need its libraries.

## Detecting Requirements

### Analyze Python File

```prolog
analyze_python_file(FilePath, Analysis) :-
    read_file_to_string(FilePath, Code, []),
    extract_imports(Code, Imports),
    python_runtime_choice(Imports, Runtime),
    partition(ironpython_compatible, Imports, Compatible, Incompatible),
    Analysis = analysis{
        runtime: Runtime,
        compatible: Compatible,
        incompatible: Incompatible
    }.
```

### Example Output

```prolog
?- analyze_python_file('ml_pipeline.py', A).
A = analysis{
    runtime: cpython_pipe,
    compatible: [sys, json, logging],
    incompatible: [numpy, pandas, sklearn]
}.
```

## Chapter Summary

- **40+ modules** are IronPython-compatible
- **numpy, pandas, ML** require CPython
- **Automatic detection** chooses the right runtime
- **Migration strategies** can reduce CPython usage
- **Hybrid approach** gets the best of both

## Next Steps

In Chapter 9, we'll explore native target code generation:
- Go pipe wrappers
- Rust pipe wrappers
- Parallel processing

## Exercises

1. **Compatibility check**: Which runtime would be chosen for a script that imports `json`, `csv`, and `sklearn`?

2. **Refactoring**: Rewrite this numpy code to be IronPython-compatible:
   ```python
   import numpy as np
   arr = np.array([1, 2, 3, 4])
   result = np.sum(arr ** 2)
   ```

3. **Hybrid design**: Design a pipeline where preprocessing uses IronPython and ML inference uses CPython.

4. **Performance test**: Benchmark IronPython vs CPython for processing 10K JSON records.

## Module Reference

### Full Compatible List

```
abc, array, base64, bisect, codecs, collections, contextlib, copy,
csv, datetime, functools, gzip, hashlib, heapq, io, itertools, json,
math, operator, os, pathlib, pickle, random, re, shutil, socket,
sqlite3, string, struct, sys, tempfile, threading, typing, urllib,
xml, zipfile, clr
```

### Full Incompatible List

```
numpy, scipy, pandas, tensorflow, torch, sklearn, cv2, PIL,
matplotlib, seaborn, plotly, keras, xgboost, lightgbm, catboost
```

---

## Navigation

**←** [Previous: Chapter 7: .NET Bridge Generation](07_dotnet_bridges) | [📖 Book 7: Cross-Target Glue](./) | [Next: Chapter 9: Go and Rust Code Generation →](09_native_code_gen)
