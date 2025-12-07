<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)
-->

# Chapter 7: .NET Bridge Generation

## Overview

The .NET family (C#, PowerShell, IronPython) can communicate in-process, avoiding serialization overhead. This chapter covers:

- Runtime detection
- PowerShell bridge generation
- IronPython bridge generation
- CPython fallback via pipes
- Choosing the right approach

## The .NET Glue Module

Located at `src/unifyweaver/glue/dotnet_glue.pl`:

```prolog
:- module(dotnet_glue, [
    % Runtime detection
    detect_dotnet_runtime/1,
    detect_ironpython/1,
    detect_powershell/1,

    % Bridge generation
    generate_powershell_bridge/2,
    generate_ironpython_bridge/2,
    generate_cpython_bridge/2,
    generate_csharp_host/3,
    generate_dotnet_pipeline/3
]).
```

## Runtime Detection

### Detecting .NET Runtime

```prolog
?- detect_dotnet_runtime(Runtime).
Runtime = dotnet_modern.  % .NET 5/6/7/8
```

Possible values:
- `dotnet_modern` - .NET 5+
- `dotnet_core` - .NET Core 2.x/3.x
- `dotnet_framework` - .NET Framework 4.x
- `mono` - Mono runtime
- `none` - No .NET installed

### Detecting PowerShell

```prolog
?- detect_powershell(Version).
Version = core('7.4.0').
```

Possible values:
- `core(Version)` - PowerShell Core (cross-platform)
- `windows(Version)` - Windows PowerShell
- `none` - Not installed

### Detecting IronPython

```prolog
?- detect_ironpython(Available).
Available = true.
```

## PowerShell Bridge

The PowerShell bridge allows C# to host PowerShell scripts in-process.

### Generation

```prolog
?- generate_powershell_bridge([namespace('MyApp')], Code).
```

### Generated Code

```csharp
using System;
using System.Collections.Generic;
using System.Management.Automation;
using System.Management.Automation.Runspaces;

namespace MyApp
{
    /// <summary>
    /// Bridge for executing PowerShell scripts in-process.
    /// </summary>
    public static class PowerShellBridge
    {
        private static readonly Runspace _runspace;

        static PowerShellBridge()
        {
            _runspace = RunspaceFactory.CreateRunspace();
            _runspace.Open();
        }

        /// <summary>
        /// Execute a PowerShell script and return results.
        /// </summary>
        public static IEnumerable<T> Invoke<T>(string script)
        {
            using var ps = PowerShell.Create();
            ps.Runspace = _runspace;
            ps.AddScript(script);

            foreach (var result in ps.Invoke<T>())
            {
                yield return result;
            }
        }

        /// <summary>
        /// Execute a script with input and return results.
        /// </summary>
        public static IEnumerable<TOutput> Invoke<TInput, TOutput>(
            string script,
            TInput input)
        {
            using var ps = PowerShell.Create();
            ps.Runspace = _runspace;
            ps.AddScript(script);
            ps.AddParameter("input", input);

            foreach (var result in ps.Invoke<TOutput>())
            {
                yield return result;
            }
        }

        /// <summary>
        /// Process a stream of inputs.
        /// </summary>
        public static IEnumerable<TOutput> InvokeStream<TInput, TOutput>(
            string script,
            IEnumerable<TInput> inputs)
        {
            foreach (var input in inputs)
            {
                foreach (var output in Invoke<TInput, TOutput>(script, input))
                {
                    yield return output;
                }
            }
        }
    }
}
```

### Usage Example

```csharp
// In your C# code
var results = PowerShellBridge.Invoke<FileInfo>(
    "Get-ChildItem -Path C:\\ -Filter *.txt"
);

foreach (var file in results)
{
    Console.WriteLine($"{file.Name}: {file.Length} bytes");
}
```

### Streaming Example

```csharp
// Process a stream of records
var inputs = new[] { "file1.txt", "file2.txt", "file3.txt" };

var results = PowerShellBridge.InvokeStream<string, FileInfo>(
    "param($input) Get-Item $input",
    inputs
);

foreach (var file in results)
{
    ProcessFile(file);
}
```

## IronPython Bridge

For Python scripts that don't require CPython-specific libraries.

### Generation

```prolog
?- generate_ironpython_bridge([namespace('MyApp')], Code).
```

### Generated Code

```csharp
using System;
using System.Collections.Generic;
using IronPython.Hosting;
using Microsoft.Scripting.Hosting;

namespace MyApp
{
    /// <summary>
    /// Bridge for executing Python scripts via IronPython.
    /// </summary>
    public static class IronPythonBridge
    {
        private static readonly ScriptEngine _engine;
        private static readonly ScriptScope _scope;

        static IronPythonBridge()
        {
            _engine = Python.CreateEngine();
            _scope = _engine.CreateScope();
        }

        /// <summary>
        /// Execute Python code.
        /// </summary>
        public static void Execute(string code)
        {
            _engine.Execute(code, _scope);
        }

        /// <summary>
        /// Execute code and return a result.
        /// </summary>
        public static T Execute<T>(string code)
        {
            return _engine.Execute<T>(code, _scope);
        }

        /// <summary>
        /// Execute code with input data.
        /// </summary>
        public static TOutput ExecuteWithInput<TInput, TOutput>(
            string code,
            TInput input)
        {
            _scope.SetVariable("input_data", input);
            return _engine.Execute<TOutput>(code, _scope);
        }

        /// <summary>
        /// Set a variable in the Python scope.
        /// </summary>
        public static void SetVariable(string name, object value)
        {
            _scope.SetVariable(name, value);
        }

        /// <summary>
        /// Get a variable from the Python scope.
        /// </summary>
        public static T GetVariable<T>(string name)
        {
            return _scope.GetVariable<T>(name);
        }
    }
}
```

### Usage Example

```csharp
// Execute Python code
IronPythonBridge.Execute(@"
import clr
clr.AddReference('System.Core')

def process(data):
    return [x * 2 for x in data]

result = process([1, 2, 3, 4, 5])
");

var result = IronPythonBridge.GetVariable<IList<int>>("result");
// result = [2, 4, 6, 8, 10]
```

### .NET Integration

IronPython can access .NET types directly:

```python
# Python code running in IronPython
import clr
clr.AddReference('System.Core')
from System.Collections.Generic import List

# Create a .NET List
my_list = List[str]()
my_list.Add("Hello")
my_list.Add("World")

# Use LINQ (via clr)
from System.Linq import Enumerable
count = Enumerable.Count(my_list)
```

## CPython Bridge (Fallback)

When IronPython can't be used (numpy, pandas, etc.), fall back to CPython via pipes.

### Generation

```prolog
?- generate_cpython_bridge([python_path('python3')], Code).
```

### Generated Code

```csharp
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text.Json;

namespace UnifyWeaver.Glue
{
    /// <summary>
    /// Bridge for executing CPython scripts via subprocess.
    /// </summary>
    public static class CPythonBridge
    {
        public static string PythonPath { get; set; } = "python3";

        /// <summary>
        /// Execute a Python script with input and return output.
        /// </summary>
        public static TOutput Execute<TInput, TOutput>(string script, TInput input)
        {
            var wrappedScript = $@"
import sys
import json

input_data = json.loads(sys.stdin.read())

{script}

print(json.dumps(result))
";

            var psi = new ProcessStartInfo
            {
                FileName = PythonPath,
                Arguments = "-c \"" + wrappedScript.Replace("\"", "\\\"") + "\"",
                UseShellExecute = false,
                RedirectStandardInput = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };

            using var process = Process.Start(psi);

            // Send input
            var inputJson = JsonSerializer.Serialize(input);
            process.StandardInput.Write(inputJson);
            process.StandardInput.Close();

            // Read output
            var output = process.StandardOutput.ReadToEnd();
            var errors = process.StandardError.ReadToEnd();

            process.WaitForExit();

            if (process.ExitCode != 0)
            {
                throw new InvalidOperationException($"Python error: {errors}");
            }

            return JsonSerializer.Deserialize<TOutput>(output);
        }

        /// <summary>
        /// Process a stream of inputs.
        /// </summary>
        public static IEnumerable<TOutput> ExecuteStream<TInput, TOutput>(
            string script,
            IEnumerable<TInput> inputStream)
        {
            // ... streaming implementation
        }
    }
}
```

### Usage Example

```csharp
// Use numpy via CPython
var result = CPythonBridge.Execute<double[], double[]>(
    @"
import numpy as np
arr = np.array(input_data)
result = (arr * 2 + 1).tolist()
",
    new[] { 1.0, 2.0, 3.0, 4.0, 5.0 }
);
// result = [3.0, 5.0, 7.0, 9.0, 11.0]
```

## Complete C# Host

Generate a C# class that includes all bridges:

```prolog
?- generate_csharp_host(
    [powershell, ironpython, cpython],
    [namespace('MyPipeline')],
    Code
).
```

This generates a complete file with all three bridges and helper methods.

## .NET Pipeline Generation

Generate a complete .NET pipeline:

```prolog
?- generate_dotnet_pipeline(
    [
        step(load, csharp, 'LoadData', []),
        step(filter, powershell, 'Where-Object {$_.Status -eq "Active"}', []),
        step(transform, ironpython, 'transform_records(records)', []),
        step(save, csharp, 'SaveResults', [])
    ],
    [],
    Code
).
```

Generated:

```csharp
public static IEnumerable<Output> RunPipeline(IEnumerable<Input> input)
{
    // Step 1: Load (C#)
    var step1 = LoadData(input);

    // Step 2: Filter (PowerShell in-process)
    var step2 = PowerShellBridge.InvokeStream<Record, Record>(
        "Where-Object {$_.Status -eq \"Active\"}",
        step1
    );

    // Step 3: Transform (IronPython in-process)
    IronPythonBridge.SetVariable("records", step2.ToList());
    IronPythonBridge.Execute("result = transform_records(records)");
    var step3 = IronPythonBridge.GetVariable<IList<Record>>("result");

    // Step 4: Save (C#)
    return SaveResults(step3);
}
```

## Choosing the Right Bridge

| Scenario | Bridge | Why |
|----------|--------|-----|
| Windows automation | PowerShell | Native cmdlets |
| Simple Python logic | IronPython | Zero serialization |
| numpy/pandas/ML | CPython | Library compatibility |
| Performance critical | IronPython or C# | No subprocess overhead |
| Cross-platform | CPython | Works everywhere |

### Decision Tree

```
Need Python?
├── No → Use PowerShell or C#
└── Yes → Check libraries
    ├── Only stdlib + .NET → IronPython (fast)
    └── numpy/pandas/etc → CPython (compatible)
```

## Chapter Summary

- **PowerShell bridge** for Windows automation and .NET integration
- **IronPython bridge** for Python code that uses stdlib
- **CPython bridge** for numpy, pandas, and other C extensions
- **In-process** communication has zero serialization
- **Choice depends** on library requirements

## Next Steps

In Chapter 8, we'll deep-dive into IronPython compatibility:
- Which modules work
- Which don't
- Migration strategies

## Exercises

1. **PowerShell bridge**: Generate a bridge and use it to list running processes.

2. **IronPython bridge**: Execute Python code that creates a .NET Dictionary.

3. **CPython bridge**: Use numpy to calculate statistics on an array.

4. **Pipeline**: Create a .NET pipeline that loads CSV, filters with PowerShell, and transforms with Python.

## Code Examples

See `examples/03-dotnet-bridge/` for:
- `PowerShellExample.cs` - PowerShell integration
- `IronPythonExample.cs` - IronPython integration
- `CPythonExample.cs` - CPython fallback
- `MixedPipeline.cs` - Combined example

---

## Navigation

**←** [Previous: Chapter 6: Shell Pipeline Orchestration](06_shell_pipelines) | [📖 Book 7: Cross-Target Glue](README) | [Next: Chapter 8: IronPython Compatibility →](08_ironpython_compat)
