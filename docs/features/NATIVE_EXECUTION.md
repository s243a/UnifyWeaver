# Native Bash Execution Support

**Version:** 0.0.2+
**Feature Status:** NEW
**Platforms:** Linux, WSL, Docker, macOS

---

## Overview

UnifyWeaver now includes native bash execution support for Linux-like environments. This allows you to execute compiled bash scripts directly from Prolog without requiring external wrapper scripts or compatibility layers.

### Platform Support Matrix

| Platform | Execution Mode | Module Required | Notes |
|----------|---------------|-----------------|-------|
| Windows PowerShell | `powershell_wsl` | PowerShell compat layer | Uses `uw-*` wrappers |
| WSL | `direct_bash` | `bash_executor` | Native execution |
| Native Linux | `direct_bash` | `bash_executor` | Native execution |
| Docker (in WSL) | `direct_bash` | `bash_executor` | Native execution |
| macOS | `direct_bash` | `bash_executor` | Native execution |

---

## Quick Start

### 1. Platform Detection

```prolog
?- use_module(library(unifyweaver/core/platform_detection)).
?- detect_platform(Platform).
Platform = docker.  % or wsl, linux, macos, windows

?- can_execute_bash_directly.
true.  % If on Linux/WSL/Docker/macOS
```

### 2. Execute Bash Code

```prolog
?- use_module(library(unifyweaver/core/bash_executor)).
?- execute_bash('echo "Hello from bash!"', Output).
Output = "Hello from bash!\n".
```

### 3. Execute with Input

```prolog
?- execute_bash('grep "foo"', 'foo\nbar\nbaz\nfoo again', Output).
Output = "foo\nfoo again\n".
```

---

## API Reference

### Platform Detection Module

**Module:** `unifyweaver/core/platform_detection`

#### detect_platform(-Platform)
Detect the current platform.

**Platforms:**
- `windows` - Native Windows (not WSL)
- `wsl` - Windows Subsystem for Linux
- `docker` - Docker container
- `linux` - Native Linux
- `macos` - macOS
- `unknown` - Could not determine

**Example:**
```prolog
?- detect_platform(P).
P = docker.
```

#### detect_execution_mode(-Mode)
Detect the appropriate execution mode for bash scripts.

**Modes:**
- `direct_bash` - Can execute bash directly
- `powershell_wsl` - Need PowerShell + WSL wrapper
- `powershell_cygwin` - Need PowerShell + Cygwin wrapper
- `unknown` - Cannot determine

**Example:**
```prolog
?- detect_execution_mode(M).
M = direct_bash.
```

#### Platform Check Predicates

```prolog
is_windows/0         % True if native Windows
is_wsl/0             % True if Windows Subsystem for Linux
is_docker/0          % True if Docker container
is_linux/0           % True if native Linux (not WSL/Docker)
is_native_linux/0    % True if any Linux-like (includes WSL/Docker)
is_macos/0           % True if macOS
can_execute_bash_directly/0  % True if bash execution supported
```

**Example:**
```prolog
?- is_docker.
true.

?- is_wsl.
false.

?- can_execute_bash_directly.
true.
```

---

### Bash Executor Module

**Module:** `unifyweaver/core/bash_executor`

#### execute_bash(+BashCode, -Output)
Execute bash code and capture stdout.

**Parameters:**
- `BashCode`: String or atom containing bash code
- `Output`: Stdout output as a string

**Example:**
```prolog
?- execute_bash('uname -o', Output).
Output = "GNU/Linux\n".

?- execute_bash('echo "The answer is $((6*7))"', Output).
Output = "The answer is 42\n".
```

#### execute_bash(+BashCode, +Input, -Output)
Execute bash code with stdin input.

**Parameters:**
- `BashCode`: String or atom containing bash code
- `Input`: String to send to stdin
- `Output`: Stdout output as a string

**Example:**
```prolog
?- execute_bash('wc -l', 'line1\nline2\nline3\n', Output).
Output = "3\n".

?- execute_bash('grep "ERROR"', 'INFO: ok\nERROR: bad\nWARN: check', Output).
Output = "ERROR: bad\n".
```

#### execute_bash_file(+FilePath, -Output)
Execute a bash script file.

**Parameters:**
- `FilePath`: Path to bash script file
- `Output`: Stdout output as a string

**Example:**
```prolog
% Assuming script.sh exists and is executable
?- execute_bash_file('scripts/my_script.sh', Output).
Output = "Script output\n".
```

#### execute_bash_file(+FilePath, +Input, -Output)
Execute a bash script file with stdin input.

**Parameters:**
- `FilePath`: Path to bash script file
- `Input`: String to send to stdin
- `Output`: Stdout output as a string

**Example:**
```prolog
?- execute_bash_file('scripts/filter.sh', 'input\ndata\n', Output).
Output = "filtered result\n".
```

#### write_and_execute_bash(+BashCode, -Output)
Write bash code to a temporary file and execute it.

Useful for longer scripts or when you need a real file (e.g., for shebang processing).

**Parameters:**
- `BashCode`: String or atom containing bash code (can include shebang)
- `Output`: Stdout output as a string

**Example:**
```prolog
?- write_and_execute_bash('#!/bin/bash\necho "Multi-line\nscript\nhere"', Output).
Output = "Multi-line\nscript\nhere\n".
```

#### write_and_execute_bash(+BashCode, +Input, -Output)
Write bash code to a temporary file with input and execute it.

**Parameters:**
- `BashCode`: String or atom containing bash code
- `Input`: String to send to stdin
- `Output`: Stdout output as a string

#### can_execute_natively/0
True if bash can be executed natively on this platform.

**Example:**
```prolog
?- can_execute_natively.
true.  % On Linux/WSL/Docker/macOS
```

---

## Usage Patterns

### Pattern 1: Platform-Adaptive Execution

Write code that adapts to the platform:

```prolog
:- use_module(library(unifyweaver/core/platform_detection)).
:- use_module(library(unifyweaver/core/bash_executor)).

execute_cross_platform(Command, Output) :-
    detect_execution_mode(Mode),
    (   Mode = direct_bash
    ->  % Use native execution
        execute_bash(Command, Output)
    ;   Mode = powershell_wsl
    ->  % Use PowerShell wrapper
        format(atom(Cmd), 'wsl bash -c "~w"', [Command]),
        shell(Cmd, Output)
    ;   % Unsupported
        throw(error(platform_error, 'Bash execution not supported'))
    ).
```

### Pattern 2: Execute Compiled Sources

Execute bash code generated by UnifyWeaver's source compiler:

```prolog
:- use_module(library(unifyweaver/sources)).
:- use_module(library(unifyweaver/core/bash_executor)).
:- use_module(library(unifyweaver/core/dynamic_source_compiler)).

% Define a source
:- source(csv, users, [csv_file('data.csv'), has_header(true)]).

% Compile and execute it
run_compiled_source(Name/Arity, Output) :-
    % Compile to bash
    compile_dynamic_source(Name/Arity, [], BashCode),

    % Execute on appropriate platform
    (   can_execute_natively
    ->  execute_bash(BashCode, Output)
    ;   throw(error(platform_error, 'Native execution required'))
    ).

% Example usage
?- run_compiled_source(users/4, Output).
Output = "alice,25,nyc,true\nbob,30,sf,true\n...".
```

### Pattern 3: Pipeline Execution

Execute multi-stage bash pipelines:

```prolog
execute_pipeline(Stages, Input, Output) :-
    % Build pipeline command
    atomic_list_concat(Stages, ' | ', Pipeline),

    % Execute
    execute_bash(Pipeline, Input, Output).

% Example: CSV processing pipeline
?- execute_pipeline(
    ['cat data.csv', 'grep "active"', 'cut -d, -f1,2'],
    '',
    Output
).
Output = "alice,25\nbob,30\n".
```

### Pattern 4: Error Handling

```prolog
safe_execute_bash(Command, Output) :-
    catch(
        execute_bash(Command, Output),
        error(execution_error(Code), Context),
        (   format('Bash command failed with exit code ~w~n', [Code]),
            format('Context: ~w~n', [Context]),
            fail
        )
    ).

% Example usage
?- safe_execute_bash('exit 1', Output).
Bash command failed with exit code 1
Context: context(execute_bash/3, 'Bash script exited with code 1')
false.
```

---

## Integration with Data Sources

The bash executor integrates seamlessly with UnifyWeaver's data source system:

```prolog
:- use_module(library(unifyweaver/sources)).
:- use_module(library(unifyweaver/core/bash_executor)).

% Define sources
:- source(csv, users, [csv_file('users.csv'), has_header(true)]).
:- source(json, api_data, [json_file('data.json'), jq_filter('.items[]')]).

% Execute a data source directly
execute_source(SourceName, Output) :-
    % Get the compiled bash for this source
    SourceName/Arity = SourceSpec,
    compile_dynamic_source(SourceSpec, [], BashCode),

    % Execute natively
    execute_bash(BashCode, Output).

% Example usage
?- execute_source(users, Output).
Output = "alice\t25\tnyc\ttrue\nbob\t30\tsf\ttrue\n...".
```

---

## Platform Detection Details

### Docker Detection

The module detects Docker containers by checking:
1. Existence of `/.dockerenv` file
2. Docker entries in `/proc/1/cgroup`

### WSL Detection

The module detects WSL by checking:
1. `WSL_DISTRO_NAME` environment variable
2. `WSL_INTEROP` environment variable
3. "Microsoft" in `/proc/version`

### Native Linux Detection

Detected when:
- Running on Unix platform
- Not in Docker
- Not in WSL
- Not on macOS

---

## Error Handling

### Platform Errors

```prolog
?- execute_bash('echo test', Output).
ERROR: Unhandled exception: platform_error(context(execute_bash/2,
       'Cannot execute bash natively on this platform'))
```

**Solution:** Check platform first:
```prolog
?- can_execute_natively.
false.  % Need to use PowerShell compatibility layer
```

### Execution Errors

```prolog
?- execute_bash('exit 42', Output).
ERROR: Unhandled exception: execution_error(42)(context(execute_bash/2,
       'Bash script exited with code 42'))
```

**Solution:** Use error handling:
```prolog
?- catch(execute_bash('exit 42', _), error(execution_error(Code), _), true).
Code = 42.
```

### File Not Found Errors

```prolog
?- execute_bash_file('nonexistent.sh', Output).
ERROR: Unhandled exception: existence_error(file, 'nonexistent.sh')
```

---

## Performance Considerations

### Direct Execution vs File Execution

**Direct execution** (`execute_bash/2-3`):
- Faster for short commands
- No file I/O overhead
- Limited by command-line length

**File execution** (`execute_bash_file/2-3`):
- Better for long scripts
- Supports shebang
- Slightly slower due to file I/O

**Temporary file execution** (`write_and_execute_bash/2-3`):
- Convenience wrapper
- Automatic cleanup
- Good for programmatically generated scripts

### Caching Compiled Sources

For frequently executed sources, cache the compiled bash:

```prolog
:- dynamic compiled_source_cache/2.

get_compiled_source(Name/Arity, BashCode) :-
    (   compiled_source_cache(Name/Arity, BashCode)
    ->  true  % Use cached version
    ;   compile_dynamic_source(Name/Arity, [], BashCode),
        assertz(compiled_source_cache(Name/Arity, BashCode))
    ).
```

---

## Testing

### Unit Tests

Both modules include test predicates:

```prolog
% Test platform detection
?- use_module(library(unifyweaver/core/platform_detection)).
?- platform_detection:test_platform_detection.

% Test bash executor
?- use_module(library(unifyweaver/core/bash_executor)).
?- bash_executor:test_bash_executor.
```

### Integration Tests

```prolog
% Test end-to-end: platform detection + execution
test_integration :-
    % Detect platform
    detect_platform(Platform),
    format('Platform: ~w~n', [Platform]),

    % Check capability
    (   can_execute_bash_directly
    ->  format('Can execute bash directly~n', [])
    ;   format('Need compatibility layer~n', []),
        halt(1)
    ),

    % Execute test command
    execute_bash('echo "Integration test passed"', Output),
    format('~w', [Output]).
```

---

## Comparison with PowerShell Compatibility Layer

| Feature | Native Execution | PowerShell Layer |
|---------|------------------|------------------|
| **Platform** | Linux/WSL/Docker/macOS | Windows PowerShell |
| **Performance** | Fast (direct) | Slower (via wsl.exe) |
| **Dependencies** | None | WSL or Cygwin |
| **Complexity** | Simple | Moderate |
| **Setup** | Auto-detected | Manual sourcing |
| **Use Case** | Development on Linux | Deployment on Windows |

---

## Best Practices

1. **Always check platform first:**
   ```prolog
   (   can_execute_natively
   ->  execute_bash(...)
   ;   use_powershell_wrapper(...)
   )
   ```

2. **Handle errors explicitly:**
   ```prolog
   catch(execute_bash(...), Error, handle_error(Error))
   ```

3. **Cache compiled sources:**
   - Avoid recompiling the same source repeatedly
   - Use dynamic predicates for caching

4. **Use appropriate execution method:**
   - Short commands → `execute_bash/2`
   - Long scripts → `write_and_execute_bash/2`
   - Existing files → `execute_bash_file/2`

5. **Close resources properly:**
   - Both modules handle cleanup automatically
   - Temporary files are deleted after use

---

## Examples

### Example 1: CSV Processing

```prolog
:- use_module(library(unifyweaver/sources)).
:- use_module(library(unifyweaver/core/bash_executor)).

% Define CSV source
:- source(csv, sales, [csv_file('sales.csv'), has_header(true)]).

% Process and analyze
analyze_sales :-
    % Compile source
    compile_dynamic_source(sales/4, [], BashCode),

    % Execute and process
    execute_bash(BashCode, RawOutput),

    % Parse output
    split_string(RawOutput, "\n", "\n", Lines),
    length(Lines, Count),
    format('Total sales records: ~w~n', [Count]).
```

### Example 2: HTTP + JSON Pipeline

```prolog
:- use_module(library(unifyweaver/core/bash_executor)).

fetch_and_parse_json(URL, Output) :-
    % Build pipeline: fetch with curl, parse with jq
    format(atom(Pipeline), 'curl -s "~w" | jq ".items[]"', [URL]),

    % Execute
    execute_bash(Pipeline, Output).

% Usage
?- fetch_and_parse_json('https://api.example.com/data', Output).
```

### Example 3: Multi-Stage ETL

```prolog
etl_pipeline(InputFile, OutputFile) :-
    % Stage 1: Extract
    format(atom(Extract), 'cat ~w', [InputFile]),
    execute_bash(Extract, RawData),

    % Stage 2: Transform
    execute_bash('grep -v "^#" | sort | uniq', RawData, Cleaned),

    % Stage 3: Load
    format(atom(Load), 'cat > ~w', [OutputFile]),
    execute_bash(Load, Cleaned, _),

    format('ETL pipeline complete: ~w -> ~w~n', [InputFile, OutputFile]).
```

---

## Troubleshooting

### "Cannot execute bash natively on this platform"

**Cause:** Running on Windows without WSL/Cygwin
**Solution:** Use PowerShell compatibility layer or install WSL

### "Command not found" errors

**Cause:** Required tool not in PATH
**Solution:** Check tool availability with `which`:
```prolog
?- execute_bash('which jq', Output).
Output = "/usr/bin/jq\n".  % Found
% OR
Output = "".  % Not found
```

### Permission denied errors

**Cause:** Script file not executable
**Solution:** Module automatically runs `chmod +x`, but check file permissions

---

## Future Enhancements

Planned features:
- Async execution with process monitoring
- Timeout support for long-running commands
- Streaming output for large datasets
- Built-in error recovery strategies
- Performance profiling integration

---

**Documentation Version:** 1.0
**Last Updated:** October 20, 2025
**Module Version:** 0.0.2+
