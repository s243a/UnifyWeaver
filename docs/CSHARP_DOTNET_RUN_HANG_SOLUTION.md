# C# dotnet run Hang Issue - Solution Documentation

## Problem Summary

When running `dotnet run --no-restore` from SWI-Prolog's `process_create/3` in WSL, the command would hang indefinitely. The same command worked perfectly when run directly from bash.

**Location:** `tests/core/test_csharp_query_target.pl` - `dotnet_command/4` predicate

## Root Cause

The `dotnet run` command doesn't behave like a simple executable. Instead, it:
1. Resolves dependencies
2. Spawns the actual compiled application as a child process
3. Monitors the child process
4. May wait for all child processes in its process group to exit

When launched from SWI-Prolog's `process_create/3`, the process tree and job control behavior conflicts with Prolog's process management, causing the parent `dotnet run` process to never exit even though the child application completes successfully.

## Working Solution

**Strategy:** Build first, then execute the compiled binary directly

Instead of calling `dotnet run`, we now:
1. Run `dotnet build --no-restore` to compile the project
2. Find the compiled executable in `bin/Debug/net9.0/`
3. Execute the binary directly

### Implementation

```prolog
dotnet_command(Dotnet, Args, Dir, Status, Output) :-
    (   Args = ['run', '--no-restore']
    ->  % Special handling - build first, then execute binary directly
        % Step 1: Build the project
        process_create(Dotnet, ['build', '--no-restore'],
                       [ cwd(Dir),
                         env(Env),
                         stdin(null),
                         stdout(null),
                         stderr(null),
                         process(BuildPID)
                       ]),
        process_wait(BuildPID, exit(BuildStatus)),
        
        (   BuildStatus =:= 0
        ->  % Step 2: Find and execute the compiled binary
            find_compiled_executable(Dir, ExePath),
            process_create(ExePath, [],
                           [ cwd(Dir),
                             env(Env),
                             stdin(pipe(In)),
                             stdout(pipe(Out)),
                             stderr(pipe(Out)),
                             process(PID)
                           ]),
            close(In),
            read_string(Out, _, Output),
            close(Out),
            process_wait(PID, exit(Status))
        ;   Status = BuildStatus,
            Output = ""
        )
    ;   % Normal handling for other dotnet commands
        ...
    ).
```

### Helper Predicate

```prolog
find_compiled_executable(Dir, ExePath) :-
    directory_file_path(Dir, 'bin/Debug/net9.0', DebugDir),
    (   exists_directory(DebugDir)
    ->  directory_files(DebugDir, Files),
        member(File, Files),
        \+ atom_concat(_, '.dll', File),
        \+ atom_concat(_, '.pdb', File),
        \+ atom_concat(_, '.deps.json', File),
        \+ atom_concat(_, '.runtimeconfig.json', File),
        File \= '.',
        File \= '..',
        directory_file_path(DebugDir, File, ExePath),
        exists_file(ExePath),
        !
    ;   fail
    ).
```

## What We Tried (That Didn't Work)

### 1. stdin(null)
```prolog
process_create(Dotnet, ['run','--no-restore'],
               [stdin(null), stdout(pipe(Out)), stderr(null), ...])
```
**Result:** Still hung. This is the standard approach for closing stdin, but didn't help.

### 2. Bash Wrapper with Shell Redirection
```prolog
ShellCmd = '/usr/bin/dotnet run --no-restore < /dev/null',
process_create(path(bash), ['-c', ShellCmd], ...)
```
**Result:** Still hung. Even shell-level stdin redirection didn't solve it.

### 3. Merged stderr into stdout
```prolog
process_create(Dotnet, ['run','--no-restore'],
               [stdin(null), stdout(pipe(Out)), stderr(pipe(Out)), ...])
```
**Result:** Still hung. This was Perplexity's first recommendation to avoid pipe buffer issues.

### 4. stdin(pipe(In)) with Immediate Close
```prolog
process_create(Dotnet, ['run','--no-restore'],
               [stdin(pipe(In)), stdout(pipe(Out)), stderr(pipe(Out)), ...]),
close(In),  % Explicit EOF signal
```
**Result:** Still hung. This is the "explicit EOF" pattern.

### 5. Threading for Concurrent stderr/stdout Reading
```prolog
thread_create(read_stream(Err, Stderr0), ErrThread, []),
read_stream(Out, Stdout0),
thread_join(ErrThread, true),
```
**Result:** Still hung. The issue wasn't pipe buffer deadlock.

## What We Didn't Try

### 1. File Redirection
Redirect stdout/stderr to files instead of pipes, then read the files:
```prolog
% Not tried - could work as workaround
process_create(Dotnet, ['run','--no-restore'],
               [stdin(null), stdout(file(OutFile)), stderr(file(ErrFile)), ...])
```

### 2. Python/Script Wrapper
Use an intermediate script to launch dotnet run:
```python
#!/usr/bin/env python3
import subprocess
import sys
result = subprocess.run(['dotnet', 'run', '--no-restore'], 
                       capture_output=True, text=True, cwd=sys.argv[1])
print(result.stdout)
```

### 3. setsid or Process Group Control
```prolog
% Not tried - may help with process group issues
process_create(path(setsid), [Dotnet, 'run', '--no-restore'], ...)
```

### 4. Timeout with Forced Termination
```prolog
% Not tried - workaround but doesn't solve root cause
call_with_time_limit(30.0, read_string(Out, _, Output))
```

## Current Status

✅ **Working:** Build+execute strategy completely eliminates the hang
✅ **Suitable for:** Automated testing (fast, reliable)
⚠️ **Limitation:** `dotnet run` still doesn't work from Prolog

## Future Work / TODO

### High Priority
- [ ] Investigate Python wrapper approach to make `dotnet run` work
  - Create a helper script that launches `dotnet run` and properly streams output
  - Would allow seeing source-level output during manual testing
  
### Medium Priority  
- [ ] Test with `setsid` or explicit process group control
- [ ] Try file redirection approach as fallback
- [ ] Investigate if SWI-Prolog version matters (check for known bugs)

### Low Priority
- [ ] Report to SWI-Prolog community (may be WSL-specific issue)
- [ ] Test on native Linux to confirm WSL-specific behavior

## Recommendations

**For Automated Tests:** Use the current build+execute strategy. It's fast, reliable, and avoids all process management complexity.

**For Manual Tests/Development:** Consider adding a separate manual test mode that uses a Python wrapper to call `dotnet run`, allowing developers to see the full .NET SDK output and diagnostics.

## Environment Notes

- **OS:** WSL (Windows Subsystem for Linux) - Ubuntu 20.04
- **SWI-Prolog:** Version not recorded (should be noted for future reference)
- **.NET SDK:** 9.0.203 (Linux native version at `/usr/bin/dotnet`)
- **Runtime:** 9.0.4

## References

- Perplexity research confirmed this is a known `dotnet run` process tree issue
- SWI-Prolog process_create documentation: https://www.swi-prolog.org/pldoc/doc_for?object=process_create/3
- Related .NET issue reports suggest similar problems in other process control scenarios
