# Language Idiosyncrasies

This document captures non-obvious behaviors, design patterns, and workarounds for language-specific quirks encountered in UnifyWeaver development.

## Unicode Handling

See [UNICODE_SPECIFICATION.md](../UNICODE_SPECIFICATION.md) for comprehensive documentation on:
- Terminal emoji support detection
- BMP vs non-BMP character handling
- Platform-specific unicode rendering behavior
- Fallback strategies for limited unicode support

## Prolog Anti-Declarative Patterns

### The Declarative Philosophy vs Imperative Reality

Prolog is fundamentally declarative: you describe *what* should be true, and Prolog figures out *how* to make it true. This includes automatic resource management - if you create a temporary file, Prolog's declarative philosophy says it should be cleaned up automatically.

However, when interfacing with external tools (bash, PowerShell, etc.), we sometimes need **imperative control** over file lifecycle that conflicts with Prolog's automatic cleanup.

### Case Study: Temporary File Cleanup Issue

**Problem**: In `bash_executor.pl`, we needed to create temporary bash scripts that external bash processes could execute. Initial implementations using Prolog's native file I/O failed mysteriously:

```prolog
% This approach FAILS on Windows+Cygwin
tmp_file('plbash', TmpFile),
atomics_to_string([TmpFile, '.sh'], TmpFileWithExt),
open(TmpFileWithExt, write, Stream),
write(Stream, BashCode),
close(Stream),
% File exists according to exists_file/1...
% But bash cannot see it!
```

**Symptoms**:
- `exists_file/1` returns `true`
- External bash process reports "No such file or directory"
- File appears to vanish between Prolog writing it and bash trying to read it

**Root Cause** (most likely): SWI-Prolog's automatic cleanup mechanisms. When using `tmp_file/2` or similar predicates, Prolog:
1. Registers the file for automatic deletion
2. May maintain internal file handles/locks
3. Can trigger cleanup based on scope/context changes
4. May use OS-specific mechanisms that conflict with external process access

**Alternative Hypothesis**: Path namespace issues between Windows and Cygwin filesystems (see Path Conversion section below).

**Solution**: Bypass Prolog's file I/O entirely and delegate to external tools:

```prolog
% This approach WORKS - let bash create its own files
write_script_via_cygwin(BashTarget, BashCode) :-
    setup_call_cleanup(
        process_create(path(bash), ['-lc', 'cat > "$1"', 'bash_executor', BashTarget],
                      [stdin(pipe(In)), stdout(null), stderr(pipe(Err)), process(PID)]),
        (   write(In, BashCode),
            close(In),
            read_string(Err, _, ErrMsg),
            close(Err),
            process_wait(PID, exit(ExitCode))
        ),
        true
    ),
    (   ExitCode = 0
    ->  true
    ;   format(atom(Msg), 'Failed to write temp script via bash (exit ~w): ~s',
              [ExitCode, ErrMsg]),
        throw(error(execution_error(ExitCode), context(write_script_via_cygwin/2, Msg)))
    ).
```

**Why This Works**:
- ✅ Bash creates the file in its own filesystem namespace
- ✅ No Prolog automatic cleanup hooks are registered
- ✅ No file locking from Prolog's I/O system
- ✅ The file truly exists where bash expects it
- ✅ We maintain imperative control over the file lifecycle

**Cleanup**: Manual deletion with error handling:
```prolog
delete_temp_file_path(Path) :-
    normalize_path_atom(Path, PathAtom),
    catch(delete_file(PathAtom), _, true).  % Fail silently if file doesn't exist
```

### Path Conversion Between Windows and Cygwin

**Challenge**: Windows and Cygwin maintain separate filesystem namespaces:
- Windows: `C:/Users/johnc/AppData/Local/Temp/file.sh`
- Cygwin: `/tmp/file.sh` or `/cygdrive/c/Users/johnc/AppData/Local/Temp/file.sh`

**Solution**: Dual-path system with conversion utilities:

```prolog
% Create paired paths for Windows and Cygwin
build_temp_paths(powershell_cygwin, TimeStamp, temp_paths(TmpFileWin, TmpFileBash)) :-
    % TmpFileWin: C:/cygwin64/tmp/plbash_XXX.sh (for Prolog cleanup)
    % TmpFileBash: /tmp/plbash_XXX.sh (for bash execution)
    ...
```

**Conversion Tools** (see `bash_executor.pl`):
- `convert_to_cygwin_path/2` - Converts Windows paths to Cygwin POSIX paths
- `find_cygpath/1` - Locates the `cygpath` utility in standard locations
- `convert_with_cygpath/2` - Uses `cygpath -u` for accurate conversion
- `fallback_cygwin_path/2` - Manual conversion when `cygpath` unavailable
- `normalize_windows_path/2` - Normalizes backslashes to forward slashes

**Path Conversion Strategy**:
1. Try `cygpath -u` for accurate conversion (Cygwin's official tool)
2. Fall back to manual conversion: `C:/foo/bar` → `/cygdrive/c/foo/bar`
3. Handle special case: `C:/cygwin64/tmp/X` → `/tmp/X`

### General Pattern: When to Use External Tools

**Use external tools when**:
- You need files to persist beyond Prolog's automatic cleanup
- External processes must access resources you create
- You need imperative control over resource lifecycle
- Platform-specific namespaces complicate direct Prolog I/O

**Use Prolog's native I/O when**:
- Files are purely internal to Prolog processing
- You want automatic cleanup
- Portability across platforms without external dependencies
- Declarative resource management is sufficient

### Related Documentation

- `HANDOFF_BASH_EXECUTOR_BUG.md` - Original problem analysis
- `HANDOFF_BASH_EXECUTOR_FIX.md` - Solution implementation details
- `bash_executor.pl` - Implementation of dual-path file management

## Summary

Prolog's declarative nature is powerful, but when interfacing with imperative external tools, we sometimes need to:
1. **Recognize when declarative patterns conflict** with external tool requirements
2. **Delegate to external tools** for operations they understand better
3. **Build bridges** between Prolog's world and the external world (path conversion, etc.)
4. **Document the impedance mismatch** so future developers understand the tradeoffs

The key insight: **Prolog is declarative, but we can embed imperative operations by delegating to external processes.**
