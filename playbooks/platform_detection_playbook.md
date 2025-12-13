# Playbook: Platform and Environment Detection

## Audience
This playbook demonstrates UnifyWeaver's platform_detection module for detecting the execution environment.

## Overview
The `platform_detection` module determines:
- Platform type (Windows, WSL, Docker, Linux, macOS)
- Execution mode
- Bash execution capability
- Container environments

## When to Use

✅ **Use platform_detection when:**
- Need platform-specific compilation
- Detecting Docker/WSL/native environments
- Checking bash availability
- Adapting behavior to platform

## Example Usage

### Detect Platform

```prolog
:- use_module('src/unifyweaver/core/platform_detection').

% Detect current platform
?- detect_platform(Platform).
Platform = linux.

% Check specific platforms
?- is_docker.
true.

?- is_wsl.
false.

?- is_windows.
false.
```

### Check Bash Availability

```prolog
% Can we execute bash directly?
?- can_execute_bash_directly.
true.
```

### Platform-Specific Compilation

```prolog
% Compile differently based on platform
compile_for_platform(Pred, Code) :-
    detect_platform(Platform),
    (   Platform = windows
    ->  compile_for_windows(Pred, Code)
    ;   Platform = wsl
    ->  compile_for_wsl(Pred, Code)
    ;   compile_for_unix(Pred, Code)
    ).
```

## Detected Platforms

- **windows**: Native Windows
- **wsl**: Windows Subsystem for Linux
- **docker**: Docker container
- **linux**: Native Linux
- **macos**: macOS
- **unknown**: Unrecognized platform

## Execution Modes

- Direct bash execution
- WSL bash wrapper
- Docker entrypoint
- PowerShell wrapper

## See Also

- `playbooks/powershell_inline_dotnet_playbook.md` - Windows-specific
- `playbooks/bash_parallel_playbook.md` - Unix-specific

## Summary

**Key Concepts:**
- ✅ Platform detection (Windows/WSL/Docker/Linux/macOS)
- ✅ Bash availability checking
- ✅ Container environment detection
- ✅ Platform-specific compilation
