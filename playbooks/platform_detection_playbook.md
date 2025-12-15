# Playbook: Platform and Environment Detection

## Audience
This playbook is a high-level guide for coding agents. It demonstrates UnifyWeaver's platform_detection module for detecting execution environments and adapting compilation accordingly.

## Workflow Overview
Use platform_detection for environment-aware compilation:
1. Detect current platform (Windows, WSL, Docker, Linux, macOS)
2. Check bash availability and execution mode
3. Adapt compilation strategy based on platform

## Agent Inputs
Reference the following artifacts:
1. **Core Module** – `src/unifyweaver/core/platform_detection.pl` contains platform detection predicates
2. **Module Documentation** – See module header for API details

## Key Features

- Platform type detection (windows, wsl, docker, linux, macos)
- Bash execution capability checking
- Container environment detection
- Platform-specific compilation strategies

## Execution Guidance

Consult the module for predicate usage:

```prolog
:- use_module('src/unifyweaver/core/platform_detection').

% Detect current platform
?- detect_platform(Platform).
Platform = linux.

% Check specific platforms
?- is_docker.
?- is_wsl.
?- can_execute_bash_directly.

% Platform-specific compilation
compile_for_platform(Pred, Code) :-
    detect_platform(Platform),
    (   Platform = windows
    ->  compile_for_windows(Pred, Code)
    ;   compile_for_unix(Pred, Code)
    ).
```

## Expected Outcome
- Platform correctly identified
- Bash availability determined
- Compilation adapted to platform capabilities
- Cross-platform compatibility achieved

## Citations
[1] src/unifyweaver/core/platform_detection.pl

