:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% platform_detection.pl - Detect execution environment and platform
% Determines whether we're on Windows, WSL, Docker, or native Linux

:- module(platform_detection, [
    detect_platform/1,           % -Platform
    detect_execution_mode/1,     % -Mode
    is_windows/0,
    is_wsl/0,
    is_docker/0,
    is_native_linux/0,
    can_execute_bash_directly/0
]).

:- use_module(library(process)).

%% ============================================
%% PLATFORM DETECTION
%% ============================================

%% detect_platform(-Platform)
%  Detect the current platform
%  Platform is one of: windows, wsl, docker, linux, macos, unknown
detect_platform(Platform) :-
    (   is_docker
    ->  Platform = docker
    ;   is_wsl
    ->  Platform = wsl
    ;   is_windows
    ->  Platform = windows
    ;   is_macos
    ->  Platform = macos
    ;   is_linux
    ->  Platform = linux
    ;   Platform = unknown
    ).

%% is_windows/0
%  True if running on native Windows (not WSL)
is_windows :-
    current_prolog_flag(windows, true),
    \+ is_wsl.

%% is_wsl/0
%  True if running in Windows Subsystem for Linux
is_wsl :-
    (   getenv('WSL_DISTRO_NAME', _)
    ->  true
    ;   getenv('WSL_INTEROP', _)
    ->  true
    ;   % Check for WSL in /proc/version
        catch(
            (   open('/proc/version', read, Stream),
                read_string(Stream, _, Content),
                close(Stream),
                sub_string(Content, _, _, _, "Microsoft")
            ),
            _,
            fail
        )
    ).

%% is_docker/0
%  True if running inside a Docker container
is_docker :-
    % Check for .dockerenv file
    (   exists_file('/.dockerenv')
    ->  true
    % Check /proc/1/cgroup for docker
    ;   catch(
            (   open('/proc/1/cgroup', read, Stream),
                read_string(Stream, _, Content),
                close(Stream),
                (   sub_string(Content, _, _, _, "docker")
                ;   sub_string(Content, _, _, _, "/lxc/")
                )
            ),
            _,
            fail
        )
    ).

%% is_linux/0
%  True if running on native Linux (not Docker, not WSL)
is_linux :-
    current_prolog_flag(unix, true),
    \+ is_wsl,
    \+ is_docker,
    \+ is_macos.

%% is_native_linux/0
%  True if running on any Linux-like environment (includes WSL and Docker)
is_native_linux :-
    current_prolog_flag(unix, true).

%% is_macos/0
%  True if running on macOS
is_macos :-
    current_prolog_flag(apple, true).

%% ============================================
%% EXECUTION MODE DETECTION
%% ============================================

%% detect_execution_mode(-Mode)
%  Detect the appropriate execution mode for bash scripts
%  Mode is one of:
%    - direct_bash    (Linux, WSL, Docker, macOS - execute bash directly)
%    - powershell_wsl (Windows PowerShell via WSL compatibility layer)
%    - powershell_cygwin (Windows PowerShell via Cygwin compatibility layer)
%    - unknown
detect_execution_mode(Mode) :-
    detect_platform(Platform),
    platform_execution_mode(Platform, Mode).

platform_execution_mode(docker, direct_bash).
platform_execution_mode(wsl, direct_bash).
platform_execution_mode(linux, direct_bash).
platform_execution_mode(macos, direct_bash).
platform_execution_mode(windows, powershell_wsl).  % Default to WSL on Windows
platform_execution_mode(unknown, unknown).

%% can_execute_bash_directly/0
%  True if we can execute bash scripts directly without wrappers
can_execute_bash_directly :-
    detect_execution_mode(Mode),
    Mode = direct_bash.

%% ============================================
%% UTILITY PREDICATES
%% ============================================

%% get_platform_info(-Info)
%  Get detailed platform information
get_platform_info(info(
    platform(Platform),
    execution_mode(ExecMode),
    can_exec_bash(CanExecBash),
    prolog_flags(Flags)
)) :-
    detect_platform(Platform),
    detect_execution_mode(ExecMode),
    (   can_execute_bash_directly
    ->  CanExecBash = true
    ;   CanExecBash = false
    ),
    findall(Flag=Value, relevant_prolog_flag(Flag, Value), Flags).

relevant_prolog_flag(windows, Value) :- current_prolog_flag(windows, Value).
relevant_prolog_flag(unix, Value) :- current_prolog_flag(unix, Value).
relevant_prolog_flag(apple, Value) :- current_prolog_flag(apple, Value).
relevant_prolog_flag(arch, Value) :- current_prolog_flag(arch, Value).
relevant_prolog_flag(version_data, Value) :- current_prolog_flag(version_data, Value).

%% ============================================
%% TESTING
%% ============================================

test_platform_detection :-
    format('~n=== Platform Detection Test ===~n', []),

    detect_platform(Platform),
    format('Platform: ~w~n', [Platform]),

    detect_execution_mode(Mode),
    format('Execution Mode: ~w~n', [Mode]),

    format('~nPlatform Checks:~n', []),
    test_platform_check('Windows', is_windows),
    test_platform_check('WSL', is_wsl),
    test_platform_check('Docker', is_docker),
    test_platform_check('Native Linux', is_linux),
    test_platform_check('macOS', is_macos),

    format('~nExecution Capability:~n', []),
    (   can_execute_bash_directly
    ->  format('  Can execute bash directly: YES~n', [])
    ;   format('  Can execute bash directly: NO (need compatibility layer)~n', [])
    ),

    format('~n=== Test Complete ===~n', []).

test_platform_check(Name, Goal) :-
    (   call(Goal)
    ->  format('  ~w: YES~n', [Name])
    ;   format('  ~w: NO~n', [Name])
    ).
