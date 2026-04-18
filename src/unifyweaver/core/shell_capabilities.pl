% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% This file is part of UnifyWeaver.
% Licensed under either MIT or Apache-2.0 at your option.

:- encoding(utf8).
%% shell_capabilities.pl - Shell Environment Capability Detection
%
% Declares what the target shell environment supports. The compiler
% uses these capabilities to choose between idiomatic bash and
% portable workarounds (e.g. avoiding $$ on WASM where getpid()
% is unavailable).
%
% Capabilities are auto-detected at load time based on the Prolog
% runtime's architecture flags. Users can override via assert/retract
% or compile options like shell_env(Cap).
%
% See: docs/design/BRUSH_WASM_COMPATIBILITY.md

:- module(shell_capabilities, [
    shell_has/1,
    shell_lacks/1,
    shell_is_wasm/0,
    set_shell_capability/1,
    clear_shell_capability/1
]).

:- dynamic shell_capability/1.

%% shell_has(+Capability) is semidet.
%
%  True if the shell environment has the given capability.
%
%  Known capabilities:
%    signals     - OS signal delivery (SIGPIPE, SIGTERM, etc.)
%    procfs      - /dev/fd/NN file descriptor pseudo-filesystem
%    syscalls    - POSIX syscalls (getpid, fork, exec, etc.)
%    processes   - multiple concurrent processes / subshells
%
shell_has(Cap) :- shell_capability(Cap).

%% shell_lacks(+Capability) is semidet.
%
%  True if the shell environment does NOT have the given capability.
%
shell_lacks(Cap) :- \+ shell_capability(Cap).

%% shell_is_wasm is semidet.
%
%  True if running on a WebAssembly target (wasm32).
%
shell_is_wasm :-
    current_prolog_flag(arch, Arch),
    atom_string(Arch, ArchStr),
    sub_string(ArchStr, _, _, _, "wasm").

%% set_shell_capability(+Capability) is det.
%
%  Declare that the shell environment has a capability.
%
set_shell_capability(Cap) :-
    (   shell_capability(Cap) -> true
    ;   assertz(shell_capability(Cap))
    ).

%% clear_shell_capability(+Capability) is det.
%
%  Declare that the shell environment lacks a capability.
%
clear_shell_capability(Cap) :-
    retractall(shell_capability(Cap)).

% ============================================================================
% AUTO-DETECTION AT LOAD TIME
% ============================================================================

:- (   shell_is_wasm
   ->  % WASM: no OS-level features available.
       % All I/O goes through JS VFS bridge.
       true
   ;   % Native: full POSIX capabilities.
       set_shell_capability(signals),
       set_shell_capability(procfs),
       set_shell_capability(syscalls),
       set_shell_capability(processes)
   ).
