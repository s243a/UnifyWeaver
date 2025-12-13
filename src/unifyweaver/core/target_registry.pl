/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 John William Creighton (s243a)
 *
 * Target Registry - Central registry for UnifyWeaver compilation targets
 *
 * This module manages metadata about available targets, their runtime families,
 * and capabilities. It enables cross-target glue to determine how targets
 * should communicate.
 */

:- module(target_registry, [
    % Target registration
    register_target/3,          % register_target(Name, Family, Capabilities)
    unregister_target/1,        % unregister_target(Name)

    % Target queries
    target_exists/1,            % target_exists(Name)
    target_family/2,            % target_family(Target, Family)
    target_capabilities/2,      % target_capabilities(Target, Capabilities)
    targets_same_family/2,      % targets_same_family(Target1, Target2)

    % Family queries
    family_targets/2,           % family_targets(Family, Targets)

    % Listing
    list_targets/1,             % list_targets(Targets)
    list_families/1,            % list_families(Families)

    % Location defaults
    default_location/2,         % default_location(Target, Location)
    default_transport/3         % default_transport(Location1, Location2, Transport)
]).

:- use_module(library(lists)).

%% ============================================
%% Dynamic storage for targets
%% ============================================

:- dynamic registered_target/3.  % registered_target(Name, Family, Capabilities)

%% ============================================
%% Target Registration
%% ============================================

%% register_target(+Name, +Family, +Capabilities)
%  Register a new target with its runtime family and capabilities.
%  Capabilities is a list of atoms describing what the target supports.
%
%  Example:
%    register_target(python, python, [streaming, pipes, libraries, ml])
%
register_target(Name, Family, Capabilities) :-
    atom(Name),
    atom(Family),
    is_list(Capabilities),
    (   registered_target(Name, _, _)
    ->  retract(registered_target(Name, _, _))
    ;   true
    ),
    assertz(registered_target(Name, Family, Capabilities)).

%% unregister_target(+Name)
%  Remove a target from the registry.
%
unregister_target(Name) :-
    retractall(registered_target(Name, _, _)).

%% ============================================
%% Target Queries
%% ============================================

%% target_exists(+Name)
%  True if a target with the given name is registered.
%
target_exists(Name) :-
    registered_target(Name, _, _).

%% target_family(?Target, ?Family)
%  Query or check the runtime family of a target.
%
target_family(Target, Family) :-
    registered_target(Target, Family, _).

%% target_capabilities(?Target, ?Capabilities)
%  Query the capabilities list of a target.
%
target_capabilities(Target, Capabilities) :-
    registered_target(Target, _, Capabilities).

%% targets_same_family(+Target1, +Target2)
%  True if both targets belong to the same runtime family.
%  Targets in the same family can communicate in-process.
%
targets_same_family(Target1, Target2) :-
    target_family(Target1, Family),
    target_family(Target2, Family).

%% ============================================
%% Family Queries
%% ============================================

%% family_targets(+Family, -Targets)
%  Get all targets belonging to a runtime family.
%
family_targets(Family, Targets) :-
    findall(T, target_family(T, Family), Targets).

%% ============================================
%% Listing
%% ============================================

%% list_targets(-Targets)
%  Get a list of all registered target names.
%
list_targets(Targets) :-
    findall(Name, registered_target(Name, _, _), Targets).

%% list_families(-Families)
%  Get a list of all unique runtime families.
%
list_families(Families) :-
    findall(F, registered_target(_, F, _), AllFamilies),
    sort(AllFamilies, Families).

%% ============================================
%% Location and Transport Defaults
%% ============================================

%% default_location(+Target, -Location)
%  Get the default location type for a target.
%  Most targets default to local_process.
%
default_location(Target, in_process) :-
    target_family(Target, Family),
    in_process_family(Family),
    !.
default_location(_, local_process).

%% in_process_family(+Family)
%  Families that support in-process communication.
%
in_process_family(dotnet).
in_process_family(jvm).

%% default_transport(+Location1, +Location2, -Transport)
%  Get the default transport for communicating between two locations.
%
default_transport(in_process, in_process, direct) :- !.
default_transport(local_process, local_process, pipe) :- !.
default_transport(local_process, remote(_), http) :- !.
default_transport(remote(_), local_process, http) :- !.
default_transport(remote(_), remote(_), http) :- !.
default_transport(_, _, pipe).  % Fallback

%% ============================================
%% Built-in Target Definitions
%% ============================================

%% Initialize built-in targets on module load
register_builtin_targets :-
    % Shell family - always separate processes, pipe communication
    register_target(bash, shell, [streaming, pipes, process_control, scripting]),
    register_target(awk, shell, [streaming, pipes, regex, aggregation, text_processing]),

    % Python family
    register_target(python, python, [streaming, pipes, libraries, ml, data_science]),
    register_target(ironpython, dotnet, [streaming, dotnet_interop, scripting]),

    % .NET family - can share process
    register_target(csharp, dotnet, [compiled, streaming, linq, async]),
    register_target(fsharp, dotnet, [compiled, streaming, functional, linq]),
    register_target(powershell, dotnet, [scripting, streaming, system_admin, dotnet_interop]),

    % JVM family - can share process
    register_target(java, jvm, [compiled, streaming, enterprise]),
    register_target(scala, jvm, [compiled, streaming, functional]),
    register_target(clojure, jvm, [compiled, streaming, functional, lisp]),
    register_target(jython, jvm, [scripting, jvm_interop]),
    register_target(kotlin, jvm, [compiled, streaming, android, coroutines]),

    % Native family - compiled binaries
    register_target(go, native, [compiled, streaming, concurrency, cross_platform]),
    register_target(rust, native, [compiled, streaming, memory_safe, performance]),
    register_target(c, native, [compiled, performance, low_level]),

    % Database
    register_target(sql, database, [queries, transactions, relational]).

:- register_builtin_targets.

%% ============================================
%% Capability Queries
%% ============================================

%% target_has_capability(+Target, +Capability)
%  Check if a target has a specific capability.
%
target_has_capability(Target, Capability) :-
    target_capabilities(Target, Caps),
    member(Capability, Caps).

%% targets_with_capability(+Capability, -Targets)
%  Find all targets that have a specific capability.
%
targets_with_capability(Capability, Targets) :-
    findall(T, target_has_capability(T, Capability), Targets).
