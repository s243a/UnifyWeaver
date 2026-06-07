:- encoding(utf8).
%% Load-isolation test for recurrence_inputs.pl.
%%
%% Spawns a FRESH swipl process that loads ONLY the helper module
%% (plus its core dependencies). If the helper transitively depends
%% on any target module, the fresh load will fail with a missing-source
%% error and the spawned process exits non-zero.
%%
%% This is the stronger of the two target-agnosticism enforcement
%% layers (the other is the grep tripwire). Catches the failure
%% mode where a helper-of-a-helper accidentally imports a target
%% module that grep would miss.
%%
%% Usage:
%%   swipl -g run_tests -t halt tests/core/test_recurrence_inputs_isolated.pl

:- use_module(library(lists)).
:- use_module(library(process), [process_create/3, process_wait/2]).
:- use_module(library(filesex), [directory_file_path/3]).

%% Capture this test file's directory at load time so we can build
%% an absolute path to the helper module.
:- prolog_load_context(directory, Dir),
   asserta(test_file_directory(Dir)).

run_tests :-
    format("~n========================================~n"),
    format("recurrence_inputs Load-Isolation Test~n"),
    format("========================================~n~n"),
    test_file_directory(Dir),
    directory_file_path(Dir, '../../src/unifyweaver/core/recurrence_inputs.pl',
                        HelperPath),
    %% Spawn a fresh swipl that loads ONLY the helper and tries the
    %% basic predicates. If any transitive dependency on a target
    %% module exists, the fresh load fails.
    %%
    %% The goal: load the module, run a smoke check, halt with the
    %% smoke result's status.
    format(atom(Goal), "use_module('~w'), recurrence_inputs:build_workload_signals([], []), recurrence_inputs:build_recurrence_term(recursive_kernel(transitive_closure2, foo/2, []), [], _)",
           [HelperPath]),
    process_create(path(swipl),
                   ['-g', Goal, '-t', 'halt'],
                   [process(PID)]),
    process_wait(PID, exit(ExitCode)),
    (   ExitCode =:= 0
    ->  format("[PASS] recurrence_inputs loaded + smoke-tested in a fresh swipl with no target modules pre-loaded~n"),
        format("~nAll tests passed~n"),
        format("========================================~n")
    ;   format("[FAIL] fresh-swipl load isolation failed (exit ~w)~n", [ExitCode]),
        format("  This means recurrence_inputs.pl has a transitive dependency on a target~n"),
        format("  module that prevented it from loading in isolation.~n"),
        format("Tests FAILED~n"),
        halt(1)
    ).
