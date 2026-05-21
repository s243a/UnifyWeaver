:- encoding(utf8).
% run_all_tests.pl - Control plane test runner

:- asserta(user:file_search_path(library, 'src/unifyweaver/core')).
:- asserta(user:file_search_path(library, 'src/unifyweaver/targets')).
:- asserta(user:file_search_path(library, 'tests/core')).

:- use_module(library(test_policy)).
:- use_module(library(test_data)).
:- use_module(library(test_compiler_driver)).
:- use_module(library(test_recursive_constraints)).
:- use_module(library(test_recursive_csharp_target)).
:- use_module(library(test_csharp_query_target)).
:- use_module(library(test_common_generator)).
:- use_module(library(test_cross_generator)).
:- use_module(library(process)).

%% The F# WAM target tests use `:- initialization(run_tests, main)` +
%% `halt(0/1)` and so can't be loaded as control-plane modules.  Spawn
%% the aggregate runner as a subprocess; it returns 0 iff every F# WAM
%% test exits 0.  The dotnet smokes inside it skip gracefully when
%% `dotnet` isn't on PATH.  A non-zero status fails the whole suite.
run_wam_fsharp_suite :-
    writeln('--- F# WAM target suite (subprocess) ---'),
    catch(
        (   process_create(path(swipl),
                ['-q', '-g', 'main', '-t', 'halt',
                 'tests/run_wam_fsharp_tests.pl'],
                [stdout(std), stderr(std), process(Pid)]),
            process_wait(Pid, exit(EC))
        ),
        Err,
        (   format('  [orchestrator spawn failed] ~q~n', [Err]),
            EC = 99
        )),
    (   EC == 0
    ->  writeln('--- F# WAM target suite: OK ---')
    ;   format('--- F# WAM target suite: FAILED (exit ~w) ---~n', [EC]),
        fail
    ).

main :-
    writeln('--- Starting Control Plane Test Suite ---'),
    %% run_wam_fsharp_suite runs FIRST.  Downstream tests
    %% (test_compiler_driver in particular) call halt/1 on failure,
    %% which exits the whole swipl process before later steps can
    %% run.  Spawning the F# WAM suite up front guarantees it
    %% executes regardless of what halts later.  The F# suite itself
    %% is subprocessed, so a failure there returns control to main/0
    %% rather than aborting.
    run_wam_fsharp_suite,
    test_policy,
    test_compiler_driver,
    test_recursive_constraints,
    test_recursive_csharp_target,
    test_csharp_query_target,
    test_common_generator,
    test_cross_generator,
    writeln('--- Control Plane Test Suite Finished ---').
