:- encoding(utf8).
% run_all_tests.pl - Control plane test runner

:- asserta(user:file_search_path(library, 'src/unifyweaver/core')).
:- asserta(user:file_search_path(library, 'tests/core')).

:- use_module(library(test_policy)).
:- use_module(library(test_data)).
:- use_module(library(test_compiler_driver)).
:- use_module(library(test_recursive_constraints)).

main :-
    writeln('--- Starting Control Plane Test Suite ---'),
    test_policy,
    test_compiler_driver,
    test_recursive_constraints,
    writeln('--- Control Plane Test Suite Finished ---').
