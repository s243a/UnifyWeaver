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

main :-
    writeln('--- Starting Control Plane Test Suite ---'),
    test_policy,
    test_compiler_driver,
    test_recursive_constraints,
    test_recursive_csharp_target,
    test_csharp_query_target,
    test_common_generator,
    test_cross_generator,
    writeln('--- Control Plane Test Suite Finished ---').
