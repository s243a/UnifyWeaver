:- encoding(utf8).
% run_all_tests.pl - Control plane test runner

:- asserta(user:file_search_path(library, 'src/unifyweaver/core')).
:- asserta(user:file_search_path(library, 'tests/core')).

:- use_module(library(test_policy)).

main :-
    writeln('--- Starting Control Plane Test Suite ---'),
    test_policy,
    writeln('--- Control Plane Test Suite Finished ---').
