:- module(test_csharp_generator_aggregates, [
    test_csharp_generator_rejects_aggregates/0
]).

:- use_module(library(plunit)).
:- use_module('../../src/unifyweaver/targets/csharp_target.pl').

setup_test_data :-
    cleanup_test_data,
    assertz(user:deg_edge(a, b)),
    assertz(user:(deg_count(X, N) :- aggregate_all(count, Y, deg_edge(X, Y), N))).

cleanup_test_data :-
    maplist(retractall, [
        user:deg_edge(_, _),
        user:deg_count(_, _)
    ]),
    catch(abolish(user:deg_edge/2), _, true),
    catch(abolish(user:deg_count/2), _, true).

:- begin_tests(csharp_generator_aggregates).

test(rejects_aggregate_all, [
    setup(setup_test_data),
    cleanup(cleanup_test_data)
]) :-
    \+ csharp_target:compile_predicate_to_csharp(deg_count/2, [mode(generator)], _).

:- end_tests(csharp_generator_aggregates).

test_csharp_generator_rejects_aggregates :-
    run_tests(csharp_generator_aggregates).
