:- module(test_csharp_generator_strat_neg, [
    test_csharp_generator_stratified_negation/0
]).

:- use_module(library(plunit)).
:- use_module('../../src/unifyweaver/targets/csharp_target.pl').

setup_stratified :-
    cleanup_all,
    assertz(user:s_edge(a, b)),
    assertz(user:s_blocked(a, b)),
    assertz(user:(s_path(X, Y) :- s_edge(X, Y), \+ s_blocked(X, Y))).

setup_unstratified :-
    cleanup_all,
    assertz(user:(u_p(X) :- \+ u_q(X))),
    assertz(user:(u_q(X) :- u_p(X))).

cleanup_all :-
    maplist(retractall, [
        user:s_edge(_, _),
        user:s_blocked(_, _),
        user:s_path(_, _),
        user:u_p(_),
        user:u_q(_)
    ]),
    catch(abolish(user:s_edge/2), _, true),
    catch(abolish(user:s_blocked/2), _, true),
    catch(abolish(user:s_path/2), _, true),
    catch(abolish(user:u_p/1), _, true),
    catch(abolish(user:u_q/1), _, true).

:- begin_tests(csharp_generator_strat_neg).

test(stratified_negation_allowed, [
    setup(setup_stratified),
    cleanup(cleanup_all)
]) :-
    csharp_target:compile_predicate_to_csharp(s_path/2, [mode(generator)], Code),
    string_length(Code, Len),
    Len > 0.

test(non_stratified_negation_rejected, [
    setup(setup_unstratified),
    cleanup(cleanup_all)
]) :-
    \+ csharp_target:compile_predicate_to_csharp(u_p/1, [mode(generator)], _).

:- end_tests(csharp_generator_strat_neg).

test_csharp_generator_stratified_negation :-
    run_tests(csharp_generator_strat_neg).

