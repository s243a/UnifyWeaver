:- module(test_prolog_target, [run_prolog_target_tests/0]).

:- use_module(library(plunit)).
:- use_module('../../src/unifyweaver/targets/prolog_target').

run_prolog_target_tests :-
    run_tests([prolog_target]).

:- begin_tests(prolog_target).

setup_branch_pruning_fixture :-
    cleanup_branch_pruning_fixture,
    assertz(user:test_ppv_edge(a, b)),
    assertz(user:test_ppv_edge(b, c)),
    assertz(user:test_ppv_edge(b, blocked)),
    assertz(user:(test_ppv_reach(Cat, Ancestor, Hops, Visited) :-
        test_ppv_edge(Cat, Ancestor),
        \+ member(Ancestor, Visited),
        Hops = 1)),
    assertz(user:(test_ppv_reach(Cat, Ancestor, Hops, Visited) :-
        test_ppv_edge(Cat, Mid),
        Mid \= blocked,
        \+ member(Mid, Visited),
        test_ppv_reach(Mid, Ancestor, H1, [Mid|Visited]),
        Hops is H1 + 1)),
    assertz(user:mode(test_ppv_reach(-, +, -, +))).

cleanup_branch_pruning_fixture :-
    retractall(user:test_ppv_edge(_, _)),
    retractall(user:test_ppv_reach(_, _, _, _)),
    retractall(user:mode(test_ppv_reach(_, _, _, _))).

setup_no_mode_fixture :-
    setup_branch_pruning_fixture,
    retractall(user:mode(test_ppv_reach(_, _, _, _))).

test(emits_branch_pruning_helpers_for_parameterized_ppv,
        [setup(setup_branch_pruning_fixture),
         cleanup(cleanup_branch_pruning_fixture)]) :-
    once(generate_prolog_script([test_ppv_reach/4], [dialect(swi)], Code)),
    once(sub_atom(Code, _, _, _, 'test_ppv_reach$prune')),
    once(sub_atom(Code, _, _, _, 'test_ppv_reach$prune_guard')),
    once(sub_atom(Code, _, _, _, 'test_ppv_reach$prune_cache')),
    once(sub_atom(Code, _, _, _, 'test_ppv_reach$pruned')),
    once(sub_atom(Code, _, _, _, ':- table')).

test(skips_branch_pruning_without_mode_signal,
        [setup(setup_no_mode_fixture),
         cleanup(cleanup_branch_pruning_fixture)]) :-
    once(generate_prolog_script([test_ppv_reach/4], [dialect(swi)], Code)),
    \+ sub_atom(Code, _, _, _, 'test_ppv_reach$prune'),
    \+ sub_atom(Code, _, _, _, 'test_ppv_reach$pruned').

:- end_tests(prolog_target).
