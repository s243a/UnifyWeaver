:- module(test_csharp_generator_depgroups, [
    test_csharp_generator_multi_support/0
]).

:- use_module(library(plunit)).
:- use_module('../../src/unifyweaver/targets/csharp_target.pl').

setup_data :-
    cleanup_data,
    assertz(user:dg_base(a)),
    assertz(user:dg_base(b)),
    assertz(user:dg_link(a, b)),
    assertz(user:(dg_mid(X, Y) :- dg_base(X), dg_link(X, Y))),
    assertz(user:(dg_target(X, Y) :- dg_mid(X, Y), dg_base(Y))),
    % join scenario: base1/base2 supports dg_join
    assertz(user:dg_b1(a, 1)),
    assertz(user:dg_b1(b, 2)),
    assertz(user:dg_b2(1, x)),
    assertz(user:dg_b2(2, y)),
    assertz(user:(dg_join(X, Z) :- dg_b1(X, N), dg_b2(N, Z))).

cleanup_data :-
    maplist(retractall, [
        user:dg_base(_),
        user:dg_link(_, _),
        user:dg_mid(_, _),
        user:dg_target(_, _),
        user:dg_b1(_, _),
        user:dg_b2(_, _),
        user:dg_join(_, _)
    ]),
    catch(abolish(user:dg_base/1), _, true),
    catch(abolish(user:dg_link/2), _, true),
    catch(abolish(user:dg_mid/2), _, true),
    catch(abolish(user:dg_target/2), _, true),
    catch(abolish(user:dg_b1/2), _, true),
    catch(abolish(user:dg_b2/2), _, true),
    catch(abolish(user:dg_join/2), _, true).

:- begin_tests(csharp_generator_depgroups).

test(multi_supporting_predicates, [
    setup(setup_data),
    cleanup(cleanup_data)
]) :-
    csharp_target:compile_predicate_to_csharp(dg_target/2, [mode(generator)], Cs),
    sub_string(Cs, _, _, _, "class DgTarget_Module"),
    sub_string(Cs, _, _, _, "dg_mid"),
    sub_string(Cs, _, _, _, "dg_base"),
    sub_string(Cs, _, _, _, "dg_link"),
    !.

test(join_supporting_predicates, [
    setup(setup_data),
    cleanup(cleanup_data)
]) :-
    csharp_target:compile_predicate_to_csharp(dg_join/2, [mode(generator)], Cs),
    sub_string(Cs, _, _, _, "class DgJoin_Module"),
    sub_string(Cs, _, _, _, "dg_b1"),
    sub_string(Cs, _, _, _, "dg_b2"),
    sub_string(Cs, _, _, _, "ApplyRule_1"),
    !.

:- end_tests(csharp_generator_depgroups).

test_csharp_generator_multi_support :-
    run_tests(csharp_generator_depgroups).
