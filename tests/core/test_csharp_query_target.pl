:- module(test_csharp_query_target, [
    test_csharp_query_target/0
]).

:- use_module(library(csharp_query_target)).

test_csharp_query_target :-
    writeln('=== Testing C# query target (fact plans) ==='),
    setup_test_data,
    (   csharp_query_target:build_query_plan(test_fact/2, [target(csharp_query)], Plan)
    ->  verify_fact_plan(Plan),
        csharp_query_target:render_plan_to_csharp(Plan, Source),
        verify_csharp_stub(Source),
        writeln('  OK: fact-only predicate emits relation scan plan and C# stub')
    ;   writeln('  FAIL: build_query_plan/3 did not succeed'),
        fail
    ),
    cleanup_test_data,
    writeln('=== C# query target tests complete ===').

setup_test_data :-
    retractall(user:test_fact(_, _)),
    assertz(user:test_fact(alice, bob)),
    assertz(user:test_fact(bob, charlie)).

cleanup_test_data :-
    retractall(user:test_fact(_, _)).

verify_fact_plan(Plan) :-
    get_dict(head, Plan, predicate{name:test_fact, arity:2}),
    get_dict(is_recursive, Plan, false),
    get_dict(root, Plan, relation_scan{predicate:predicate{name:test_fact, arity:2}}),
    get_dict(relations, Plan, [Relation]),
    get_dict(facts, Relation, Facts),
    Facts == [[alice, bob], [bob, charlie]].

verify_csharp_stub(Source) :-
    sub_string(Source, _, _, _, 'provider.AddFact'),
    sub_string(Source, _, _, _, 'QueryPlan').
