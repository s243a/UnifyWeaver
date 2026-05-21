:- module(test_recursive_csharp_target, [
    test_recursive_csharp_target/0
]).

:- use_module(library(recursive_compiler)).
:- use_module(library(stream_compiler)).
:- use_module(library(constraint_analyzer)).

test_recursive_csharp_target :-
    format('~n=== Testing recursive compiler C# target integration ===~n'),
    setup_test_data,
    test_non_recursive_facts_to_csharp,
    test_recursive_tail_to_csharp,
    cleanup_test_data,
    format('~n=== C# target recursive compiler tests complete ===~n').

setup_test_data :-
    cleanup_test_data,
    assertz(user:(test_cf_fact(a, b))),
    assertz(user:(test_cf_fact(b, c))),
    assertz(user:(test_cf_rec(X, Y) :- test_cf_fact(X, Y))),
    assertz(user:(test_cf_rec(X, Z) :- test_cf_fact(X, Y), test_cf_rec(Y, Z))).

cleanup_test_data :-
    catch(abolish(user:test_cf_fact/2), _, true),
    catch(abolish(user:test_cf_rec/2), _, true),
    clear_constraints(test_cf_fact/2),
    clear_constraints(test_cf_rec/2).

test_non_recursive_facts_to_csharp :-
    compile_recursive(test_cf_fact/2, [target(csharp)], Code),
    % Non-recursive csharp now routes through the query-plan backend
    % (same path as recursive csharp); facts compile to a QueryPlan
    % wrapping a RelationScanNode.  Distinct() only appears in
    % aggregation/set-recursion contexts — not for plain fact lookup —
    % so check for the distinctive query-plan markers instead.
    (   sub_string(Code, _, _, _, "namespace UnifyWeaver.Generated"),
        sub_string(Code, _, _, _, "RelationScanNode"),
        sub_string(Code, _, _, _, "QueryPlan") ->
        format('  ✓ Non-recursive predicate compiled to C# query plan~n')
    ;   format('  ✗ FAILED: Expected C# query-plan output~n'),
        fail
    ).

test_recursive_tail_to_csharp :-
    compile_recursive(test_cf_rec/2, [target(csharp)], Code),
    % test_cf_rec/2 is classified as transitive_closure(test_cf_fact),
    % so it lowers to the specialized TransitiveClosureNode rather
    % than the general FixpointNode + RecursiveRefNode + Delta form
    % (which the classifier only picks for predicates that aren't
    % shape-matched to transitive closure).  Check the markers the
    % renderer actually emits for this shape.
    (   sub_string(Code, _, _, _, "namespace UnifyWeaver.Generated"),
        sub_string(Code, _, _, _, "TransitiveClosureNode"),
        sub_string(Code, _, _, _, "QueryPlan") ->
        format('  ✓ Recursive predicate compiled to C# query plan~n')
    ;   format('  ✗ FAILED: Expected C# transitive-closure plan output~n'),
        fail
    ).
