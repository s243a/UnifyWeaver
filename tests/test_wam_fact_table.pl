:- use_module('../src/unifyweaver/core/wam_fact_table').
:- use_module('../src/unifyweaver/core/wam_dict').

:- dynamic test_failed/0.

% Test facts
:- dynamic test_parent/2.
test_parent(a, b).
test_parent(a, c).
test_parent(b, d).
test_parent(b, e).
test_parent(c, f).

:- dynamic test_single/1.
test_single(hello).
test_single(world).

:- dynamic test_rule/2.
test_rule(X, Y) :- test_parent(X, Y).  % This is a rule, not a fact

pass(Test) :- format('[PASS] ~w~n', [Test]).
fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (test_failed -> true ; assert(test_failed)).

test_build_fact_table :-
    Test = 'fact_table: build from asserted facts',
    (   build_fact_table(test_parent, 2, Table),
        fact_table_size(Table, 5)
    ->  pass(Test)
    ;   fail_test(Test, 'should have 5 facts')
    ).

test_lookup_bound :-
    Test = 'fact_table: lookup with bound first arg',
    (   build_fact_table(test_parent, 2, Table),
        wam_dict_lookup_all(a, Table, Matches),
        length(Matches, 2),
        Matches == [[b], [c]]
    ->  pass(Test)
    ;   fail_test(Test, 'should find 2 matches for a')
    ).

test_lookup_bound_single :-
    Test = 'fact_table: lookup with single match',
    (   build_fact_table(test_parent, 2, Table),
        wam_dict_lookup_all(c, Table, Matches),
        Matches == [[f]]
    ->  pass(Test)
    ;   fail_test(Test, 'should find 1 match for c')
    ).

test_lookup_missing :-
    Test = 'fact_table: lookup with no match',
    (   build_fact_table(test_parent, 2, Table),
        wam_dict_lookup_all(z, Table, Matches),
        Matches == []
    ->  pass(Test)
    ;   fail_test(Test, 'should find 0 matches for z')
    ).

test_is_fact_predicate :-
    Test = 'fact_table: is_fact_predicate for facts vs rules',
    (   is_fact_predicate(test_parent, 2),
        \+ is_fact_predicate(test_rule, 2)
    ->  pass(Test)
    ;   fail_test(Test, 'should distinguish facts from rules')
    ).

test_single_arg_facts :-
    Test = 'fact_table: single-argument facts',
    (   build_fact_table(test_single, 1, Table),
        fact_table_size(Table, 2),
        wam_dict_lookup_all(hello, Table, M1),
        M1 == [[]]  % empty rest-args for arity-1
    ->  pass(Test)
    ;   fail_test(Test, 'single-arg facts failed')
    ).

test_fact_table_with_real_data :-
    Test = 'fact_table: performance with many facts',
    % Create 1000 dynamic facts
    forall(between(1, 1000, I), (
        format(atom(Key), 'cat_~w', [I]),
        Mod is I mod 10,
        format(atom(Parent), 'parent_~w', [Mod]),
        assert(perf_test(Key, Parent))
    )),
    (   build_fact_table(perf_test, 2, Table),
        fact_table_size(Table, 1000),
        % Lookup should be fast
        wam_dict_lookup_all('cat_500', Table, Matches),
        length(Matches, 1)
    ->  pass(Test)
    ;   fail_test(Test, '1000-fact table failed')
    ),
    retractall(perf_test(_, _)).

run_tests :-
    format('~n========================================~n'),
    format('WAM Fact Table Tests~n'),
    format('========================================~n~n'),
    test_build_fact_table,
    test_lookup_bound,
    test_lookup_bound_single,
    test_lookup_missing,
    test_is_fact_predicate,
    test_single_arg_facts,
    test_fact_table_with_real_data,
    format('~n========================================~n'),
    (   test_failed
    ->  format('Some tests FAILED~n'), halt(1)
    ;   format('All tests passed~n')
    ).

:- initialization(run_tests, main).
