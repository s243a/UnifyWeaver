:- use_module('../src/unifyweaver/core/wam_rule_index').
:- use_module('../src/unifyweaver/core/wam_dict').
:- use_module('../src/unifyweaver/targets/wam_target').

:- dynamic test_failed/0.

% Test predicates — small facts only
:- discontiguous category_parent/2.
:- dynamic max_depth/1.
dimension_n(5). max_depth(10).
category_parent(a, b). category_parent(b, physics).
category_ancestor(Cat, Parent, 1, Visited) :-
    category_parent(Cat, Parent), \+ member(Parent, Visited).
category_ancestor(Cat, Ancestor, Hops, Visited) :-
    max_depth(MaxD), length(Visited, Depth), Depth < MaxD, !,
    category_parent(Cat, Mid), \+ member(Mid, Visited),
    category_ancestor(Mid, Ancestor, H1, [Mid|Visited]), Hops is H1 + 1.

pass(Test) :- format('[PASS] ~w~n', [Test]).
fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (test_failed -> true ; assert(test_failed)).

run_tests :-
    format('~n========================================~n'),
    format('WAM Rule Index Tests~n'),
    format('========================================~n~n'),

    % Test 1: Build rule index
    (   compile_predicate_to_wam(user:category_ancestor/4, [], Code),
        build_rule_index(Code, RI, _), length(RI, 2)
    ->  pass('build rule index: 2 rules')
    ;   fail_test('build rule index', 'expected 2 rules')),

    % Test 2: Clause 1 head pattern
    (   build_rule_index(Code, [rule(HP1, _, _)|_], _),
        member(match('A3', 1), HP1)
    ->  pass('clause 1 has match(A3, 1)')
    ;   fail_test('clause 1 head', 'missing A3=1 match')),

    % Test 3: Clause 2 no constant matches
    (   build_rule_index(Code, [_, rule(HP2, _, _)], _),
        HP2 == []
    ->  pass('clause 2 has no constant matches')
    ;   fail_test('clause 2 head', 'should have no matches')),

    % Test 4: Clause 1 bindings
    (   build_rule_index(Code, [rule(_, HB1, _)|_], _),
        member(bind('X1', 'A1'), HB1),
        member(bind('Y1', 'A2'), HB1),
        member(bind('Y2', 'A4'), HB1)
    ->  pass('clause 1 bindings correct')
    ;   fail_test('clause 1 bindings', 'wrong bindings')),

    % Test 5: Body starts with put_value
    (   build_rule_index(Code, [rule(_, _, [pc(_, generic(["put_value"|_]))|_])|_], _)
    ->  pass('clause 1 body starts with put_value')
    ;   fail_test('clause 1 body', 'wrong first body instr')),

    % Test 6: Dispatch to clause 1 when A3=1
    (   build_rule_index(Code, RI6, _),
        wam_dict_from_list(['A1'-a, 'A2'-physics, 'A3'-1, 'A4'-[a]], Regs6),
        rule_index_dispatch(RI6, Regs6, rule(HP6, _, _), _),
        member(match('A3', 1), HP6)
    ->  pass('dispatch selects clause 1 when A3=1')
    ;   fail_test('dispatch clause 1', 'wrong clause selected')),

    % Test 7: Dispatch to clause 2 when A3 is variable
    (   build_rule_index(Code, RI7, _),
        wam_dict_from_list(['A1'-a, 'A2'-physics, 'A3'-'_Q1', 'A4'-[a]], Regs7),
        rule_index_dispatch(RI7, Regs7, rule(HP7, _, _), _),
        HP7 == []
    ->  pass('dispatch selects clause 2 when A3 is variable')
    ;   fail_test('dispatch clause 2', 'wrong clause selected')),

    % Test 8: dimension_n/1 single rule
    (   compile_predicate_to_wam(user:dimension_n/1, [], Code8),
        build_rule_index(Code8, RI8, _),
        length(RI8, 1),
        RI8 = [rule(HP8, _, _)],
        member(match('A1', 5), HP8)
    ->  pass('dimension_n/1 has 1 rule with match(A1, 5)')
    ;   fail_test('dimension_n rule index', 'wrong')),

    format('~n========================================~n'),
    (   test_failed
    ->  format('Some tests FAILED~n'), halt(1)
    ;   format('All tests passed~n')
    ).

:- initialization(run_tests, main).
