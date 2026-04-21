:- encoding(utf8).
%% Test suite for demand_analysis.pl
%%
%% Usage:
%%   swipl -g run_tests -t halt tests/core/test_demand_analysis.pl

:- use_module('../../src/unifyweaver/core/demand_analysis').
:- use_module(library(lists)).

%% ========================================================================
%% Test runner
%% ========================================================================

run_tests :-
    format("~n========================================~n"),
    format("Demand Analysis Tests~n"),
    format("========================================~n~n"),
    findall(Test, test(Test), Tests),
    length(Tests, Total),
    run_all(Tests, 0, Passed),
    format("~n========================================~n"),
    (   Passed =:= Total
    ->  format("All ~w tests passed~n", [Total])
    ;   Failed is Total - Passed,
        format("~w of ~w tests FAILED~n", [Failed, Total]),
        format("Tests FAILED~n"),
        halt(1)
    ),
    format("========================================~n").

run_all([], Passed, Passed).
run_all([Test|Rest], Acc, Passed) :-
    (   catch(call(Test), Error,
            (format("[FAIL] ~w: ~w~n", [Test, Error]), fail))
    ->  Acc1 is Acc + 1,
        run_all(Rest, Acc1, Passed)
    ;   run_all(Rest, Acc, Passed)
    ).

pass(Name) :- format("[PASS] ~w~n", [Name]).
fail_test(Name, Reason) :- format("[FAIL] ~w: ~w~n", [Name, Reason]), fail.

%% ========================================================================
%% Test declarations
%% ========================================================================

test(test_category_ancestor_eligible).
test(test_category_ancestor_spec_fields).
test(test_non_recursive_rejected).
test(test_no_mode_rejected).
test(test_aggregate_in_prefix_rejected).
test(test_edge_pred_extraction).
test(test_target_arg_detection).
test(test_guard_point_detection).
test(test_edge_direction_detection).
test(test_simple_transitive_closure).
test(test_is_demand_eligible_convenience).
test(test_guard_pred_name).
test(test_generate_guarded_clauses).
test(test_generate_demand_init).

%% ========================================================================
%% Setup: mock mode declarations
%% ========================================================================

%% category_ancestor(+Cat, +Root, -Hops, +Visited)
:- assert(user:mode(category_ancestor(+, +, -, +))).

%% closure(+Source, -Target)
:- assert(user:mode(closure(+, -))).

%% ========================================================================
%% Test clauses — mirror effective_distance.pl's category_ancestor/4
%% ========================================================================

ca_clauses([
    % Base clause
    (category_ancestor(Cat, Parent, 1, Visited) :-
        (category_parent(Cat, Parent),
         \+ member(Parent, Visited))),
    % Recursive clause
    (category_ancestor(Cat, Ancestor, Hops, Visited) :-
        (max_depth(MaxD),
         length(Visited, Depth),
         Depth < MaxD, !,
         category_parent(Cat, Mid),
         \+ member(Mid, Visited),
         category_ancestor(Mid, Ancestor, H1, [Mid|Visited]),
         Hops is H1 + 1))
]).

%% Helper to convert full clauses to Head-Body pairs
clauses_to_pairs([], []).
clauses_to_pairs([(H :- B)|Rest], [H-B|RestPairs]) :-
    clauses_to_pairs(Rest, RestPairs).

%% ========================================================================
%% Tests
%% ========================================================================

test_category_ancestor_eligible :-
    Test = 'Demand: category_ancestor/4 is demand-eligible',
    ca_clauses(Clauses),
    clauses_to_pairs(Clauses, Pairs),
    (   detect_demand_eligible(category_ancestor, 4, Pairs, _Spec)
    ->  pass(Test)
    ;   fail_test(Test, 'detection failed')
    ).

test_category_ancestor_spec_fields :-
    Test = 'Demand: spec has correct edge pred and target arg',
    ca_clauses(Clauses),
    clauses_to_pairs(Clauses, Pairs),
    (   detect_demand_eligible(category_ancestor, 4, Pairs, Spec),
        demand_spec_edge_pred(Spec, EP),
        EP = category_parent/2,
        demand_spec_target_arg(Spec, TA),
        integer(TA),
        TA =:= 2,
        demand_spec_input_positions(Spec, IPs),
        IPs = [0, 1, 3]   % +Cat(0), +Root(1), +Visited(3)
    ->  pass(Test)
    ;   fail_test(Test, 'spec fields incorrect')
    ).

test_non_recursive_rejected :-
    Test = 'Demand: non-recursive predicate rejected',
    Clauses = [
        (foo(X, Y) :- bar(X, Y))
    ],
    clauses_to_pairs(Clauses, Pairs),
    (   \+ detect_demand_eligible(foo, 2, Pairs, _)
    ->  pass(Test)
    ;   fail_test(Test, 'should have been rejected')
    ).

test_no_mode_rejected :-
    Test = 'Demand: predicate without mode declaration rejected',
    Clauses = [
        (no_mode_pred(X, Y) :- (edge(X, Z), no_mode_pred(Z, Y)))
    ],
    clauses_to_pairs(Clauses, Pairs),
    (   \+ detect_demand_eligible(no_mode_pred, 2, Pairs, _)
    ->  pass(Test)
    ;   fail_test(Test, 'should have been rejected (no mode)')
    ).

:- assert(user:mode(agg_pred(+, -))).

test_aggregate_in_prefix_rejected :-
    Test = 'Demand: aggregate before recursive call rejected',
    Clauses = [
        (agg_pred(X, Y) :-
            (findall(Z, edge(X, Z), Zs),
             member(Z, Zs),
             agg_pred(Z, Y)))
    ],
    clauses_to_pairs(Clauses, Pairs),
    (   \+ detect_demand_eligible(agg_pred, 2, Pairs, _)
    ->  pass(Test)
    ;   fail_test(Test, 'should have been rejected (aggregate in prefix)')
    ).

test_edge_pred_extraction :-
    Test = 'Demand: extracts category_parent/2 as edge predicate',
    ca_clauses(Clauses),
    clauses_to_pairs(Clauses, Pairs),
    (   detect_demand_eligible(category_ancestor, 4, Pairs, Spec),
        demand_spec_edge_pred(Spec, category_parent/2)
    ->  pass(Test)
    ;   fail_test(Test, 'wrong edge predicate')
    ).

test_target_arg_detection :-
    Test = 'Demand: detects arg 2 (Ancestor/Root) as target',
    ca_clauses(Clauses),
    clauses_to_pairs(Clauses, Pairs),
    (   detect_demand_eligible(category_ancestor, 4, Pairs, Spec),
        demand_spec_target_arg(Spec, 2)
    ->  pass(Test)
    ;   fail_test(Test, 'wrong target arg')
    ).

test_guard_point_detection :-
    Test = 'Demand: guard insertion point found after edge call',
    ca_clauses(Clauses),
    clauses_to_pairs(Clauses, Pairs),
    (   detect_demand_eligible(category_ancestor, 4, Pairs, Spec),
        demand_spec_guard_points(Spec, GuardPoints),
        GuardPoints \= [],
        member(guard_point(_, GoalIdx, _), GuardPoints),
        integer(GoalIdx)
    ->  pass(Test)
    ;   fail_test(Test, 'no guard points found')
    ).

test_edge_direction_detection :-
    Test = 'Demand: edge direction detected (from=1, to=2 for category_parent)',
    ca_clauses(Clauses),
    clauses_to_pairs(Clauses, Pairs),
    (   detect_demand_eligible(category_ancestor, 4, Pairs, Spec),
        demand_spec_edge_direction(Spec, direction(FromPos, ToPos)),
        % category_parent(Cat, Mid) — Cat is from (pos 1), Mid is to (pos 2)
        % but Mid feeds into recursive call arg 1, so ToPos should match
        integer(FromPos),
        integer(ToPos)
    ->  pass(Test)
    ;   fail_test(Test, 'edge direction detection failed')
    ).

test_simple_transitive_closure :-
    Test = 'Demand: simple closure/2 is demand-eligible',
    Clauses = [
        (closure(X, Y) :- edge(X, Y)),
        (closure(X, Y) :- (edge(X, Z), closure(Z, Y)))
    ],
    clauses_to_pairs(Clauses, Pairs),
    (   detect_demand_eligible(closure, 2, Pairs, Spec),
        demand_spec_edge_pred(Spec, edge/2)
    ->  pass(Test)
    ;   fail_test(Test, 'simple closure detection failed')
    ).

test_is_demand_eligible_convenience :-
    Test = 'Demand: is_demand_eligible/3 convenience predicate',
    ca_clauses(Clauses),
    clauses_to_pairs(Clauses, Pairs),
    (   is_demand_eligible(category_ancestor, 4, Pairs)
    ->  pass(Test)
    ;   fail_test(Test, 'convenience predicate failed')
    ).

%% ========================================================================
%% Phase D2 tests — guard emission
%% ========================================================================

test_guard_pred_name :-
    Test = 'Demand D2: guard pred name derived from edge pred',
    ca_clauses(Clauses),
    clauses_to_pairs(Clauses, Pairs),
    (   detect_demand_eligible(category_ancestor, 4, Pairs, Spec),
        demand_guard_pred_name(Spec, Name),
        Name == can_reach_via_category_parent
    ->  pass(Test)
    ;   fail_test(Test, 'wrong guard pred name')
    ).

test_generate_guarded_clauses :-
    Test = 'Demand D2: guarded clauses have guard goal inserted',
    ca_clauses(Clauses),
    clauses_to_pairs(Clauses, Pairs),
    (   detect_demand_eligible(category_ancestor, 4, Pairs, Spec),
        generate_guarded_clauses(Spec, Pairs, GuardedPairs),
        length(GuardedPairs, 2),
        % The recursive clause (index 1) should contain the guard
        nth0(1, GuardedPairs, _GHead-GBody),
        term_to_atom(GBody, BodyAtom),
        sub_atom(BodyAtom, _, _, _, 'can_reach_via_category_parent')
    ->  pass(Test)
    ;   fail_test(Test, 'guard not inserted in recursive clause')
    ).

test_generate_demand_init :-
    Test = 'Demand D2: generate_demand_init produces init_demand clause',
    ca_clauses(Clauses),
    clauses_to_pairs(Clauses, Pairs),
    (   detect_demand_eligible(category_ancestor, 4, Pairs, Spec),
        generate_demand_init(Spec, root_category/1, InitClause),
        InitClause = (init_demand :- _InitBody),
        term_to_atom(InitClause, ClauseAtom),
        sub_atom(ClauseAtom, _, _, _, 'retractall'),
        sub_atom(ClauseAtom, _, _, _, 'can_reach_via_category_parent')
    ->  pass(Test)
    ;   fail_test(Test, 'init clause generation failed')
    ).
