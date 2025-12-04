%% test_phase_9c3.pl
%  Test suite for Phase 9c-3: Nested Grouping (Multiple Group Fields)
%
%  Tests grouping by multiple fields simultaneously

:- use_module('src/unifyweaver/targets/go_target').

%% Test predicates with nested grouping

% Test 1: Two-field grouping with count
state_city_counts(State, City, Count) :-
    group_by([State, City],
             json_record([state-State, city-City, name-_]),
             count, Count).

% Test 2: Two-field grouping with multiple aggregations
region_stats(Region, Category, Count, AvgPrice) :-
    group_by([Region, Category],
             json_record([region-Region, category-Category, price-Price]),
             [count(Count), avg(Price, AvgPrice)]).

%% Test execution

test_two_field_count :-
    write('=== Test 1: Two-field grouping with count ==='), nl,
    compile_predicate_to_go(state_city_counts/3, [
        db_backend(bbolt),
        db_file('test_data.db'),
        db_bucket(records)
    ], Code),
    write('Generated Go code:'), nl,
    write(Code), nl,
    % Verify key patterns
    (   sub_string(Code, _, _, _, "strings.Join")
    ->  write('✓ Composite key generation found'), nl
    ;   write('✗ Composite key generation NOT found'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "strings.Split")
    ->  write('✓ Composite key parsing found'), nl
    ;   write('✗ Composite key parsing NOT found'), nl, fail
    ),
    write('✓ Test 1 PASSED'), nl, nl.

test_two_field_multi_agg :-
    write('=== Test 2: Two-field grouping with multiple aggregations ==='), nl,
    compile_predicate_to_go(region_stats/4, [
        db_backend(bbolt),
        db_file('test_data.db'),
        db_bucket(records)
    ], Code),
    write('Generated Go code:'), nl,
    write(Code), nl,
    % Verify patterns
    (   sub_string(Code, _, _, _, "Nested grouping detected")
    ->  write('✓ Nested grouping detected'), nl
    ;   write('Note: Nested grouping detection message not in output'), nl
    ),
    (   sub_string(Code, _, _, _, "strings.Join")
    ->  write('✓ Composite key generation found'), nl
    ;   write('✗ Composite key generation NOT found'), nl, fail
    ),
    (   sub_string(Code, _, _, _, '"region": parts[0]')
    ->  write('✓ First field parsing found'), nl
    ;   write('✗ First field parsing NOT found'), nl, fail
    ),
    (   sub_string(Code, _, _, _, '"category": parts[1]')
    ->  write('✓ Second field parsing found'), nl
    ;   write('✗ Second field parsing NOT found'), nl, fail
    ),
    write('✓ Test 2 PASSED'), nl, nl.

%% Run all tests
run_all_tests :-
    write(''), nl,
    write('╔════════════════════════════════════════════════════╗'), nl,
    write('║   Phase 9c-3: Nested Grouping Test Suite         ║'), nl,
    write('╚════════════════════════════════════════════════════╝'), nl,
    nl,
    test_two_field_count,
    test_two_field_multi_agg,
    write('╔════════════════════════════════════════════════════╗'), nl,
    write('║   ALL TESTS PASSED ✓                             ║'), nl,
    write('╚════════════════════════════════════════════════════╝'), nl,
    nl.

%% Entry point
:- initialization(run_all_tests).
