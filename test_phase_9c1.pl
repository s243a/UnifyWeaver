%% test_phase_9c1.pl
%  Test suite for Phase 9c-1: Multiple Aggregations
%
%  Tests multiple aggregation operations in a single query:
%  - [count, avg]: Count and average together
%  - [count, sum, avg]: Three aggregations
%  - [count, avg, max, min]: Four aggregations
%  - [count, sum, avg, max, min]: All five operations

:- use_module('src/unifyweaver/targets/go_target').

%% Test predicates with multiple aggregations

% Test 1: Two aggregations (count + avg)
city_count_and_avg(City, Count, AvgAge) :-
    group_by(City, json_record([city-City, age-Age]),
             [count(Count), avg(Age, AvgAge)]).

% Test 2: Three aggregations (count + sum + avg)
city_three_stats(City, Count, TotalAge, AvgAge) :-
    group_by(City, json_record([city-City, age-Age]),
             [count(Count), sum(Age, TotalAge), avg(Age, AvgAge)]).

% Test 3: Four aggregations (count + avg + max + min)
city_age_stats(City, Count, AvgAge, MaxAge, MinAge) :-
    group_by(City, json_record([city-City, age-Age]),
             [count(Count), avg(Age, AvgAge), max(Age, MaxAge), min(Age, MinAge)]).

% Test 4: All five aggregations
city_complete_stats(City, Count, TotalAge, AvgAge, MaxAge, MinAge) :-
    group_by(City, json_record([city-City, age-Age]),
             [count(Count), sum(Age, TotalAge), avg(Age, AvgAge),
              max(Age, MaxAge), min(Age, MinAge)]).

%% Test execution

test_count_and_avg :-
    write('=== Test 1: Count + Average ==='), nl,
    compile_predicate_to_go(city_count_and_avg/3, [
        db_backend(bbolt),
        db_file('test_users.db'),
        db_bucket(users)
    ], Code),
    write('Generated Go code:'), nl,
    write(Code), nl,
    % Verify key patterns in generated code
    (   sub_string(Code, _, _, _, "type GroupStats struct")
    ->  write('✓ GroupStats struct defined'), nl
    ;   write('✗ GroupStats struct NOT defined'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "count    int")
    ->  write('✓ count field in struct'), nl
    ;   write('✗ count field NOT in struct'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "sum      float64")
    ->  write('✓ sum field in struct'), nl
    ;   write('✗ sum field NOT in struct'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "stats[groupStr].count++")
    ->  write('✓ Count accumulation found'), nl
    ;   write('✗ Count accumulation NOT found'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "stats[groupStr].sum += valueFloat")
    ->  write('✓ Sum accumulation found'), nl
    ;   write('✗ Sum accumulation NOT found'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "avg := 0.0")
    ->  write('✓ Average calculation found'), nl
    ;   write('✗ Average calculation NOT found'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "\"count\": s.count")
    ->  write('✓ Count in output'), nl
    ;   write('✗ Count NOT in output'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "\"avg\": avg")
    ->  write('✓ Avg in output'), nl
    ;   write('✗ Avg NOT in output'), nl, fail
    ),
    write('✓ Test 1 PASSED'), nl, nl.

test_three_stats :-
    write('=== Test 2: Count + Sum + Average ==='), nl,
    compile_predicate_to_go(city_three_stats/4, [
        db_backend(bbolt),
        db_file('test_users.db'),
        db_bucket(users)
    ], Code),
    write('Generated Go code:'), nl,
    write(Code), nl,
    % Verify key patterns
    (   sub_string(Code, _, _, _, "type GroupStats struct")
    ->  write('✓ GroupStats struct defined'), nl
    ;   write('✗ GroupStats struct NOT defined'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "count    int")
    ->  write('✓ count field in struct'), nl
    ;   write('✗ count field NOT in struct'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "sum      float64")
    ->  write('✓ sum field in struct'), nl
    ;   write('✗ sum field NOT in struct'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "\"count\": s.count")
    ->  write('✓ Count in output'), nl
    ;   write('✗ Count NOT in output'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "\"sum\": s.sum")
    ->  write('✓ Sum in output'), nl
    ;   write('✗ Sum NOT in output'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "\"avg\": avg")
    ->  write('✓ Avg in output'), nl
    ;   write('✗ Avg NOT in output'), nl, fail
    ),
    write('✓ Test 2 PASSED'), nl, nl.

test_age_stats :-
    write('=== Test 3: Count + Avg + Max + Min ==='), nl,
    compile_predicate_to_go(city_age_stats/5, [
        db_backend(bbolt),
        db_file('test_users.db'),
        db_bucket(users)
    ], Code),
    write('Generated Go code:'), nl,
    write(Code), nl,
    % Verify key patterns
    (   sub_string(Code, _, _, _, "type GroupStats struct")
    ->  write('✓ GroupStats struct defined'), nl
    ;   write('✗ GroupStats struct NOT defined'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "maxValue float64")
    ->  write('✓ maxValue field in struct'), nl
    ;   write('✗ maxValue field NOT in struct'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "maxFirst bool")
    ->  write('✓ maxFirst field in struct'), nl
    ;   write('✗ maxFirst field NOT in struct'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "minValue float64")
    ->  write('✓ minValue field in struct'), nl
    ;   write('✗ minValue field NOT in struct'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "minFirst bool")
    ->  write('✓ minFirst field in struct'), nl
    ;   write('✗ minFirst field NOT in struct'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "maxFirst: true, minFirst: true")
    ->  write('✓ Struct initialization with both flags'), nl
    ;   write('✗ Struct initialization NOT correct'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "\"max\": s.maxValue")
    ->  write('✓ Max in output'), nl
    ;   write('✗ Max NOT in output'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "\"min\": s.minValue")
    ->  write('✓ Min in output'), nl
    ;   write('✗ Min NOT in output'), nl, fail
    ),
    write('✓ Test 3 PASSED'), nl, nl.

test_all_five :-
    write('=== Test 4: All Five Aggregations ==='), nl,
    compile_predicate_to_go(city_complete_stats/6, [
        db_backend(bbolt),
        db_file('test_users.db'),
        db_bucket(users)
    ], Code),
    write('Generated Go code:'), nl,
    write(Code), nl,
    % Verify key patterns
    (   sub_string(Code, _, _, _, "type GroupStats struct")
    ->  write('✓ GroupStats struct defined'), nl
    ;   write('✗ GroupStats struct NOT defined'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "count    int")
    ->  write('✓ count field in struct'), nl
    ;   write('✗ count field NOT in struct'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "sum      float64")
    ->  write('✓ sum field in struct'), nl
    ;   write('✗ sum field NOT in struct'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "maxValue float64")
    ->  write('✓ maxValue field in struct'), nl
    ;   write('✗ maxValue field NOT in struct'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "minValue float64")
    ->  write('✓ minValue field in struct'), nl
    ;   write('✗ minValue field NOT in struct'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "\"count\": s.count")
    ->  write('✓ Count in output'), nl
    ;   write('✗ Count NOT in output'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "\"sum\": s.sum")
    ->  write('✓ Sum in output'), nl
    ;   write('✗ Sum NOT in output'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "\"avg\": avg")
    ->  write('✓ Avg in output'), nl
    ;   write('✗ Avg NOT in output'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "\"max\": s.maxValue")
    ->  write('✓ Max in output'), nl
    ;   write('✗ Max NOT in output'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "\"min\": s.minValue")
    ->  write('✓ Min in output'), nl
    ;   write('✗ Min NOT in output'), nl, fail
    ),
    write('✓ Test 4 PASSED'), nl, nl.

%% Run all tests
run_all_tests :-
    write(''), nl,
    write('╔════════════════════════════════════════════════════╗'), nl,
    write('║   Phase 9c-1: Multiple Aggregations Test Suite   ║'), nl,
    write('╚════════════════════════════════════════════════════╝'), nl,
    nl,
    test_count_and_avg,
    test_three_stats,
    test_age_stats,
    test_all_five,
    write('╔════════════════════════════════════════════════════╗'), nl,
    write('║   ALL TESTS PASSED ✓                             ║'), nl,
    write('╚════════════════════════════════════════════════════╝'), nl,
    nl.

%% Entry point
:- initialization(run_all_tests).
