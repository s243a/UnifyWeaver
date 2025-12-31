%% test_phase_9b.pl
%  Test suite for Phase 9b: GROUP BY Aggregations
%
%  Tests all five GROUP BY aggregation operations:
%  - group_by + count: Count records per group
%  - group_by + sum: Sum field values per group
%  - group_by + avg: Calculate average per group
%  - group_by + max: Find maximum per group
%  - group_by + min: Find minimum per group

:- use_module('src/unifyweaver/targets/go_target').

%% Test predicates with GROUP BY

% Test 1: GROUP BY count
city_counts(City, Count) :-
    group_by(City, json_record([city-City, name-_Name]), count, Count).

% Test 2: GROUP BY sum
city_total_age(City, TotalAge) :-
    group_by(City, json_record([city-City, age-Age]), sum(Age), TotalAge).

% Test 3: GROUP BY avg
city_avg_age(City, AvgAge) :-
    group_by(City, json_record([city-City, age-Age]), avg(Age), AvgAge).

% Test 4: GROUP BY max
city_max_age(City, MaxAge) :-
    group_by(City, json_record([city-City, age-Age]), max(Age), MaxAge).

% Test 5: GROUP BY min
city_min_age(City, MinAge) :-
    group_by(City, json_record([city-City, age-Age]), min(Age), MinAge).

%% Test execution

test_group_by_count :-
    write('=== Test 1: GROUP BY Count ==='), nl,
    compile_predicate_to_go(city_counts/2, [
        db_backend(bbolt),
        db_file('test_users.db'),
        db_bucket(users)
    ], Code),
    write('Generated Go code:'), nl,
    write(Code), nl,
    % Verify key patterns in generated code
    (   sub_string(Code, _, _, _, "counts := make(map[string]int)")
    ->  write('✓ Counts map initialized'), nl
    ;   write('✗ Counts map NOT initialized'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "counts[groupStr]++")
    ->  write('✓ Count increment found'), nl
    ;   write('✗ Count increment NOT found'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "for group, count := range counts")
    ->  write('✓ Group iteration found'), nl
    ;   write('✗ Group iteration NOT found'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "\"city\": group")
    ->  write('✓ City field in output'), nl
    ;   write('✗ City field NOT in output'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "\"count\": count")
    ->  write('✓ Count field in output'), nl
    ;   write('✗ Count field NOT in output'), nl, fail
    ),
    write('✓ Test 1 PASSED'), nl, nl.

test_group_by_sum :-
    write('=== Test 2: GROUP BY Sum ==='), nl,
    compile_predicate_to_go(city_total_age/2, [
        db_backend(bbolt),
        db_file('test_users.db'),
        db_bucket(users)
    ], Code),
    write('Generated Go code:'), nl,
    write(Code), nl,
    % Verify key patterns
    (   sub_string(Code, _, _, _, "sums := make(map[string]float64)")
    ->  write('✓ Sums map initialized'), nl
    ;   write('✗ Sums map NOT initialized'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "sums[groupStr] += valueFloat")
    ->  write('✓ Sum accumulation found'), nl
    ;   write('✗ Sum accumulation NOT found'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "for group, sum := range sums")
    ->  write('✓ Group iteration found'), nl
    ;   write('✗ Group iteration NOT found'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "\"sum\": sum")
    ->  write('✓ Sum field in output'), nl
    ;   write('✗ Sum field NOT in output'), nl, fail
    ),
    write('✓ Test 2 PASSED'), nl, nl.

test_group_by_avg :-
    write('=== Test 3: GROUP BY Average ==='), nl,
    compile_predicate_to_go(city_avg_age/2, [
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
    (   sub_string(Code, _, _, _, "stats := make(map[string]*GroupStats)")
    ->  write('✓ Stats map initialized'), nl
    ;   write('✗ Stats map NOT initialized'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "stats[groupStr].sum += valueFloat")
    ->  write('✓ Sum accumulation found'), nl
    ;   write('✗ Sum accumulation NOT found'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "stats[groupStr].count++")
    ->  write('✓ Count increment found'), nl
    ;   write('✗ Count increment NOT found'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "avg = s.sum / float64(s.count)")
    ->  write('✓ Average calculation found'), nl
    ;   write('✗ Average calculation NOT found'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "\"avg\": avg")
    ->  write('✓ Avg field in output'), nl
    ;   write('✗ Avg field NOT in output'), nl, fail
    ),
    write('✓ Test 3 PASSED'), nl, nl.

test_group_by_max :-
    write('=== Test 4: GROUP BY Maximum ==='), nl,
    compile_predicate_to_go(city_max_age/2, [
        db_backend(bbolt),
        db_file('test_users.db'),
        db_bucket(users)
    ], Code),
    write('Generated Go code:'), nl,
    write(Code), nl,
    % Verify key patterns
    (   sub_string(Code, _, _, _, "type GroupMax struct")
    ->  write('✓ GroupMax struct defined'), nl
    ;   write('✗ GroupMax struct NOT defined'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "maxes := make(map[string]*GroupMax)")
    ->  write('✓ Maxes map initialized'), nl
    ;   write('✗ Maxes map NOT initialized'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "maxes[groupStr].first || valueFloat > maxes[groupStr].maxValue")
    ->  write('✓ Max comparison found'), nl
    ;   write('✗ Max comparison NOT found'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "\"max\": m.maxValue")
    ->  write('✓ Max field in output'), nl
    ;   write('✗ Max field NOT in output'), nl, fail
    ),
    write('✓ Test 4 PASSED'), nl, nl.

test_group_by_min :-
    write('=== Test 5: GROUP BY Minimum ==='), nl,
    compile_predicate_to_go(city_min_age/2, [
        db_backend(bbolt),
        db_file('test_users.db'),
        db_bucket(users)
    ], Code),
    write('Generated Go code:'), nl,
    write(Code), nl,
    % Verify key patterns
    (   sub_string(Code, _, _, _, "type GroupMin struct")
    ->  write('✓ GroupMin struct defined'), nl
    ;   write('✗ GroupMin struct NOT defined'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "mins := make(map[string]*GroupMin)")
    ->  write('✓ Mins map initialized'), nl
    ;   write('✗ Mins map NOT initialized'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "mins[groupStr].first || valueFloat < mins[groupStr].minValue")
    ->  write('✓ Min comparison found'), nl
    ;   write('✗ Min comparison NOT found'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "\"min\": m.minValue")
    ->  write('✓ Min field in output'), nl
    ;   write('✗ Min field NOT in output'), nl, fail
    ),
    write('✓ Test 5 PASSED'), nl, nl.

%% Run all tests
run_all_tests :-
    write(''), nl,
    write('╔════════════════════════════════════════════════════╗'), nl,
    write('║   Phase 9b: GROUP BY Aggregations Test Suite     ║'), nl,
    write('╚════════════════════════════════════════════════════╝'), nl,
    nl,
    test_group_by_count,
    test_group_by_sum,
    test_group_by_avg,
    test_group_by_max,
    test_group_by_min,
    write('╔════════════════════════════════════════════════════╗'), nl,
    write('║   ALL TESTS PASSED ✓                             ║'), nl,
    write('╚════════════════════════════════════════════════════╝'), nl,
    nl.

%% Entry point
:- initialization(run_all_tests).
