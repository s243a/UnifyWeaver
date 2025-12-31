%% test_phase_9a.pl
%  Test suite for Phase 9a: Simple Aggregations
%
%  Tests all five aggregation operations:
%  - count: Count records
%  - sum: Sum numeric field values
%  - avg: Calculate average of field
%  - max: Find maximum value
%  - min: Find minimum value

:- use_module('src/unifyweaver/targets/go_target').

%% Test predicates with aggregations

% Test 1: Count all users
user_count(Count) :-
    aggregate(count, json_record([name-_Name]), Count).

% Test 2: Sum all ages
total_age(Sum) :-
    aggregate(sum(Age), json_record([age-Age]), Sum).

% Test 3: Average age
avg_age(Avg) :-
    aggregate(avg(Age), json_record([age-Age]), Avg).

% Test 4: Maximum age
max_age(Max) :-
    aggregate(max(Age), json_record([age-Age]), Max).

% Test 5: Minimum age
min_age(Min) :-
    aggregate(min(Age), json_record([age-Age]), Min).

%% Test execution

test_count :-
    write('=== Test 1: Count Aggregation ==='), nl,
    compile_predicate_to_go(user_count/1, [
        db_backend(bbolt),
        db_file('test_users.db'),
        db_bucket(users)
    ], Code),
    write('Generated Go code:'), nl,
    write(Code), nl,
    % Verify key patterns in generated code
    (   sub_string(Code, _, _, _, "count := 0")
    ->  write('✓ Count variable initialized'), nl
    ;   write('✗ Count variable NOT initialized'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "count++")
    ->  write('✓ Count increment found'), nl
    ;   write('✗ Count increment NOT found'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "fmt.Println(count)")
    ->  write('✓ Count output found'), nl
    ;   write('✗ Count output NOT found'), nl, fail
    ),
    write('✓ Test 1 PASSED'), nl, nl.

test_sum :-
    write('=== Test 2: Sum Aggregation ==='), nl,
    compile_predicate_to_go(total_age/1, [
        db_backend(bbolt),
        db_file('test_users.db'),
        db_bucket(users)
    ], Code),
    write('Generated Go code:'), nl,
    write(Code), nl,
    % Verify key patterns in generated code
    (   sub_string(Code, _, _, _, "sum := 0.0")
    ->  write('✓ Sum variable initialized'), nl
    ;   write('✗ Sum variable NOT initialized'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "sum +=")
    ->  write('✓ Sum accumulation found'), nl
    ;   write('✗ Sum accumulation NOT found'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "valueFloat, ok := valueRaw.(float64)")
    ->  write('✓ Type conversion found'), nl
    ;   write('✗ Type conversion NOT found'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "fmt.Println(sum)")
    ->  write('✓ Sum output found'), nl
    ;   write('✗ Sum output NOT found'), nl, fail
    ),
    write('✓ Test 2 PASSED'), nl, nl.

test_avg :-
    write('=== Test 3: Average Aggregation ==='), nl,
    compile_predicate_to_go(avg_age/1, [
        db_backend(bbolt),
        db_file('test_users.db'),
        db_bucket(users)
    ], Code),
    write('Generated Go code:'), nl,
    write(Code), nl,
    % Verify key patterns in generated code
    (   sub_string(Code, _, _, _, "sum := 0.0")
    ->  write('✓ Sum variable initialized'), nl
    ;   write('✗ Sum variable NOT initialized'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "count := 0")
    ->  write('✓ Count variable initialized'), nl
    ;   write('✗ Count variable NOT initialized'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "sum +=")
    ->  write('✓ Sum accumulation found'), nl
    ;   write('✗ Sum accumulation NOT found'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "count++")
    ->  write('✓ Count increment found'), nl
    ;   write('✗ Count increment NOT found'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "avg = sum / float64(count)")
    ->  write('✓ Average calculation found'), nl
    ;   write('✗ Average calculation NOT found'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "fmt.Println(avg)")
    ->  write('✓ Average output found'), nl
    ;   write('✗ Average output NOT found'), nl, fail
    ),
    write('✓ Test 3 PASSED'), nl, nl.

test_max :-
    write('=== Test 4: Maximum Aggregation ==='), nl,
    compile_predicate_to_go(max_age/1, [
        db_backend(bbolt),
        db_file('test_users.db'),
        db_bucket(users)
    ], Code),
    write('Generated Go code:'), nl,
    write(Code), nl,
    % Verify key patterns in generated code
    (   sub_string(Code, _, _, _, "maxValue := 0.0")
    ->  write('✓ Max variable initialized'), nl
    ;   write('✗ Max variable NOT initialized'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "first := true")
    ->  write('✓ First flag initialized'), nl
    ;   write('✗ First flag NOT initialized'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "first || valueFloat > maxValue")
    ->  write('✓ Max comparison found'), nl
    ;   write('✗ Max comparison NOT found'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "first = false")
    ->  write('✓ First flag update found'), nl
    ;   write('✗ First flag update NOT found'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "fmt.Println(maxValue)")
    ->  write('✓ Max output found'), nl
    ;   write('✗ Max output NOT found'), nl, fail
    ),
    write('✓ Test 4 PASSED'), nl, nl.

test_min :-
    write('=== Test 5: Minimum Aggregation ==='), nl,
    compile_predicate_to_go(min_age/1, [
        db_backend(bbolt),
        db_file('test_users.db'),
        db_bucket(users)
    ], Code),
    write('Generated Go code:'), nl,
    write(Code), nl,
    % Verify key patterns in generated code
    (   sub_string(Code, _, _, _, "minValue := 0.0")
    ->  write('✓ Min variable initialized'), nl
    ;   write('✗ Min variable NOT initialized'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "first := true")
    ->  write('✓ First flag initialized'), nl
    ;   write('✗ First flag NOT initialized'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "first || valueFloat < minValue")
    ->  write('✓ Min comparison found'), nl
    ;   write('✗ Min comparison NOT found'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "first = false")
    ->  write('✓ First flag update found'), nl
    ;   write('✗ First flag update NOT found'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "fmt.Println(minValue)")
    ->  write('✓ Min output found'), nl
    ;   write('✗ Min output NOT found'), nl, fail
    ),
    write('✓ Test 5 PASSED'), nl, nl.

%% Run all tests
run_all_tests :-
    write(''), nl,
    write('╔════════════════════════════════════════════════════╗'), nl,
    write('║   Phase 9a: Simple Aggregations Test Suite       ║'), nl,
    write('╚════════════════════════════════════════════════════╝'), nl,
    nl,
    test_count,
    test_sum,
    test_avg,
    test_max,
    test_min,
    write('╔════════════════════════════════════════════════════╗'), nl,
    write('║   ALL TESTS PASSED ✓                             ║'), nl,
    write('╚════════════════════════════════════════════════════╝'), nl,
    nl.

%% Entry point
:- initialization(run_all_tests).
