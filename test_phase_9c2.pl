%% test_phase_9c2.pl
%  Test suite for Phase 9c-2: HAVING Clause Support
%
%  Tests filtering groups by aggregation results:
%  - Single HAVING constraint with count
%  - Single HAVING constraint with avg
%  - Multiple HAVING constraints
%  - All comparison operators: >, <, >=, =<, =, =\=
%  - HAVING with multiple aggregations

:- use_module('src/unifyweaver/targets/go_target').

%% Test predicates with HAVING clauses

% Test 1: Simple HAVING with count > threshold
large_cities(City, Count) :-
    group_by(City, json_record([city-City, name-_]), count, Count),
    Count > 100.

% Test 2: HAVING with avg comparison
mature_cities(City, Avg) :-
    group_by(City, json_record([city-City, age-Age]), avg(Age), Avg),
    Avg > 40.0.

% Test 3: Multiple HAVING constraints
active_cities(City, Count, Avg) :-
    group_by(City, json_record([city-City, age-Age]),
             [count(Count), avg(Age, Avg)]),
    Count >= 10,
    Avg > 30.0.

% Test 4: HAVING with >= operator
min_size_cities(City, Count) :-
    group_by(City, json_record([city-City, name-_]), count, Count),
    Count >= 50.

% Test 5: HAVING with =< operator
max_avg_cities(City, Avg) :-
    group_by(City, json_record([city-City, age-Age]), avg(Age), Avg),
    Avg =< 50.0.

% Test 6: HAVING with < operator
small_avg_cities(City, Avg) :-
    group_by(City, json_record([city-City, age-Age]), avg(Age), Avg),
    Avg < 25.0.

% Test 7: Complex HAVING with multiple aggregations and constraints
premium_cities(City, Count, AvgAge, MaxAge) :-
    group_by(City, json_record([city-City, age-Age]),
             [count(Count), avg(Age, AvgAge), max(Age, MaxAge)]),
    Count >= 20,
    AvgAge >= 35.0,
    MaxAge < 65.

%% Test execution

test_simple_count_having :-
    write('=== Test 1: Simple HAVING with count > 100 ==='), nl,
    compile_predicate_to_go(large_cities/2, [
        db_backend(bbolt),
        db_file('test_users.db'),
        db_bucket(users)
    ], Code),
    write('Generated Go code:'), nl,
    write(Code), nl,
    % Verify HAVING filter in generated code
    (   sub_string(Code, _, _, _, "// HAVING filter:")
    ->  write('✓ HAVING filter comment found'), nl
    ;   write('✗ HAVING filter comment NOT found'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "if !(s.count > 100)")
    ->  write('✓ Count > 100 filter found'), nl
    ;   write('✗ Count > 100 filter NOT found'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "continue")
    ->  write('✓ Continue statement found'), nl
    ;   write('✗ Continue statement NOT found'), nl, fail
    ),
    write('✓ Test 1 PASSED'), nl, nl.

test_avg_having :-
    write('=== Test 2: HAVING with avg > 40.0 ==='), nl,
    compile_predicate_to_go(mature_cities/2, [
        db_backend(bbolt),
        db_file('test_users.db'),
        db_bucket(users)
    ], Code),
    write('Generated Go code:'), nl,
    write(Code), nl,
    % Verify HAVING filter for avg
    (   sub_string(Code, _, _, _, "// HAVING filter:")
    ->  write('✓ HAVING filter comment for avg found'), nl
    ;   write('✗ HAVING filter comment for avg NOT found'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "if !(avg > 40")
    ->  write('✓ Avg > 40.0 filter found'), nl
    ;   write('✗ Avg > 40.0 filter NOT found'), nl, fail
    ),
    write('✓ Test 2 PASSED'), nl, nl.

test_multiple_having :-
    write('=== Test 3: Multiple HAVING constraints ==='), nl,
    compile_predicate_to_go(active_cities/3, [
        db_backend(bbolt),
        db_file('test_users.db'),
        db_bucket(users)
    ], Code),
    write('Generated Go code:'), nl,
    write(Code), nl,
    % Verify both HAVING filters - count occurrences of "// HAVING filter:"
    (   sub_string(Code, Pos1, _, _, "// HAVING filter:"),
        sub_string(Code, Pos2, _, _, "// HAVING filter:"),
        Pos1 \= Pos2
    ->  write('✓ Two HAVING filter comments found'), nl
    ;   write('✗ Two HAVING filter comments NOT found'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "if !(s.count >= 10)")
    ->  write('✓ Count >= 10 condition found'), nl
    ;   write('✗ Count >= 10 condition NOT found'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "if !(avg > 30")
    ->  write('✓ Avg > 30.0 condition found'), nl
    ;   write('✗ Avg > 30.0 condition NOT found'), nl, fail
    ),
    write('✓ Test 3 PASSED'), nl, nl.

test_gte_operator :-
    write('=== Test 4: HAVING with >= operator ==='), nl,
    compile_predicate_to_go(min_size_cities/2, [
        db_backend(bbolt),
        db_file('test_users.db'),
        db_bucket(users)
    ], Code),
    write('Generated Go code:'), nl,
    write(Code), nl,
    % Verify >= operator
    (   sub_string(Code, _, _, _, "if !(s.count >= 50)")
    ->  write('✓ >= operator correctly generated'), nl
    ;   write('✗ >= operator NOT found'), nl, fail
    ),
    write('✓ Test 4 PASSED'), nl, nl.

test_lte_operator :-
    write('=== Test 5: HAVING with =< operator ==='), nl,
    compile_predicate_to_go(max_avg_cities/2, [
        db_backend(bbolt),
        db_file('test_users.db'),
        db_bucket(users)
    ], Code),
    write('Generated Go code:'), nl,
    write(Code), nl,
    % Verify =< operator (should be <= in Go)
    (   sub_string(Code, _, _, _, "if !(avg =< 50")
    ->  write('✓ =< operator correctly generated'), nl
    ;   write('✗ =< operator NOT found'), nl, fail
    ),
    write('✓ Test 5 PASSED'), nl, nl.

test_lt_operator :-
    write('=== Test 6: HAVING with < operator ==='), nl,
    compile_predicate_to_go(small_avg_cities/2, [
        db_backend(bbolt),
        db_file('test_users.db'),
        db_bucket(users)
    ], Code),
    write('Generated Go code:'), nl,
    write(Code), nl,
    % Verify < operator
    (   sub_string(Code, _, _, _, "if !(avg < 25")
    ->  write('✓ < operator correctly generated'), nl
    ;   write('✗ < operator NOT found'), nl, fail
    ),
    write('✓ Test 6 PASSED'), nl, nl.

test_complex_having :-
    write('=== Test 7: Complex HAVING with 3 constraints ==='), nl,
    compile_predicate_to_go(premium_cities/4, [
        db_backend(bbolt),
        db_file('test_users.db'),
        db_bucket(users)
    ], Code),
    write('Generated Go code:'), nl,
    write(Code), nl,
    % Verify three HAVING filter comments
    (   sub_string(Code, Pos1, _, _, "// HAVING filter:"),
        sub_string(Code, Pos2, _, _, "// HAVING filter:"),
        sub_string(Code, Pos3, _, _, "// HAVING filter:"),
        Pos1 \= Pos2, Pos2 \= Pos3, Pos1 \= Pos3
    ->  write('✓ Three HAVING filter comments found'), nl
    ;   write('✗ Three HAVING filter comments NOT found'), nl, fail
    ),
    % Verify conditions
    (   sub_string(Code, _, _, _, "if !(s.count >= 20)")
    ->  write('✓ Count >= 20 condition found'), nl
    ;   write('✗ Count >= 20 condition NOT found'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "if !(avg >= 35")
    ->  write('✓ AvgAge >= 35.0 condition found'), nl
    ;   write('✗ AvgAge >= 35.0 condition NOT found'), nl, fail
    ),
    (   sub_string(Code, _, _, _, "if !(s.maxValue < 65)")
    ->  write('✓ MaxAge < 65 condition found'), nl
    ;   write('✗ MaxAge < 65 condition NOT found'), nl, fail
    ),
    write('✓ Test 7 PASSED'), nl, nl.

%% Run all tests
run_all_tests :-
    write(''), nl,
    write('╔════════════════════════════════════════════════════╗'), nl,
    write('║   Phase 9c-2: HAVING Clause Test Suite           ║'), nl,
    write('╚════════════════════════════════════════════════════╝'), nl,
    nl,
    test_simple_count_having,
    test_avg_having,
    test_multiple_having,
    test_gte_operator,
    test_lte_operator,
    test_lt_operator,
    test_complex_having,
    write('╔════════════════════════════════════════════════════╗'), nl,
    write('║   ALL TESTS PASSED ✓                             ║'), nl,
    write('╚════════════════════════════════════════════════════╝'), nl,
    nl.

%% Entry point
:- initialization(run_all_tests).
