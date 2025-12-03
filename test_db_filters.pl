#!/usr/bin/env swipl

% Test database filtering predicates (Phase 8a)

:- use_module('src/unifyweaver/targets/go_target').

%% Test Schema - Users
:- json_schema(user_data, [
    field(name, string),
    field(age, integer),
    field(city, string),
    field(salary, integer)
]).

%% ============================================
%% WRITE MODE: Populate test database
%% ============================================

user_write(Name, Age, City, Salary) :-
    json_record([name-Name, age-Age, city-City, salary-Salary]).

test_populate_db :-
    format('~n=== Test 0: Populate Database ===~n'),

    compile_predicate_to_go(user_write/4, [
        json_input(true),
        json_schema(user_data),
        db_backend(bbolt),
        db_file('filters_test.db'),
        db_bucket(users),
        db_key_field(name),
        package(main)
    ], Code),

    % Write generated code
    make_directory_path('output_filters_populate'),
    open('output_filters_populate/populate.go', write, Stream),
    write(Stream, Code),
    close(Stream),

    format('Generated: output_filters_populate/populate.go~n'),
    format('Purpose: Populate test database with sample users~n').

%% ============================================
%% TEST 1: Simple age filter (Age >= 30)
%% ============================================

adults(Name, Age) :-
    json_record([name-Name, age-Age, city-_City, salary-_Salary]),
    Age >= 30.

test_age_filter :-
    format('~n=== Test 1: Age Filter (Age >= 30) ===~n'),

    compile_predicate_to_go(adults/2, [
        db_backend(bbolt),
        db_file('filters_test.db'),
        db_bucket(users),
        db_mode(read),
        package(main)
    ], Code),

    % Write generated code
    make_directory_path('output_filters_age'),
    open('output_filters_age/read_adults.go', write, Stream),
    write(Stream, Code),
    close(Stream),

    format('Generated: output_filters_age/read_adults.go~n'),
    format('Expected: Only users with age >= 30~n').

%% ============================================
%% TEST 2: Multi-field filter (Age > 25 AND City = "NYC")
%% ============================================

nyc_young_adults(Name, Age) :-
    json_record([name-Name, age-Age, city-City, salary-_Salary]),
    Age > 25,
    City = "NYC".

test_multi_filter :-
    format('~n=== Test 2: Multi-Field Filter (Age > 25 AND City = NYC) ===~n'),

    compile_predicate_to_go(nyc_young_adults/2, [
        db_backend(bbolt),
        db_file('filters_test.db'),
        db_bucket(users),
        db_mode(read),
        package(main)
    ], Code),

    % Write generated code
    make_directory_path('output_filters_multi'),
    open('output_filters_multi/read_nyc_young.go', write, Stream),
    write(Stream, Code),
    close(Stream),

    format('Generated: output_filters_multi/read_nyc_young.go~n'),
    format('Expected: Only NYC users with age > 25~n').

%% ============================================
%% TEST 3: Salary range (30000 =< Salary, Salary =< 80000)
%% ============================================

middle_income(Name, Salary) :-
    json_record([name-Name, age-_Age, city-_City, salary-Salary]),
    30000 =< Salary,
    Salary =< 80000.

test_salary_range :-
    format('~n=== Test 3: Salary Range (30000 =< Salary =< 80000) ===~n'),

    compile_predicate_to_go(middle_income/2, [
        db_backend(bbolt),
        db_file('filters_test.db'),
        db_bucket(users),
        db_mode(read),
        package(main)
    ], Code),

    % Write generated code
    make_directory_path('output_filters_salary'),
    open('output_filters_salary/read_middle_income.go', write, Stream),
    write(Stream, Code),
    close(Stream),

    format('Generated: output_filters_salary/read_middle_income.go~n'),
    format('Expected: Only users with salary between 30000 and 80000~n').

%% ============================================
%% TEST 4: Not equal filter (City \= "NYC")
%% ============================================

non_nyc_users(Name, City) :-
    json_record([name-Name, age-_Age, city-City, salary-_Salary]),
    City \= "NYC".

test_not_equal :-
    format('~n=== Test 4: Not Equal Filter (City \\= NYC) ===~n'),

    compile_predicate_to_go(non_nyc_users/2, [
        db_backend(bbolt),
        db_file('filters_test.db'),
        db_bucket(users),
        db_mode(read),
        package(main)
    ], Code),

    % Write generated code
    make_directory_path('output_filters_not_equal'),
    open('output_filters_not_equal/read_non_nyc.go', write, Stream),
    write(Stream, Code),
    close(Stream),

    format('Generated: output_filters_not_equal/read_non_nyc.go~n'),
    format('Expected: Only users not from NYC~n').

%% ============================================
%% TEST 5: All comparison operators
%% ============================================

test_all_operators(Name, Age, Salary) :-
    json_record([name-Name, age-Age, city-_City, salary-Salary]),
    Age > 20,           % Greater than
    Age < 60,           % Less than
    Salary >= 25000,    % Greater than or equal
    Salary =< 100000.   % Less than or equal

test_all_comparisons :-
    format('~n=== Test 5: All Comparison Operators ===~n'),

    compile_predicate_to_go(test_all_operators/3, [
        db_backend(bbolt),
        db_file('filters_test.db'),
        db_bucket(users),
        db_mode(read),
        package(main)
    ], Code),

    % Write generated code
    make_directory_path('output_filters_all_ops'),
    open('output_filters_all_ops/read_all_ops.go', write, Stream),
    write(Stream, Code),
    close(Stream),

    format('Generated: output_filters_all_ops/read_all_ops.go~n'),
    format('Expected: Users matching all 4 constraints~n').

%% ============================================
%% TEST 6: No filter - read all (baseline)
%% ============================================

all_users(Name, Age, City, Salary) :-
    json_record([name-Name, age-Age, city-City, salary-Salary]).

test_no_filter :-
    format('~n=== Test 6: No Filter (Read All) ===~n'),

    compile_predicate_to_go(all_users/4, [
        db_backend(bbolt),
        db_file('filters_test.db'),
        db_bucket(users),
        db_mode(read),
        package(main)
    ], Code),

    % Write generated code
    make_directory_path('output_filters_all'),
    open('output_filters_all/read_all.go', write, Stream),
    write(Stream, Code),
    close(Stream),

    format('Generated: output_filters_all/read_all.go~n'),
    format('Expected: All users without filtering~n').

%% Main test runner
main :-
    format('~n===== Database Filter Tests (Phase 8a) =====~n'),

    % Run all tests
    test_populate_db,
    test_age_filter,
    test_multi_filter,
    test_salary_range,
    test_not_equal,
    test_all_comparisons,
    test_no_filter,

    format('~n===== All Tests Generated Successfully =====~n'),
    format('~nNext steps:~n'),
    format('1. Build and run populate program with test data~n'),
    format('2. Build and run each filter test~n'),
    format('3. Verify output matches expected filtered results~n').

:- initialization(main, main).
