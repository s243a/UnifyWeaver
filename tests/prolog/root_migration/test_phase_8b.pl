#!/usr/bin/env swipl

% Test Phase 8b: Enhanced Filtering (String operations + List membership)

:- use_module('src/unifyweaver/targets/go_target').

%% Test Schema - Users
:- json_schema(user_data_8b, [
    field(name, string),
    field(age, integer),
    field(city, string),
    field(status, string)
]).

%% ============================================
%% WRITE MODE: Populate test database
%% ============================================

populate_users(Name, Age, City, Status) :-
    json_record([name-Name, age-Age, city-City, status-Status]).

test_populate :-
    format('~n=== Test 0: Populate Database ===~n'),

    compile_predicate_to_go(populate_users/4, [
        json_input(true),
        json_schema(user_data_8b),
        db_backend(bbolt),
        db_file('test_phase_8b.db'),
        db_bucket(users),
        db_key_field(name),
        package(main)
    ], Code),

    % Write generated code
    make_directory_path('output_phase_8b_populate'),
    open('output_phase_8b_populate/populate.go', write, Stream),
    write(Stream, Code),
    close(Stream),

    format('Generated: output_phase_8b_populate/populate.go~n'),
    format('Purpose: Populate test database with sample users~n').

%% ============================================
%% TEST 1: Case-Insensitive City Search (=@=)
%% ============================================

user_by_city_insensitive(Name, City) :-
    json_record([name-Name, age-_Age, city-City, status-_Status]),
    City =@= "nyc".

test_case_insensitive :-
    format('~n=== Test 1: Case-Insensitive City (=@=) ===~n'),

    compile_predicate_to_go(user_by_city_insensitive/2, [
        db_backend(bbolt),
        db_file('test_phase_8b.db'),
        db_bucket(users),
        db_mode(read),
        package(main)
    ], Code),

    % Write generated code
    make_directory_path('output_phase_8b_insensitive'),
    open('output_phase_8b_insensitive/query.go', write, Stream),
    write(Stream, Code),
    close(Stream),

    format('Generated: output_phase_8b_insensitive/query.go~n'),
    format('Expected: Alice, Charlie, Eve, Julia (all NYC case variants)~n').

%% ============================================
%% TEST 2: Name Contains Substring (contains/2)
%% ============================================

users_with_substring(Name) :-
    json_record([name-Name, age-_Age, city-_City, status-_Status]),
    contains(Name, "ali").

test_contains :-
    format('~n=== Test 2: Contains Substring ===~n'),

    compile_predicate_to_go(users_with_substring/1, [
        db_backend(bbolt),
        db_file('test_phase_8b.db'),
        db_bucket(users),
        db_mode(read),
        package(main)
    ], Code),

    % Write generated code
    make_directory_path('output_phase_8b_contains'),
    open('output_phase_8b_contains/query.go', write, Stream),
    write(Stream, Code),
    close(Stream),

    format('Generated: output_phase_8b_contains/query.go~n'),
    format('Expected: Alice, Natalie, Kalina, Julia~n').

%% ============================================
%% TEST 3: City Membership (String List)
%% ============================================

major_city_users(Name, City) :-
    json_record([name-Name, age-_Age, city-City, status-_Status]),
    member(City, ["NYC", "SF", "LA", "Chicago"]).

test_city_membership :-
    format('~n=== Test 3: Major Cities (String List) ===~n'),

    compile_predicate_to_go(major_city_users/2, [
        db_backend(bbolt),
        db_file('test_phase_8b.db'),
        db_bucket(users),
        db_mode(read),
        package(main)
    ], Code),

    % Write generated code
    make_directory_path('output_phase_8b_city_member'),
    open('output_phase_8b_city_member/query.go', write, Stream),
    write(Stream, Code),
    close(Stream),

    format('Generated: output_phase_8b_city_member/query.go~n'),
    format('Expected: Users from NYC, SF, LA, Chicago~n').

%% ============================================
%% TEST 4: Age Membership (Numeric List)
%% ============================================

specific_age_users(Name, Age) :-
    json_record([name-Name, age-Age, city-_City, status-_Status]),
    member(Age, [25, 30, 35, 40]).

test_age_membership :-
    format('~n=== Test 4: Specific Ages (Numeric List) ===~n'),

    compile_predicate_to_go(specific_age_users/2, [
        db_backend(bbolt),
        db_file('test_phase_8b.db'),
        db_bucket(users),
        db_mode(read),
        package(main)
    ], Code),

    % Write generated code
    make_directory_path('output_phase_8b_age_member'),
    open('output_phase_8b_age_member/query.go', write, Stream),
    write(Stream, Code),
    close(Stream),

    format('Generated: output_phase_8b_age_member/query.go~n'),
    format('Expected: Users aged 25, 30, 35, or 40~n').

%% ============================================
%% TEST 5: Status Membership (Atom List)
%% ============================================

active_users(Name, Status) :-
    json_record([name-Name, age-_Age, city-_City, status-Status]),
    member(Status, ["active", "premium"]).

test_status_membership :-
    format('~n=== Test 5: Active/Premium Users (Atom List) ===~n'),

    compile_predicate_to_go(active_users/2, [
        db_backend(bbolt),
        db_file('test_phase_8b.db'),
        db_bucket(users),
        db_mode(read),
        package(main)
    ], Code),

    % Write generated code
    make_directory_path('output_phase_8b_status_member'),
    open('output_phase_8b_status_member/query.go', write, Stream),
    write(Stream, Code),
    close(Stream),

    format('Generated: output_phase_8b_status_member/query.go~n'),
    format('Expected: Users with status active or premium~n').

%% ============================================
%% TEST 6: Mixed String + Numeric Filter
%% ============================================

nyc_young_adults(Name, Age, City) :-
    json_record([name-Name, age-Age, city-City, status-_Status]),
    City =@= "nyc",
    Age > 25.

test_mixed_filter :-
    format('~n=== Test 6: NYC Young Adults (Mixed) ===~n'),

    compile_predicate_to_go(nyc_young_adults/3, [
        db_backend(bbolt),
        db_file('test_phase_8b.db'),
        db_bucket(users),
        db_mode(read),
        package(main)
    ], Code),

    % Write generated code
    make_directory_path('output_phase_8b_mixed'),
    open('output_phase_8b_mixed/query.go', write, Stream),
    write(Stream, Code),
    close(Stream),

    format('Generated: output_phase_8b_mixed/query.go~n'),
    format('Expected: NYC users aged > 25~n').

%% ============================================
%% TEST 7: Contains + Membership
%% ============================================

major_city_a_names(Name, City) :-
    json_record([name-Name, age-_Age, city-City, status-_Status]),
    contains(Name, "a"),
    member(City, ["NYC", "SF", "LA"]).

test_contains_member :-
    format('~n=== Test 7: Major City + Name Contains "a" ===~n'),

    compile_predicate_to_go(major_city_a_names/2, [
        db_backend(bbolt),
        db_file('test_phase_8b.db'),
        db_bucket(users),
        db_mode(read),
        package(main)
    ], Code),

    % Write generated code
    make_directory_path('output_phase_8b_contains_member'),
    open('output_phase_8b_contains_member/query.go', write, Stream),
    write(Stream, Code),
    close(Stream),

    format('Generated: output_phase_8b_contains_member/query.go~n'),
    format('Expected: Major city users with "a" in name~n').

%% ============================================
%% TEST 8: Complex Query
%% ============================================

complex_string_query(Name, City, Status) :-
    json_record([name-Name, age-_Age, city-City, status-Status]),
    contains(Name, "i"),
    City =@= "NYC",
    member(Status, ["active", "premium"]).

test_complex :-
    format('~n=== Test 8: Complex String Query ===~n'),

    compile_predicate_to_go(complex_string_query/3, [
        db_backend(bbolt),
        db_file('test_phase_8b.db'),
        db_bucket(users),
        db_mode(read),
        package(main)
    ], Code),

    % Write generated code
    make_directory_path('output_phase_8b_complex'),
    open('output_phase_8b_complex/query.go', write, Stream),
    write(Stream, Code),
    close(Stream),

    format('Generated: output_phase_8b_complex/query.go~n'),
    format('Expected: NYC active/premium users with "i" in name~n').

%% Main test runner
main :-
    format('~n===== Phase 8b: Enhanced Filtering Tests =====~n'),

    % Run all tests
    test_populate,
    test_case_insensitive,
    test_contains,
    test_city_membership,
    test_age_membership,
    test_status_membership,
    test_mixed_filter,
    test_contains_member,
    test_complex,

    format('~n===== All Tests Generated Successfully =====~n'),
    format('~nNext steps:~n'),
    format('1. Build and run populate program with test data~n'),
    format('2. Build and run each test~n'),
    format('3. Verify output matches expected results~n').

:- initialization(main, main).
