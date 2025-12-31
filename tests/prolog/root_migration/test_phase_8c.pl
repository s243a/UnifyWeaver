#!/usr/bin/env swipl

% Test Phase 8c: Key Optimization Detection

:- use_module('src/unifyweaver/targets/go_target').

%% Test Schema - Simple users
:- json_schema(user_simple, [
    field(name, string),
    field(age, integer),
    field(city, string)
]).

%% ============================================
%% TEST 1: Direct Lookup (Single Key)
%% ============================================

% Predicate that should trigger direct lookup optimization
user_by_name_exact(Name, Age, City) :-
    json_record([name-Name, age-Age, city-City]),
    Name = "Alice".

test_direct_lookup :-
    format('~n=== Test 1: Direct Lookup Optimization ===~n'),

    compile_predicate_to_go(user_by_name_exact/3, [
        json_schema(user_simple),
        db_backend(bbolt),
        db_file('test_phase_8c.db'),
        db_bucket(users),
        db_key_field(name),  % Single key field
        package(main)
    ], Code),

    % Write generated code
    make_directory_path('output_phase_8c_direct'),
    open('output_phase_8c_direct/query.go', write, Stream),
    write(Stream, Code),
    close(Stream),

    format('Generated: output_phase_8c_direct/query.go~n'),
    format('Purpose: Direct lookup with Name="Alice"~n'),
    format('Expected: Uses bucket.Get() instead of ForEach()~n').

%% ============================================
%% TEST 2: Prefix Scan (Composite Key)
%% ============================================

% Predicate that should trigger prefix scan optimization
users_in_city(Name, Age, City) :-
    json_record([name-Name, city-City, age-Age]),
    City = "NYC".

test_prefix_scan :-
    format('~n=== Test 2: Prefix Scan Optimization ===~n'),

    compile_predicate_to_go(users_in_city/3, [
        json_schema(user_simple),
        db_backend(bbolt),
        db_file('test_phase_8c.db'),
        db_bucket(users),
        db_key_field([city, name]),  % Composite key
        package(main)
    ], Code),

    % Write generated code
    make_directory_path('output_phase_8c_prefix'),
    open('output_phase_8c_prefix/query.go', write, Stream),
    write(Stream, Code),
    close(Stream),

    format('Generated: output_phase_8c_prefix/query.go~n'),
    format('Purpose: Prefix scan with City="NYC"~n'),
    format('Expected: Uses cursor.Seek() with prefix~n').

%% ============================================
%% TEST 3: Full Scan Fallback (Non-Key Field)
%% ============================================

% Predicate that should use full scan (age not in key)
users_by_age(Name, Age) :-
    json_record([name-Name, age-Age]),
    Age > 30.

test_full_scan :-
    format('~n=== Test 3: Full Scan Fallback ===~n'),

    compile_predicate_to_go(users_by_age/2, [
        json_schema(user_simple),
        db_backend(bbolt),
        db_file('test_phase_8c.db'),
        db_bucket(users),
        db_key_field(name),  % Key is name, but constraint is on age
        package(main)
    ], Code),

    % Write generated code
    make_directory_path('output_phase_8c_fullscan'),
    open('output_phase_8c_fullscan/query.go', write, Stream),
    write(Stream, Code),
    close(Stream),

    format('Generated: output_phase_8c_fullscan/query.go~n'),
    format('Purpose: Full scan with Age>30 (non-key constraint)~n'),
    format('Expected: Uses bucket.ForEach()~n').

%% ============================================
%% TEST 4: No Optimization (Case-Insensitive)
%% ============================================

% Predicate with case-insensitive match (can't use direct lookup)
user_by_name_ci(Name, Age) :-
    json_record([name-Name, age-Age]),
    Name =@= "alice".

test_no_optimization :-
    format('~n=== Test 4: No Optimization (Case-Insensitive) ===~n'),

    compile_predicate_to_go(user_by_name_ci/2, [
        json_schema(user_simple),
        db_backend(bbolt),
        db_file('test_phase_8c.db'),
        db_bucket(users),
        db_key_field(name),
        package(main)
    ], Code),

    % Write generated code
    make_directory_path('output_phase_8c_noopt'),
    open('output_phase_8c_noopt/query.go', write, Stream),
    write(Stream, Code),
    close(Stream),

    format('Generated: output_phase_8c_noopt/query.go~n'),
    format('Purpose: Case-insensitive match (Name =@= "alice")~n'),
    format('Expected: Uses bucket.ForEach() (can''t optimize)~n').

%% ============================================
%% TEST 5: Composite Key Direct Lookup
%% ============================================

% Predicate with exact match on both key fields
user_exact_match(Name, Age) :-
    json_record([city-City, name-Name, age-Age]),
    City = "NYC",
    Name = "Alice".

test_composite_direct :-
    format('~n=== Test 5: Composite Key Direct Lookup ===~n'),

    compile_predicate_to_go(user_exact_match/2, [
        json_schema(user_simple),
        db_backend(bbolt),
        db_file('test_phase_8c.db'),
        db_bucket(users),
        db_key_field([city, name]),  % Both fields in constraint
        package(main)
    ], Code),

    % Write generated code
    make_directory_path('output_phase_8c_composite_direct'),
    open('output_phase_8c_composite_direct/query.go', write, Stream),
    write(Stream, Code),
    close(Stream),

    format('Generated: output_phase_8c_composite_direct/query.go~n'),
    format('Purpose: Direct lookup with City="NYC", Name="Alice"~n'),
    format('Expected: Uses bucket.Get() with composite key "NYC:Alice"~n').

%% Run all tests
:- initialization((
    test_direct_lookup,
    test_prefix_scan,
    test_full_scan,
    test_no_optimization,
    test_composite_direct,
    format('~n=== All Phase 8c tests complete ===~n'),
    halt
)).
