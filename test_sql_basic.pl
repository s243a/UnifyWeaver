% test_sql_basic.pl - Basic SQL Target Tests
% Tests Phase 1: Simple SELECT queries with WHERE clauses

:- use_module('src/unifyweaver/targets/sql_target').

% Define table schema
:- sql_table(person, [name-text, age-integer, city-text]).

% Test predicates

% Test 1: Simple filter (adults aged 18+)
test_adult :-
    write('Test 1: Adult filter (age >= 18)'), nl,
    compile_predicate_to_sql(adult/1, [format(view)], SQL),
    write(SQL), nl,
    % Verify SQL contains expected elements
    (   sub_string(SQL, _, _, _, "SELECT"),
        sub_string(SQL, _, _, _, "FROM person"),
        sub_string(SQL, _, _, _, "WHERE"),
        sub_string(SQL, _, _, _, "age >= 18")
    ->  write('✓ Test 1 passed'), nl
    ;   write('✗ Test 1 failed'), nl, fail
    ).

adult(Name) :- person(Name, Age, _), Age >= 18.

% Test 2: Filter with city
test_city_filter :-
    write('Test 2: NYC residents'), nl,
    compile_predicate_to_sql(nyc_residents/1, [format(view)], SQL),
    write(SQL), nl,
    (   sub_string(SQL, _, _, _, "SELECT"),
        sub_string(SQL, _, _, _, "FROM person")
    ->  write('✓ Test 2 passed'), nl
    ;   write('✗ Test 2 failed'), nl, fail
    ).

nyc_residents(Name) :- person(Name, _, "NYC").

% Test 3: Multiple constraints
test_multiple_constraints :-
    write('Test 3: NYC adults'), nl,
    compile_predicate_to_sql(nyc_adults/1, [format(view)], SQL),
    write(SQL), nl,
    (   sub_string(SQL, _, _, _, "SELECT"),
        sub_string(SQL, _, _, _, "FROM person"),
        sub_string(SQL, _, _, _, "WHERE"),
        sub_string(SQL, _, _, _, "AND")
    ->  write('✓ Test 3 passed'), nl
    ;   write('✗ Test 3 failed'), nl, fail
    ).

nyc_adults(Name) :- person(Name, Age, "NYC"), Age >= 21.

% Test 4: Range query
test_range :-
    write('Test 4: Age range (30-50)'), nl,
    compile_predicate_to_sql(middle_aged/2, [format(view)], SQL),
    write(SQL), nl,
    (   sub_string(SQL, _, _, _, "SELECT"),
        sub_string(SQL, _, _, _, "FROM person"),
        sub_string(SQL, _, _, _, "WHERE")
    ->  write('✓ Test 4 passed'), nl
    ;   write('✗ Test 4 failed'), nl, fail
    ).

middle_aged(Name, Age) :-
    person(Name, Age, _),
    Age >= 30,
    Age =< 50.

% Run all tests
run_tests :-
    write('======================================'), nl,
    write('  SQL Target Basic Tests'), nl,
    write('======================================'), nl, nl,
    test_adult,
    nl,
    test_city_filter,
    nl,
    test_multiple_constraints,
    nl,
    test_range,
    nl,
    write('======================================'), nl,
    write('  All tests passed!'), nl,
    write('======================================'), nl.

% Test helper - compile and save to file
test_save_to_file :-
    write('Generating SQL file...'), nl,
    compile_predicate_to_sql(adult/1, [format(view)], SQL),
    write_sql_file(SQL, 'output_sql_test/adult.sql'),
    write('SQL file created: output_sql_test/adult.sql'), nl.
