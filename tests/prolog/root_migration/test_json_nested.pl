:- use_module('src/unifyweaver/targets/go_target').

%% Nested JSON Field Access Tests

% Test 1: Simple nested access (2 levels)
city(City) :- json_get([user, city], City).

test_simple_nested :-
    write('=== Test 1: Simple nested (2 levels) ===\n'),
    compile_predicate_to_go(city/1, [json_input(true)], Code),
    format('~n~s~n', [Code]),
    write_go_program(Code, 'nested_simple.go'),
    write('Generated: nested_simple.go\n').

% Test 2: Deep nested (3 levels)
location(City) :- json_get([user, address, city], City).

test_deep_nested :-
    write('=== Test 2: Deep nested (3 levels) ===\n'),
    compile_predicate_to_go(location/1, [json_input(true)], Code),
    write_go_program(Code, 'nested_deep.go'),
    write('Generated: nested_deep.go\n').

% Test 3: Multiple nested fields
user_info(Name, City) :-
    json_get([user, name], Name),
    json_get([user, address, city], City).

test_multiple_nested :-
    write('=== Test 3: Multiple nested fields ===\n'),
    compile_predicate_to_go(user_info/2, [json_input(true)], Code),
    write_go_program(Code, 'nested_multiple.go'),
    write('Generated: nested_multiple.go\n').

% Test 4: Mixed flat and nested
mixed_data(Id, City) :-
    json_record([id-Id]),
    json_get([location, city], City).

test_mixed :-
    write('=== Test 4: Mixed flat and nested ===\n'),
    compile_predicate_to_go(mixed_data/2, [json_input(true)], Code),
    write_go_program(Code, 'nested_mixed.go'),
    write('Generated: nested_mixed.go\n').

% Test 5: Very deep nesting (4 levels)
team_lead(Lead) :- json_get([company, department, team, lead], Lead).

test_very_deep :-
    write('=== Test 5: Very deep nesting (4 levels) ===\n'),
    compile_predicate_to_go(team_lead/1, [json_input(true)], Code),
    write_go_program(Code, 'nested_very_deep.go'),
    write('Generated: nested_very_deep.go\n').

% Run all tests
run_all_tests :-
    test_simple_nested,
    test_deep_nested,
    test_multiple_nested,
    test_mixed,
    test_very_deep,
    write('\n=== All nested field test programs generated ===\n').
