:- use_module('src/unifyweaver/targets/go_target').

%% JSON Output Tests
%% Transform delimiter-based input to JSON

% Test 1: Two fields - default field names
user(Name, Age).

test_default_names :-
    write('=== Test 1: JSON output with default field names ===\n'),
    compile_predicate_to_go(user/2, [json_output(true)], Code),
    format('~n~s~n', [Code]),
    write_go_program(Code, 'output_default.go'),
    write('Generated: output_default.go\n').

% Test 2: Custom field names
person(N, A).

test_custom_names :-
    write('=== Test 2: JSON output with custom field names ===\n'),
    compile_predicate_to_go(person/2, [
        json_output(true),
        json_fields([name, age])
    ], Code),
    write_go_program(Code, 'output_custom.go'),
    write('Generated: output_custom.go\n').

% Test 3: Three fields
record(Id, Name, Active).

test_three_fields :-
    write('=== Test 3: Three fields ===\n'),
    compile_predicate_to_go(record/3, [
        json_output(true),
        json_fields([id, name, active])
    ], Code),
    write_go_program(Code, 'output_three.go'),
    write('Generated: output_three.go\n').

% Test 4: Tab-delimited input
employee(Id, Name, Dept, Salary).

test_tab_input :-
    write('=== Test 4: Tab-delimited input to JSON ===\n'),
    compile_predicate_to_go(employee/4, [
        json_output(true),
        field_delimiter(tab),
        json_fields([id, name, department, salary])
    ], Code),
    write_go_program(Code, 'output_tab.go'),
    write('Generated: output_tab.go\n').

% Run all tests
run_all_tests :-
    test_default_names,
    test_custom_names,
    test_three_fields,
    test_tab_input,
    write('\n=== All JSON output test programs generated ===\n').
