:- use_module('src/unifyweaver/targets/go_target').

%% Comprehensive JSON Input Tests

% Test 1: Two string fields
user(Name, Age) :- json_record([name-Name, age-Age]).

test_two_fields :-
    write('=== Test 1: Two fields (string) ===\n'),
    compile_predicate_to_go(user/2, [json_input(true)], Code),
    write_go_program(Code, 'test_two_fields.go'),
    write('Generated: test_two_fields.go\n').

% Test 2: Three fields with mixed types (number, string, boolean)
record(Id, Name, Active) :- json_record([id-Id, name-Name, active-Active]).

test_three_fields :-
    write('=== Test 2: Three fields (mixed types) ===\n'),
    compile_predicate_to_go(record/3, [json_input(true)], Code),
    write_go_program(Code, 'test_three_fields.go'),
    write('Generated: test_three_fields.go\n').

% Test 3: Single field
product(Name) :- json_record([product-Name]).

test_single_field :-
    write('=== Test 3: Single field ===\n'),
    compile_predicate_to_go(product/1, [json_input(true)], Code),
    write_go_program(Code, 'test_single_field.go'),
    write('Generated: test_single_field.go\n').

% Test 4: Four fields
employee(Id, Name, Dept, Salary) :-
    json_record([id-Id, name-Name, department-Dept, salary-Salary]).

test_four_fields :-
    write('=== Test 4: Four fields ===\n'),
    compile_predicate_to_go(employee/4, [json_input(true)], Code),
    write_go_program(Code, 'test_four_fields.go'),
    write('Generated: test_four_fields.go\n').

% Test 5: With unique=false
duplicates(Name, Value) :- json_record([name-Name, value-Value]).

test_allow_duplicates :-
    write('=== Test 5: Allow duplicates (unique=false) ===\n'),
    compile_predicate_to_go(duplicates/2, [json_input(true), unique(false)], Code),
    write_go_program(Code, 'test_duplicates.go'),
    write('Generated: test_duplicates.go\n').

% Test 6: With different delimiter (tab)
tab_record(A, B) :- json_record([field1-A, field2-B]).

test_tab_delimiter :-
    write('=== Test 6: Tab delimiter ===\n'),
    compile_predicate_to_go(tab_record/2, [json_input(true), field_delimiter(tab)], Code),
    write_go_program(Code, 'test_tab.go'),
    write('Generated: test_tab.go\n').

% Run all tests
run_all_tests :-
    test_two_fields,
    test_three_fields,
    test_single_field,
    test_four_fields,
    test_allow_duplicates,
    test_tab_delimiter,
    write('\n=== All test programs generated ===\n').
