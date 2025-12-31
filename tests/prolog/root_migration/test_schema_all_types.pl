% Test JSON schema support - all primitive types
% Tests string, integer, float, boolean validation

:- use_module('src/unifyweaver/targets/go_target').

% Define schema with all primitive types
:- json_schema(employee, [
    field(name, string),
    field(salary, float),
    field(age, integer),
    field(active, boolean)
]).

% Predicate using all types
employee(Name, Salary, Age, Active) :-
    json_record([name-Name, salary-Salary, age-Age, active-Active]).

test_all_types :-
    format('~n=== Test: All Primitive Types ===~n'),

    % Compile with schema
    compile_predicate_to_go(employee/4, [
        json_input(true),
        json_schema(employee)
    ], Code),

    % Write to file
    write_go_program(Code, 'schema_all_types.go'),
    format('Generated schema_all_types.go~n').

% Run test
:- initialization((test_all_types, halt)).
