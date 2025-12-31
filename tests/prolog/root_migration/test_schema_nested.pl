% Test JSON schema support - nested fields with types
% Tests type validation in nested JSON structures

:- use_module('src/unifyweaver/targets/go_target').

% Define schema for nested fields
:- json_schema(profile, [
    field(name, string),
    field(city, string),
    field(age, integer)
]).

% Predicate with nested field access
user_info(Name, City) :-
    json_get([user, name], Name),
    json_get([user, address, city], City).

test_nested_schema :-
    format('~n=== Test: Nested Fields with Schema ===~n'),

    % Compile with schema
    compile_predicate_to_go(user_info/2, [
        json_input(true),
        json_schema(profile)
    ], Code),

    % Write to file
    write_go_program(Code, 'schema_nested.go'),
    format('Generated schema_nested.go~n').

% Run test
:- initialization((test_nested_schema, halt)).
