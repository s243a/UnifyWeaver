% Test JSON schema support - basic typed fields
% Tests string and integer type validation

:- use_module('src/unifyweaver/targets/go_target').

% Define schema with basic types
:- json_schema(user, [
    field(name, string),
    field(age, integer)
]).

% Predicate using schema
user(Name, Age) :- json_record([name-Name, age-Age]).

test_basic_schema :-
    format('~n=== Test: Basic Schema (string, integer) ===~n'),

    % Compile with schema
    compile_predicate_to_go(user/2, [
        json_input(true),
        json_schema(user)
    ], Code),

    % Write to file
    write_go_program(Code, 'schema_basic.go'),
    format('Generated schema_basic.go~n').

% Run test
:- initialization((test_basic_schema, halt)).
