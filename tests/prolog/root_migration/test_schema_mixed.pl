% Test JSON schema support - mixed flat and nested with types
% Tests type validation in mixed field access patterns

:- use_module('src/unifyweaver/targets/go_target').

% Define schema
:- json_schema(data_record, [
    field(id, integer),
    field(city, string)
]).

% Predicate with mixed flat and nested access
mixed_data(Id, City) :-
    json_record([id-Id]),
    json_get([location, city], City).

test_mixed_schema :-
    format('~n=== Test: Mixed Flat and Nested with Schema ===~n'),

    % Compile with schema
    compile_predicate_to_go(mixed_data/2, [
        json_input(true),
        json_schema(data_record)
    ], Code),

    % Write to file
    write_go_program(Code, 'schema_mixed.go'),
    format('Generated schema_mixed.go~n').

% Run test
:- initialization((test_mixed_schema, halt)).
