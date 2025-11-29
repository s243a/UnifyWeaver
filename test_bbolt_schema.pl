#!/usr/bin/env swipl

% Test bbolt database storage with JSON schema validation

:- use_module('src/unifyweaver/targets/go_target').

% Define user schema
:- json_schema(user, [
    field(name, string),
    field(age, integer),
    field(email, string)
]).

% User predicate with schema
user(Name, Age, Email) :-
    json_record([name-Name, age-Age, email-Email]).

test_schema_bbolt :-
    % Compile with schema and bbolt backend
    compile_predicate_to_go(user/3, [
        json_input(true),
        json_schema(user),
        db_backend(bbolt),
        db_file('test_users_schema.db'),
        db_bucket(users),
        db_key_field(name),
        package(main)
    ], Code),

    % Write generated code
    open('output_bbolt_schema/user_store.go', write, Stream),
    write(Stream, Code),
    close(Stream),

    format('Generated: output_bbolt_schema/user_store.go~n').

:- initialization(test_schema_bbolt, main).
