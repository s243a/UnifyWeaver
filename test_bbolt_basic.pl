#!/usr/bin/env swipl

% Test basic bbolt database storage without schema validation

:- use_module('src/unifyweaver/targets/go_target').

% Simple user predicate - no schema
user(Name, Age) :-
    json_record([name-Name, age-Age]).

test_basic_bbolt :-
    % Compile with bbolt backend
    compile_predicate_to_go(user/2, [
        json_input(true),
        db_backend(bbolt),
        db_file('test_users.db'),
        db_bucket(users),
        db_key_field(name),
        package(main)
    ], Code),

    % Write generated code
    open('output_bbolt_basic/user_store.go', write, Stream),
    write(Stream, Code),
    close(Stream),

    format('Generated: output_bbolt_basic/user_store.go~n').

:- initialization(test_basic_bbolt, main).
