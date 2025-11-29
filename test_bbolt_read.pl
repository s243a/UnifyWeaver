#!/usr/bin/env swipl

% Test bbolt database read mode

:- use_module('src/unifyweaver/targets/go_target').

% Simple read predicate - outputs all records from database
read_users :-
    true.  % No body needed for read mode

test_read_bbolt :-
    % Compile with bbolt backend in read mode
    compile_predicate_to_go(read_users/0, [
        db_backend(bbolt),
        db_file('test_users.db'),
        db_bucket(users),
        db_mode(read),
        package(main)
    ], Code),

    % Write generated code
    open('output_bbolt_read/read_users.go', write, Stream),
    write(Stream, Code),
    close(Stream),

    format('Generated: output_bbolt_read/read_users.go~n').

:- initialization(test_read_bbolt, main).
