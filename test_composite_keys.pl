#!/usr/bin/env swipl

% Test composite key strategies for bbolt database

:- use_module('src/unifyweaver/targets/go_target').

%% Test 1: Simple composite key (name + city)
:- json_schema(user_location, [
    field(name, string),
    field(age, integer),
    field(city, string)
]).

user_location(Name, Age, City) :-
    json_record([name-Name, age-Age, city-City]).

test_composite_simple :-
    format('~n=== Test 1: Simple Composite Key (name + city) ===~n'),

    compile_predicate_to_go(user_location/3, [
        json_input(true),
        json_schema(user_location),
        db_backend(bbolt),
        db_file('composite_simple.db'),
        db_bucket(users),
        db_key_strategy(composite([field(name), field(city)])),
        db_key_delimiter(':'),
        package(main)
    ], Code),

    % Write generated code
    open('output_composite_simple/user_store.go', write, Stream),
    write(Stream, Code),
    close(Stream),

    format('Generated: output_composite_simple/user_store.go~n'),
    format('Expected keys: Alice:NYC, Bob:SF, etc.~n').

%% Test 2: Backward compatibility - db_key_field (legacy)
user_simple(Name, Age) :-
    json_record([name-Name, age-Age]).

test_backward_compat :-
    format('~n=== Test 2: Backward Compatibility (db_key_field) ===~n'),

    compile_predicate_to_go(user_simple/2, [
        json_input(true),
        db_backend(bbolt),
        db_file('backward_compat.db'),
        db_bucket(users),
        db_key_field(name),  % Legacy syntax
        package(main)
    ], Code),

    % Write generated code
    open('output_backward_compat/user_store.go', write, Stream),
    write(Stream, Code),
    close(Stream),

    format('Generated: output_backward_compat/user_store.go~n'),
    format('Expected keys: Alice, Bob, etc. (single field)~n').

%% Test 3: Hash of content field
document(DocName, Content) :-
    json_record([name-DocName, content-Content]).

test_hash_key :-
    format('~n=== Test 3: Hash Key Strategy ===~n'),

    compile_predicate_to_go(document/2, [
        json_input(true),
        db_backend(bbolt),
        db_file('hash_keys.db'),
        db_bucket(documents),
        db_key_strategy(hash(field(content))),  % Hash the content field
        package(main)
    ], Code),

    % Write generated code
    open('output_hash_keys/doc_store.go', write, Stream),
    write(Stream, Code),
    close(Stream),

    format('Generated: output_hash_keys/doc_store.go~n'),
    format('Expected keys: SHA-256 hashes of content~n').

%% Test 4: Composite with hash (name + hash of content)
test_name_plus_hash :-
    format('~n=== Test 4: Composite Name + Hash(Content) ===~n'),

    compile_predicate_to_go(document/2, [
        json_input(true),
        db_backend(bbolt),
        db_file('name_hash.db'),
        db_bucket(documents),
        db_key_strategy(composite([
            field(name),
            hash(field(content))
        ])),
        db_key_delimiter(':'),
        package(main)
    ], Code),

    % Write generated code
    open('output_name_hash/doc_store.go', write, Stream),
    write(Stream, Code),
    close(Stream),

    format('Generated: output_name_hash/doc_store.go~n'),
    format('Expected keys: mydoc:a3f5e8b9..., etc.~n').

%% Main test runner
main :-
    format('~n===== Database Key Strategy Tests =====~n'),

    % Create output directories
    make_directory_path('output_composite_simple'),
    make_directory_path('output_backward_compat'),
    make_directory_path('output_hash_keys'),
    make_directory_path('output_name_hash'),

    % Run tests
    test_composite_simple,
    test_backward_compat,
    test_hash_key,
    test_name_plus_hash,

    format('~n===== All Tests Generated Successfully =====~n').

:- initialization(main, main).
