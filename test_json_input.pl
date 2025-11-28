:- use_module('src/unifyweaver/targets/go_target').

% Test 1: Simple JSON input with 2 fields
% Input: {"name": "Alice", "age": 25}
% Output: Alice:25

user(Name, Age) :- json_record([name-Name, age-Age]).

test_simple :-
    compile_predicate_to_go(user/2, [json_input(true)], Code),
    format('~n=== Generated Go Code ===~n~s~n', [Code]).

test_write :-
    compile_predicate_to_go(user/2, [json_input(true)], Code),
    write_go_program(Code, 'user_json.go'),
    format('Generated user_json.go~n').

% Test 2: Three fields
% Input: {"id": 1, "name": "Bob", "active": true}
% Output: 1:Bob:true

record(Id, Name, Active) :- json_record([id-Id, name-Name, active-Active]).

test_three_fields :-
    compile_predicate_to_go(record/3, [json_input(true)], Code),
    write_go_program(Code, 'record_json.go'),
    format('Generated record_json.go~n').
