:- encoding(utf8).
:- use_module(library(filesex)).
:- use_module('src/unifyweaver/targets/go_target').

person(alice, 25).
person(bob, 17).
person(charlie, 45).

adult(X, Age) :- person(X, Age), Age > 18.

test :-
    make_directory_path('output_test'),
    go_target:compile_predicate_to_go(adult/2, [], Code),
    go_target:write_go_program(Code, 'output_test/adult.go').
