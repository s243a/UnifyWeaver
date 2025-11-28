:- encoding(utf8).
:- use_module('src/unifyweaver/targets/go_target').

person(alice, 25).
person(bob, 17).
person(charlie, 45).

child(X, Age) :- person(X, Age), Age < 18.

test :-
    go_target:compile_predicate_to_go(child/2, [], Code),
    go_target:write_go_program(Code, 'child.go').
