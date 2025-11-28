:- encoding(utf8).
:- use_module('src/unifyweaver/targets/go_target').

person(alice, 25).
person(bob, 17).
person(charlie, 45).

working_age(X, Age) :- person(X, Age), Age >= 18, Age =< 65.

test :-
    go_target:compile_predicate_to_go(working_age/2, [], Code),
    go_target:write_go_program(Code, 'working_age.go').
