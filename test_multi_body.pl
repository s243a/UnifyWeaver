:- use_module('src/unifyweaver/targets/go_target').

% Test multiple rules with different body predicates

% Different source predicates with different arities
user(alice).
employee(bob, engineering).
contractor(charlie, design, hourly).

% Multiple rules with different bodies
person(Name) :- user(Name).
person(Name) :- employee(Name, _).
person(Name) :- contractor(Name, _, _).

test :-
    compile_predicate_to_go(person/1, [], Code),
    format('~n=== Generated Go Code ===~n~s~n', [Code]).

test_write :-
    compile_predicate_to_go(person/1, [], Code),
    write_go_program(Code, 'person.go'),
    format('Generated person.go~n').
