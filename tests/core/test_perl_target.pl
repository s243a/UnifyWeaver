:- module(test_perl_target, [
    test_perl_target/0
]).

:- use_module(library(plunit)).
:- use_module('../../src/unifyweaver/targets/perl_target').

test_perl_target :-
    run_tests(perl_target).

:- begin_tests(perl_target).

test(compile_facts) :-
    retractall(user:parent_perl(_, _)),
    assertz(user:parent_perl(a, b)),
    assertz(user:parent_perl(b, c)),
    perl_target:compile_predicate_to_perl(parent_perl/2, [], Code),
    sub_string(Code, _, _, _, "sub parent_perl"),
    sub_string(Code, _, _, _, "['a', 'b']"),
    sub_string(Code, _, _, _, "['b', 'c']").

:- end_tests(perl_target).
