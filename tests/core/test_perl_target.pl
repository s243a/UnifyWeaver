:- module(test_perl_target, [
    test_perl_target/0
]).

:- use_module(library(plunit)).
:- use_module('../../src/unifyweaver/targets/perl_target').

test_perl_target :-
    run_tests(perl_target).

:- begin_tests(perl_target, [
    setup(cleanup_test_preds),
    cleanup(cleanup_test_preds)
]).

cleanup_test_preds :-
    retractall(user:parent_perl(_, _)),
    retractall(user:grandparent_perl(_, _)).

test(compile_facts) :-
    assertz(user:parent_perl(a, b)),
    assertz(user:parent_perl(b, c)),
    perl_target:compile_predicate_to_perl(parent_perl/2, [], Code),
    sub_string(Code, _, _, _, "sub parent_perl"),
    sub_string(Code, _, _, _, "['a', 'b']"),
    sub_string(Code, _, _, _, "['b', 'c']").

test(compile_join) :-
    assertz((user:grandparent_perl(X, Z) :- user:parent_perl(X, Y), user:parent_perl(Y, Z))),
    perl_target:compile_predicate_to_perl(grandparent_perl/2, [], Code),
    sub_string(Code, _, _, _, "sub grandparent_perl"),
    sub_string(Code, _, _, _, 'my ($arg1, $arg2) = @_;'),
    sub_string(Code, _, _, _, "parent_perl(sub {"),
    sub_string(Code, _, _, _, 'my ($arg1, $v0) = @_;'),
    sub_string(Code, _, _, _, 'my ($v0, $arg2) = @_;').

:- end_tests(perl_target).
