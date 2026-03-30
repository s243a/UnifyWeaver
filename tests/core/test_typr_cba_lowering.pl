:- module(test_typr_cba_lowering, [test_typr_cba_lowering/0]).
:- use_module(library(plunit)).
:- use_module('../../src/unifyweaver/targets/typr_target').

test_typr_cba_lowering :-
    run_tests([typr_cba_lowering]).

:- begin_tests(typr_cba_lowering).

compile_typr(Pred/Arity, Code) :-
    typr_target:compile_predicate_to_typr(Pred/Arity, [], Code).

has(Code, Substr) :-
    once(sub_string(Code, _, _, _, Substr)).

% Tier 1: Multi-clause → if/else chain
test(multi_clause_guard_chain) :-
    assert(user:(classify(X, small) :- X > 0, X < 10)),
    assert(user:(classify(X, large) :- X >= 10)),
    compile_typr(classify/2, Code),
    has(Code, "classify"),
    has(Code, "fn("),
    has(Code, "arg1 > 0"),
    has(Code, "arg1 < 10"),
    has(Code, "\"small\""),
    has(Code, "arg1 >= 10"),
    has(Code, "\"large\""),
    retractall(user:classify(_, _)).

test(single_clause_guard) :-
    assert(user:(positive(X, yes) :- X > 0)),
    assert(user:(positive(X, no) :- X =< 0)),
    compile_typr(positive/2, Code),
    has(Code, "positive"),
    has(Code, "arg1 > 0"),
    has(Code, "\"yes\""),
    retractall(user:positive(_, _)).

test(arithmetic_output) :-
    assert(user:(double(X, R) :- R is X * 2)),
    compile_typr(double/2, Code),
    has(Code, "double"),
    has(Code, "arg1"),
    has(Code, "* 2"),
    retractall(user:double(_, _)).

test(assignment_output) :-
    assert(user:(identity(X, R) :- R = X)),
    compile_typr(identity/2, Code),
    has(Code, "fn("),
    has(Code, "arg1"),
    retractall(user:identity(_, _)).

% Tier 2: If-then-else
test(if_then_else_simple) :-
    assert(user:(abs_val(X, R) :- (X >= 0 -> R = X ; R is -X))),
    compile_typr(abs_val/2, Code),
    has(Code, ">= 0"),
    has(Code, "arg1"),
    retractall(user:abs_val(_, _)).

test(nested_if_then_else) :-
    assert(user:(range_classify(X, R) :-
        (X < 0 -> R = negative
        ; (X =:= 0 -> R = zero
        ; R = positive)))),
    compile_typr(range_classify/2, Code),
    has(Code, "< 0"),
    has(Code, "\"negative\""),
    has(Code, "== 0"),
    has(Code, "\"zero\""),
    has(Code, "\"positive\""),
    retractall(user:range_classify(_, _)).

% TypR-specific syntax
test(typr_uses_let_fn) :-
    assert(user:(inc(X, Y) :- Y is X + 1)),
    compile_typr(inc/2, Code),
    has(Code, "let inc <- fn("),
    retractall(user:inc(_, _)).

test(typr_uses_r_embedded) :-
    assert(user:(double_it(X, Y) :- Y is X * 2)),
    compile_typr(double_it/2, Code),
    has(Code, "@{"),
    has(Code, "}@"),
    retractall(user:double_it(_, _)).

test(three_clause) :-
    assert(user:(grade(X, low) :- X < 50)),
    assert(user:(grade(X, mid) :- X >= 50, X < 80)),
    assert(user:(grade(X, high) :- X >= 80)),
    compile_typr(grade/2, Code),
    has(Code, "< 50"),
    has(Code, ">= 80"),
    has(Code, "\"low\""),
    has(Code, "\"high\""),
    retractall(user:grade(_, _)).

test(complex_arithmetic) :-
    assert(user:(formula(X, Y) :- Y is (X * X) + (X * 2) + 1)),
    compile_typr(formula/2, Code),
    has(Code, "arg1"),
    has(Code, "* 2"),
    retractall(user:formula(_, _)).

test(mod_guard) :-
    assert(user:(parity(X, even) :- 0 =:= X mod 2)),
    assert(user:(parity(X, odd) :- 0 =\= X mod 2)),
    compile_typr(parity/2, Code),
    has(Code, "%%"),
    has(Code, "\"even\""),
    retractall(user:parity(_, _)).

:- end_tests(typr_cba_lowering).
