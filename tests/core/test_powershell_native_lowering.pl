:- module(test_powershell_native_lowering, [test_powershell_native_lowering/0]).
:- use_module(library(plunit)).
:- use_module('../../src/unifyweaver/targets/powershell_target').

test_powershell_native_lowering :-
    run_tests([powershell_native_lowering]).

:- begin_tests(powershell_native_lowering).

compile_ps(Pred/Arity, Code) :-
    powershell_target:compile_predicate_to_powershell(Pred/Arity, [], Code).

has(Code, Substr) :-
    once(sub_string(Code, _, _, _, Substr)).

% Tier 1: Multi-clause → if/else chain
test(multi_clause_guard_chain) :-
    assert(user:(classify(X, small) :- X > 0, X < 10)),
    assert(user:(classify(X, large) :- X >= 10)),
    compile_ps(classify/2, Code),
    has(Code, "function classify"),
    has(Code, "$arg1 -gt 0 -and $arg1 -lt 10"),
    has(Code, "\"small\""),
    has(Code, "$arg1 -ge 10"),
    has(Code, "\"large\""),
    retractall(user:classify(_, _)).

test(single_clause_guard) :-
    assert(user:(positive(X, yes) :- X > 0)),
    assert(user:(positive(X, no) :- X =< 0)),
    compile_ps(positive/2, Code),
    has(Code, "function positive"),
    has(Code, "$arg1 -gt 0"),
    has(Code, "\"yes\""),
    retractall(user:positive(_, _)).

test(arithmetic_output) :-
    assert(user:(double(X, R) :- R is X * 2)),
    compile_ps(double/2, Code),
    has(Code, "function double"),
    has(Code, "($arg1 * 2)"),
    retractall(user:double(_, _)).

test(assignment_output) :-
    assert(user:(identity(X, R) :- R = X)),
    compile_ps(identity/2, Code),
    has(Code, "function identity"),
    has(Code, "$arg1"),
    retractall(user:identity(_, _)).

% Tier 2: If-then-else
test(if_then_else_simple) :-
    assert(user:(abs_val(X, R) :- (X >= 0 -> R = X ; R is -X))),
    compile_ps(abs_val/2, Code),
    has(Code, "$arg1 -ge 0"),
    has(Code, "$arg1"),
    has(Code, "(-$arg1)"),
    retractall(user:abs_val(_, _)).

test(nested_if_then_else) :-
    assert(user:(range_classify(X, R) :-
        (X < 0 -> R = negative
        ; (X =:= 0 -> R = zero
        ; R = positive)))),
    compile_ps(range_classify/2, Code),
    has(Code, "$arg1 -lt 0"),
    has(Code, "\"negative\""),
    has(Code, "$arg1 -eq 0"),
    has(Code, "\"zero\""),
    has(Code, "\"positive\""),
    retractall(user:range_classify(_, _)).

% PowerShell-specific syntax
test(ps_uses_dollar_vars) :-
    assert(user:(inc(X, Y) :- Y is X + 1)),
    compile_ps(inc/2, Code),
    has(Code, "$arg1"),
    retractall(user:inc(_, _)).

test(ps_uses_dash_operators) :-
    assert(user:(cmp(X, big) :- X > 100)),
    assert(user:(cmp(X, small) :- X =< 100)),
    compile_ps(cmp/2, Code),
    has(Code, "-gt"),
    has(Code, "-le"),
    retractall(user:cmp(_, _)).

test(three_clause) :-
    assert(user:(grade(X, low) :- X < 50)),
    assert(user:(grade(X, mid) :- X >= 50, X < 80)),
    assert(user:(grade(X, high) :- X >= 80)),
    compile_ps(grade/2, Code),
    has(Code, "$arg1 -lt 50"),
    has(Code, "$arg1 -ge 80"),
    has(Code, "\"low\""),
    has(Code, "\"high\""),
    retractall(user:grade(_, _)).

test(complex_arithmetic) :-
    assert(user:(formula(X, Y) :- Y is (X * X) + (X * 2) + 1)),
    compile_ps(formula/2, Code),
    has(Code, "($arg1 * $arg1)"),
    has(Code, "($arg1 * 2)"),
    retractall(user:formula(_, _)).

test(mod_guard) :-
    assert(user:(parity(X, even) :- 0 =:= X mod 2)),
    assert(user:(parity(X, odd) :- 0 =\= X mod 2)),
    compile_ps(parity/2, Code),
    has(Code, "%"),
    has(Code, "\"even\""),
    retractall(user:parity(_, _)).

:- end_tests(powershell_native_lowering).
