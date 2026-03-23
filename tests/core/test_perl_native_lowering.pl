:- module(test_perl_native_lowering, [test_perl_native_lowering/0]).
:- use_module(library(plunit)).
:- use_module('../../src/unifyweaver/targets/perl_target').

test_perl_native_lowering :-
    run_tests([perl_native_lowering]).

:- begin_tests(perl_native_lowering).

% Helper: compile using the public API
compile_pl(Pred/Arity, Code) :-
    perl_target:compile_predicate_to_perl(Pred/Arity, [], Code).

% Helper: check substring exists (deterministic)
has(Code, Substr) :-
    once(sub_string(Code, _, _, _, Substr)).

% ============================================================================
% Tier 1: Multi-clause predicates → if/elsif/else chains
% ============================================================================

test(multi_clause_guard_chain) :-
    assert(user:(classify(X, small) :- X > 0, X < 10)),
    assert(user:(classify(X, large) :- X >= 10)),
    compile_pl(classify/2, Code),
    has(Code, "sub classify"),
    has(Code, "$arg1 > 0 && $arg1 < 10"),
    has(Code, "\"small\""),
    has(Code, "elsif ($arg1 >= 10)"),
    has(Code, "\"large\""),
    retractall(user:classify(_, _)).

test(single_clause_guard) :-
    assert(user:(positive(X, yes) :- X > 0)),
    compile_pl(positive/2, Code),
    has(Code, "sub positive"),
    has(Code, "$arg1 > 0"),
    has(Code, "\"yes\""),
    retractall(user:positive(_, _)).

test(arithmetic_output) :-
    assert(user:(double(X, R) :- R is X * 2)),
    compile_pl(double/2, Code),
    has(Code, "sub double"),
    has(Code, "($arg1 * 2)"),
    retractall(user:double(_, _)).

test(assignment_output) :-
    assert(user:(identity(X, R) :- R = X)),
    compile_pl(identity/2, Code),
    has(Code, "sub identity"),
    has(Code, "$arg1"),
    retractall(user:identity(_, _)).

test(multi_clause_rules) :-
    assert(user:(color2(X, warm) :- X == red)),
    assert(user:(color2(X, cool) :- X == blue)),
    assert(user:(color2(X, cool) :- X == green)),
    compile_pl(color2/2, Code),
    has(Code, "sub color2"),
    has(Code, "$arg1 eq \"red\""),
    has(Code, "$arg1 eq \"blue\""),
    has(Code, "$arg1 eq \"green\""),
    retractall(user:color2(_, _)).

% ============================================================================
% Tier 2: If-then-else and nested conditionals
% ============================================================================

test(if_then_else_simple) :-
    assert(user:(abs_val(X, R) :- (X >= 0 -> R = X ; R is -X))),
    compile_pl(abs_val/2, Code),
    has(Code, "sub abs_val"),
    has(Code, "$arg1 >= 0"),
    has(Code, "$arg1"),
    has(Code, "(-$arg1)"),
    retractall(user:abs_val(_, _)).

test(nested_if_then_else) :-
    assert(user:(range_classify(X, R) :-
        (X < 0 -> R = negative
        ; (X =:= 0 -> R = zero
        ; R = positive)))),
    compile_pl(range_classify/2, Code),
    has(Code, "sub range_classify"),
    has(Code, "$arg1 < 0"),
    has(Code, "\"negative\""),
    has(Code, "$arg1 == 0"),
    has(Code, "\"zero\""),
    has(Code, "\"positive\""),
    retractall(user:range_classify(_, _)).

test(three_way_nested) :-
    assert(user:(sign(X, R) :-
        (X > 0 -> R = positive
        ; (X < 0 -> R = negative
        ; R = zero)))),
    compile_pl(sign/2, Code),
    has(Code, "sub sign"),
    has(Code, "$arg1 > 0"),
    has(Code, "$arg1 < 0"),
    retractall(user:sign(_, _)).

% ============================================================================
% Tier 1: Guard separation with arity > 2
% ============================================================================

test(guard_with_computation) :-
    assert(user:(safe_div(X, Y, R) :- Y > 0, R is X / Y)),
    compile_pl(safe_div/3, Code),
    has(Code, "sub safe_div"),
    has(Code, "my ($arg1, $arg2) = @_;"),
    has(Code, "$arg2 > 0"),
    retractall(user:safe_div(_, _, _)).

% ============================================================================
% Perl-specific syntax
% ============================================================================

test(perl_uses_elsif) :-
    assert(user:(grade(X, pass) :- X >= 50)),
    assert(user:(grade(X, fail) :- X < 50)),
    compile_pl(grade/2, Code),
    has(Code, "elsif"),
    retractall(user:grade(_, _)).

test(perl_uses_die) :-
    assert(user:(only_pos(X, yes) :- X > 0)),
    compile_pl(only_pos/2, Code),
    has(Code, "die"),
    retractall(user:only_pos(_, _)).

% ============================================================================
% Verify shared module is loaded
% ============================================================================

test(uses_shared_analysis_module) :-
    current_predicate(clause_body_analysis:normalize_goals/2),
    current_predicate(clause_body_analysis:if_then_else_goal/4),
    current_predicate(clause_body_analysis:build_head_varmap/3).

:- end_tests(perl_native_lowering).
