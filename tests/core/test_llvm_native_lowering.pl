:- module(test_llvm_native_lowering, [test_llvm_native_lowering/0]).
:- use_module(library(plunit)).
:- use_module('../../src/unifyweaver/targets/llvm_target').

test_llvm_native_lowering :-
    run_tests([llvm_native_lowering]).

:- begin_tests(llvm_native_lowering).

% Helper: compile using the public API
compile_ll(Pred/Arity, Code) :-
    llvm_target:compile_predicate_to_llvm(Pred/Arity, [], Code).

% Helper: check substring exists (deterministic)
has(Code, Substr) :-
    once(sub_string(Code, _, _, _, Substr)).

% ============================================================================
% Tier 1: Multi-clause predicates → chained basic blocks
% ============================================================================

test(multi_clause_guard_chain) :-
    assert(user:(classify(X, small) :- X > 0, X < 10)),
    assert(user:(classify(X, large) :- X >= 10)),
    compile_ll(classify/2, Code),
    has(Code, "define i64 @classify"),
    has(Code, "icmp sgt i64"),
    has(Code, "icmp slt i64"),
    has(Code, "icmp sge i64"),
    has(Code, "ret i64"),
    retractall(user:classify(_, _)).

test(single_clause_guard) :-
    assert(user:(positive(X, yes) :- X > 0)),
    compile_ll(positive/2, Code),
    has(Code, "define i64 @positive"),
    has(Code, "icmp sgt i64"),
    retractall(user:positive(_, _)).

test(arithmetic_output) :-
    assert(user:(double(X, R) :- R is X * 2)),
    compile_ll(double/2, Code),
    has(Code, "define i64 @double"),
    has(Code, "mul i64"),
    retractall(user:double(_, _)).

test(assignment_output) :-
    assert(user:(identity(X, R) :- R = X)),
    compile_ll(identity/2, Code),
    has(Code, "define i64 @identity"),
    has(Code, "ret i64"),
    retractall(user:identity(_, _)).

test(multi_clause_numeric) :-
    assert(user:(color2(X, warm) :- X == red)),
    assert(user:(color2(X, cool) :- X == blue)),
    compile_ll(color2/2, Code),
    has(Code, "define i64 @color2"),
    has(Code, "ret i64"),
    retractall(user:color2(_, _)).

% ============================================================================
% Tier 2: If-then-else
% ============================================================================

test(if_then_else_simple) :-
    assert(user:(abs_val(X, R) :- (X >= 0 -> R = X ; R is -X))),
    compile_ll(abs_val/2, Code),
    has(Code, "define i64 @abs_val(i64 %arg1)"),
    has(Code, "ret i64"),
    retractall(user:abs_val(_, _)).

test(nested_if_then_else) :-
    assert(user:(range_classify(X, R) :-
        (X < 0 -> R = negative
        ; (X =:= 0 -> R = zero
        ; R = positive)))),
    compile_ll(range_classify/2, Code),
    has(Code, "define i64 @range_classify(i64 %arg1)"),
    retractall(user:range_classify(_, _)).

test(three_way_nested) :-
    assert(user:(sign(X, R) :-
        (X > 0 -> R = positive
        ; (X < 0 -> R = negative
        ; R = zero)))),
    compile_ll(sign/2, Code),
    has(Code, "define i64 @sign(i64 %arg1)"),
    retractall(user:sign(_, _)).

% ============================================================================
% Tier 1: Guard separation with arity > 2
% ============================================================================

test(guard_with_computation) :-
    assert(user:(safe_div(X, Y, R) :- Y > 0, R is X / Y)),
    compile_ll(safe_div/3, Code),
    has(Code, "define i64 @safe_div(i64 %arg1, i64 %arg2)"),
    has(Code, "icmp sgt i64 %arg2, 0"),
    has(Code, "sdiv i64 %arg1"),
    retractall(user:safe_div(_, _, _)).

% ============================================================================
% LLVM-specific syntax
% ============================================================================

test(llvm_uses_icmp) :-
    assert(user:(grade(X, pass) :- X >= 50)),
    assert(user:(grade(X, fail) :- X < 50)),
    compile_ll(grade/2, Code),
    has(Code, "icmp"),
    retractall(user:grade(_, _)).

test(llvm_uses_exit) :-
    assert(user:(only_pos(X, yes) :- X > 0)),
    compile_ll(only_pos/2, Code),
    has(Code, "call void @exit(i32 1)"),
    retractall(user:only_pos(_, _)).

% ============================================================================
% Verify shared module is loaded
% ============================================================================

test(uses_shared_analysis_module) :-
    current_predicate(clause_body_analysis:normalize_goals/2),
    current_predicate(clause_body_analysis:if_then_else_goal/4),
    current_predicate(clause_body_analysis:build_head_varmap/3).

:- end_tests(llvm_native_lowering).
