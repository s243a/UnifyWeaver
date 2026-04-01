:- module(test_go_native_lowering, [test_go_native_lowering/0]).
:- use_module(library(plunit)).
:- use_module('../../src/unifyweaver/targets/go_target').
:- use_module('../../src/unifyweaver/core/constraint_analyzer').

:- dynamic user:'table'/1.

test_go_native_lowering :-
    run_tests([go_native_lowering]).

:- begin_tests(go_native_lowering).

% Helper: compile using user module
compile_go(Pred/Arity, Code) :-
    go_target:compile_predicate_to_go(user:Pred/Arity, [], Code).

% Helper: check substring exists (deterministic)
has(Code, Substr) :-
    once(sub_string(Code, _, _, _, Substr)).

% ============================================================================
% Tier 1: Multi-clause predicates → if/else if/else chains
% ============================================================================

test(multi_clause_guard_chain) :-
    assert(user:(classify(X, small) :- X > 0, X < 10)),
    assert(user:(classify(X, large) :- X >= 10)),
    compile_go(classify/2, Code),
    has(Code, "func classify(arg1 interface{})"),
    has(Code, "if arg1 > 0 && arg1 < 10"),
    has(Code, "return \"small\""),
    has(Code, "else if arg1 >= 10"),
    has(Code, "return \"large\""),
    retractall(user:classify(_, _)).

test(single_clause_guard) :-
    assert(user:(positive(X, yes) :- X > 0)),
    compile_go(positive/2, Code),
    has(Code, "func positive(arg1 interface{})"),
    has(Code, "arg1 > 0"),
    has(Code, "return \"yes\""),
    retractall(user:positive(_, _)).

test(arithmetic_output) :-
    assert(user:(double(X, R) :- R is X * 2)),
    compile_go(double/2, Code),
    has(Code, "func double(arg1 interface{})"),
    has(Code, "(arg1 * 2)"),
    retractall(user:double(_, _)).

test(assignment_output) :-
    assert(user:(identity(X, R) :- R = X)),
    compile_go(identity/2, Code),
    has(Code, "func identity(arg1 interface{})"),
    has(Code, "arg1"),
    retractall(user:identity(_, _)).

test(multi_fact_native) :-
    assert(user:color2(red, warm)),
    assert(user:color2(blue, cool)),
    assert(user:color2(green, cool)),
    compile_go(color2/2, Code),
    has(Code, "func color2(arg1 interface{})"),
    has(Code, "if arg1 == \"red\""),
    has(Code, "else if arg1 == \"blue\""),
    has(Code, "else if arg1 == \"green\""),
    retractall(user:color2(_, _)).

% ============================================================================
% Tier 2: If-then-else and nested conditionals
% ============================================================================

test(if_then_else_simple) :-
    assert(user:(abs_val(X, R) :- (X >= 0 -> R = X ; R is -X))),
    compile_go(abs_val/2, Code),
    has(Code, "func abs_val(arg1 interface{})"),
    has(Code, "if arg1 >= 0"),
    has(Code, "return arg1"),
    has(Code, "return (-arg1)"),
    retractall(user:abs_val(_, _)).

test(nested_if_then_else) :-
    assert(user:(range_classify(X, R) :-
        (X < 0 -> R = negative
        ; (X =:= 0 -> R = zero
        ; R = positive)))),
    compile_go(range_classify/2, Code),
    has(Code, "func range_classify(arg1 interface{})"),
    has(Code, "arg1 < 0"),
    has(Code, "\"negative\""),
    has(Code, "arg1 == 0"),
    has(Code, "\"zero\""),
    has(Code, "\"positive\""),
    retractall(user:range_classify(_, _)).

test(three_way_nested) :-
    assert(user:(sign(X, R) :-
        (X > 0 -> R = positive
        ; (X < 0 -> R = negative
        ; R = zero)))),
    compile_go(sign/2, Code),
    has(Code, "func sign(arg1 interface{})"),
    has(Code, "arg1 > 0"),
    has(Code, "arg1 < 0"),
    retractall(user:sign(_, _)).

% ============================================================================
% Tier 1: Guard separation with arity > 2
% ============================================================================

test(guard_with_computation) :-
    assert(user:(safe_div(X, Y, R) :- Y > 0, R is X / Y)),
    compile_go(safe_div/3, Code),
    has(Code, "func safe_div(arg1 interface{}, arg2 interface{})"),
    has(Code, "arg2 > 0"),
    retractall(user:safe_div(_, _, _)).

% ============================================================================
% Go-specific syntax
% ============================================================================

test(go_package_and_import) :-
    assert(user:(double2(X, R) :- R is X * 2)),
    compile_go(double2/2, Code),
    has(Code, "package main"),
    has(Code, "import \"fmt\""),
    retractall(user:double2(_, _)).

test(go_uses_interface_type) :-
    assert(user:(id2(X, R) :- R = X)),
    compile_go(id2/2, Code),
    has(Code, "interface{}"),
    retractall(user:id2(_, _)).

% ============================================================================
% Verify shared module is loaded
% ============================================================================

test(uses_shared_analysis_module) :-
    current_predicate(clause_body_analysis:normalize_goals/2),
    current_predicate(clause_body_analysis:if_then_else_goal/4),
    current_predicate(clause_body_analysis:build_head_varmap/3).

test(weighted_recursive_accumulation_lowering) :-
    assert(user:(weighted_edge(a, b))),
    assert(user:(weighted_edge(b, c))),
    assert(user:(weighted_cost(a, 2))),
    assert(user:(weighted_cost(b, 5))),
    assert(user:(weighted_path(X, Y, Acc) :-
        weighted_edge(X, Y),
        weighted_cost(X, Cost),
        Acc is Cost)),
    assert(user:(weighted_path(X, Z, Acc) :-
        weighted_edge(X, Y),
        weighted_cost(X, Cost),
        weighted_path(Y, Z, PrevAcc),
        Acc is PrevAcc + Cost)),
    compile_go(weighted_path/3, Code),
    has(Code, "Path-aware recursive accumulation"),
    has(Code, "var weighted_pathAux = map[string]int{}"),
    has(Code, "results = append(results, weighted_pathResult{nb, stepCost})"),
    has(Code, "results = append(results, weighted_pathResult{sub.Value, (sub.Acc + stepCost)})"),
    retractall(user:weighted_edge(_, _)),
    retractall(user:weighted_cost(_, _)),
    retractall(user:weighted_path(_, _, _)).

test(log_recursive_accumulation_lowering) :-
    assert(user:(semantic_edge(a, b))),
    assert(user:(semantic_edge(b, c))),
    assert(user:(semantic_degree(a, 2))),
    assert(user:(semantic_degree(b, 5))),
    assert(user:(semantic_path(X, Y, Acc) :-
        semantic_edge(X, Y),
        semantic_degree(X, Deg),
        Acc is log(Deg) / log(5))),
    assert(user:(semantic_path(X, Z, Acc) :-
        semantic_edge(X, Y),
        semantic_degree(X, Deg),
        semantic_path(Y, Z, PrevAcc),
        Acc is PrevAcc + (log(Deg) / log(5)))),
    compile_go(semantic_path/3, Code),
    has(Code, "var semantic_pathAux = map[string]float64{}"),
    has(Code, "\"math\""),
    has(Code, "math.Log(stepCost) / math.Log(5.0)"),
    retractall(user:semantic_edge(_, _)),
    retractall(user:semantic_degree(_, _)),
    retractall(user:semantic_path(_, _, _)).

test(min_counted_recursive_lowering) :-
    assert(user:(min_edge(a, b))),
    assert(user:(min_edge(b, c))),
    assert(user:(min_reach(X, Y, H) :-
        min_edge(X, Y),
        H is 1)),
    assert(user:(min_reach(X, Z, H) :-
        min_edge(X, Y),
        min_reach(Y, Z, H1),
        H is H1 + 1)),
    assertz(user:'table'(min_reach(_, _, min))),
    compile_go(min_reach/3, Code),
    has(Code, "minimum-hop transitive closure"),
    has(Code, "container/list"),
    has(Code, "func min_reachMin"),
    has(Code, "best := map[string]int{}"),
    retractall(user:min_edge(_, _)),
    retractall(user:min_reach(_, _, _)),
    retractall(user:'table'(min_reach(_, _, min))).

test(weighted_min_recursive_accumulation_lowering) :-
    assert(user:(weighted_min_edge(a, b))),
    assert(user:(weighted_min_edge(b, c))),
    assert(user:(weighted_min_cost(a, 2))),
    assert(user:(weighted_min_cost(b, 5))),
    assert(user:(weighted_min_path(X, Y, Acc) :-
        weighted_min_edge(X, Y),
        weighted_min_cost(X, Cost),
        Acc is Cost)),
    assert(user:(weighted_min_path(X, Z, Acc) :-
        weighted_min_edge(X, Y),
        weighted_min_cost(X, Cost),
        weighted_min_path(Y, Z, PrevAcc),
        Acc is PrevAcc + Cost)),
    assertz(user:'table'(weighted_min_path(_, _, min))),
    declare_constraint(weighted_min_path/3, [positive_step(3)]),
    compile_go(weighted_min_path/3, Code),
    has(Code, "positive weighted minimum accumulation"),
    has(Code, "\"container/heap\""),
    has(Code, "func weighted_min_pathMin"),
    has(Code, "best := map[string]int{}"),
    has(Code, "Cost: stepCost"),
    has(Code, "Cost: (cost + stepCost)"),
    retractall(user:weighted_min_edge(_, _)),
    retractall(user:weighted_min_cost(_, _)),
    retractall(user:weighted_min_path(_, _, _)),
    clear_constraints(weighted_min_path/3),
    retractall(user:'table'(weighted_min_path(_, _, min))).

test(weighted_min_guarded_recursive_accumulation_lowering) :-
    assert(user:(weighted_guard_edge(a, b))),
    assert(user:(weighted_guard_edge(b, c))),
    assert(user:(weighted_guard_cost(a, 2))),
    assert(user:(weighted_guard_cost(b, 5))),
    assert(user:(weighted_guard_path(X, Y, Acc) :-
        weighted_guard_edge(X, Y),
        weighted_guard_cost(X, Cost),
        Cost > 0,
        Acc is Cost)),
    assert(user:(weighted_guard_path(X, Z, Acc) :-
        weighted_guard_edge(X, Y),
        weighted_guard_cost(X, Cost),
        Cost > 0,
        weighted_guard_path(Y, Z, PrevAcc),
        Acc is PrevAcc + Cost)),
    assertz(user:'table'(weighted_guard_path(_, _, min))),
    compile_go(weighted_guard_path/3, Code),
    has(Code, "positive weighted minimum accumulation"),
    has(Code, "func weighted_guard_pathMin"),
    has(Code, "\"container/heap\""),
    retractall(user:weighted_guard_edge(_, _)),
    retractall(user:weighted_guard_cost(_, _)),
    retractall(user:weighted_guard_path(_, _, _)),
    retractall(user:'table'(weighted_guard_path(_, _, min))).

:- end_tests(go_native_lowering).
