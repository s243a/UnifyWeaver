:- module(test_rust_native_lowering, [test_rust_native_lowering/0]).
:- use_module(library(plunit)).
:- use_module('../../src/unifyweaver/targets/rust_target').

test_rust_native_lowering :-
    run_tests([rust_native_lowering]).

:- begin_tests(rust_native_lowering).

% Helper: compile using compile_predicate_to_rust_normal to bypass semantic routing
compile_rs(Pred/Arity, Code) :-
    rust_target:compile_predicate_to_rust_normal(Pred, Arity, [], Code).

% Helper: check substring exists (deterministic)
has(Code, Substr) :-
    once(sub_string(Code, _, _, _, Substr)).

% ============================================================================
% Tier 1: Multi-clause predicates → if/else if/else chains
% ============================================================================

test(multi_clause_guard_chain) :-
    assert(user:(classify(X, small) :- X > 0, X < 10)),
    assert(user:(classify(X, large) :- X >= 10)),
    compile_rs(classify/2, Code),
    has(Code, "fn classify(arg1: i64)"),
    has(Code, "if arg1 > 0 && arg1 < 10"),
    has(Code, "\"small\""),
    has(Code, "else if arg1 >= 10"),
    has(Code, "\"large\""),
    retractall(user:classify(_, _)).

test(single_clause_guard) :-
    assert(user:(positive(X, yes) :- X > 0)),
    compile_rs(positive/2, Code),
    has(Code, "fn positive(arg1: i64)"),
    has(Code, "arg1 > 0"),
    has(Code, "\"yes\""),
    retractall(user:positive(_, _)).

test(arithmetic_output) :-
    assert(user:(double(X, R) :- R is X * 2)),
    compile_rs(double/2, Code),
    has(Code, "fn double(arg1: i64)"),
    has(Code, "(arg1 * 2)"),
    retractall(user:double(_, _)).

test(assignment_output) :-
    assert(user:(identity(X, R) :- R = X)),
    compile_rs(identity/2, Code),
    has(Code, "fn identity(arg1: i64)"),
    has(Code, "arg1"),
    retractall(user:identity(_, _)).

test(multi_fact_native) :-
    assert(user:color2(red, warm)),
    assert(user:color2(blue, cool)),
    assert(user:color2(green, cool)),
    compile_rs(color2/2, Code),
    has(Code, "fn color2(arg1: i64)"),
    has(Code, "if arg1 == \"red\""),
    has(Code, "else if arg1 == \"blue\""),
    has(Code, "else if arg1 == \"green\""),
    retractall(user:color2(_, _)).

% ============================================================================
% Tier 2: If-then-else and nested conditionals
% ============================================================================

test(if_then_else_simple) :-
    assert(user:(abs_val(X, R) :- (X >= 0 -> R = X ; R is -X))),
    compile_rs(abs_val/2, Code),
    has(Code, "fn abs_val(arg1: i64)"),
    has(Code, "if arg1 >= 0"),
    has(Code, "arg1"),
    has(Code, "(-arg1)"),
    retractall(user:abs_val(_, _)).

test(nested_if_then_else) :-
    assert(user:(range_classify(X, R) :-
        (X < 0 -> R = negative
        ; (X =:= 0 -> R = zero
        ; R = positive)))),
    compile_rs(range_classify/2, Code),
    has(Code, "fn range_classify(arg1: i64)"),
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
    compile_rs(sign/2, Code),
    has(Code, "fn sign(arg1: i64)"),
    has(Code, "arg1 > 0"),
    has(Code, "arg1 < 0"),
    retractall(user:sign(_, _)).

% ============================================================================
% Tier 1: Guard separation with arity > 2
% ============================================================================

test(guard_with_computation) :-
    assert(user:(safe_div(X, Y, R) :- Y > 0, R is X / Y)),
    compile_rs(safe_div/3, Code),
    has(Code, "fn safe_div(arg1: i64, arg2: i64)"),
    has(Code, "arg2 > 0"),
    retractall(user:safe_div(_, _, _)).

% ============================================================================
% Rust-specific syntax
% ============================================================================

test(rust_uses_panic_for_no_match) :-
    assert(user:(only_pos(X, yes) :- X > 0)),
    compile_rs(only_pos/2, Code),
    has(Code, "panic!"),
    retractall(user:only_pos(_, _)).

test(rust_uses_i64_type) :-
    assert(user:(id2(X, R) :- R = X)),
    compile_rs(id2/2, Code),
    has(Code, "i64"),
    retractall(user:id2(_, _)).

% ============================================================================
% Expanded: three-clause classify
% ============================================================================

test(three_clause_classify) :-
    assert(user:(grade(X, low) :- X < 50)),
    assert(user:(grade(X, mid) :- X >= 50, X < 80)),
    assert(user:(grade(X, high) :- X >= 80)),
    compile_rs(grade/2, Code),
    has(Code, "arg1 < 50"),
    has(Code, "arg1 >= 80"),
    has(Code, "\"low\""),
    has(Code, "\"high\""),
    retractall(user:grade(_, _)).

test(mod_guard) :-
    assert(user:(parity(X, even) :- 0 =:= X mod 2)),
    assert(user:(parity(X, odd) :- 0 =\= X mod 2)),
    compile_rs(parity/2, Code),
    has(Code, "%"),
    has(Code, "\"even\""),
    retractall(user:parity(_, _)).

test(complex_arithmetic) :-
    assert(user:(formula(X, Y) :- Y is (X * X) + (X * 2) + 1)),
    compile_rs(formula/2, Code),
    has(Code, "arg1 * arg1"),
    has(Code, "arg1 * 2"),
    retractall(user:formula(_, _)).

test(negation_output) :-
    assert(user:(negate(X, Y) :- Y is 0 - X)),
    compile_rs(negate/2, Code),
    has(Code, "0 - arg1"),
    retractall(user:negate(_, _)).

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
    compile_rs(weighted_path/3, Code),
    has(Code, "Path-aware recursive accumulation"),
    has(Code, "let mut aux: HashMap<String, i64> = HashMap::new();"),
    has(Code, "aux.insert(\"a\".to_string(), 2);"),
    has(Code, "aux.insert(\"b\".to_string(), 5);"),
    has(Code, "results.push((nb.clone(), *step_cost));"),
    has(Code, "results.push((ancestor, (acc + *step_cost)));"),
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
    compile_rs(semantic_path/3, Code),
    has(Code, "let mut aux: HashMap<String, f64> = HashMap::new();"),
    has(Code, "aux.insert(\"a\".to_string(), 2.0);"),
    has(Code, "aux.insert(\"b\".to_string(), 5.0);"),
    has(Code, "((*step_cost).ln() / (5.0).ln())"),
    has(Code, "(acc + ((*step_cost).ln() / (5.0).ln()))"),
    retractall(user:semantic_edge(_, _)),
    retractall(user:semantic_degree(_, _)),
    retractall(user:semantic_path(_, _, _)).

:- end_tests(rust_native_lowering).
