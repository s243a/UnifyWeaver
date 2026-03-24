:- module(test_wat_native_lowering, [test_wat_native_lowering/0]).
:- use_module(library(plunit)).
:- use_module('../../src/unifyweaver/targets/wat_target').

test_wat_native_lowering :-
    run_tests([wat_native_lowering]).

:- begin_tests(wat_native_lowering).

% Helper: compile using the public API
compile_wat(Pred/Arity, Code) :-
    wat_target:compile_predicate_to_wat(Pred/Arity, [], Code).

% Helper: check substring exists (deterministic)
has(Code, Substr) :-
    once(sub_string(Code, _, _, _, Substr)).

% ============================================================================
% Tier 1: Multi-clause predicates → nested if/else chains
% ============================================================================

test(multi_clause_guard_chain) :-
    assert(user:(classify(X, small) :- X > 0, X < 10)),
    assert(user:(classify(X, large) :- X >= 10)),
    compile_wat(classify/2, Code),
    has(Code, "(func $classify"),
    has(Code, "i64.gt_s"),
    has(Code, "i64.lt_s"),
    has(Code, "i32.and"),
    has(Code, "i64.ge_s"),
    has(Code, "(if (result i64)"),
    retractall(user:classify(_, _)).

test(single_clause_guard) :-
    assert(user:(positive(X, yes) :- X > 0)),
    compile_wat(positive/2, Code),
    has(Code, "(func $positive"),
    has(Code, "i64.gt_s"),
    retractall(user:positive(_, _)).

test(arithmetic_output) :-
    assert(user:(double(X, R) :- R is X * 2)),
    compile_wat(double/2, Code),
    has(Code, "(func $double"),
    has(Code, "i64.mul"),
    retractall(user:double(_, _)).

test(assignment_output) :-
    assert(user:(identity(X, R) :- R = X)),
    compile_wat(identity/2, Code),
    has(Code, "(func $identity"),
    has(Code, "local.get $arg1"),
    retractall(user:identity(_, _)).

test(multi_clause_rules) :-
    assert(user:(color2(X, warm) :- X == red)),
    assert(user:(color2(X, cool) :- X == blue)),
    assert(user:(color2(X, cool) :- X == green)),
    compile_wat(color2/2, Code),
    has(Code, "(func $color2"),
    has(Code, "i64.eq"),
    retractall(user:color2(_, _)).

% ============================================================================
% Tier 2: If-then-else and nested conditionals
% ============================================================================

test(if_then_else_simple) :-
    assert(user:(abs_val(X, R) :- (X >= 0 -> R = X ; R is -X))),
    compile_wat(abs_val/2, Code),
    has(Code, "(func $abs_val"),
    has(Code, "i64.ge_s"),
    has(Code, "i64.sub"),
    retractall(user:abs_val(_, _)).

test(nested_if_then_else) :-
    assert(user:(range_classify(X, R) :-
        (X < 0 -> R = negative
        ; (X =:= 0 -> R = zero
        ; R = positive)))),
    compile_wat(range_classify/2, Code),
    has(Code, "(func $range_classify"),
    has(Code, "i64.lt_s"),
    has(Code, "i64.eq"),
    retractall(user:range_classify(_, _)).

test(three_way_nested) :-
    assert(user:(sign(X, R) :-
        (X > 0 -> R = positive
        ; (X < 0 -> R = negative
        ; R = zero)))),
    compile_wat(sign/2, Code),
    has(Code, "(func $sign"),
    has(Code, "i64.gt_s"),
    has(Code, "i64.lt_s"),
    retractall(user:sign(_, _)).

% ============================================================================
% Tier 1: Guard separation with arity > 2
% ============================================================================

test(guard_with_computation) :-
    assert(user:(safe_div(X, Y, R) :- Y > 0, R is X / Y)),
    compile_wat(safe_div/3, Code),
    has(Code, "(func $safe_div"),
    has(Code, "(param $arg1 i64)"),
    has(Code, "(param $arg2 i64)"),
    has(Code, "i64.gt_s"),
    has(Code, "i64.div_s"),
    retractall(user:safe_div(_, _, _)).

% ============================================================================
% WAT-specific syntax
% ============================================================================

test(wat_uses_structured_if) :-
    assert(user:(grade(X, pass) :- X >= 50)),
    assert(user:(grade(X, fail) :- X < 50)),
    compile_wat(grade/2, Code),
    has(Code, "(if (result i64)"),
    has(Code, "(then"),
    has(Code, "(else"),
    retractall(user:grade(_, _)).

test(wat_uses_module_wrapper) :-
    assert(user:(only_pos(X, yes) :- X > 0)),
    compile_wat(only_pos/2, Code),
    has(Code, "(module"),
    has(Code, "(export"),
    retractall(user:only_pos(_, _)).

% ============================================================================
% Recursion patterns
% ============================================================================

test(tail_recursion_loop_br) :-
    wat_target:compile_tail_recursion_wat(sum/2, [], Code),
    has(Code, "(module"),
    has(Code, "(func $sum"),
    has(Code, "(loop $continue"),
    has(Code, "(br $continue)"),
    has(Code, "i64.add"),
    has(Code, "$sum_entry").

test(linear_recursion_memo) :-
    wat_target:compile_linear_recursion_wat(fib/2, [], Code),
    has(Code, "(module"),
    has(Code, "(func $fib"),
    has(Code, "(memory"),
    has(Code, "i64.store"),
    has(Code, "i64.load"),
    has(Code, "i32.store").

test(tree_recursion_two_calls) :-
    wat_target:compile_tree_recursion_wat(fib/2, [], Code),
    has(Code, "(module"),
    has(Code, "(func $fib"),
    has(Code, "(memory"),
    has(Code, "(call $fib (i64.sub (local.get $n) (i64.const 1)))"),
    has(Code, "(call $fib (i64.sub (local.get $n) (i64.const 2)))").

test(mutual_recursion_cross_call) :-
    wat_target:compile_mutual_recursion_wat([is_even/1, is_odd/1], [], Code),
    has(Code, "(module"),
    has(Code, "(func $is_even"),
    has(Code, "(func $is_odd"),
    has(Code, "(call $is_odd"),
    has(Code, "(call $is_even").

test(transitive_closure_bfs) :-
    wat_target:compile_transitive_closure_wat(reachable/2, [], Code),
    has(Code, "(module"),
    has(Code, "(func $reachable"),
    has(Code, "$add_edge"),
    has(Code, "(loop $bfs"),
    has(Code, "$edge_scan"),
    has(Code, "$mark_visited").

test(multicall_recursion_memo) :-
    wat_target:compile_multicall_recursion_wat(fib/2, [], Code),
    has(Code, "(module"),
    has(Code, "(func $fib"),
    has(Code, "Multicall"),
    has(Code, "(call $fib (i64.sub (local.get $n) (i64.const 1)))"),
    has(Code, "(call $fib (i64.sub (local.get $n) (i64.const 2)))"),
    has(Code, "i64.store"),
    has(Code, "i32.store").

test(direct_multicall_memo) :-
    wat_target:compile_direct_multicall_wat(fib/2, [], Code),
    has(Code, "(module"),
    has(Code, "(func $fib"),
    has(Code, "Direct Multi-Call"),
    has(Code, "(call $fib"),
    has(Code, "i64.store").

% ============================================================================
% Template integration
% ============================================================================

test(template_transitive_closure) :-
    wat_target:compile_transitive_closure_wat_from_template(ancestor, parent, Code),
    has(Code, "(module"),
    has(Code, "ancestor"),
    has(Code, "(loop $bfs").

% ============================================================================
% Multifile dispatch hooks
% ============================================================================

test(multifile_tail_hook) :-
    tail_recursion:compile_tail_pattern(wat, "sum", 2, [], [], 2, add, false, Code),
    has(Code, "(module"),
    has(Code, "(func $sum"),
    has(Code, "(loop $continue").

test(multifile_linear_hook) :-
    linear_recursion:compile_linear_pattern(wat, "fib", 2, [], [], true, table, Code),
    has(Code, "(module"),
    has(Code, "(func $fib"),
    has(Code, "i64.load").

test(multifile_tree_hook) :-
    tree_recursion:compile_tree_pattern(wat, fibonacci, fib, 2, true, Code),
    has(Code, "(module"),
    has(Code, "(func $fib"),
    has(Code, "(call $fib (i64.sub (local.get $n) (i64.const 1)))").

test(multifile_multicall_hook) :-
    multicall_linear_recursion:compile_multicall_pattern(wat, "fib", [base1], [], true, Code),
    has(Code, "(module"),
    has(Code, "(func $fib").

test(multifile_direct_multicall_hook) :-
    direct_multi_call_recursion:compile_direct_multicall_pattern(wat, "fib", [], clause(fib(n,f), true), Code),
    has(Code, "(module"),
    has(Code, "(func $fib").

test(multifile_mutual_hook) :-
    mutual_recursion:compile_mutual_pattern(wat, [is_even/1, is_odd/1], true, table, Code),
    has(Code, "(module"),
    has(Code, "(func $is_even"),
    has(Code, "(func $is_odd").

% ============================================================================
% Component system
% ============================================================================

test(component_compile) :-
    wat_target:wat_compile_component(test_comp,
        [code("    (local.get $input)")],
        [],
        Code),
    has(Code, "comp_test_comp"),
    has(Code, "(export"),
    has(Code, "local.get $input").

% ============================================================================
% Target registry
% ============================================================================

test(target_registered) :-
    use_module('src/unifyweaver/core/target_registry'),
    target_registry:registered_target(wat, lowlevel, Caps),
    once(member(structured_control_flow, Caps)).

% ============================================================================
% LLVM WASM fallback
% ============================================================================

test(llvm_fallback_on_unsupported) :-
    % The LLVM fallback should activate when native lowering fails
    % and LLVM is available
    wat_target:compile_wasm_module(
        [func(test_func, 2, linear_recursion)],
        [module_name(test_func)],
        LLVMCode
    ),
    has(LLVMCode, "wasm32").

% ============================================================================
% Verify shared module is loaded
% ============================================================================

test(uses_shared_analysis_module) :-
    current_predicate(clause_body_analysis:normalize_goals/2),
    current_predicate(clause_body_analysis:if_then_else_goal/4),
    current_predicate(clause_body_analysis:build_head_varmap/3).

:- end_tests(wat_native_lowering).
