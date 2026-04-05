:- encoding(utf8).
:- use_module(library(plunit)).
:- use_module('../../src/unifyweaver/targets/wam_ilasm_target').

:- begin_tests(ilasm_wam_e2e).

% ============================================================================
% Step function structure
% ============================================================================

test(step_function_is_valid_cil) :-
    compile_step_wam_to_cil([], StepCode),
    assertion(sub_atom(StepCode, _, _, _, '.method public static bool step')),
    assertion(sub_atom(StepCode, _, _, _, '.maxstack')),
    assertion(sub_atom(StepCode, _, _, _, 'switch (')),
    assertion(sub_atom(StepCode, _, _, _, 'L_get_constant')),
    assertion(sub_atom(StepCode, _, _, _, 'L_put_constant')),
    assertion(sub_atom(StepCode, _, _, _, 'L_proceed')),
    assertion(sub_atom(StepCode, _, _, _, 'L_try_me_else')),
    assertion(sub_atom(StepCode, _, _, _, 'L_trust_me')),
    assertion(sub_atom(StepCode, _, _, _, 'L_builtin_call')).

% ============================================================================
% Helper functions all defined
% ============================================================================

test(helpers_define_all_required_methods) :-
    compile_wam_helpers_to_cil([], Code),
    assertion(sub_atom(Code, _, _, _, '.method public static bool backtrack')),
    assertion(sub_atom(Code, _, _, _, '.method public static void unwind_trail')),
    assertion(sub_atom(Code, _, _, _, '.method public static bool execute_builtin')),
    assertion(sub_atom(Code, _, _, _, '.method public static int64 eval_arith')).

% ============================================================================
% WAM predicate wrapper structure
% ============================================================================

test(predicate_wrapper_has_all_parts) :-
    WamCode = "parent/2:\n    get_constant john, A1\n    get_constant mary, A2\n    proceed\nL_p2:\n    get_constant bob, A1\n    proceed",
    compile_wam_predicate_to_cil(parent/2, WamCode, [], CILCode),
    assertion(sub_atom(CILCode, _, _, _, 'parent_code')),
    assertion(sub_atom(CILCode, _, _, _, 'parent_labels')),
    assertion(sub_atom(CILCode, _, _, _, '.method public static bool parent')),
    assertion(sub_atom(CILCode, _, _, _, 'newobj instance void WamState::.ctor')),
    assertion(sub_atom(CILCode, _, _, _, 'run_loop')).

% ============================================================================
% Argument register setup
% ============================================================================

test(arg_setup_3arity) :-
    WamCode = "add/3:\n    proceed",
    compile_wam_predicate_to_cil(add/3, WamCode, [], CILCode),
    assertion(sub_atom(CILCode, _, _, _, 'ldc.i4 0')),  % reg 0 = A1
    assertion(sub_atom(CILCode, _, _, _, 'ldc.i4 1')),  % reg 1 = A2
    assertion(sub_atom(CILCode, _, _, _, 'ldc.i4 2')),  % reg 2 = A3
    assertion(sub_atom(CILCode, _, _, _, 'SetReg')).

% ============================================================================
% Zero-arity predicate
% ============================================================================

test(zero_arity_no_arg_setup) :-
    WamCode = "main/0:\n    proceed",
    compile_wam_predicate_to_cil(main/0, WamCode, [], CILCode),
    assertion(sub_atom(CILCode, _, _, _, 'bool main(class WamState vm)')),
    assertion(\+ sub_atom(CILCode, _, _, _, 'SetReg')).

% ============================================================================
% Multi-clause label resolution
% ============================================================================

test(multi_clause_labels_resolved) :-
    WamCode = "anc/2:\n    try_me_else L_c2\n    get_constant p, A1\n    proceed\nL_c2:\n    trust_me\n    get_constant gp, A1\n    proceed",
    compile_wam_predicate_to_cil(anc/2, WamCode, [], CILCode),
    assertion(sub_atom(CILCode, _, _, _, 'new Instruction(22, 1L, 0L)')).

% ============================================================================
% Full runtime assembly
% ============================================================================

test(full_runtime_has_all_components) :-
    compile_wam_runtime_to_cil([], RuntimeCode),
    assertion(sub_atom(RuntimeCode, _, _, _, '.method public static bool step')),
    assertion(sub_atom(RuntimeCode, _, _, _, '.method public static bool backtrack')),
    assertion(sub_atom(RuntimeCode, _, _, _, '.method public static void unwind_trail')),
    assertion(sub_atom(RuntimeCode, _, _, _, '.method public static bool execute_builtin')),
    assertion(sub_atom(RuntimeCode, _, _, _, '.method public static int64 eval_arith')).

% ============================================================================
% WAM fallback disable
% ============================================================================

test(wam_fallback_disabled) :-
    wam_ilasm_target:compile_predicates_collect_cil(
        [nonexistent_pred/2],
        [wam_fallback(false)],
        NativeParts, WamParts),
    assertion(NativeParts == []),
    (   WamParts = [FailMsg|_]
    ->  assertion(sub_atom(FailMsg, _, _, _, 'compilation failed'))
    ;   true
    ).

% ============================================================================
% Choice point IR uses Array.Clone
% ============================================================================

test(try_me_else_uses_clone) :-
    compile_step_wam_to_cil([], StepCode),
    assertion(sub_atom(StepCode, _, _, _, 'Array::Clone')).

% ============================================================================
% eval_arith handles compound ops
% ============================================================================

test(eval_arith_compound_ops) :-
    compile_wam_helpers_to_cil([], Code),
    assertion(sub_atom(Code, _, _, _, 'L_ea_add')),
    assertion(sub_atom(Code, _, _, _, 'L_ea_sub')),
    assertion(sub_atom(Code, _, _, _, 'L_ea_mul')),
    assertion(sub_atom(Code, _, _, _, 'L_ea_div')).

% ============================================================================
% builtin_is calls eval_arith
% ============================================================================

test(builtin_is_uses_eval_arith) :-
    compile_wam_helpers_to_cil([], Code),
    assertion(sub_atom(Code, _, _, _, 'call int64 PrologGenerated.Program::eval_arith')).

% ============================================================================
% .tail call in runtime template
% ============================================================================

test(runtime_template_uses_tail_call) :-
    wam_ilasm_target:read_template_file(
        'templates/targets/ilasm_wam/runtime.il.mustache', Template),
    assertion(sub_atom(Template, _, _, _, '.tail')),
    assertion(sub_atom(Template, _, _, _, 'call bool')),
    assertion(sub_atom(Template, _, _, _, 'run_loop')).

:- end_tests(ilasm_wam_e2e).
