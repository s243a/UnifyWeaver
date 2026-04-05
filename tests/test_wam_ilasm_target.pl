:- encoding(utf8).
:- use_module(library(plunit)).
:- use_module('../src/unifyweaver/targets/wam_ilasm_target').
:- use_module('../src/unifyweaver/bindings/cil_wam_bindings').

:- begin_tests(wam_ilasm_target).

% ============================================================================
% Phase 0: Binding registry tests
% ============================================================================

test(cil_reg_name_to_index_a1) :-
    cil_reg_name_to_index('A1', Idx),
    assertion(Idx == 0).

test(cil_reg_name_to_index_x1) :-
    cil_reg_name_to_index('X1', Idx),
    assertion(Idx == 16).

test(cil_type_map_value) :-
    cil_wam_type_map(value, T),
    assertion(T == 'class Value').

test(cil_type_map_integer) :-
    cil_wam_type_map(integer, T),
    assertion(T == 'int64').

test(cil_binding_exists_atom_check) :-
    cil_wam_binding(atom/1, Expr, _, _, _),
    assertion(Expr == 'isinst AtomValue').

test(cil_binding_exists_add) :-
    cil_wam_binding('+'/3, Expr, _, _, _),
    assertion(Expr == 'add').

test(cil_binding_get_assoc) :-
    cil_wam_binding(get_assoc/3, Expr, _, _, _),
    assertion(Expr == 'ldelem.ref').

test(cil_binding_var_check) :-
    cil_wam_binding(var/1, Expr, _, _, _),
    assertion(sub_atom(Expr, _, _, _, 'IsUnbound')).

% ============================================================================
% Phase 2: Step function generation
% ============================================================================

test(step_cil_generation) :-
    compile_step_wam_to_cil([], StepCode),
    assertion(sub_atom(StepCode, _, _, _, '.method public static bool step')),
    assertion(sub_atom(StepCode, _, _, _, 'switch (')),
    assertion(sub_atom(StepCode, _, _, _, 'L_get_constant')),
    assertion(sub_atom(StepCode, _, _, _, 'L_put_constant')),
    assertion(sub_atom(StepCode, _, _, _, 'L_proceed')),
    assertion(sub_atom(StepCode, _, _, _, 'L_try_me_else')),
    assertion(sub_atom(StepCode, _, _, _, 'L_trust_me')).

test(step_has_isinst) :-
    compile_step_wam_to_cil([], StepCode),
    assertion(sub_atom(StepCode, _, _, _, 'isinst IntegerValue')).

test(step_has_virtual_calls) :-
    compile_step_wam_to_cil([], StepCode),
    assertion(sub_atom(StepCode, _, _, _, 'callvirt instance bool Value::IsUnbound()')).

% ============================================================================
% Phase 3: Helper function generation
% ============================================================================

test(helpers_generation) :-
    compile_wam_helpers_to_cil([], Code),
    assertion(sub_atom(Code, _, _, _, '.method public static bool backtrack')),
    assertion(sub_atom(Code, _, _, _, '.method public static void unwind_trail')),
    assertion(sub_atom(Code, _, _, _, '.method public static bool execute_builtin')),
    assertion(sub_atom(Code, _, _, _, '.method public static int64 eval_arith')).

test(backtrack_uses_array_clone) :-
    compile_wam_helpers_to_cil([], Code),
    assertion(sub_atom(Code, _, _, _, 'Array::Clone')).

test(backtrack_uses_list_api) :-
    compile_wam_helpers_to_cil([], Code),
    assertion(sub_atom(Code, _, _, _, 'get_Count')).

test(execute_builtin_has_switch) :-
    compile_wam_helpers_to_cil([], Code),
    assertion(sub_atom(Code, _, _, _, 'switch (')),
    assertion(sub_atom(Code, _, _, _, 'L_bi_is')),
    assertion(sub_atom(Code, _, _, _, 'L_bi_gt')),
    assertion(sub_atom(Code, _, _, _, 'L_bi_cut')).

test(eval_arith_has_isinst_dispatch) :-
    compile_wam_helpers_to_cil([], Code),
    assertion(sub_atom(Code, _, _, _, 'isinst IntegerValue')),
    assertion(sub_atom(Code, _, _, _, 'isinst FloatValue')),
    assertion(sub_atom(Code, _, _, _, 'isinst CompoundValue')).

test(eval_arith_has_compound_ops) :-
    compile_wam_helpers_to_cil([], Code),
    assertion(sub_atom(Code, _, _, _, 'L_ea_add')),
    assertion(sub_atom(Code, _, _, _, 'L_ea_sub')),
    assertion(sub_atom(Code, _, _, _, 'L_ea_mul')),
    assertion(sub_atom(Code, _, _, _, 'L_ea_div')).

test(builtin_is_calls_eval_arith) :-
    compile_wam_helpers_to_cil([], Code),
    assertion(sub_atom(Code, _, _, _, 'call int64 PrologGenerated.Program::eval_arith')).

% ============================================================================
% Phase 4: Instruction lowering
% ============================================================================

test(cil_get_constant_literal) :-
    wam_instruction_to_cil_literal(get_constant(integer(42), 'A1'), Lit),
    assertion(sub_atom(Lit, _, _, _, 'new Instruction(0, 42L, 0L)')).

test(cil_get_variable_literal) :-
    wam_instruction_to_cil_literal(get_variable('X1', 'A1'), Lit),
    assertion(sub_atom(Lit, _, _, _, 'new Instruction(1, 16L, 0L)')).

test(cil_allocate_literal) :-
    wam_instruction_to_cil_literal(allocate, Lit),
    assertion(Lit == 'new Instruction(16, 0L, 0L)').

test(cil_proceed_literal) :-
    wam_instruction_to_cil_literal(proceed, Lit),
    assertion(Lit == 'new Instruction(20, 0L, 0L)').

test(cil_call_errors_without_labelmap, [throws(error(label_resolution_required(call, _), _))]) :-
    wam_instruction_to_cil_literal(call('parent/2', 2), _).

% ============================================================================
% Phase 4: WAM text line parsing
% ============================================================================

test(parse_cil_line_get_constant) :-
    wam_line_to_cil_literal(["get_constant", "john,", "A1"], Lit),
    assertion(sub_atom(Lit, _, _, _, 'new Instruction(0,')).

test(parse_cil_line_allocate) :-
    wam_line_to_cil_literal(["allocate"], Lit),
    assertion(Lit == 'new Instruction(16, 0L, 0L)').

test(parse_cil_label_resolution) :-
    WamCode = "parent/2:\n    try_me_else L_alt\nL_alt:\n    proceed",
    compile_wam_predicate_to_cil(parent/2, WamCode, [], CILCode),
    % try_me_else should resolve L_alt (label index 1)
    assertion(sub_atom(CILCode, _, _, _, 'new Instruction(22, 1L, 0L)')).

% ============================================================================
% Phase 5: Full runtime + predicate wrapper
% ============================================================================

test(full_runtime_generation) :-
    compile_wam_runtime_to_cil([], RuntimeCode),
    assertion(sub_atom(RuntimeCode, _, _, _, '.method public static bool step')),
    assertion(sub_atom(RuntimeCode, _, _, _, '.method public static bool backtrack')).

test(predicate_wrapper_structure) :-
    WamCode = "test/1:\n    get_constant a, A1\n    proceed",
    compile_wam_predicate_to_cil(test/1, WamCode, [], CILCode),
    assertion(sub_atom(CILCode, _, _, _, 'test_code')),
    assertion(sub_atom(CILCode, _, _, _, 'test_labels')),
    assertion(sub_atom(CILCode, _, _, _, '.method public static bool test')),
    assertion(sub_atom(CILCode, _, _, _, 'newobj instance void WamState::.ctor')),
    assertion(sub_atom(CILCode, _, _, _, 'run_loop')).

test(zero_arity_no_arg_setup) :-
    WamCode = "main/0:\n    proceed",
    compile_wam_predicate_to_cil(main/0, WamCode, [], CILCode),
    assertion(sub_atom(CILCode, _, _, _, 'bool main(class WamState vm)')),
    assertion(\+ sub_atom(CILCode, _, _, _, 'SetReg')).

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
% Builtin op ID mapping
% ============================================================================

test(builtin_cil_op_is) :-
    builtin_op_to_cil_id('is/2', Id),
    assertion(Id == 0).

test(builtin_cil_op_gt) :-
    builtin_op_to_cil_id('>/2', Id),
    assertion(Id == 1).

test(builtin_cil_op_cut) :-
    builtin_op_to_cil_id('!/0', Id),
    assertion(Id == 10).

% ============================================================================
% Templates
% ============================================================================

test(types_template_has_value_hierarchy) :-
    wam_ilasm_target:read_template_file(
        'templates/targets/ilasm_wam/types.il.mustache', Template),
    assertion(sub_atom(Template, _, _, _, 'class Value')),
    assertion(sub_atom(Template, _, _, _, 'AtomValue')),
    assertion(sub_atom(Template, _, _, _, 'IntegerValue')),
    assertion(sub_atom(Template, _, _, _, 'CompoundValue')),
    assertion(sub_atom(Template, _, _, _, 'UnboundValue')),
    assertion(sub_atom(Template, _, _, _, 'WamState')),
    assertion(sub_atom(Template, _, _, _, 'ChoicePoint')).

test(runtime_template_has_tail_call) :-
    wam_ilasm_target:read_template_file(
        'templates/targets/ilasm_wam/runtime.il.mustache', Template),
    assertion(sub_atom(Template, _, _, _, '.tail')),
    assertion(sub_atom(Template, _, _, _, 'run_loop')).

test(state_template_has_array_clone) :-
    wam_ilasm_target:read_template_file(
        'templates/targets/ilasm_wam/state.il.mustache', Template),
    assertion(sub_atom(Template, _, _, _, 'GetReg')),
    assertion(sub_atom(Template, _, _, _, 'SetReg')),
    assertion(sub_atom(Template, _, _, _, 'TrailBinding')).

% ============================================================================
% Atom table reset
% ============================================================================

% ============================================================================
% Compound term instruction cases in step dispatch
% ============================================================================

test(step_has_compound_instructions) :-
    compile_step_wam_to_cil([], StepCode),
    assertion(sub_atom(StepCode, _, _, _, 'L_get_structure:')),
    assertion(sub_atom(StepCode, _, _, _, 'L_get_list:')),
    assertion(sub_atom(StepCode, _, _, _, 'L_unify_variable:')),
    assertion(sub_atom(StepCode, _, _, _, 'L_unify_value:')),
    assertion(sub_atom(StepCode, _, _, _, 'L_unify_constant:')),
    assertion(sub_atom(StepCode, _, _, _, 'L_put_structure:')),
    assertion(sub_atom(StepCode, _, _, _, 'L_put_list:')),
    assertion(sub_atom(StepCode, _, _, _, 'L_set_variable:')),
    assertion(sub_atom(StepCode, _, _, _, 'L_set_value:')),
    assertion(sub_atom(StepCode, _, _, _, 'L_set_constant:')).

test(compound_instrs_use_heap_push) :-
    compile_step_wam_to_cil([], StepCode),
    assertion(sub_atom(StepCode, _, _, _, 'HeapPush')).

test(get_structure_has_write_and_read_mode) :-
    compile_step_wam_to_cil([], StepCode),
    assertion(sub_atom(StepCode, _, _, _, 'L_gs_read:')).

test(unify_variable_creates_unbound) :-
    compile_step_wam_to_cil([], StepCode),
    assertion(sub_atom(StepCode, _, _, _, 'UnboundValue')).

test(atom_table_reset) :-
    cil_atom_table_reset,
    wam_ilasm_target:cil_intern_atom(reset_test_a, IdA),
    wam_ilasm_target:cil_intern_atom(reset_test_b, IdB),
    assertion(IdA == 1),
    assertion(IdB == 2),
    % Reset and verify IDs restart
    cil_atom_table_reset,
    wam_ilasm_target:cil_intern_atom(reset_test_c, IdC),
    assertion(IdC == 1).

test(state_template_documents_heap_push) :-
    wam_ilasm_target:read_template_file(
        'templates/targets/ilasm_wam/state.il.mustache', Template),
    assertion(sub_atom(Template, _, _, _, 'HeapPush')),
    assertion(sub_atom(Template, _, _, _, 'compound term construction')).

:- end_tests(wam_ilasm_target).
