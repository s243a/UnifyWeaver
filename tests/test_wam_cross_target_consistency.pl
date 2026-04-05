:- encoding(utf8).
% test_wam_cross_target_consistency.pl
%
% Validates that all hybrid WAM targets share consistent instruction sets,
% register ABIs, and builtin op ID mappings. Catches drift between targets.

:- use_module(library(plunit)).
:- use_module('../src/unifyweaver/targets/wam_llvm_target').
:- use_module('../src/unifyweaver/targets/wam_ilasm_target').
:- use_module('../src/unifyweaver/bindings/llvm_wam_bindings').
:- use_module('../src/unifyweaver/bindings/cil_wam_bindings').

:- begin_tests(wam_cross_target_consistency).

% ============================================================================
% Register ABI: A1→0, X1→16 must match across LLVM and ILAsm
% ============================================================================

test(register_abi_a1_consistent) :-
    reg_name_to_index('A1', LLVMIdx),
    cil_reg_name_to_index('A1', CILIdx),
    assertion(LLVMIdx == CILIdx),
    assertion(LLVMIdx == 0).

test(register_abi_a16_consistent) :-
    reg_name_to_index('A16', LLVMIdx),
    cil_reg_name_to_index('A16', CILIdx),
    assertion(LLVMIdx == CILIdx),
    assertion(LLVMIdx == 15).

test(register_abi_x1_consistent) :-
    reg_name_to_index('X1', LLVMIdx),
    cil_reg_name_to_index('X1', CILIdx),
    assertion(LLVMIdx == CILIdx),
    assertion(LLVMIdx == 16).

test(register_abi_x16_consistent) :-
    reg_name_to_index('X16', LLVMIdx),
    cil_reg_name_to_index('X16', CILIdx),
    assertion(LLVMIdx == CILIdx),
    assertion(LLVMIdx == 31).

test(register_abi_full_range) :-
    % Verify all 32 register slots are consistent
    numlist(1, 16, AIndices),
    forall(member(I, AIndices), (
        atom_concat('A', I, RegName),
        reg_name_to_index(RegName, LIdx),
        cil_reg_name_to_index(RegName, CIdx),
        assertion(LIdx == CIdx)
    )),
    numlist(1, 16, XIndices),
    forall(member(I, XIndices), (
        atom_concat('X', I, RegName),
        reg_name_to_index(RegName, LIdx),
        cil_reg_name_to_index(RegName, CIdx),
        assertion(LIdx == CIdx)
    )).

% ============================================================================
% Builtin op IDs: must match between LLVM and ILAsm
% ============================================================================

test(builtin_op_ids_consistent) :-
    % All builtin ops that have IDs in both targets
    Builtins = [
        'is/2', '>/2', '</2', '>=/2', '=</2', '=:=/2', '=\\=/2',
        '==/2', 'true/0', 'fail/0', '!/0'
    ],
    forall(member(Op, Builtins), (
        builtin_op_to_id(Op, LLVMId),
        builtin_op_to_cil_id(Op, CILId),
        assertion(LLVMId == CILId)
    )).

test(builtin_op_is_id_zero) :-
    builtin_op_to_id('is/2', LLVMId),
    builtin_op_to_cil_id('is/2', CILId),
    assertion(LLVMId == 0),
    assertion(CILId == 0).

test(builtin_op_cut_id_ten) :-
    builtin_op_to_id('!/0', LLVMId),
    builtin_op_to_cil_id('!/0', CILId),
    assertion(LLVMId == 10),
    assertion(CILId == 10).

test(builtin_op_unknown_consistent) :-
    builtin_op_to_id('nonexistent/7', LLVMId),
    builtin_op_to_cil_id('nonexistent/7', CILId),
    assertion(LLVMId == 99),
    assertion(CILId == 99).

% ============================================================================
% Instruction set: all targets must have the same 25 instruction cases
% ============================================================================

% The canonical instruction set (by name, alphabetical order)
canonical_instruction_set(InstrSet) :-
    InstrSet = [
        allocate, builtin_call, call, deallocate, execute,
        get_constant, get_list, get_structure, get_value, get_variable,
        proceed, put_constant, put_list, put_structure, put_value, put_variable,
        retry_me_else, set_constant, set_value, set_variable,
        trust_me, try_me_else,
        unify_constant, unify_value, unify_variable
    ].

test(llvm_step_has_all_instructions) :-
    compile_step_wam_to_llvm([], StepCode),
    canonical_instruction_set(Instrs),
    forall(member(Instr, Instrs), (
        atom_string(Instr, InstrStr),
        (   sub_atom(StepCode, _, _, _, InstrStr)
        ->  true
        ;   format(user_error, 'LLVM missing instruction: ~w~n', [Instr]),
            fail
        )
    )).

test(ilasm_step_has_all_instructions) :-
    compile_step_wam_to_cil([], StepCode),
    canonical_instruction_set(Instrs),
    forall(member(Instr, Instrs), (
        atom_string(Instr, InstrStr),
        % ILAsm uses L_ prefix for labels
        format(atom(Label), 'L_~w', [InstrStr]),
        (   sub_atom(StepCode, _, _, _, Label)
        ->  true
        ;   format(user_error, 'ILAsm missing instruction: ~w~n', [Instr]),
            fail
        )
    )).

% ============================================================================
% Instruction tag numbering: LLVM and ILAsm must use same tags
% ============================================================================

% Canonical tag mapping
canonical_instruction_tag(get_constant, 0).
canonical_instruction_tag(get_variable, 1).
canonical_instruction_tag(get_value, 2).
canonical_instruction_tag(get_structure, 3).
canonical_instruction_tag(get_list, 4).
canonical_instruction_tag(unify_variable, 5).
canonical_instruction_tag(unify_value, 6).
canonical_instruction_tag(unify_constant, 7).
canonical_instruction_tag(put_constant, 8).
canonical_instruction_tag(put_variable, 9).
canonical_instruction_tag(put_value, 10).
canonical_instruction_tag(put_structure, 11).
canonical_instruction_tag(put_list, 12).
canonical_instruction_tag(set_variable, 13).
canonical_instruction_tag(set_value, 14).
canonical_instruction_tag(set_constant, 15).
canonical_instruction_tag(allocate, 16).
canonical_instruction_tag(deallocate, 17).
canonical_instruction_tag(call, 18).
canonical_instruction_tag(execute, 19).
canonical_instruction_tag(proceed, 20).
canonical_instruction_tag(builtin_call, 21).
canonical_instruction_tag(try_me_else, 22).
canonical_instruction_tag(retry_me_else, 23).
canonical_instruction_tag(trust_me, 24).

test(llvm_instruction_tags_in_switch) :-
    compile_step_wam_to_llvm([], StepCode),
    % Verify key tags appear in the switch statement
    forall(canonical_instruction_tag(_, Tag), (
        format(atom(TagStr), 'i32 ~w, label', [Tag]),
        (   sub_atom(StepCode, _, _, _, TagStr)
        ->  true
        ;   format(user_error, 'LLVM missing tag: ~w~n', [Tag]),
            fail
        )
    )).

test(ilasm_instruction_tags_in_switch) :-
    compile_step_wam_to_cil([], StepCode),
    % ILAsm switch lists labels in order — the switch itself covers tags 0-24
    assertion(sub_atom(StepCode, _, _, _, 'switch (')),
    assertion(sub_atom(StepCode, _, _, _, 'L_get_constant')),
    assertion(sub_atom(StepCode, _, _, _, 'L_trust_me')).

% ============================================================================
% Instruction lowering: LLVM and ILAsm produce consistent literals
% ============================================================================

test(allocate_literal_consistent) :-
    wam_instruction_to_llvm_literal(allocate, LLVMLit),
    wam_instruction_to_cil_literal(allocate, CILLit),
    % Both should encode tag 16
    assertion(sub_atom(LLVMLit, _, _, _, '16')),
    assertion(sub_atom(CILLit, _, _, _, '16')).

test(proceed_literal_consistent) :-
    wam_instruction_to_llvm_literal(proceed, LLVMLit),
    wam_instruction_to_cil_literal(proceed, CILLit),
    assertion(sub_atom(LLVMLit, _, _, _, '20')),
    assertion(sub_atom(CILLit, _, _, _, '20')).

test(get_variable_literal_tag_consistent) :-
    wam_instruction_to_llvm_literal(get_variable('X1', 'A1'), LLVMLit),
    wam_instruction_to_cil_literal(get_variable('X1', 'A1'), CILLit),
    % Both should have tag 1, X1=16, A1=0
    assertion(sub_atom(LLVMLit, _, _, _, '1')),
    assertion(sub_atom(LLVMLit, _, _, _, '16')),
    assertion(sub_atom(CILLit, _, _, _, '1')),
    assertion(sub_atom(CILLit, _, _, _, '16')).

% ============================================================================
% Type mappings: both targets map the same Prolog types
% ============================================================================

test(type_map_keys_consistent) :-
    CoreTypes = [value, integer, float, bool, atom, string, list, assoc],
    forall(member(T, CoreTypes), (
        (   llvm_wam_type_map(T, _)
        ->  true
        ;   format(user_error, 'LLVM missing type map for: ~w~n', [T]), fail
        ),
        (   cil_wam_type_map(T, _)
        ->  true
        ;   format(user_error, 'CIL missing type map for: ~w~n', [T]), fail
        )
    )).

% ============================================================================
% Binding coverage: both targets have bindings for core operations
% ============================================================================

test(core_bindings_exist_in_both) :-
    CoreOps = [
        get_assoc/3, put_assoc/4, atom/1, integer/1, var/1,
        '+'/3, '-'/3, '*'/3, '=='/2
    ],
    forall(member(Op, CoreOps), (
        (   llvm_wam_binding(Op, _, _, _, _)
        ->  true
        ;   format(user_error, 'LLVM missing binding: ~w~n', [Op]), fail
        ),
        (   cil_wam_binding(Op, _, _, _, _)
        ->  true
        ;   format(user_error, 'CIL missing binding: ~w~n', [Op]), fail
        )
    )).

% ============================================================================
% Step function structure: both targets have read/write mode dispatch
% ============================================================================

test(llvm_has_read_write_mode) :-
    compile_step_wam_to_llvm([], Code),
    assertion(sub_atom(Code, _, _, _, 'wam_peek_stack_type')),
    assertion(sub_atom(Code, _, _, _, 'wam_unify_ctx_next')),
    assertion(sub_atom(Code, _, _, _, 'wam_write_ctx_dec')).

test(ilasm_has_read_write_mode) :-
    compile_step_wam_to_cil([], Code),
    assertion(sub_atom(Code, _, _, _, 'PeekStackType')),
    assertion(sub_atom(Code, _, _, _, 'UnifyCtxNext')),
    assertion(sub_atom(Code, _, _, _, 'WriteCtxDec')).

% ============================================================================
% Helper functions: both targets have the same set
% ============================================================================

test(llvm_has_all_helpers) :-
    compile_wam_helpers_to_llvm([], Code),
    assertion(sub_atom(Code, _, _, _, 'backtrack')),
    assertion(sub_atom(Code, _, _, _, 'unwind_trail')),
    assertion(sub_atom(Code, _, _, _, 'execute_builtin')),
    assertion(sub_atom(Code, _, _, _, 'eval_arith')).

test(ilasm_has_all_helpers) :-
    compile_wam_helpers_to_cil([], Code),
    assertion(sub_atom(Code, _, _, _, 'backtrack')),
    assertion(sub_atom(Code, _, _, _, 'unwind_trail')),
    assertion(sub_atom(Code, _, _, _, 'execute_builtin')),
    assertion(sub_atom(Code, _, _, _, 'eval_arith')).

:- end_tests(wam_cross_target_consistency).
