:- encoding(utf8).
% test_llvm_wam_e2e.pl - End-to-end tests for WAM-to-LLVM hybrid target
%
% Tests that the complete pipeline (Prolog → WAM → LLVM IR) produces
% structurally valid LLVM IR modules. Verifies:
%   - Type definitions present
%   - Value functions present
%   - State management present
%   - Step dispatch generated
%   - Helper functions generated
%   - WAM-compiled predicate wrappers generated
%   - Module structure is self-consistent
%
% For external validation (opt -verify, lli), see test_llvm_wam_pipeline.sh

:- use_module(library(plunit)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../../src/unifyweaver/targets/wam_llvm_target').
:- use_module('../../src/unifyweaver/targets/wam_target', [compile_predicate_to_wam/3]).

:- begin_tests(llvm_wam_e2e).

% ============================================================================
% Test 1: Step function generates valid LLVM switch
% ============================================================================

test(step_function_is_valid_llvm) :-
    compile_step_wam_to_llvm([], StepCode),
    % Must be a valid function definition
    assertion(sub_atom(StepCode, _, _, _, 'define i1 @step(%WamState* %vm, %Instruction* %instr)')),
    % Must have entry block with switch
    assertion(sub_atom(StepCode, _, _, _, 'entry:')),
    assertion(sub_atom(StepCode, _, _, _, 'switch i32 %tag, label %default')),
    % Must have default block
    assertion(sub_atom(StepCode, _, _, _, 'default:')),
    assertion(sub_atom(StepCode, _, _, _, 'ret i1 false')),
    % Must have key instruction blocks
    assertion(sub_atom(StepCode, _, _, _, 'get_constant:')),
    assertion(sub_atom(StepCode, _, _, _, 'put_constant:')),
    assertion(sub_atom(StepCode, _, _, _, 'proceed:')),
    assertion(sub_atom(StepCode, _, _, _, 'try_me_else:')),
    assertion(sub_atom(StepCode, _, _, _, 'trust_me:')).

% ============================================================================
% Test 2: Helper functions all defined
% ============================================================================

test(helpers_define_all_required_functions) :-
    compile_wam_helpers_to_llvm([], HelpersCode),
    assertion(sub_atom(HelpersCode, _, _, _, 'define i1 @backtrack(%WamState* %vm)')),
    assertion(sub_atom(HelpersCode, _, _, _, 'define void @unwind_trail(%WamState* %vm, i32 %saved_mark)')),
    assertion(sub_atom(HelpersCode, _, _, _, 'define i1 @execute_builtin(%WamState* %vm, i32 %op, i32 %arity)')),
    assertion(sub_atom(HelpersCode, _, _, _, 'define i64 @eval_arith(%WamState* %vm, %Value %expr)')).

% ============================================================================
% Test 3: WAM predicate compilation produces valid wrapper
% ============================================================================

test(wam_predicate_wrapper_structure) :-
    % Simulate WAM code for a simple two-clause predicate
    WamCode = "parent/2:\n    get_constant john, A1\n    get_constant mary, A2\n    proceed\nL_parent_2_2:\n    get_constant bob, A1\n    get_constant alice, A2\n    proceed",
    compile_wam_predicate_to_llvm(parent/2, WamCode, [], LLVMCode),
    % Must have code array
    assertion(sub_atom(LLVMCode, _, _, _, '@parent_code = private constant')),
    assertion(sub_atom(LLVMCode, _, _, _, '%Instruction')),
    % Must have label array
    assertion(sub_atom(LLVMCode, _, _, _, '@parent_labels = private constant')),
    % Must have wrapper function
    assertion(sub_atom(LLVMCode, _, _, _, 'define i1 @parent')),
    % Must call wam_state_new and run_loop
    assertion(sub_atom(LLVMCode, _, _, _, 'call %WamState* @wam_state_new')),
    assertion(sub_atom(LLVMCode, _, _, _, 'call i1 @run_loop')).

% ============================================================================
% Test 4: WAM predicate with arguments sets registers
% ============================================================================

test(wam_predicate_with_args_sets_regs) :-
    WamCode = "add/3:\n    proceed",
    compile_wam_predicate_to_llvm(add/3, WamCode, [], LLVMCode),
    % 3-arity: should set A1 (reg 0), A2 (reg 1), A3 (reg 2)
    assertion(sub_atom(LLVMCode, _, _, _, 'call void @wam_set_reg(%WamState* %vm, i32 0, %Value %a1)')),
    assertion(sub_atom(LLVMCode, _, _, _, 'call void @wam_set_reg(%WamState* %vm, i32 1, %Value %a2)')),
    assertion(sub_atom(LLVMCode, _, _, _, 'call void @wam_set_reg(%WamState* %vm, i32 2, %Value %a3)')).

% ============================================================================
% Test 5: Full runtime assembly (step + helpers combined)
% ============================================================================

test(full_runtime_has_all_components) :-
    compile_wam_runtime_to_llvm([], RuntimeCode),
    % Step function
    assertion(sub_atom(RuntimeCode, _, _, _, 'define i1 @step')),
    % All helpers
    assertion(sub_atom(RuntimeCode, _, _, _, 'define i1 @backtrack')),
    assertion(sub_atom(RuntimeCode, _, _, _, 'define void @unwind_trail')),
    assertion(sub_atom(RuntimeCode, _, _, _, 'define i1 @execute_builtin')),
    assertion(sub_atom(RuntimeCode, _, _, _, 'define i64 @eval_arith')).

% ============================================================================
% Test 6: WAM fallback can be disabled
% ============================================================================

test(wam_fallback_disabled) :-
    % With wam_fallback(false), WAM compilation should not be attempted
    % compile_predicates_collect should produce a "compilation failed" comment
    wam_llvm_target:compile_predicates_collect(
        [nonexistent_pred/2],
        [wam_fallback(false)],
        NativeParts, WamParts),
    assertion(NativeParts == []),
    % Should have a failure comment
    (   WamParts = [FailMsg|_]
    ->  assertion(sub_atom(FailMsg, _, _, _, 'compilation failed'))
    ;   true  % Empty is also acceptable — means it was skipped
    ).

% ============================================================================
% Test 7: Label resolution in multi-clause predicate
% ============================================================================

test(multi_clause_label_resolution) :-
    WamCode = "ancestor/2:\n    try_me_else L_clause2\n    get_constant parent, A1\n    proceed\nL_clause2:\n    trust_me\n    get_constant grandparent, A1\n    proceed",
    compile_wam_predicate_to_llvm(ancestor/2, WamCode, [], LLVMCode),
    % try_me_else should reference L_clause2 (label index 1)
    assertion(sub_atom(LLVMCode, _, _, _, 'i32 22, i64 1')).

% ============================================================================
% Test 8: Instruction count matches in code array
% ============================================================================

test(instruction_count_matches) :-
    WamCode = "fact/1:\n    get_constant a, A1\n    proceed",
    compile_wam_predicate_to_llvm(fact/1, WamCode, [], LLVMCode),
    % 2 instructions → [2 x %Instruction]
    assertion(sub_atom(LLVMCode, _, _, _, '[2 x %Instruction]')).

% ============================================================================
% Test 9: Choice point instructions produce correct IR
% ============================================================================

test(choice_point_ir_structure) :-
    compile_step_wam_to_llvm([], StepCode),
    % try_me_else should save registers via memcpy
    assertion(sub_atom(StepCode, _, _, _, 'tme.dst_raw')),
    assertion(sub_atom(StepCode, _, _, _, 'llvm.memcpy')),
    % backtrack restores via memcpy
    compile_wam_helpers_to_llvm([], HelpersCode),
    assertion(sub_atom(HelpersCode, _, _, _, 'llvm.memcpy')).

% ============================================================================
% Test 10: eval_arith handles compound ops
% ============================================================================

test(eval_arith_compound_ops) :-
    compile_wam_helpers_to_llvm([], Code),
    % Binary arithmetic dispatch
    assertion(sub_atom(Code, _, _, _, 'do_add:')),
    assertion(sub_atom(Code, _, _, _, 'add i64 %a, %b')),
    assertion(sub_atom(Code, _, _, _, 'do_sub:')),
    assertion(sub_atom(Code, _, _, _, 'sub i64 %a, %b')),
    assertion(sub_atom(Code, _, _, _, 'do_mul:')),
    assertion(sub_atom(Code, _, _, _, 'mul i64 %a, %b')),
    assertion(sub_atom(Code, _, _, _, 'do_div:')),
    assertion(sub_atom(Code, _, _, _, 'sdiv i64 %a, %b')),
    % Division by zero guard
    assertion(sub_atom(Code, _, _, _, 'do_div_ok')),
    % Unary negation
    assertion(sub_atom(Code, _, _, _, 'do_neg:')).

% ============================================================================
% Test 11: builtin_is calls eval_arith
% ============================================================================

test(builtin_is_uses_eval_arith) :-
    compile_wam_helpers_to_llvm([], Code),
    assertion(sub_atom(Code, _, _, _, 'call i64 @eval_arith(%WamState* %vm, %Value %is.a2)')),
    assertion(sub_atom(Code, _, _, _, 'call %Value @value_integer(i64 %is.result)')).

% ============================================================================
% Test 12: Zero-arity predicate (no arg setup)
% ============================================================================

test(zero_arity_predicate) :-
    WamCode = "main/0:\n    proceed",
    compile_wam_predicate_to_llvm(main/0, WamCode, [], LLVMCode),
    % Should have %WamState* %vm as only param
    assertion(sub_atom(LLVMCode, _, _, _, 'define i1 @main(%WamState* %vm)')),
    % Should NOT have wam_set_reg calls (no args)
    assertion(\+ sub_atom(LLVMCode, _, _, _, 'wam_set_reg')).

:- end_tests(llvm_wam_e2e).
