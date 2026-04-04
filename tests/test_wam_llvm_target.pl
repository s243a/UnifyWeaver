:- encoding(utf8).
:- use_module(library(plunit)).
:- use_module('../src/unifyweaver/targets/wam_llvm_target').
:- use_module('../src/unifyweaver/bindings/llvm_wam_bindings').

:- begin_tests(wam_llvm_target).

% ============================================================================
% Phase 0: Binding registry tests
% ============================================================================

test(reg_name_to_index_a1) :-
    reg_name_to_index('A1', Idx),
    assertion(Idx == 0).

test(reg_name_to_index_a2) :-
    reg_name_to_index('A2', Idx),
    assertion(Idx == 1).

test(reg_name_to_index_x1) :-
    reg_name_to_index('X1', Idx),
    assertion(Idx == 16).

test(reg_name_to_index_x3) :-
    reg_name_to_index('X3', Idx),
    assertion(Idx == 18).

test(type_map_value) :-
    llvm_wam_type_map(value, LLVMType),
    assertion(LLVMType == '%Value').

test(type_map_integer) :-
    llvm_wam_type_map(integer, LLVMType),
    assertion(LLVMType == 'i64').

test(type_map_assoc) :-
    llvm_wam_type_map(assoc, LLVMType),
    assertion(LLVMType == '[32 x %Value]').

test(binding_exists_get_assoc) :-
    llvm_wam_binding(get_assoc/3, _, _, _, _).

test(binding_exists_atom_check) :-
    llvm_wam_binding(atom/1, Expr, _, _, _),
    assertion(Expr == 'icmp eq i32 ~tag, 0').

test(binding_exists_add) :-
    llvm_wam_binding('+'/3, Expr, _, _, _),
    assertion(Expr == 'add i64 ~a, ~b').

% ============================================================================
% Phase 2: Instruction lowering tests
% ============================================================================

test(llvm_get_constant) :-
    wam_instruction_to_llvm_literal(get_constant(atom(john), 'A1'), Lit),
    assertion(sub_atom(Lit, _, _, _, '{ i32 0,')),
    assertion(sub_atom(Lit, _, _, _, 'i64 0 }')).  % A1 → index 0

test(llvm_get_variable) :-
    wam_instruction_to_llvm_literal(get_variable('X1', 'A1'), Lit),
    assertion(sub_atom(Lit, _, _, _, '{ i32 1, i64 16, i64 0 }')).  % X1=16, A1=0

test(llvm_get_value) :-
    wam_instruction_to_llvm_literal(get_value('X2', 'A2'), Lit),
    assertion(sub_atom(Lit, _, _, _, '{ i32 2, i64 17, i64 1 }')).  % X2=17, A2=1

test(llvm_put_constant) :-
    wam_instruction_to_llvm_literal(put_constant(integer(42), 'A1'), Lit),
    assertion(sub_atom(Lit, _, _, _, '{ i32 8, i64 42, i64 0 }')).

test(llvm_put_variable) :-
    wam_instruction_to_llvm_literal(put_variable('X1', 'A2'), Lit),
    assertion(sub_atom(Lit, _, _, _, '{ i32 9, i64 16, i64 1 }')).

test(llvm_allocate) :-
    wam_instruction_to_llvm_literal(allocate, Lit),
    assertion(Lit == '{ i32 16, i64 0, i64 0 }').

test(llvm_proceed) :-
    wam_instruction_to_llvm_literal(proceed, Lit),
    assertion(Lit == '{ i32 20, i64 0, i64 0 }').

test(llvm_trust_me) :-
    wam_instruction_to_llvm_literal(trust_me, Lit),
    assertion(Lit == '{ i32 24, i64 0, i64 0 }').

test(llvm_builtin_call) :-
    wam_instruction_to_llvm_literal(builtin_call('>/2', 2), Lit),
    assertion(sub_atom(Lit, _, _, _, '{ i32 21, i64 1, i64 2 }')).  % >/2 → id 1

% Label-referencing instructions error on /2, work on /3
test(llvm_call_errors_without_labelmap, [throws(error(label_resolution_required(call, _), _))]) :-
    wam_instruction_to_llvm_literal(call('parent/2', 2), _).

test(llvm_call_with_labelmap) :-
    LabelMap = ['parent/2'-0, 'ancestor/2'-1],
    wam_instruction_to_llvm_literal(call('parent/2', 2), LabelMap, Lit),
    assertion(sub_atom(Lit, _, _, _, '{ i32 18, i64 0, i64 2 }')).

test(llvm_try_me_else_with_labelmap) :-
    LabelMap = ['L_alt'-0, 'L_clause2'-1],
    wam_instruction_to_llvm_literal(try_me_else('L_clause2'), LabelMap, Lit),
    assertion(sub_atom(Lit, _, _, _, '{ i32 22, i64 1, i64 0 }')).

test(llvm_execute_with_labelmap) :-
    LabelMap = ['main/0'-0],
    wam_instruction_to_llvm_literal(execute('main/0'), LabelMap, Lit),
    assertion(sub_atom(Lit, _, _, _, '{ i32 19, i64 0, i64 0 }')).

test(llvm_non_label_instr_works_with_labelmap) :-
    wam_instruction_to_llvm_literal(allocate, [], Lit),
    assertion(Lit == '{ i32 16, i64 0, i64 0 }').

% ============================================================================
% Phase 2: WAM text line parsing
% ============================================================================

test(parse_line_get_constant) :-
    wam_line_to_llvm_literal(["get_constant", "john,", "A1"], Lit),
    assertion(sub_atom(Lit, _, _, _, 'i32 0')).

test(parse_line_get_variable) :-
    wam_line_to_llvm_literal(["get_variable", "X1,", "A1"], Lit),
    assertion(sub_atom(Lit, _, _, _, 'i32 1')),
    assertion(sub_atom(Lit, _, _, _, 'i64 16')),  % X1
    assertion(sub_atom(Lit, _, _, _, 'i64 0')).    % A1

test(parse_line_allocate) :-
    wam_line_to_llvm_literal(["allocate"], Lit),
    assertion(Lit == '%Instruction { i32 16, i64 0, i64 0 }').

test(parse_line_proceed) :-
    wam_line_to_llvm_literal(["proceed"], Lit),
    assertion(Lit == '%Instruction { i32 20, i64 0, i64 0 }').

test(parse_label_and_instr) :-
    WamCode = "parent/2:\n    get_constant john, A1\n    proceed",
    compile_wam_predicate_to_llvm(parent/2, WamCode, [], LLVMCode),
    assertion(sub_atom(LLVMCode, _, _, _, '@parent_code')),
    assertion(sub_atom(LLVMCode, _, _, _, '@parent_labels')),
    assertion(sub_atom(LLVMCode, _, _, _, 'define i1 @parent')).

% ============================================================================
% Phase 2: Step function generation
% ============================================================================

test(step_wam_generation) :-
    compile_step_wam_to_llvm([], StepCode),
    % Should contain LLVM switch dispatch
    assertion(sub_atom(StepCode, _, _, _, 'define i1 @step')),
    assertion(sub_atom(StepCode, _, _, _, 'switch i32 %tag')),
    assertion(sub_atom(StepCode, _, _, _, 'get_constant')),
    assertion(sub_atom(StepCode, _, _, _, 'put_constant')),
    assertion(sub_atom(StepCode, _, _, _, 'proceed')),
    assertion(sub_atom(StepCode, _, _, _, 'try_me_else')),
    assertion(sub_atom(StepCode, _, _, _, 'trust_me')).

% ============================================================================
% Phase 3: Helper function generation
% ============================================================================

test(helpers_generation) :-
    compile_wam_helpers_to_llvm([], HelpersCode),
    assertion(sub_atom(HelpersCode, _, _, _, 'define i1 @backtrack')),
    assertion(sub_atom(HelpersCode, _, _, _, 'define void @unwind_trail')),
    assertion(sub_atom(HelpersCode, _, _, _, 'define i1 @execute_builtin')),
    assertion(sub_atom(HelpersCode, _, _, _, 'define i64 @eval_arith')).

test(backtrack_has_memcpy) :-
    compile_wam_helpers_to_llvm([], Code),
    assertion(sub_atom(Code, _, _, _, 'llvm.memcpy')).

test(builtin_dispatch_has_switch) :-
    compile_wam_helpers_to_llvm([], Code),
    assertion(sub_atom(Code, _, _, _, 'switch i32 %op')).

% ============================================================================
% Phase 5: Full runtime generation
% ============================================================================

test(full_runtime_generation) :-
    compile_wam_runtime_to_llvm([], RuntimeCode),
    % Step function
    assertion(sub_atom(RuntimeCode, _, _, _, 'define i1 @step')),
    % Helpers
    assertion(sub_atom(RuntimeCode, _, _, _, 'define i1 @backtrack')).

% ============================================================================
% Builtin op ID mapping
% ============================================================================

test(builtin_op_is) :-
    builtin_op_to_id('is/2', Id),
    assertion(Id == 0).

test(builtin_op_gt) :-
    builtin_op_to_id('>/2', Id),
    assertion(Id == 1).

test(builtin_op_true) :-
    builtin_op_to_id('true/0', Id),
    assertion(Id == 8).

test(builtin_op_cut) :-
    builtin_op_to_id('!/0', Id),
    assertion(Id == 10).

test(builtin_op_unknown) :-
    builtin_op_to_id('unknown/3', Id),
    assertion(Id == 99).

% ============================================================================
% Review fixes: label resolution in text parser
% ============================================================================

test(label_resolution_try_me_else) :-
    WamCode = "parent/2:\n    try_me_else L_alt\nL_alt:\n    proceed",
    compile_wam_predicate_to_llvm(parent/2, WamCode, [], LLVMCode),
    % try_me_else should have label index 1 (L_alt is the 2nd label)
    assertion(sub_atom(LLVMCode, _, _, _, 'i32 22, i64 1')).

test(label_resolution_call) :-
    WamCode = "main/0:\n    call main/0, 0\n    proceed",
    compile_wam_predicate_to_llvm(main/0, WamCode, [], LLVMCode),
    % call should resolve main/0 to label index 0
    assertion(sub_atom(LLVMCode, _, _, _, 'i32 18, i64 0')).

% ============================================================================
% Review fixes: allocate/deallocate are not stubs
% ============================================================================

test(allocate_pushes_env_frame) :-
    compile_step_wam_to_llvm([], StepCode),
    % Should reference stack operations, not just inc_pc
    assertion(sub_atom(StepCode, _, _, _, 'alloc.ss_ptr')),
    assertion(sub_atom(StepCode, _, _, _, 'alloc.cp')).

test(deallocate_scans_backward) :-
    compile_step_wam_to_llvm([], StepCode),
    % Should have a backward scan loop, not just top-of-stack check
    assertion(sub_atom(StepCode, _, _, _, 'dealloc.loop')),
    assertion(sub_atom(StepCode, _, _, _, 'dealloc.check')),
    assertion(sub_atom(StepCode, _, _, _, 'dealloc.skip')),
    assertion(sub_atom(StepCode, _, _, _, 'dealloc.restore')),
    assertion(sub_atom(StepCode, _, _, _, 'dealloc.saved_cp')).

% ============================================================================
% Review fixes: eval_arith handles compound expressions
% ============================================================================

test(eval_arith_has_compound_handling) :-
    compile_wam_helpers_to_llvm([], Code),
    assertion(sub_atom(Code, _, _, _, 'compound_arith')),
    assertion(sub_atom(Code, _, _, _, 'do_add')),
    assertion(sub_atom(Code, _, _, _, 'do_sub')),
    assertion(sub_atom(Code, _, _, _, 'do_mul')),
    assertion(sub_atom(Code, _, _, _, 'do_div')).

% ============================================================================
% Review fixes: atom interning (no hash collisions)
% ============================================================================

test(atom_interning_unique_ids) :-
    wam_llvm_target:intern_atom(test_atom_a, IdA),
    wam_llvm_target:intern_atom(test_atom_b, IdB),
    assertion(IdA \== IdB),
    % Same atom gets same ID
    wam_llvm_target:intern_atom(test_atom_a, IdA2),
    assertion(IdA == IdA2).

% ============================================================================
% Review fixes: =\=/2 in execute_builtin
% ============================================================================

test(builtin_arith_ne_in_dispatch) :-
    compile_wam_helpers_to_llvm([], Code),
    assertion(sub_atom(Code, _, _, _, 'builtin_arith_ne')),
    assertion(sub_atom(Code, _, _, _, 'i32 6, label %builtin_arith_ne')).

% ============================================================================
% Phase 7: WASM variant
% ============================================================================

test(wasm_runtime_uses_loop_not_musttail) :-
    % Read the WASM runtime template
    wam_llvm_target:read_template_file(
        'templates/targets/llvm_wam_wasm/runtime.ll.mustache', Template),
    % Should NOT contain 'musttail call' (the actual IR instruction)
    assertion(\+ sub_atom(Template, _, _, _, 'musttail call')),
    % Should contain a loop-based run_loop
    assertion(sub_atom(Template, _, _, _, 'br label %loop')).

test(wasm_types_use_wasm32) :-
    wam_llvm_target:read_template_file(
        'templates/targets/llvm_wam_wasm/types.ll.mustache', Template),
    assertion(sub_atom(Template, _, _, _, 'wasm32')).

test(wasm_allocator_has_bump_alloc) :-
    wam_llvm_target:read_template_file(
        'templates/targets/llvm_wam_wasm/allocator.ll.mustache', Template),
    assertion(sub_atom(Template, _, _, _, '@wam_heap_ptr')),
    assertion(sub_atom(Template, _, _, _, 'define i32 @wam_alloc')),
    assertion(sub_atom(Template, _, _, _, 'define void @wam_alloc_rewind')),
    assertion(sub_atom(Template, _, _, _, 'define void @wam_memcpy')),
    % Pointer provenance: uses getelementptr from base, not raw inttoptr
    assertion(sub_atom(Template, _, _, _, '@wam_linear_base')),
    assertion(sub_atom(Template, _, _, _, 'getelementptr i8, i8* %base')).

test(wasm_module_template_structure) :-
    wam_llvm_target:read_template_file(
        'templates/targets/llvm_wam_wasm/module.ll.mustache', Template),
    assertion(sub_atom(Template, _, _, _, 'wasm32-unknown-unknown')),
    assertion(sub_atom(Template, _, _, _, '{{allocator_functions}}')),
    assertion(sub_atom(Template, _, _, _, '{{wasm_exports}}')).

test(wasm_build_commands) :-
    build_wam_wasm_module('test.ll', 'test_out', Commands),
    assertion(sub_atom(Commands, _, _, _, 'llc --mtriple=wasm32-unknown-unknown')),
    assertion(sub_atom(Commands, _, _, _, 'wasm-ld')),
    assertion(sub_atom(Commands, _, _, _, '--no-entry')),
    assertion(sub_atom(Commands, _, _, _, '--export-all')),
    % Toolchain check
    assertion(sub_atom(Commands, _, _, _, 'command -v')).

test(wasm_exports_generation) :-
    wam_llvm_target:generate_wasm_exports([parent/2, ancestor/2], ExportCode),
    assertion(sub_atom(ExportCode, _, _, _, '@parent_wasm')),
    assertion(sub_atom(ExportCode, _, _, _, '@ancestor_wasm')),
    assertion(sub_atom(ExportCode, _, _, _, 'ret i32')),
    % Export visibility: dso_local + attribute group
    assertion(sub_atom(ExportCode, _, _, _, 'dso_local')),
    assertion(sub_atom(ExportCode, _, _, _, 'wasm-export-name')).

% ============================================================================
% Fix #3: lookup_label_index warning
% ============================================================================

test(lookup_label_warns_on_unknown, [true]) :-
    % Should succeed with Index=0 (and print a warning to stderr)
    wam_llvm_target:lookup_label_index('nonexistent_label', [], Index),
    assertion(Index == 0).

test(lookup_label_strict_throws, [throws(error(unknown_label(_), _))]) :-
    wam_llvm_target:lookup_label_index(
        'nonexistent_label', [], [wam_strict_labels(true)], _).

:- end_tests(wam_llvm_target).
