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
    % The assoc register array was widened from 32 to 64 %Value slots.
    assertion(LLVMType == '[64 x %Value]').

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

test(parse_line_switch_on_constant_a2_fallthrough) :-
    wam_line_to_llvm_literal(["switch_on_constant_a2_fallthrough", "end_of_file:default"], Lit),
    assertion(Lit == '%Instruction { i32 27, i64 0, i64 0 }').

test(resolve_line_switch_on_constant_a2_fallthrough) :-
    wam_llvm_target:wam_line_to_llvm_literal_resolved(["switch_on_constant_a2_fallthrough", "end_of_file:default"], [], Lit),
    assertion(Lit == '%Instruction { i32 27, i64 0, i64 0 }').

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
    assertion(sub_atom(RuntimeCode, _, _, _, 'define i1 @backtrack')),
    assertion(sub_atom(RuntimeCode, _, _, _, 'define i1 @wam_atom_field_eq_value')),
    assertion(sub_atom(RuntimeCode, _, _, _, 'define %WamSlice @wam_atom_field_slice_value')),
    assertion(sub_atom(RuntimeCode, _, _, _, 'define i64 @wam_atom_field_count_value')),
    assertion(sub_atom(RuntimeCode, _, _, _, 'define i1 @wam_is_field_whitespace')),
    assertion(sub_atom(RuntimeCode, _, _, _, 'define i64 @wam_atom_field_length_value')),
    assertion(sub_atom(RuntimeCode, _, _, _, 'define %WamSlice @wam_subslice_value')),
    assertion(sub_atom(RuntimeCode, _, _, _, 'define %WamSlice @wam_atom_field_subslice_value')),
    assertion(sub_atom(RuntimeCode, _, _, _, 'define i64 @wam_slice_index_value')),
    assertion(sub_atom(RuntimeCode, _, _, _, 'define i64 @wam_atom_field_index_value')),
    assertion(sub_atom(RuntimeCode, _, _, _, 'define void @wam_print_ascii_lower_slice')),
    assertion(sub_atom(RuntimeCode, _, _, _, 'define void @wam_print_ascii_upper_slice')),
    assertion(sub_atom(RuntimeCode, _, _, _, 'define i1 @wam_atom_prefix_value')),
    assertion(sub_atom(RuntimeCode, _, _, _, 'define %WamAssocI64Table* @wam_assoc_i64_new')),
    assertion(sub_atom(RuntimeCode, _, _, _, 'define i1 @wam_assoc_i64_resize')),
    assertion(sub_atom(RuntimeCode, _, _, _, 'define i64 @wam_assoc_i64_inc')),
    assertion(sub_atom(RuntimeCode, _, _, _, 'define i64 @wam_assoc_i64_get')),
    assertion(sub_atom(RuntimeCode, _, _, _, 'define void @wam_assoc_i64_free')).

test(atom_prefix_guard_emitter) :-
    llvm_emit_atom_prefix_guard(prefix_guard_test, '%line', 'ERROR', '%ok', GlobalIR-CallIR),
    assertion(sub_atom(GlobalIR, _, _, _, '@.prefix_5Fguard_5Ftest = private constant [6 x i8] c"ERROR\\00"')),
    assertion(sub_atom(CallIR, _, _, _, '%prefix_5Fguard_5Ftest_ptr = getelementptr')),
    assertion(sub_atom(CallIR, _, _, _, '%ok = call i1 @wam_atom_prefix_value(%Value %line')),
    assertion(sub_atom(CallIR, _, _, _, 'i64 5')).

test(atom_field_eq_guard_emitter) :-
    llvm_emit_atom_field_eq_guard(field_eq_guard_test, '%line', 1, 'ERROR', 32, '%ok', GlobalIR-CallIR),
    assertion(sub_atom(GlobalIR, _, _, _, '@.field_5Feq_5Fguard_5Ftest = private constant [6 x i8] c"ERROR\\00"')),
    assertion(sub_atom(CallIR, _, _, _, '%field_5Feq_5Fguard_5Ftest_ptr = getelementptr')),
    assertion(sub_atom(CallIR, _, _, _, '%ok = call i1 @wam_atom_field_eq_value(%Value %line, i64 1')),
    assertion(sub_atom(CallIR, _, _, _, 'i64 5, i8 32')).

test(atom_field_slice_emitter) :-
    llvm_emit_atom_field_slice('%line', 2, 32, plawk_f2, CallIR),
    assertion(sub_atom(CallIR, _, _, _, '%plawk_f2 = call %WamSlice @wam_atom_field_slice_value(%Value %line, i64 2, i8 32)')),
    assertion(sub_atom(CallIR, _, _, _, '%plawk_f2_ptr = extractvalue %WamSlice %plawk_f2, 0')),
    assertion(sub_atom(CallIR, _, _, _, '%plawk_f2_len = trunc i64 %plawk_f2_len64 to i32')).

test(atom_field_count_emitter) :-
    llvm_emit_atom_field_count('%line', 58, plawk_nf, CallIR),
    assertion(sub_atom(CallIR, _, _, _, '%plawk_nf = call i64 @wam_atom_field_count_value(%Value %line, i8 58)')).

test(atom_field_length_emitter) :-
    llvm_emit_atom_field_length('%line', 2, 58, plawk_len, CallIR),
    assertion(sub_atom(CallIR, _, _, _, '%plawk_len = call i64 @wam_atom_field_length_value(%Value %line, i64 2, i8 58)')).

test(atom_field_subslice_emitter) :-
    llvm_emit_atom_field_subslice('%line', 2, 58, 1, 3, plawk_substr, CallIR),
    assertion(sub_atom(CallIR, _, _, _, '%plawk_substr = call %WamSlice @wam_atom_field_subslice_value(%Value %line, i64 2, i8 58, i64 1, i64 3)')),
    assertion(sub_atom(CallIR, _, _, _, '%plawk_substr_ptr = extractvalue %WamSlice %plawk_substr, 0')),
    assertion(sub_atom(CallIR, _, _, _, '%plawk_substr_len = trunc i64 %plawk_substr_len64 to i32')).

test(atom_field_index_emitter) :-
    llvm_emit_atom_field_index(plawk_index_needle, '%line', 2, 'disk', 58, plawk_index, GlobalIR-CallIR),
    assertion(sub_atom(GlobalIR, _, _, _, '@.plawk_5Findex_5Fneedle = private constant [5 x i8] c"disk\\00"')),
    assertion(sub_atom(CallIR, _, _, _, '%plawk_5Findex_5Fneedle_ptr = getelementptr')),
    assertion(sub_atom(CallIR, _, _, _, '%plawk_index = call i64 @wam_atom_field_index_value(%Value %line, i64 2, i8 58')),
    assertion(sub_atom(CallIR, _, _, _, 'i64 4')).

test(ascii_case_slice_print_emitter) :-
    llvm_emit_ascii_case_slice_print(lower, '%ptr', '%len', plawk_lower, LowerIR),
    llvm_emit_ascii_case_slice_print(upper, '%ptr', '%len', plawk_upper, UpperIR),
    assertion(LowerIR == '  call void @wam_print_ascii_lower_slice(i8* %ptr, i64 %len)'),
    assertion(UpperIR == '  call void @wam_print_ascii_upper_slice(i8* %ptr, i64 %len)').

test(c_string_global_emitter) :-
    llvm_emit_c_string_global(example_label, "hello\n", GlobalIR, StringLen, BytesLen),
    assertion(StringLen == 6),
    assertion(BytesLen == 7),
    assertion(GlobalIR == '@.example_label = private constant [7 x i8] c"hello\\0A\\00"').

test(printf_i64_emitter) :-
    llvm_emit_printf_i64(plawk_surface_print_i64, value_fmt, printed_value, '%n', Parts),
    assertion(Parts == [
        '  %value_fmt = getelementptr [4 x i8], [4 x i8]* @.plawk_surface_print_i64, i32 0, i32 0',
        '  %printed_value = call i32 (i8*, ...) @printf(i8* %value_fmt, i64 %n)'
    ]).

test(printf_slice_emitter) :-
    llvm_emit_printf_slice(plawk_surface_print_slice, slice_fmt, printed_slice,
        '%len', '%ptr', Parts),
    assertion(Parts == [
        '  %slice_fmt = getelementptr [5 x i8], [5 x i8]* @.plawk_surface_print_slice, i32 0, i32 0',
        '  %printed_slice = call i32 (i8*, ...) @printf(i8* %slice_fmt, i32 %len, i8* %ptr)'
    ]).

test(printf_string_emitter) :-
    llvm_emit_printf_string(plawk_surface_print_string, string_fmt, printed_string,
        '%ptr', StringParts),
    llvm_emit_printf_string(plawk_surface_print_line, 4, line_fmt, printed_line,
        '%line_s', LineParts),
    assertion(StringParts == [
        '  %string_fmt = getelementptr [3 x i8], [3 x i8]* @.plawk_surface_print_string, i32 0, i32 0',
        '  %printed_string = call i32 (i8*, ...) @printf(i8* %string_fmt, i8* %ptr)'
    ]),
    assertion(LineParts == [
        '  %line_fmt = getelementptr [4 x i8], [4 x i8]* @.plawk_surface_print_line, i32 0, i32 0',
        '  %printed_line = call i32 (i8*, ...) @printf(i8* %line_fmt, i8* %line_s)'
    ]).

test(printf0_emitter) :-
    llvm_emit_printf0(plawk_surface_print_newline, 2, newline_fmt, printed_newline, Parts),
    assertion(Parts == [
        '  %newline_fmt = getelementptr [2 x i8], [2 x i8]* @.plawk_surface_print_newline, i32 0, i32 0',
        '  %printed_newline = call i32 (i8*, ...) @printf(i8* %newline_fmt)'
    ]).

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
    % Unknown ops map to the reserved sentinel 200. (It was 99 until the
    % builtin table grew past that — id 99 is now getenv/2 — so the
    % sentinel was moved above the op range.)
    assertion(Id == 200).

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

% ============================================================================
% Compound term instruction cases in step dispatch
% ============================================================================

test(step_has_compound_instructions) :-
    compile_step_wam_to_llvm([], StepCode),
    assertion(sub_atom(StepCode, _, _, _, 'get_structure:')),
    assertion(sub_atom(StepCode, _, _, _, 'get_list:')),
    assertion(sub_atom(StepCode, _, _, _, 'unify_variable:')),
    assertion(sub_atom(StepCode, _, _, _, 'unify_value:')),
    assertion(sub_atom(StepCode, _, _, _, 'unify_constant:')),
    assertion(sub_atom(StepCode, _, _, _, 'put_structure:')),
    assertion(sub_atom(StepCode, _, _, _, 'put_list:')),
    assertion(sub_atom(StepCode, _, _, _, 'set_variable:')),
    assertion(sub_atom(StepCode, _, _, _, 'set_value:')),
    assertion(sub_atom(StepCode, _, _, _, 'set_constant:')).

test(compound_instrs_use_heap_push) :-
    compile_step_wam_to_llvm([], StepCode),
    assertion(sub_atom(StepCode, _, _, _, 'wam_heap_push')).

test(get_structure_has_write_and_read_mode) :-
    compile_step_wam_to_llvm([], StepCode),
    assertion(sub_atom(StepCode, _, _, _, 'gs.write:')),
    assertion(sub_atom(StepCode, _, _, _, 'gs.read:')).

test(unify_variable_has_read_write_dispatch) :-
    compile_step_wam_to_llvm([], StepCode),
    assertion(sub_atom(StepCode, _, _, _, 'uv.read:')),
    assertion(sub_atom(StepCode, _, _, _, 'uv.write:')),
    assertion(sub_atom(StepCode, _, _, _, 'wam_peek_stack_type')),
    assertion(sub_atom(StepCode, _, _, _, 'wam_unify_ctx_next')),
    % Write mode advances the cursor via wam_write_ctx_set_arg (this
    % replaced the older standalone wam_write_ctx_dec helper).
    assertion(sub_atom(StepCode, _, _, _, 'wam_write_ctx_set_arg')).

test(unify_value_has_read_write_dispatch) :-
    compile_step_wam_to_llvm([], StepCode),
    assertion(sub_atom(StepCode, _, _, _, 'uvl.read:')),
    assertion(sub_atom(StepCode, _, _, _, 'uvl.write:')),
    assertion(sub_atom(StepCode, _, _, _, 'uvl.fail:')).

test(write_ctx_pushed_by_structures) :-
    compile_step_wam_to_llvm([], StepCode),
    assertion(sub_atom(StepCode, _, _, _, 'wam_push_write_ctx')).

test(lookup_label_warns_on_unknown, [true]) :-
    % Should succeed with Index=0 (and print a warning to stderr)
    wam_llvm_target:lookup_label_index('nonexistent_label', [], Index),
    assertion(Index == 0).

test(lookup_label_strict_throws, [throws(error(unknown_label(_), _))]) :-
    wam_llvm_target:lookup_label_index(
        'nonexistent_label', [], [wam_strict_labels(true)], _).

:- end_tests(wam_llvm_target).
