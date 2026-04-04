:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wam_llvm_target.pl - WAM-to-LLVM IR Transpilation Target
%
% Transpiles WAM runtime predicates (wam_runtime.pl) to LLVM IR code.
% Phase 2: step_wam/3 → LLVM switch dispatch
% Phase 3: helper predicates → LLVM functions
% Phase 4: WAM instructions → LLVM struct literals
% Phase 5: Hybrid module assembly
%
% Key LLVM-specific design choices:
%   - Value = { i32 tag, i64 payload } tagged union (not enum/interface)
%   - Instruction dispatch via LLVM switch on integer tag
%   - Registers = [32 x %Value] fixed array (not HashMap/map)
%   - Run loop uses musttail for constant-stack execution
%   - Arena-style memory (malloc + backtrack rewind)
%
% See: docs/design/WAM_LLVM_TRANSPILATION_IMPLEMENTATION_PLAN.md

:- module(wam_llvm_target, [
    compile_step_wam_to_llvm/2,          % +Options, -LLVMCode
    compile_wam_helpers_to_llvm/2,       % +Options, -LLVMCode
    compile_wam_runtime_to_llvm/2,       % +Options, -LLVMCode (step + helpers combined)
    compile_wam_predicate_to_llvm/4,     % +Pred/Arity, +WamCode, +Options, -LLVMCode
    wam_instruction_to_llvm_literal/2,   % +WamInstr, -LLVMLiteral
    wam_line_to_llvm_literal/2,          % +Parts, -LLVMLit
    write_wam_llvm_project/3,            % +Predicates, +Options, +OutputFile
    builtin_op_to_id/2                   % +OpName, -IntId
]).

:- use_module(library(lists)).
:- use_module(library(option)).
:- use_module('../core/template_system').
:- use_module('../bindings/llvm_wam_bindings').
:- use_module('../targets/wam_target', [compile_predicate_to_wam/3]).

:- discontiguous wam_llvm_case/2.

% ============================================================================
% PHASE 5: Hybrid Module Assembly
% ============================================================================

%% write_wam_llvm_project(+Predicates, +Options, +OutputFile)
%  Generates a complete LLVM IR module for the given predicates.
write_wam_llvm_project(Predicates, Options, OutputFile) :-
    option(module_name(ModuleName), Options, 'wam_generated'),
    option(target_triple(Triple), Options, 'x86_64-pc-linux-gnu'),
    option(target_datalayout(DataLayout), Options,
        'e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128'),
    get_time(TimeStamp),
    format_time(string(Date), "%Y-%m-%d %H:%M:%S", TimeStamp),

    % Read and render type definitions template
    read_template_file('templates/targets/llvm_wam/types.ll.mustache', TypesTemplate),
    render_template(TypesTemplate, [module_name=ModuleName, date=Date], TypesDef),

    % Read and render value functions template
    read_template_file('templates/targets/llvm_wam/value.ll.mustache', ValueTemplate),
    render_template(ValueTemplate, [], ValueFuncs),

    % Read and render state functions template
    read_template_file('templates/targets/llvm_wam/state.ll.mustache', StateTemplate),
    render_template(StateTemplate, [], StateFuncs),

    % Generate runtime (step + helpers)
    compile_step_wam_to_llvm(Options, StepFunc),
    compile_wam_helpers_to_llvm(Options, HelpersCode),
    read_template_file('templates/targets/llvm_wam/runtime.ll.mustache', RuntimeTemplate),
    render_template(RuntimeTemplate, [
        step_function=StepFunc,
        backtrack_function=HelpersCode
    ], RuntimeFuncs),

    % Compile predicates (native or WAM fallback)
    compile_predicates_for_llvm(Predicates, Options, NativeCode, WamCode),

    % Assemble full module
    read_template_file('templates/targets/llvm_wam/module.ll.mustache', ModuleTemplate),
    render_template(ModuleTemplate, [
        module_name=ModuleName,
        date=Date,
        target_datalayout=DataLayout,
        target_triple=Triple,
        type_definitions=TypesDef,
        value_functions=ValueFuncs,
        state_functions=StateFuncs,
        runtime_functions=RuntimeFuncs,
        native_predicates=NativeCode,
        wam_predicates=WamCode,
        interop_bridge=""
    ], FullModule),

    % Write output file
    setup_call_cleanup(
        open(OutputFile, write, Stream),
        format(Stream, "~w", [FullModule]),
        close(Stream)
    ),
    format('WAM LLVM module created at: ~w~n', [OutputFile]).

%% read_template_file(+Path, -Content)
read_template_file(Path, Content) :-
    (   exists_file(Path)
    ->  read_file_to_string(Path, Content, [])
    ;   format(atom(Content), "; Template not found: ~w", [Path])
    ).

%% compile_predicates_for_llvm(+Predicates, +Options, -NativeCode, -WamCode)
compile_predicates_for_llvm([], _, "", "").
compile_predicates_for_llvm([PredIndicator|Rest], Options, NativeCode, WamCode) :-
    (   PredIndicator = Module:Pred/Arity -> true
    ;   PredIndicator = Pred/Arity, Module = user
    ),
    compile_predicates_for_llvm(Rest, Options, RestNative, RestWam),
    (   % Try native LLVM lowering first
        catch(
            llvm_target:compile_predicate_to_llvm(Module:Pred/Arity, Options, PredCode),
            _, fail)
    ->  format(user_error, '  ~w/~w: native lowering~n', [Pred, Arity]),
        format(atom(NativeCode), "~w\n\n~w", [PredCode, RestNative]),
        WamCode = RestWam
    ;   % Fall back to WAM compilation
        option(wam_fallback(WamFB), Options, true),
        WamFB \== false,
        wam_target:compile_predicate_to_wam(Module:Pred/Arity, Options, WamRaw),
        compile_wam_predicate_to_llvm(Pred/Arity, WamRaw, Options, PredCode)
    ->  format(user_error, '  ~w/~w: WAM fallback~n', [Pred, Arity]),
        NativeCode = RestNative,
        format(atom(WamCode), "~w\n\n~w", [PredCode, RestWam])
    ;   % Neither worked
        format(user_error, '  ~w/~w: compilation failed~n', [Pred, Arity]),
        NativeCode = RestNative,
        format(atom(WamCode), "; ~w/~w: compilation failed\n~w", [Pred, Arity, RestWam])
    ).

% ============================================================================
% PHASE 2: step_wam/3 → LLVM switch dispatch
% ============================================================================

%% compile_step_wam_to_llvm(+Options, -LLVMCode)
%  Generates the step() function body as an LLVM switch on instruction tag.
compile_step_wam_to_llvm(_Options, LLVMCode) :-
    findall(Case, compile_llvm_step_case(Case), Cases),
    atomic_list_concat(Cases, '\n', CasesCode),
    format(atom(LLVMCode),
'define i1 @step(%WamState* %vm, %Instruction* %instr) {
entry:
  %tag_ptr = getelementptr %Instruction, %Instruction* %instr, i32 0, i32 0
  %tag = load i32, i32* %tag_ptr
  %op1_ptr = getelementptr %Instruction, %Instruction* %instr, i32 0, i32 1
  %op1 = load i64, i64* %op1_ptr
  %op2_ptr = getelementptr %Instruction, %Instruction* %instr, i32 0, i32 2
  %op2 = load i64, i64* %op2_ptr
  switch i32 %tag, label %default [
    i32 0, label %get_constant
    i32 1, label %get_variable
    i32 2, label %get_value
    i32 8, label %put_constant
    i32 9, label %put_variable
    i32 10, label %put_value
    i32 16, label %allocate
    i32 17, label %deallocate
    i32 18, label %do_call
    i32 19, label %do_execute
    i32 20, label %proceed
    i32 21, label %builtin_call
    i32 22, label %try_me_else
    i32 23, label %retry_me_else
    i32 24, label %trust_me
  ]

~w

default:
  ret i1 false
}', [CasesCode]).

compile_llvm_step_case(CaseCode) :-
    wam_llvm_case(Label, BodyCode),
    format(atom(CaseCode), '~w:\n~w', [Label, BodyCode]).

% --- Head Unification Instructions ---

wam_llvm_case('get_constant',
'  ; op1 = constant value (packed), op2 = register index
  %gc.reg_idx = trunc i64 %op2 to i32
  %gc.current = call %Value @wam_get_reg(%WamState* %vm, i32 %gc.reg_idx)
  %gc.is_unb = call i1 @value_is_unbound(%Value %gc.current)
  br i1 %gc.is_unb, label %gc.bind, label %gc.check_eq

gc.bind:
  ; Unbound: bind to constant
  call void @wam_trail_binding(%WamState* %vm, i32 %gc.reg_idx)
  %gc.const_val = insertvalue %Value undef, i32 0, 0           ; tag from op1 high bits
  %gc.const_v2 = insertvalue %Value %gc.const_val, i64 %op1, 1
  call void @wam_set_reg(%WamState* %vm, i32 %gc.reg_idx, %Value %gc.const_v2)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true

gc.check_eq:
  ; Bound: check equality
  %gc.expected = insertvalue %Value undef, i32 0, 0
  %gc.expected2 = insertvalue %Value %gc.expected, i64 %op1, 1
  %gc.eq = call i1 @value_equals(%Value %gc.current, %Value %gc.expected2)
  br i1 %gc.eq, label %gc.match, label %gc.fail

gc.match:
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true

gc.fail:
  ret i1 false').

wam_llvm_case('get_variable',
'  ; op1 = Xn index, op2 = Ai index
  %gv.ai = trunc i64 %op2 to i32
  %gv.xn = trunc i64 %op1 to i32
  %gv.val = call %Value @wam_get_reg(%WamState* %vm, i32 %gv.ai)
  call void @wam_trail_binding(%WamState* %vm, i32 %gv.xn)
  call void @wam_set_reg(%WamState* %vm, i32 %gv.xn, %Value %gv.val)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').

wam_llvm_case('get_value',
'  ; op1 = Xn index, op2 = Ai index
  %gval.ai = trunc i64 %op2 to i32
  %gval.xn = trunc i64 %op1 to i32
  %gval.va = call %Value @wam_get_reg(%WamState* %vm, i32 %gval.ai)
  %gval.vx = call %Value @wam_get_reg(%WamState* %vm, i32 %gval.xn)
  ; Check if either is unbound
  %gval.a_unb = call i1 @value_is_unbound(%Value %gval.va)
  br i1 %gval.a_unb, label %gval.bind_a, label %gval.check_x

gval.bind_a:
  call void @wam_trail_binding(%WamState* %vm, i32 %gval.ai)
  call void @wam_set_reg(%WamState* %vm, i32 %gval.ai, %Value %gval.vx)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true

gval.check_x:
  %gval.x_unb = call i1 @value_is_unbound(%Value %gval.vx)
  br i1 %gval.x_unb, label %gval.bind_x, label %gval.check_eq

gval.bind_x:
  call void @wam_trail_binding(%WamState* %vm, i32 %gval.xn)
  call void @wam_set_reg(%WamState* %vm, i32 %gval.xn, %Value %gval.va)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true

gval.check_eq:
  %gval.eq = call i1 @value_equals(%Value %gval.va, %Value %gval.vx)
  br i1 %gval.eq, label %gval.match, label %gval.fail

gval.match:
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true

gval.fail:
  ret i1 false').

% --- Body Construction Instructions ---

wam_llvm_case('put_constant',
'  ; op1 = constant value (packed), op2 = register index
  %pc.reg_idx = trunc i64 %op2 to i32
  %pc.val = insertvalue %Value undef, i32 0, 0
  %pc.val2 = insertvalue %Value %pc.val, i64 %op1, 1
  call void @wam_trail_binding(%WamState* %vm, i32 %pc.reg_idx)
  call void @wam_set_reg(%WamState* %vm, i32 %pc.reg_idx, %Value %pc.val2)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').

wam_llvm_case('put_variable',
'  ; op1 = Xn index, op2 = Ai index
  %pv.xn = trunc i64 %op1 to i32
  %pv.ai = trunc i64 %op2 to i32
  ; Create unbound variable
  %pv.pc = call i32 @wam_get_pc(%WamState* %vm)
  %pv.pc_ext = zext i32 %pv.pc to i64
  %pv.var = call %Value @value_unbound(i8* null)
  call void @wam_trail_binding(%WamState* %vm, i32 %pv.xn)
  call void @wam_trail_binding(%WamState* %vm, i32 %pv.ai)
  call void @wam_set_reg(%WamState* %vm, i32 %pv.xn, %Value %pv.var)
  call void @wam_set_reg(%WamState* %vm, i32 %pv.ai, %Value %pv.var)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').

wam_llvm_case('put_value',
'  ; op1 = Xn index, op2 = Ai index
  %pvl.xn = trunc i64 %op1 to i32
  %pvl.ai = trunc i64 %op2 to i32
  %pvl.val = call %Value @wam_get_reg(%WamState* %vm, i32 %pvl.xn)
  call void @wam_trail_binding(%WamState* %vm, i32 %pvl.ai)
  call void @wam_set_reg(%WamState* %vm, i32 %pvl.ai, %Value %pvl.val)
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').

% --- Control Instructions ---

wam_llvm_case('allocate',
'  ; Push environment frame: save CP on stack
  %alloc.ss_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 3
  %alloc.ss = load i32, i32* %alloc.ss_ptr
  %alloc.stack_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 2
  %alloc.stack = load %StackEntry*, %StackEntry** %alloc.stack_ptr
  %alloc.entry = getelementptr %StackEntry, %StackEntry* %alloc.stack, i32 %alloc.ss
  ; type = 0 (EnvFrame)
  %alloc.type_ptr = getelementptr %StackEntry, %StackEntry* %alloc.entry, i32 0, i32 0
  store i32 0, i32* %alloc.type_ptr
  ; aux = current CP
  %alloc.cp = call i32 @wam_get_cp(%WamState* %vm)
  %alloc.cp_ext = zext i32 %alloc.cp to i64
  %alloc.aux_ptr = getelementptr %StackEntry, %StackEntry* %alloc.entry, i32 0, i32 1
  store i64 %alloc.cp_ext, i64* %alloc.aux_ptr
  ; Increment stack size
  %alloc.new_ss = add i32 %alloc.ss, 1
  store i32 %alloc.new_ss, i32* %alloc.ss_ptr
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').

wam_llvm_case('deallocate',
'  ; Pop environment frame: restore CP from stack
  %dealloc.ss_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 3
  %dealloc.ss = load i32, i32* %dealloc.ss_ptr
  %dealloc.has_frames = icmp sgt i32 %dealloc.ss, 0
  br i1 %dealloc.has_frames, label %dealloc.pop, label %dealloc.done

dealloc.pop:
  %dealloc.top_idx = sub i32 %dealloc.ss, 1
  %dealloc.stack_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 2
  %dealloc.stack = load %StackEntry*, %StackEntry** %dealloc.stack_ptr
  ; Scan backward for an EnvFrame (type == 0)
  %dealloc.entry = getelementptr %StackEntry, %StackEntry* %dealloc.stack, i32 %dealloc.top_idx
  %dealloc.type_ptr = getelementptr %StackEntry, %StackEntry* %dealloc.entry, i32 0, i32 0
  %dealloc.type = load i32, i32* %dealloc.type_ptr
  %dealloc.is_env = icmp eq i32 %dealloc.type, 0
  br i1 %dealloc.is_env, label %dealloc.restore, label %dealloc.done

dealloc.restore:
  ; Restore CP from saved value
  %dealloc.aux_ptr = getelementptr %StackEntry, %StackEntry* %dealloc.entry, i32 0, i32 1
  %dealloc.saved_cp = load i64, i64* %dealloc.aux_ptr
  %dealloc.cp = trunc i64 %dealloc.saved_cp to i32
  call void @wam_set_cp(%WamState* %vm, i32 %dealloc.cp)
  ; Pop stack frame
  store i32 %dealloc.top_idx, i32* %dealloc.ss_ptr
  br label %dealloc.done

dealloc.done:
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').

wam_llvm_case('do_call',
'  ; op1 = label index, op2 = arity
  %call.label = trunc i64 %op1 to i32
  %call.target_pc = call i32 @wam_label_pc(%WamState* %vm, i32 %call.label)
  %call.valid = icmp sge i32 %call.target_pc, 0
  br i1 %call.valid, label %call.go, label %call.fail

call.go:
  ; Save continuation
  %call.pc = call i32 @wam_get_pc(%WamState* %vm)
  %call.next = add i32 %call.pc, 1
  call void @wam_set_cp(%WamState* %vm, i32 %call.next)
  call void @wam_set_pc(%WamState* %vm, i32 %call.target_pc)
  ret i1 true

call.fail:
  ret i1 false').

wam_llvm_case('do_execute',
'  ; op1 = label index
  %exec.label = trunc i64 %op1 to i32
  %exec.target_pc = call i32 @wam_label_pc(%WamState* %vm, i32 %exec.label)
  %exec.valid = icmp sge i32 %exec.target_pc, 0
  br i1 %exec.valid, label %exec.go, label %exec.fail

exec.go:
  call void @wam_set_pc(%WamState* %vm, i32 %exec.target_pc)
  ret i1 true

exec.fail:
  ret i1 false').

wam_llvm_case('proceed',
'  ; Return to continuation or halt
  %proc.cp = call i32 @wam_get_cp(%WamState* %vm)
  %proc.is_halt = icmp eq i32 %proc.cp, 0
  br i1 %proc.is_halt, label %proc.halt, label %proc.return

proc.halt:
  call void @wam_set_halted(%WamState* %vm, i1 true)
  ret i1 true

proc.return:
  call void @wam_set_pc(%WamState* %vm, i32 %proc.cp)
  call void @wam_set_cp(%WamState* %vm, i32 0)
  ret i1 true').

wam_llvm_case('builtin_call',
'  ; op1 = builtin op id, op2 = arity
  %bi.op = trunc i64 %op1 to i32
  %bi.arity = trunc i64 %op2 to i32
  %bi.result = call i1 @execute_builtin(%WamState* %vm, i32 %bi.op, i32 %bi.arity)
  br i1 %bi.result, label %bi.ok, label %bi.fail

bi.ok:
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true

bi.fail:
  ret i1 false').

% --- Choice Point Instructions ---

wam_llvm_case('try_me_else',
'  ; op1 = label index for alternative
  %tme.label = trunc i64 %op1 to i32
  %tme.next_pc = call i32 @wam_label_pc(%WamState* %vm, i32 %tme.label)
  ; Push choice point
  %tme.cpn_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 13
  %tme.cpn = load i32, i32* %tme.cpn_ptr
  %tme.cps_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 12
  %tme.cps = load %ChoicePoint*, %ChoicePoint** %tme.cps_ptr
  %tme.cp_slot = getelementptr %ChoicePoint, %ChoicePoint* %tme.cps, i32 %tme.cpn
  ; Set next_pc
  %tme.npc_ptr = getelementptr %ChoicePoint, %ChoicePoint* %tme.cp_slot, i32 0, i32 0
  store i32 %tme.next_pc, i32* %tme.npc_ptr
  ; Save registers (copy 32 x %Value)
  %tme.dst_regs = getelementptr %ChoicePoint, %ChoicePoint* %tme.cp_slot, i32 0, i32 1, i32 0
  %tme.src_regs = getelementptr %WamState, %WamState* %vm, i32 0, i32 1, i32 0
  %tme.dst_raw = bitcast %Value* %tme.dst_regs to i8*
  %tme.src_raw = bitcast %Value* %tme.src_regs to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %tme.dst_raw, i8* %tme.src_raw, i64 512, i1 false)
  ; Save trail mark
  %tme.ts_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 9
  %tme.ts = load i32, i32* %tme.ts_ptr
  %tme.tm_ptr = getelementptr %ChoicePoint, %ChoicePoint* %tme.cp_slot, i32 0, i32 2
  store i32 %tme.ts, i32* %tme.tm_ptr
  ; Save cp
  %tme.saved_cp = call i32 @wam_get_cp(%WamState* %vm)
  %tme.scp_ptr = getelementptr %ChoicePoint, %ChoicePoint* %tme.cp_slot, i32 0, i32 3
  store i32 %tme.saved_cp, i32* %tme.scp_ptr
  ; Increment choice point count
  %tme.new_cpn = add i32 %tme.cpn, 1
  store i32 %tme.new_cpn, i32* %tme.cpn_ptr
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').

wam_llvm_case('retry_me_else',
'  ; op1 = label index for next alternative
  %rme.label = trunc i64 %op1 to i32
  %rme.next_pc = call i32 @wam_label_pc(%WamState* %vm, i32 %rme.label)
  ; Update top choice point next_pc
  %rme.cpn_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 13
  %rme.cpn = load i32, i32* %rme.cpn_ptr
  %rme.top_idx = sub i32 %rme.cpn, 1
  %rme.cps_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 12
  %rme.cps = load %ChoicePoint*, %ChoicePoint** %rme.cps_ptr
  %rme.top = getelementptr %ChoicePoint, %ChoicePoint* %rme.cps, i32 %rme.top_idx
  %rme.npc_ptr = getelementptr %ChoicePoint, %ChoicePoint* %rme.top, i32 0, i32 0
  store i32 %rme.next_pc, i32* %rme.npc_ptr
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').

wam_llvm_case('trust_me',
'  ; Pop top choice point
  %tm.cpn_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 13
  %tm.cpn = load i32, i32* %tm.cpn_ptr
  %tm.new_cpn = sub i32 %tm.cpn, 1
  store i32 %tm.new_cpn, i32* %tm.cpn_ptr
  call void @wam_inc_pc(%WamState* %vm)
  ret i1 true').

% ============================================================================
% PHASE 3: Helper predicates → LLVM functions
% ============================================================================

%% compile_wam_helpers_to_llvm(+Options, -LLVMCode)
%  Generates LLVM IR for WAM runtime helpers.
compile_wam_helpers_to_llvm(_Options, LLVMCode) :-
    compile_backtrack_to_llvm(BacktrackCode),
    compile_unwind_trail_to_llvm(UnwindCode),
    compile_execute_builtin_to_llvm(BuiltinCode),
    compile_eval_arith_to_llvm(ArithCode),
    atomic_list_concat([
        BacktrackCode, '\n\n',
        UnwindCode, '\n\n',
        BuiltinCode, '\n\n',
        ArithCode
    ], LLVMCode).

compile_backtrack_to_llvm(Code) :-
    Code = 'define i1 @backtrack(%WamState* %vm) {
entry:
  %cpn_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 13
  %cpn = load i32, i32* %cpn_ptr
  %has_cp = icmp sgt i32 %cpn, 0
  br i1 %has_cp, label %restore, label %fail

restore:
  %top_idx = sub i32 %cpn, 1
  %cps_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 12
  %cps = load %ChoicePoint*, %ChoicePoint** %cps_ptr
  %top = getelementptr %ChoicePoint, %ChoicePoint* %cps, i32 %top_idx

  ; Get trail mark and unwind
  %tm_ptr = getelementptr %ChoicePoint, %ChoicePoint* %top, i32 0, i32 2
  %tm = load i32, i32* %tm_ptr
  call void @unwind_trail(%WamState* %vm, i32 %tm)

  ; Restore registers
  %dst_regs = getelementptr %WamState, %WamState* %vm, i32 0, i32 1, i32 0
  %src_regs = getelementptr %ChoicePoint, %ChoicePoint* %top, i32 0, i32 1, i32 0
  %dst_raw = bitcast %Value* %dst_regs to i8*
  %src_raw = bitcast %Value* %src_regs to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst_raw, i8* %src_raw, i64 512, i1 false)

  ; Restore PC from choice point next_pc
  %npc_ptr = getelementptr %ChoicePoint, %ChoicePoint* %top, i32 0, i32 0
  %npc = load i32, i32* %npc_ptr
  call void @wam_set_pc(%WamState* %vm, i32 %npc)

  ; Restore CP
  %scp_ptr = getelementptr %ChoicePoint, %ChoicePoint* %top, i32 0, i32 3
  %scp = load i32, i32* %scp_ptr
  call void @wam_set_cp(%WamState* %vm, i32 %scp)

  ; Clear halted
  call void @wam_set_halted(%WamState* %vm, i1 false)

  ret i1 true

fail:
  ret i1 false
}'.

compile_unwind_trail_to_llvm(Code) :-
    Code = 'define void @unwind_trail(%WamState* %vm, i32 %saved_mark) {
entry:
  %ts_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 9
  %ts = load i32, i32* %ts_ptr
  %need_unwind = icmp sgt i32 %ts, %saved_mark
  br i1 %need_unwind, label %loop, label %done

loop:
  %cur_ts = load i32, i32* %ts_ptr
  %still_more = icmp sgt i32 %cur_ts, %saved_mark
  br i1 %still_more, label %unwind_one, label %done

unwind_one:
  %new_ts = sub i32 %cur_ts, 1
  store i32 %new_ts, i32* %ts_ptr
  ; Load trail entry
  %trail_arr_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 8
  %trail_arr = load %TrailEntry*, %TrailEntry** %trail_arr_ptr
  %entry = getelementptr %TrailEntry, %TrailEntry* %trail_arr, i32 %new_ts
  ; Restore old value to register
  %reg_ptr = getelementptr %TrailEntry, %TrailEntry* %entry, i32 0, i32 0
  %reg_idx = load i32, i32* %reg_ptr
  %old_val_ptr = getelementptr %TrailEntry, %TrailEntry* %entry, i32 0, i32 1
  %old_val = load %Value, %Value* %old_val_ptr
  call void @wam_set_reg(%WamState* %vm, i32 %reg_idx, %Value %old_val)
  br label %loop

done:
  ret void
}'.

compile_execute_builtin_to_llvm(Code) :-
    Code = '; Execute builtin operations
; Dispatches on integer op codes:
;   0 = is/2, 1 = >/2, 2 = </2, 3 = >=/2, 4 = =</2
;   5 = =:=/2, 6 = =\\=/2, 7 = ==/2, 8 = true/0, 9 = fail/0
;   10 = !/0, 11 = write/1, 12 = nl/0
;   13 = atom/1, 14 = integer/1, 15 = float/1, 16 = number/1
;   17 = compound/1, 18 = var/1, 19 = nonvar/1, 20 = is_list/1
define i1 @execute_builtin(%WamState* %vm, i32 %op, i32 %arity) {
entry:
  switch i32 %op, label %unknown [
    i32 0, label %builtin_is
    i32 1, label %builtin_gt
    i32 2, label %builtin_lt
    i32 3, label %builtin_ge
    i32 4, label %builtin_le
    i32 5, label %builtin_arith_eq
    i32 6, label %builtin_arith_ne
    i32 7, label %builtin_eq
    i32 8, label %builtin_true
    i32 9, label %builtin_fail
    i32 10, label %builtin_cut
    i32 14, label %builtin_integer_check
    i32 18, label %builtin_var
    i32 19, label %builtin_nonvar
  ]

builtin_is:
  ; A1 is result, A2 is expression — evaluate A2 and unify with A1
  %is.a2 = call %Value @wam_get_reg(%WamState* %vm, i32 1)
  %is.tag = call i32 @value_tag(%Value %is.a2)
  %is.is_int = icmp eq i32 %is.tag, 1
  br i1 %is.is_int, label %is.bind, label %is.fail

is.bind:
  %is.a1 = call %Value @wam_get_reg(%WamState* %vm, i32 0)
  %is.a1_unb = call i1 @value_is_unbound(%Value %is.a1)
  br i1 %is.a1_unb, label %is.do_bind, label %is.check_eq

is.do_bind:
  call void @wam_trail_binding(%WamState* %vm, i32 0)
  call void @wam_set_reg(%WamState* %vm, i32 0, %Value %is.a2)
  ret i1 true

is.check_eq:
  %is.eq = call i1 @value_equals(%Value %is.a1, %Value %is.a2)
  ret i1 %is.eq

is.fail:
  ret i1 false

builtin_gt:
  %gt.a1 = call %Value @wam_get_reg(%WamState* %vm, i32 0)
  %gt.a2 = call %Value @wam_get_reg(%WamState* %vm, i32 1)
  %gt.v1 = call i64 @value_payload(%Value %gt.a1)
  %gt.v2 = call i64 @value_payload(%Value %gt.a2)
  %gt.r = icmp sgt i64 %gt.v1, %gt.v2
  ret i1 %gt.r

builtin_lt:
  %lt.a1 = call %Value @wam_get_reg(%WamState* %vm, i32 0)
  %lt.a2 = call %Value @wam_get_reg(%WamState* %vm, i32 1)
  %lt.v1 = call i64 @value_payload(%Value %lt.a1)
  %lt.v2 = call i64 @value_payload(%Value %lt.a2)
  %lt.r = icmp slt i64 %lt.v1, %lt.v2
  ret i1 %lt.r

builtin_ge:
  %ge.a1 = call %Value @wam_get_reg(%WamState* %vm, i32 0)
  %ge.a2 = call %Value @wam_get_reg(%WamState* %vm, i32 1)
  %ge.v1 = call i64 @value_payload(%Value %ge.a1)
  %ge.v2 = call i64 @value_payload(%Value %ge.a2)
  %ge.r = icmp sge i64 %ge.v1, %ge.v2
  ret i1 %ge.r

builtin_le:
  %le.a1 = call %Value @wam_get_reg(%WamState* %vm, i32 0)
  %le.a2 = call %Value @wam_get_reg(%WamState* %vm, i32 1)
  %le.v1 = call i64 @value_payload(%Value %le.a1)
  %le.v2 = call i64 @value_payload(%Value %le.a2)
  %le.r = icmp sle i64 %le.v1, %le.v2
  ret i1 %le.r

builtin_arith_eq:
  %aeq.a1 = call %Value @wam_get_reg(%WamState* %vm, i32 0)
  %aeq.a2 = call %Value @wam_get_reg(%WamState* %vm, i32 1)
  %aeq.v1 = call i64 @value_payload(%Value %aeq.a1)
  %aeq.v2 = call i64 @value_payload(%Value %aeq.a2)
  %aeq.r = icmp eq i64 %aeq.v1, %aeq.v2
  ret i1 %aeq.r

builtin_arith_ne:
  %ane.a1 = call %Value @wam_get_reg(%WamState* %vm, i32 0)
  %ane.a2 = call %Value @wam_get_reg(%WamState* %vm, i32 1)
  %ane.v1 = call i64 @value_payload(%Value %ane.a1)
  %ane.v2 = call i64 @value_payload(%Value %ane.a2)
  %ane.r = icmp ne i64 %ane.v1, %ane.v2
  ret i1 %ane.r

builtin_eq:
  %eq.a1 = call %Value @wam_get_reg(%WamState* %vm, i32 0)
  %eq.a2 = call %Value @wam_get_reg(%WamState* %vm, i32 1)
  %eq.r = call i1 @value_equals(%Value %eq.a1, %Value %eq.a2)
  ret i1 %eq.r

builtin_true:
  ret i1 true

builtin_fail:
  ret i1 false

builtin_cut:
  ; Clear all choice points
  %cut.cpn_ptr = getelementptr %WamState, %WamState* %vm, i32 0, i32 13
  store i32 0, i32* %cut.cpn_ptr
  ret i1 true

builtin_integer_check:
  %ic.a1 = call %Value @wam_get_reg(%WamState* %vm, i32 0)
  %ic.tag = call i32 @value_tag(%Value %ic.a1)
  %ic.r = icmp eq i32 %ic.tag, 1
  ret i1 %ic.r

builtin_var:
  %var.a1 = call %Value @wam_get_reg(%WamState* %vm, i32 0)
  %var.r = call i1 @value_is_unbound(%Value %var.a1)
  ret i1 %var.r

builtin_nonvar:
  %nv.a1 = call %Value @wam_get_reg(%WamState* %vm, i32 0)
  %nv.r = call i1 @value_is_unbound(%Value %nv.a1)
  %nv.nr = xor i1 %nv.r, true
  ret i1 %nv.nr

unknown:
  ret i1 false
}'.

compile_eval_arith_to_llvm(Code) :-
    Code = '; Evaluate arithmetic expression
; Takes a Value, returns the integer payload.
; For compound ops (tag=3), extracts functor and recursively evaluates args.
; For register refs (tag=6 unbound with name starting A/X), dereferences.
define i64 @eval_arith(%WamState* %vm, %Value %expr) {
entry:
  %tag = call i32 @value_tag(%Value %expr)
  switch i32 %tag, label %fail [
    i32 1, label %return_int
    i32 2, label %return_float_as_int
    i32 3, label %compound_arith
  ]

return_int:
  %val = call i64 @value_payload(%Value %expr)
  ret i64 %val

return_float_as_int:
  %fbits = call i64 @value_payload(%Value %expr)
  %fval = bitcast i64 %fbits to double
  %ival = fptosi double %fval to i64
  ret i64 %ival

compound_arith:
  ; Compound: payload is pointer to %Compound (functor, arity, args)
  %cp_bits = call i64 @value_payload(%Value %expr)
  %cp_ptr = inttoptr i64 %cp_bits to %Compound*
  ; Load arity
  %arity_ptr = getelementptr %Compound, %Compound* %cp_ptr, i32 0, i32 1
  %arity = load i32, i32* %arity_ptr
  ; Load args array pointer
  %args_ptr = getelementptr %Compound, %Compound* %cp_ptr, i32 0, i32 2
  %args = load %Value*, %Value** %args_ptr
  ; Load functor pointer for comparison
  %fn_ptr_ptr = getelementptr %Compound, %Compound* %cp_ptr, i32 0, i32 0
  %fn_ptr = load i8*, i8** %fn_ptr_ptr
  ; Binary: evaluate both args
  %is_binary = icmp eq i32 %arity, 2
  br i1 %is_binary, label %eval_binary, label %check_unary

eval_binary:
  %arg0_ptr = getelementptr %Value, %Value* %args, i32 0
  %arg0 = load %Value, %Value* %arg0_ptr
  %arg1_ptr = getelementptr %Value, %Value* %args, i32 1
  %arg1 = load %Value, %Value* %arg1_ptr
  %a = call i64 @eval_arith(%WamState* %vm, %Value %arg0)
  %b = call i64 @eval_arith(%WamState* %vm, %Value %arg1)
  ; Dispatch on functor: check first char for +, -, *, /
  %fn_first = load i8, i8* %fn_ptr
  switch i8 %fn_first, label %fail [
    i8 43, label %do_add     ; \'+\'
    i8 45, label %do_sub     ; \'-\'
    i8 42, label %do_mul     ; \'*\'
    i8 47, label %do_div     ; \'/\'
  ]

do_add:
  %add_r = add i64 %a, %b
  ret i64 %add_r

do_sub:
  %sub_r = sub i64 %a, %b
  ret i64 %sub_r

do_mul:
  %mul_r = mul i64 %a, %b
  ret i64 %mul_r

do_div:
  %div_zero = icmp eq i64 %b, 0
  br i1 %div_zero, label %fail, label %do_div_ok

do_div_ok:
  %div_r = sdiv i64 %a, %b
  ret i64 %div_r

check_unary:
  %is_unary = icmp eq i32 %arity, 1
  br i1 %is_unary, label %eval_unary, label %fail

eval_unary:
  %u_arg_ptr = getelementptr %Value, %Value* %args, i32 0
  %u_arg = load %Value, %Value* %u_arg_ptr
  %u_val = call i64 @eval_arith(%WamState* %vm, %Value %u_arg)
  %u_fn_first = load i8, i8* %fn_ptr
  %u_is_neg = icmp eq i8 %u_fn_first, 45  ; \'-\'
  br i1 %u_is_neg, label %do_neg, label %fail

do_neg:
  %neg_r = sub i64 0, %u_val
  ret i64 %neg_r

fail:
  ret i64 0
}'.

% ============================================================================
% ASSEMBLY: Combine Phase 2 + Phase 3 into complete runtime
% ============================================================================

%% compile_wam_runtime_to_llvm(+Options, -LLVMCode)
%  Generates the combined step function + all helper functions.
compile_wam_runtime_to_llvm(Options, LLVMCode) :-
    compile_step_wam_to_llvm(Options, StepCode),
    compile_wam_helpers_to_llvm(Options, HelpersCode),
    atomic_list_concat([StepCode, '\n\n', HelpersCode], LLVMCode).

% ============================================================================
% PHASE 4: WAM instructions → LLVM struct literals
% ============================================================================

%% wam_instruction_to_llvm_literal(+WamInstr, -LLVMLiteral)
%  Converts a WAM instruction term to an LLVM %Instruction struct literal.

wam_instruction_to_llvm_literal(get_constant(C, Ai), Lit) :-
    llvm_pack_value(C, PackedVal),
    reg_name_to_index(Ai, RegIdx),
    format(atom(Lit), '{ i32 0, i64 ~w, i64 ~w }', [PackedVal, RegIdx]).
wam_instruction_to_llvm_literal(get_variable(Xn, Ai), Lit) :-
    reg_name_to_index(Xn, XnIdx),
    reg_name_to_index(Ai, AiIdx),
    format(atom(Lit), '{ i32 1, i64 ~w, i64 ~w }', [XnIdx, AiIdx]).
wam_instruction_to_llvm_literal(get_value(Xn, Ai), Lit) :-
    reg_name_to_index(Xn, XnIdx),
    reg_name_to_index(Ai, AiIdx),
    format(atom(Lit), '{ i32 2, i64 ~w, i64 ~w }', [XnIdx, AiIdx]).
wam_instruction_to_llvm_literal(get_structure(F, Ai), Lit) :-
    reg_name_to_index(Ai, AiIdx),
    format(atom(Lit), '{ i32 3, i64 0, i64 ~w } ; get_structure ~w', [AiIdx, F]).
wam_instruction_to_llvm_literal(get_list(Ai), Lit) :-
    reg_name_to_index(Ai, AiIdx),
    format(atom(Lit), '{ i32 4, i64 ~w, i64 0 }', [AiIdx]).
wam_instruction_to_llvm_literal(unify_variable(Xn), Lit) :-
    reg_name_to_index(Xn, XnIdx),
    format(atom(Lit), '{ i32 5, i64 ~w, i64 0 }', [XnIdx]).
wam_instruction_to_llvm_literal(unify_value(Xn), Lit) :-
    reg_name_to_index(Xn, XnIdx),
    format(atom(Lit), '{ i32 6, i64 ~w, i64 0 }', [XnIdx]).
wam_instruction_to_llvm_literal(unify_constant(C), Lit) :-
    llvm_pack_value(C, PackedVal),
    format(atom(Lit), '{ i32 7, i64 ~w, i64 0 }', [PackedVal]).

wam_instruction_to_llvm_literal(put_constant(C, Ai), Lit) :-
    llvm_pack_value(C, PackedVal),
    reg_name_to_index(Ai, RegIdx),
    format(atom(Lit), '{ i32 8, i64 ~w, i64 ~w }', [PackedVal, RegIdx]).
wam_instruction_to_llvm_literal(put_variable(Xn, Ai), Lit) :-
    reg_name_to_index(Xn, XnIdx),
    reg_name_to_index(Ai, AiIdx),
    format(atom(Lit), '{ i32 9, i64 ~w, i64 ~w }', [XnIdx, AiIdx]).
wam_instruction_to_llvm_literal(put_value(Xn, Ai), Lit) :-
    reg_name_to_index(Xn, XnIdx),
    reg_name_to_index(Ai, AiIdx),
    format(atom(Lit), '{ i32 10, i64 ~w, i64 ~w }', [XnIdx, AiIdx]).
wam_instruction_to_llvm_literal(put_structure(F, Ai), Lit) :-
    reg_name_to_index(Ai, AiIdx),
    format(atom(Lit), '{ i32 11, i64 0, i64 ~w } ; put_structure ~w', [AiIdx, F]).
wam_instruction_to_llvm_literal(put_list(Ai), Lit) :-
    reg_name_to_index(Ai, AiIdx),
    format(atom(Lit), '{ i32 12, i64 ~w, i64 0 }', [AiIdx]).
wam_instruction_to_llvm_literal(set_variable(Xn), Lit) :-
    reg_name_to_index(Xn, XnIdx),
    format(atom(Lit), '{ i32 13, i64 ~w, i64 0 }', [XnIdx]).
wam_instruction_to_llvm_literal(set_value(Xn), Lit) :-
    reg_name_to_index(Xn, XnIdx),
    format(atom(Lit), '{ i32 14, i64 ~w, i64 0 }', [XnIdx]).
wam_instruction_to_llvm_literal(set_constant(C), Lit) :-
    llvm_pack_value(C, PackedVal),
    format(atom(Lit), '{ i32 15, i64 ~w, i64 0 }', [PackedVal]).

wam_instruction_to_llvm_literal(allocate, '{ i32 16, i64 0, i64 0 }').
wam_instruction_to_llvm_literal(deallocate, '{ i32 17, i64 0, i64 0 }').
wam_instruction_to_llvm_literal(call(P, N), Lit) :-
    format(atom(Lit), '{ i32 18, i64 0, i64 ~w } ; call ~w', [N, P]).
wam_instruction_to_llvm_literal(execute(P), Lit) :-
    format(atom(Lit), '{ i32 19, i64 0, i64 0 } ; execute ~w', [P]).
wam_instruction_to_llvm_literal(proceed, '{ i32 20, i64 0, i64 0 }').
wam_instruction_to_llvm_literal(builtin_call(Op, N), Lit) :-
    builtin_op_to_id(Op, OpId),
    format(atom(Lit), '{ i32 21, i64 ~w, i64 ~w } ; builtin_call ~w', [OpId, N, Op]).

wam_instruction_to_llvm_literal(try_me_else(Label), Lit) :-
    format(atom(Lit), '{ i32 22, i64 0, i64 0 } ; try_me_else ~w', [Label]).
wam_instruction_to_llvm_literal(retry_me_else(Label), Lit) :-
    format(atom(Lit), '{ i32 23, i64 0, i64 0 } ; retry_me_else ~w', [Label]).
wam_instruction_to_llvm_literal(trust_me, '{ i32 24, i64 0, i64 0 }').

wam_instruction_to_llvm_literal(switch_on_constant(_), '; switch_on_constant handled via labels').
wam_instruction_to_llvm_literal(switch_on_structure(_), '; switch_on_structure handled via labels').
wam_instruction_to_llvm_literal(switch_on_constant_a2(_), '; switch_on_constant_a2 handled via labels').

% Label pseudo-instruction
wam_instruction_to_llvm_literal(label(L), Lit) :-
    format(atom(Lit), '; label: ~w', [L]).

% Fallback
wam_instruction_to_llvm_literal(Instr, Lit) :-
    format(atom(Lit), '; TODO: ~w', [Instr]).

% --- Atom table (string interning) ---
% Assigns unique sequential integer IDs to atoms. Two atoms with the
% same name always get the same ID; different names always get different IDs.
% This avoids hash collisions that would cause silent correctness bugs.

:- dynamic atom_table_entry/2.   % atom_table_entry(AtomName, Id)
:- dynamic atom_table_next_id/1. % atom_table_next_id(NextId)
atom_table_next_id(1).           % Start from 1; 0 reserved for empty

%% intern_atom(+AtomName, -Id)
%  Returns the unique integer ID for AtomName, allocating a new one if needed.
intern_atom(AtomName, Id) :-
    (   atom_table_entry(AtomName, Id)
    ->  true
    ;   retract(atom_table_next_id(Id)),
        NextId is Id + 1,
        assert(atom_table_next_id(NextId)),
        assert(atom_table_entry(AtomName, Id))
    ).

% --- Value packing helpers ---

llvm_pack_value(atom(A), Packed) :- !,
    intern_atom(A, Packed).
llvm_pack_value(integer(I), I) :- !.
llvm_pack_value(N, N) :- integer(N), !.
llvm_pack_value(N, Packed) :- float(N), !, Packed is truncate(N).
llvm_pack_value(A, Packed) :- atom(A), !, intern_atom(A, Packed).
llvm_pack_value(_, 0).

% --- Builtin op name → integer ID mapping ---

builtin_op_to_id('is/2', 0).
builtin_op_to_id('>/2', 1).
builtin_op_to_id('</2', 2).
builtin_op_to_id('>=/2', 3).
builtin_op_to_id('=</2', 4).
builtin_op_to_id('=:=/2', 5).
builtin_op_to_id('=\\=/2', 6).
builtin_op_to_id('==/2', 7).
builtin_op_to_id('true/0', 8).
builtin_op_to_id('fail/0', 9).
builtin_op_to_id('!/0', 10).
builtin_op_to_id('write/1', 11).
builtin_op_to_id('nl/0', 12).
builtin_op_to_id('atom/1', 13).
builtin_op_to_id('integer/1', 14).
builtin_op_to_id('float/1', 15).
builtin_op_to_id('number/1', 16).
builtin_op_to_id('compound/1', 17).
builtin_op_to_id('var/1', 18).
builtin_op_to_id('nonvar/1', 19).
builtin_op_to_id('is_list/1', 20).
builtin_op_to_id(_, 99).  % Unknown

% ============================================================================
% WAM line parser → LLVM struct literals (from WAM assembly text)
% ============================================================================

%% compile_wam_predicate_to_llvm(+Pred/Arity, +WamCode, +Options, -LLVMCode)
%  Takes WAM instruction output and produces LLVM IR with instruction
%  array and label table as global constants.
compile_wam_predicate_to_llvm(Pred/Arity, WamCode, _Options, LLVMCode) :-
    atom_string(Pred, PredStr),
    atom_string(WamCode, WamStr),
    split_string(WamStr, "\n", "", Lines),
    wam_lines_to_llvm(Lines, 0, LLVMLiterals, LabelEntries),
    length(LLVMLiterals, InstrCount),
    length(LabelEntries, LabelCount),
    % Build instruction array entries
    maplist([Lit, Entry]>>(format(atom(Entry), '  ~w', [Lit])), LLVMLiterals, Entries),
    atomic_list_concat(Entries, ',\n', EntriesStr),
    % Build label array entries
    maplist([_-Idx, Entry]>>(format(atom(Entry), '  i32 ~w', [Idx])), LabelEntries, LabelRows),
    (   LabelRows == []
    ->  LabelsStr = "  i32 0"
    ;   atomic_list_concat(LabelRows, ',\n', LabelsStr)
    ),
    % Build arg setup
    build_llvm_arg_setup(Arity, ArgSetup),
    build_llvm_param_list(Arity, ParamList),
    format(atom(LLVMCode),
'; WAM-compiled predicate: ~w/~w
@~w_code = private constant [~w x %Instruction] [
~w
]

@~w_labels = private constant [~w x i32] [
~w
]

define i1 @~w(~w) {
entry:
  %vm = call %WamState* @wam_state_new(
    %Instruction* getelementptr ([~w x %Instruction], [~w x %Instruction]* @~w_code, i32 0, i32 0),
    i32 ~w,
    i32* getelementptr ([~w x i32], [~w x i32]* @~w_labels, i32 0, i32 0),
    i32 ~w)
~w
  %result = call i1 @run_loop(%WamState* %vm)
  ret i1 %result
}
', [PredStr, Arity,
    PredStr, InstrCount, EntriesStr,
    PredStr, LabelCount, LabelsStr,
    PredStr, ParamList,
    InstrCount, InstrCount, PredStr, InstrCount,
    LabelCount, LabelCount, PredStr, LabelCount,
    ArgSetup]).

%% wam_lines_to_llvm(+Lines, +PC, -LLVMLits, -LabelEntries)
%  Two-pass approach: first collect all labels and raw instruction parts,
%  then generate LLVM literals with resolved label indices.
wam_lines_to_llvm(Lines, StartPC, LLVMLits, LabelEntries) :-
    % Pass 1: collect labels and raw instruction parts
    wam_lines_pass1(Lines, StartPC, RawInstrs, LabelEntries),
    % Build label name → index mapping (position in label array)
    build_label_index_map(LabelEntries, LabelMap),
    % Pass 2: generate LLVM literals with label resolution
    maplist(resolve_llvm_literal(LabelMap), RawInstrs, LLVMLits).

%% wam_lines_pass1(+Lines, +PC, -RawInstrs, -Labels)
%  First pass: separate labels from instructions, track PC.
wam_lines_pass1([], _, [], []).
wam_lines_pass1([Line|Rest], PC, RawInstrs, Labels) :-
    split_string(Line, " \t", " \t", Parts),
    delete(Parts, "", CleanParts),
    (   CleanParts == []
    ->  wam_lines_pass1(Rest, PC, RawInstrs, Labels)
    ;   CleanParts = [First|_],
        (   sub_string(First, _, 1, 0, ":")
        ->  sub_string(First, 0, _, 1, LabelName),
            Labels = [LabelName-PC | RestLabels],
            wam_lines_pass1(Rest, PC, RawInstrs, RestLabels)
        ;   RawInstrs = [CleanParts | RestInstrs],
            NPC is PC + 1,
            wam_lines_pass1(Rest, NPC, RestInstrs, Labels)
        )
    ).

%% build_label_index_map(+LabelEntries, -LabelMap)
%  Creates an assoc mapping label names to their index in the label array.
build_label_index_map(LabelEntries, LabelMap) :-
    length(LabelEntries, _),
    foldl(add_label_entry, LabelEntries, 0-[], _-LabelMap).

add_label_entry(Name-_PC, Idx-Map, NextIdx-[Name-Idx|Map]) :-
    NextIdx is Idx + 1.

%% resolve_llvm_literal(+LabelMap, +Parts, -LLVMLit)
%  Second pass: generate LLVM literal with resolved label indices.
resolve_llvm_literal(LabelMap, Parts, LLVMLit) :-
    wam_line_to_llvm_literal_resolved(Parts, LabelMap, LLVMLit).

%% wam_line_to_llvm_literal_resolved(+Parts, +LabelMap, -LLVMLit)
%  Converts parsed WAM instruction text to LLVM %Instruction struct literal,
%  with label names resolved to indices via LabelMap.

%% lookup_label_index(+LabelName, +LabelMap, -Index)
%  Find label index in map, or return 0 if not found.
lookup_label_index(LabelName, LabelMap, Index) :-
    (   member(LabelName-Index, LabelMap)
    ->  true
    ;   Index = 0
    ).

% Instructions that need label resolution:
wam_line_to_llvm_literal_resolved(["call", P, N], LabelMap, Lit) :- !,
    clean_comma(P, CP), clean_comma(N, CN),
    (   number_string(Arity, CN) -> true ; Arity = 0 ),
    atom_string(CP, CPAtom),
    lookup_label_index(CPAtom, LabelMap, LabelIdx),
    format(atom(Lit), '%Instruction { i32 18, i64 ~w, i64 ~w } ; call ~w', [LabelIdx, Arity, CP]).
wam_line_to_llvm_literal_resolved(["execute", P], LabelMap, Lit) :- !,
    clean_comma(P, CP),
    atom_string(CP, CPAtom),
    lookup_label_index(CPAtom, LabelMap, LabelIdx),
    format(atom(Lit), '%Instruction { i32 19, i64 ~w, i64 0 } ; execute ~w', [LabelIdx, CP]).
wam_line_to_llvm_literal_resolved(["try_me_else", L], LabelMap, Lit) :- !,
    clean_comma(L, CL),
    atom_string(CL, CLAtom),
    lookup_label_index(CLAtom, LabelMap, LabelIdx),
    format(atom(Lit), '%Instruction { i32 22, i64 ~w, i64 0 } ; try_me_else ~w', [LabelIdx, CL]).
wam_line_to_llvm_literal_resolved(["retry_me_else", L], LabelMap, Lit) :- !,
    clean_comma(L, CL),
    atom_string(CL, CLAtom),
    lookup_label_index(CLAtom, LabelMap, LabelIdx),
    format(atom(Lit), '%Instruction { i32 23, i64 ~w, i64 0 } ; retry_me_else ~w', [LabelIdx, CL]).
% All other instructions: delegate to existing parser (no labels needed)
wam_line_to_llvm_literal_resolved(Parts, _LabelMap, Lit) :-
    wam_line_to_llvm_literal(Parts, Lit).

%% wam_line_to_llvm_literal(+Parts, -LLVMLit)
%  Converts parsed WAM instruction text to LLVM %Instruction struct literal.
%  For non-label-referencing instructions only. Label-referencing instructions
%  are handled by wam_line_to_llvm_literal_resolved/3 above.

wam_line_to_llvm_literal(["get_constant", C, Ai], Lit) :-
    clean_comma(C, CC), clean_comma(Ai, CAi),
    llvm_pack_value_str(CC, PackedVal),
    atom_string(CAi, CAiAtom),
    reg_name_to_index(CAiAtom, RegIdx),
    format(atom(Lit), '%Instruction { i32 0, i64 ~w, i64 ~w }', [PackedVal, RegIdx]).
wam_line_to_llvm_literal(["get_variable", Xn, Ai], Lit) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    atom_string(CXn, CXnAtom), atom_string(CAi, CAiAtom),
    reg_name_to_index(CXnAtom, XnIdx),
    reg_name_to_index(CAiAtom, AiIdx),
    format(atom(Lit), '%Instruction { i32 1, i64 ~w, i64 ~w }', [XnIdx, AiIdx]).
wam_line_to_llvm_literal(["get_value", Xn, Ai], Lit) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    atom_string(CXn, CXnAtom), atom_string(CAi, CAiAtom),
    reg_name_to_index(CXnAtom, XnIdx),
    reg_name_to_index(CAiAtom, AiIdx),
    format(atom(Lit), '%Instruction { i32 2, i64 ~w, i64 ~w }', [XnIdx, AiIdx]).
wam_line_to_llvm_literal(["get_structure", FN, Ai], Lit) :-
    clean_comma(FN, _CFN), clean_comma(Ai, CAi),
    atom_string(CAi, CAiAtom),
    reg_name_to_index(CAiAtom, AiIdx),
    format(atom(Lit), '%Instruction { i32 3, i64 0, i64 ~w }', [AiIdx]).
wam_line_to_llvm_literal(["get_list", Ai], Lit) :-
    clean_comma(Ai, CAi),
    atom_string(CAi, CAiAtom),
    reg_name_to_index(CAiAtom, AiIdx),
    format(atom(Lit), '%Instruction { i32 4, i64 ~w, i64 0 }', [AiIdx]).
wam_line_to_llvm_literal(["unify_variable", Xn], Lit) :-
    clean_comma(Xn, CXn),
    atom_string(CXn, CXnAtom),
    reg_name_to_index(CXnAtom, XnIdx),
    format(atom(Lit), '%Instruction { i32 5, i64 ~w, i64 0 }', [XnIdx]).
wam_line_to_llvm_literal(["unify_value", Xn], Lit) :-
    clean_comma(Xn, CXn),
    atom_string(CXn, CXnAtom),
    reg_name_to_index(CXnAtom, XnIdx),
    format(atom(Lit), '%Instruction { i32 6, i64 ~w, i64 0 }', [XnIdx]).
wam_line_to_llvm_literal(["unify_constant", C], Lit) :-
    clean_comma(C, CC),
    llvm_pack_value_str(CC, PackedVal),
    format(atom(Lit), '%Instruction { i32 7, i64 ~w, i64 0 }', [PackedVal]).

wam_line_to_llvm_literal(["put_constant", C, Ai], Lit) :-
    clean_comma(C, CC), clean_comma(Ai, CAi),
    llvm_pack_value_str(CC, PackedVal),
    atom_string(CAi, CAiAtom),
    reg_name_to_index(CAiAtom, RegIdx),
    format(atom(Lit), '%Instruction { i32 8, i64 ~w, i64 ~w }', [PackedVal, RegIdx]).
wam_line_to_llvm_literal(["put_variable", Xn, Ai], Lit) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    atom_string(CXn, CXnAtom), atom_string(CAi, CAiAtom),
    reg_name_to_index(CXnAtom, XnIdx),
    reg_name_to_index(CAiAtom, AiIdx),
    format(atom(Lit), '%Instruction { i32 9, i64 ~w, i64 ~w }', [XnIdx, AiIdx]).
wam_line_to_llvm_literal(["put_value", Xn, Ai], Lit) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    atom_string(CXn, CXnAtom), atom_string(CAi, CAiAtom),
    reg_name_to_index(CXnAtom, XnIdx),
    reg_name_to_index(CAiAtom, AiIdx),
    format(atom(Lit), '%Instruction { i32 10, i64 ~w, i64 ~w }', [XnIdx, AiIdx]).
wam_line_to_llvm_literal(["put_structure", FN, Ai], Lit) :-
    clean_comma(FN, _CFN), clean_comma(Ai, CAi),
    atom_string(CAi, CAiAtom),
    reg_name_to_index(CAiAtom, AiIdx),
    format(atom(Lit), '%Instruction { i32 11, i64 0, i64 ~w }', [AiIdx]).
wam_line_to_llvm_literal(["put_list", Ai], Lit) :-
    clean_comma(Ai, CAi),
    atom_string(CAi, CAiAtom),
    reg_name_to_index(CAiAtom, AiIdx),
    format(atom(Lit), '%Instruction { i32 12, i64 ~w, i64 0 }', [AiIdx]).
wam_line_to_llvm_literal(["set_variable", Xn], Lit) :-
    clean_comma(Xn, CXn),
    atom_string(CXn, CXnAtom),
    reg_name_to_index(CXnAtom, XnIdx),
    format(atom(Lit), '%Instruction { i32 13, i64 ~w, i64 0 }', [XnIdx]).
wam_line_to_llvm_literal(["set_value", Xn], Lit) :-
    clean_comma(Xn, CXn),
    atom_string(CXn, CXnAtom),
    reg_name_to_index(CXnAtom, XnIdx),
    format(atom(Lit), '%Instruction { i32 14, i64 ~w, i64 0 }', [XnIdx]).
wam_line_to_llvm_literal(["set_constant", C], Lit) :-
    clean_comma(C, CC),
    llvm_pack_value_str(CC, PackedVal),
    format(atom(Lit), '%Instruction { i32 15, i64 ~w, i64 0 }', [PackedVal]).

wam_line_to_llvm_literal(["allocate"], '%Instruction { i32 16, i64 0, i64 0 }').
wam_line_to_llvm_literal(["deallocate"], '%Instruction { i32 17, i64 0, i64 0 }').
% call, execute, try_me_else, retry_me_else are handled by
% wam_line_to_llvm_literal_resolved/3 (label resolution required).
wam_line_to_llvm_literal(["proceed"], '%Instruction { i32 20, i64 0, i64 0 }').
wam_line_to_llvm_literal(["builtin_call", Op, N], Lit) :-
    clean_comma(Op, COp), clean_comma(N, CN),
    (   number_string(Num, CN) -> true ; Num = 0 ),
    atom_string(COp, COpAtom),
    builtin_op_to_id(COpAtom, OpId),
    format(atom(Lit), '%Instruction { i32 21, i64 ~w, i64 ~w }', [OpId, Num]).
wam_line_to_llvm_literal(["trust_me"], '%Instruction { i32 24, i64 0, i64 0 }').

wam_line_to_llvm_literal(["switch_on_constant"|_], '; switch_on_constant — handled via labels').
wam_line_to_llvm_literal(["switch_on_structure"|_], '; switch_on_structure — handled via labels').

wam_line_to_llvm_literal(Parts, Lit) :-
    atomic_list_concat(Parts, " ", Line),
    format(atom(Lit), '; TODO: ~w', [Line]).

% --- Utility predicates ---

clean_comma(S, Clean) :-
    (   sub_string(S, Before, 1, 0, ",")
    ->  sub_string(S, 0, Before, 1, Clean)
    ;   Clean = S
    ).

llvm_pack_value_str(Str, Packed) :-
    (   number_string(N, Str)
    ->  Packed = N
    ;   atom_string(A, Str),
        llvm_pack_value(atom(A), Packed)
    ).

build_llvm_param_list(0, "%WamState* %vm") :- !.
build_llvm_param_list(Arity, ParamList) :-
    numlist(1, Arity, Indices),
    maplist([I, S]>>(format(atom(S), "%Value %a~w", [I])), Indices, Parts),
    atomic_list_concat(['%WamState* %vm'|Parts], ', ', ParamList).

build_llvm_arg_setup(0, "") :- !.
build_llvm_arg_setup(Arity, Setup) :-
    numlist(1, Arity, Indices),
    maplist([I, S]>>(
        RegIdx is I - 1,
        format(atom(S),
            '  call void @wam_set_reg(%WamState* %vm, i32 ~w, %Value %a~w)',
            [RegIdx, I])
    ), Indices, Parts),
    atomic_list_concat(Parts, '\n', Setup).
