:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025-2026 John William Creighton (@s243a)
%
% wam_ilasm_target.pl - WAM-to-ILAsm (CIL) Transpilation Target
%
% Transpiles WAM runtime predicates (wam_runtime.pl) to CIL assembly.
% Phase 2: step_wam/3 → CIL switch dispatch
% Phase 3: helper predicates → CIL methods
% Phase 4: WAM instructions → CIL Instruction arrays
% Phase 5: Hybrid module assembly
%
% Key CIL-specific design choices:
%   - Value = class hierarchy (AtomValue, IntegerValue, etc.)
%   - Instruction dispatch via CIL switch (jump table)
%   - Registers = Value[] fixed array (shared ABI with LLVM)
%   - Run loop uses .tail call for constant-stack execution
%   - CLR GC handles all memory — no arena, no manual deallocation
%   - Type checks via isinst (no tag integers)
%   - Atoms are strings (no interning table needed)
%
% See: docs/design/WAM_ILASM_TRANSPILATION_IMPLEMENTATION_PLAN.md

:- module(wam_ilasm_target, [
    compile_step_wam_to_cil/2,          % +Options, -CILCode
    compile_wam_helpers_to_cil/2,       % +Options, -CILCode
    compile_wam_runtime_to_cil/2,       % +Options, -CILCode (step + helpers combined)
    compile_wam_predicate_to_cil/4,     % +Pred/Arity, +WamCode, +Options, -CILCode
    wam_instruction_to_cil_literal/2,   % +WamInstr, -CILLiteral
    wam_line_to_cil_literal/2,          % +Parts, -CILLit
    write_wam_ilasm_project/3,          % +Predicates, +Options, +OutputFile
    builtin_op_to_cil_id/2,            % +OpName, -IntId
    cil_atom_table_reset/0             % Reset atom table between compilations
]).

:- use_module(library(lists)).
:- use_module(library(option)).
:- use_module('../core/template_system').
:- use_module('../bindings/cil_wam_bindings').
:- use_module('../targets/wam_target', [compile_predicate_to_wam/3]).

:- discontiguous wam_cil_case/2.

% ============================================================================
% PHASE 5: Hybrid Module Assembly
% ============================================================================

%% write_wam_ilasm_project(+Predicates, +Options, +OutputFile)
write_wam_ilasm_project(Predicates, Options, OutputFile) :-
    option(module_name(ModuleName), Options, 'WamGenerated'),
    option(class_name(ClassName), Options, 'PrologGenerated.Program'),
    get_time(TimeStamp),
    format_time(string(Date), "%Y-%m-%d %H:%M:%S", TimeStamp),

    % State-management methods are rendered first and embedded INSIDE the
    % WamState class in the types template: they are emitted as
    % `instance ... WamState::...` members, so placing them in Program
    % (the old layout) made every WamState:: call site unresolvable.
    read_template_file('templates/targets/ilasm_wam/state.il.mustache', StateTemplate),
    render_template(StateTemplate, [], StateMethods),

    read_template_file('templates/targets/ilasm_wam/types.il.mustache', TypesTemplate),
    render_template(TypesTemplate, [module_name=ModuleName, date=Date,
                                    state_methods=StateMethods], TypesDef),

    compile_step_wam_to_cil(Options, StepMethod),
    compile_wam_helpers_to_cil(Options, HelperMethods),
    read_template_file('templates/targets/ilasm_wam/runtime.il.mustache', RuntimeTemplate),
    render_template(RuntimeTemplate, [
        step_method=StepMethod,
        helper_methods=HelperMethods,
        class_name=ClassName
    ], RuntimeMethods),

    compile_predicates_for_cil(Predicates, Options, NativeCode, WamCode),

    read_template_file('templates/targets/ilasm_wam/module.il.mustache', ModuleTemplate),
    render_template(ModuleTemplate, [
        module_name=ModuleName,
        date=Date,
        class_name=ClassName,
        type_definitions=TypesDef,
        runtime_methods=RuntimeMethods,
        native_predicates=NativeCode,
        wam_predicates=WamCode
    ], FullModule),

    setup_call_cleanup(
        open(OutputFile, write, Stream, [encoding(utf8)]),
        format(Stream, "~w", [FullModule]),
        close(Stream)
    ),
    format('WAM ILAsm module created at: ~w~n', [OutputFile]).

read_template_file(Path, Content) :-
    (   exists_file(Path)
    ->  read_file_to_string(Path, Content, [])
    ;   format(atom(Content), "// Template not found: ~w", [Path])
    ).

%% compile_predicates_for_cil(+Predicates, +Options, -NativeCode, -WamCode)
compile_predicates_for_cil(Predicates, Options, NativeCode, WamCode) :-
    compile_predicates_collect_cil(Predicates, Options, NativeParts, WamParts),
    atomic_list_concat(NativeParts, '\n\n', NativeCode),
    atomic_list_concat(WamParts, '\n\n', WamCode).

compile_predicates_collect_cil([], _, [], []).
compile_predicates_collect_cil([PredIndicator|Rest], Options, NativeParts, WamParts) :-
    (   PredIndicator = Module:Pred/Arity -> true
    ;   PredIndicator = Pred/Arity, Module = user
    ),
    compile_predicates_collect_cil(Rest, Options, RestNative, RestWam),
    (   catch(
            ilasm_target:compile_predicate_to_ilasm(Module:Pred/Arity,
                [include_main(false)|Options], PredCode),
            _, fail)
    ->  format(user_error, '  ~w/~w: native lowering~n', [Pred, Arity]),
        NativeParts = [PredCode | RestNative],
        WamParts = RestWam
    ;   option(wam_fallback(WamFB), Options, true),
        WamFB \== false,
        wam_target:compile_predicate_to_wam(Module:Pred/Arity, Options, WamRaw),
        compile_wam_predicate_to_cil(Pred/Arity, WamRaw, Options, PredCode)
    ->  format(user_error, '  ~w/~w: WAM fallback~n', [Pred, Arity]),
        NativeParts = RestNative,
        WamParts = [PredCode | RestWam]
    ;   format(user_error, '  ~w/~w: compilation failed~n', [Pred, Arity]),
        NativeParts = RestNative,
        format(atom(FailComment), '// ~w/~w: compilation failed', [Pred, Arity]),
        WamParts = [FailComment | RestWam]
    ).

% ============================================================================
% PHASE 2: step_wam/3 → CIL switch dispatch
% ============================================================================

compile_step_wam_to_cil(_Options, CILCode) :-
    findall(Case, compile_cil_step_case(Case), Cases),
    atomic_list_concat(Cases, '\n', CasesCode),
    format(atom(CILCode),
'.method public static bool step(class WamState vm, class Instruction instr) cil managed {
    .maxstack 8
    .locals init (int32 regIdx, int64 op1, int64 op2, class Value val, class StackEntry se)
    // Load instruction fields
    ldarg.1
    ldfld int32 Instruction::Tag
    switch (L_get_constant, L_get_variable, L_get_value,
            L_get_structure, L_get_list, L_unify_variable,
            L_unify_value, L_unify_constant,
            L_put_constant, L_put_variable, L_put_value,
            L_put_structure, L_put_list, L_set_variable,
            L_set_value, L_set_constant,
            L_allocate, L_deallocate, L_call, L_execute,
            L_proceed, L_builtin_call,
            L_try_me_else, L_retry_me_else, L_trust_me,
            L_cut_ite, L_jump)
    // Default: unknown instruction
    ldc.i4.0
    ret

~w
}', [CasesCode]).

compile_cil_step_case(CaseCode) :-
    wam_cil_case(Label, BodyCode),
    format(atom(CaseCode), '~w:\n~w', [Label, BodyCode]).

% --- Head Unification Instructions ---

wam_cil_case('L_get_constant',
'    // get_constant: op1=packed value, op2=(tag<<16)|reg_idx
    // Lower 16 bits of op2: register index. Upper bits: value tag
    // (0=Atom, 1=Integer) — same packing as the LLVM WAM target.
    ldarg.1
    ldfld int64 Instruction::Op2
    conv.i4
    ldc.i4 65535
    and
    stloc.0                          // regIdx
    ldarg.0
    ldarg.0
    ldloc.0
    callvirt instance class Value WamState::GetReg(int32)
    callvirt instance class Value WamState::Deref(class Value)
    stloc.3                          // val = dereferenced reg value
    ldloc.3
    isinst UnboundValue
    brfalse L_gc_check
    // Unbound: bind the cell to the constant (alias-preserving)
    ldarg.0
    ldloc.3
    ldarg.1
    ldfld int64 Instruction::Op2
    conv.i4
    ldc.i4 16
    shr
    ldarg.1
    ldfld int64 Instruction::Op1
    call class Value PrologGenerated.Program::make_constant(int32, int64)
    callvirt instance void WamState::BindVar(class Value, class Value)
    ldarg.0
    callvirt instance void WamState::IncPC()
    ldc.i4.1
    ret
L_gc_check:
    // Bound: compare with expected constant
    ldarg.1
    ldfld int64 Instruction::Op2
    conv.i4
    ldc.i4 16
    shr
    ldarg.1
    ldfld int64 Instruction::Op1
    call class Value PrologGenerated.Program::make_constant(int32, int64)
    ldloc.3
    callvirt instance bool Value::ValueEquals(class Value)
    brfalse L_gc_fail
    ldarg.0
    callvirt instance void WamState::IncPC()
    ldc.i4.1
    ret
L_gc_fail:
    ldc.i4.0
    ret').

wam_cil_case('L_get_variable',
'    // get_variable: copy Ai to Xn
    ldarg.1
    ldfld int64 Instruction::Op2
    conv.i4
    stloc.0                          // ai index
    ldarg.0
    ldloc.0
    callvirt instance class Value WamState::GetReg(int32)
    stloc.3                          // val
    ldarg.1
    ldfld int64 Instruction::Op1
    conv.i4
    stloc.0                          // xn index
    ldarg.0
    ldloc.0
    callvirt instance void WamState::TrailBinding(int32)
    ldarg.0
    ldloc.0
    ldloc.3
    callvirt instance void WamState::SetReg(int32, class Value)
    ldarg.0
    callvirt instance void WamState::IncPC()
    ldc.i4.1
    ret').

wam_cil_case('L_get_value',
'    // get_value: unify Ai and Xn
    ldarg.1
    ldfld int64 Instruction::Op2
    conv.i4
    stloc.0
    ldarg.0
    ldloc.0
    callvirt instance class Value WamState::GetReg(int32)
    stloc.3                          // valA
    // Check if A is unbound
    ldloc.3
    callvirt instance bool Value::IsUnbound()
    brfalse L_gv_check_x
    // Bind A to X
    ldarg.0
    ldloc.0
    callvirt instance void WamState::TrailBinding(int32)
    ldarg.1
    ldfld int64 Instruction::Op1
    conv.i4
    stloc.0
    ldarg.0
    ldloc.0
    callvirt instance class Value WamState::GetReg(int32)
    stloc.3
    ldarg.1
    ldfld int64 Instruction::Op2
    conv.i4
    stloc.0
    ldarg.0
    ldloc.0
    ldloc.3
    callvirt instance void WamState::SetReg(int32, class Value)
    ldarg.0
    callvirt instance void WamState::IncPC()
    ldc.i4.1
    ret
L_gv_check_x:
    // Check equality
    ldloc.3
    ldarg.0
    ldarg.1
    ldfld int64 Instruction::Op1
    conv.i4
    callvirt instance class Value WamState::GetReg(int32)
    callvirt instance bool Value::ValueEquals(class Value)
    brfalse L_gv_fail
    ldarg.0
    callvirt instance void WamState::IncPC()
    ldc.i4.1
    ret
L_gv_fail:
    ldc.i4.0
    ret').

% --- Structure/List Head Unification ---

wam_cil_case('L_get_structure',
'    // get_structure: write mode (unbound) or read mode (bound)
    ldarg.1
    ldfld int64 Instruction::Op2
    conv.i4
    stloc.0                          // Ai reg index
    ldarg.0
    ldloc.0
    callvirt instance class Value WamState::GetReg(int32)
    stloc.3                          // val
    ldloc.3
    callvirt instance bool Value::IsUnbound()
    brfalse L_gs_read
    // Write mode: heap marker + Ref + push WriteCtx
    ldarg.0
    ldstr "str_marker"
    newobj instance void AtomValue::.ctor(string)
    callvirt instance int32 WamState::HeapPush(class Value)
    pop
    ldarg.0
    ldloc.0
    callvirt instance void WamState::TrailBinding(int32)
    ldarg.0
    ldloc.0
    ldarg.0
    ldfld int32 WamState::HeapSize
    ldc.i4.1
    sub
    newobj instance void RefValue::.ctor(int32)
    callvirt instance void WamState::SetReg(int32, class Value)
    // Push WriteCtx with arity (default 2)
    ldarg.0
    ldc.i4.2
    callvirt instance void WamState::PushWriteCtx(int32)
    ldarg.0
    callvirt instance void WamState::IncPC()
    ldc.i4.1
    ret
L_gs_read:
    // Read mode: succeed and advance
    ldarg.0
    callvirt instance void WamState::IncPC()
    ldc.i4.1
    ret').

wam_cil_case('L_get_list',
'    // get_list: like get_structure for lists (./2, arity=2)
    ldarg.1
    ldfld int64 Instruction::Op1
    conv.i4
    stloc.0                          // Ai reg index
    ldarg.0
    ldloc.0
    callvirt instance class Value WamState::GetReg(int32)
    stloc.3
    ldloc.3
    callvirt instance bool Value::IsUnbound()
    brfalse L_gl_read
    // Write mode: heap marker + Ref + WriteCtx(2)
    ldarg.0
    ldstr "str(./2)"
    newobj instance void AtomValue::.ctor(string)
    callvirt instance int32 WamState::HeapPush(class Value)
    pop
    ldarg.0
    ldloc.0
    callvirt instance void WamState::TrailBinding(int32)
    ldarg.0
    ldloc.0
    ldarg.0
    ldfld int32 WamState::HeapSize
    ldc.i4.1
    sub
    newobj instance void RefValue::.ctor(int32)
    callvirt instance void WamState::SetReg(int32, class Value)
    ldarg.0
    ldc.i4.2
    callvirt instance void WamState::PushWriteCtx(int32)
    ldarg.0
    callvirt instance void WamState::IncPC()
    ldc.i4.1
    ret
L_gl_read:
    ldarg.0
    callvirt instance void WamState::IncPC()
    ldc.i4.1
    ret').

wam_cil_case('L_unify_variable',
'    // unify_variable: read mode (UnifyCtx) or write mode (WriteCtx)
    ldarg.1
    ldfld int64 Instruction::Op1
    conv.i4
    stloc.0                          // Xn reg index
    ldarg.0
    callvirt instance int32 WamState::PeekStackType()
    ldc.i4.1                         // 1 = UnifyCtx
    bne.un L_uv_write
    // Read mode: pop next arg from UnifyCtx, store in Xn
    ldarg.0
    callvirt instance class Value WamState::UnifyCtxNext()
    stloc.3
    ldloc.3
    brfalse L_uv_write              // null → fall through to write mode
    ldarg.0
    ldloc.0
    callvirt instance void WamState::TrailBinding(int32)
    ldarg.0
    ldloc.0
    ldloc.3
    callvirt instance void WamState::SetReg(int32, class Value)
    ldarg.0
    callvirt instance void WamState::IncPC()
    ldc.i4.1
    ret
L_uv_write:
    // Write mode: create unbound var, push on heap, store in Xn
    ldarg.0
    ldfld int32 WamState::HeapSize
    call string [mscorlib]System.Convert::ToString(int32)
    ldstr "_H"
    call string [mscorlib]System.String::Concat(string, string)
    newobj instance void UnboundValue::.ctor(string)
    stloc.3
    ldarg.0
    ldloc.3
    callvirt instance int32 WamState::HeapPush(class Value)
    pop
    ldarg.0
    callvirt instance int32 WamState::WriteCtxDec()
    pop
    ldarg.0
    ldloc.0
    callvirt instance void WamState::TrailBinding(int32)
    ldarg.0
    ldloc.0
    ldloc.3
    callvirt instance void WamState::SetReg(int32, class Value)
    ldarg.0
    callvirt instance void WamState::IncPC()
    ldc.i4.1
    ret').

wam_cil_case('L_unify_value',
'    // unify_value: read mode (check equality) or write mode (push to heap)
    ldarg.1
    ldfld int64 Instruction::Op1
    conv.i4
    stloc.0                          // Xn reg index
    ldarg.0
    callvirt instance int32 WamState::PeekStackType()
    ldc.i4.1                         // 1 = UnifyCtx
    bne.un L_uvl_write
    // Read mode: pop expected arg, compare with register value
    ldarg.0
    callvirt instance class Value WamState::UnifyCtxNext()
    stloc.3                          // expected
    ldloc.3
    brfalse L_uvl_write             // null → fall through to write
    ldarg.0
    ldloc.0
    callvirt instance class Value WamState::GetReg(int32)
    // Check: equal OR either unbound → succeed
    dup
    ldloc.3
    callvirt instance bool Value::ValueEquals(class Value)
    brtrue L_uvl_read_ok
    // Check if expected is unbound
    ldloc.3
    callvirt instance bool Value::IsUnbound()
    brtrue L_uvl_read_ok
    // Check if actual is unbound → bind
    dup
    callvirt instance bool Value::IsUnbound()
    brfalse L_uvl_fail
    // Bind actual to expected
    pop
    ldarg.0
    ldloc.0
    callvirt instance void WamState::TrailBinding(int32)
    ldarg.0
    ldloc.0
    ldloc.3
    callvirt instance void WamState::SetReg(int32, class Value)
L_uvl_read_ok:
    pop                              // clean stack
    ldarg.0
    callvirt instance void WamState::IncPC()
    ldc.i4.1
    ret
L_uvl_write:
    // Write mode: push Xn value onto heap
    ldarg.0
    ldloc.0
    callvirt instance class Value WamState::GetReg(int32)
    stloc.3
    ldarg.0
    ldloc.3
    callvirt instance int32 WamState::HeapPush(class Value)
    pop
    ldarg.0
    callvirt instance int32 WamState::WriteCtxDec()
    pop
    ldarg.0
    callvirt instance void WamState::IncPC()
    ldc.i4.1
    ret
L_uvl_fail:
    pop
    ldc.i4.0
    ret').

wam_cil_case('L_unify_constant',
'    // unify_constant: read mode (check) or write mode (push to heap)
    ldarg.0
    callvirt instance int32 WamState::PeekStackType()
    ldc.i4.1
    bne.un L_uc_write
    // Read mode: pop expected, check equality or unbound
    ldarg.0
    callvirt instance class Value WamState::UnifyCtxNext()
    stloc.3
    ldloc.3
    brfalse L_uc_write              // null → fall through to write
    ldarg.1
    ldfld int64 Instruction::Op1
    newobj instance void IntegerValue::.ctor(int64)
    ldloc.3
    callvirt instance bool Value::ValueEquals(class Value)
    brtrue L_uc_read_ok
    ldloc.3
    callvirt instance bool Value::IsUnbound()
    brtrue L_uc_read_ok
    ldc.i4.0
    ret
L_uc_read_ok:
    ldarg.0
    callvirt instance void WamState::IncPC()
    ldc.i4.1
    ret
L_uc_write:
    ldarg.0
    ldarg.1
    ldfld int64 Instruction::Op1
    newobj instance void IntegerValue::.ctor(int64)
    callvirt instance int32 WamState::HeapPush(class Value)
    pop
    ldarg.0
    callvirt instance int32 WamState::WriteCtxDec()
    pop
    ldarg.0
    callvirt instance void WamState::IncPC()
    ldc.i4.1
    ret').

% --- Body Construction Instructions ---

wam_cil_case('L_put_constant',
'    // put_constant: store constant in register
    // op1=packed value, op2=(tag<<16)|reg_idx (0=Atom, 1=Integer)
    ldarg.1
    ldfld int64 Instruction::Op2
    conv.i4
    ldc.i4 65535
    and
    stloc.0
    ldarg.0
    ldloc.0
    callvirt instance void WamState::TrailBinding(int32)
    ldarg.0
    ldloc.0
    ldarg.1
    ldfld int64 Instruction::Op2
    conv.i4
    ldc.i4 16
    shr
    ldarg.1
    ldfld int64 Instruction::Op1
    call class Value PrologGenerated.Program::make_constant(int32, int64)
    callvirt instance void WamState::SetReg(int32, class Value)
    ldarg.0
    callvirt instance void WamState::IncPC()
    ldc.i4.1
    ret').

wam_cil_case('L_put_variable',
'    // put_variable: create unbound var, store in Xn and Ai
    ldarg.0
    ldfld int32 WamState::PC
    call string [mscorlib]System.Convert::ToString(int32)
    newobj instance void UnboundValue::.ctor(string)
    stloc.3
    // Store in Xn
    ldarg.1
    ldfld int64 Instruction::Op1
    conv.i4
    stloc.0
    ldarg.0
    ldloc.0
    callvirt instance void WamState::TrailBinding(int32)
    ldarg.0
    ldloc.0
    ldloc.3
    callvirt instance void WamState::SetReg(int32, class Value)
    // Store in Ai
    ldarg.1
    ldfld int64 Instruction::Op2
    conv.i4
    stloc.0
    ldarg.0
    ldloc.0
    callvirt instance void WamState::TrailBinding(int32)
    ldarg.0
    ldloc.0
    ldloc.3
    callvirt instance void WamState::SetReg(int32, class Value)
    ldarg.0
    callvirt instance void WamState::IncPC()
    ldc.i4.1
    ret').

wam_cil_case('L_put_value',
'    // put_value: copy Xn to Ai
    ldarg.0
    ldarg.1
    ldfld int64 Instruction::Op1
    conv.i4
    callvirt instance class Value WamState::GetReg(int32)
    stloc.3
    ldarg.1
    ldfld int64 Instruction::Op2
    conv.i4
    stloc.0
    ldarg.0
    ldloc.0
    callvirt instance void WamState::TrailBinding(int32)
    ldarg.0
    ldloc.0
    ldloc.3
    callvirt instance void WamState::SetReg(int32, class Value)
    ldarg.0
    callvirt instance void WamState::IncPC()
    ldc.i4.1
    ret').

wam_cil_case('L_put_structure',
'    // put_structure: heap marker + Ref + WriteCtx
    ldarg.1
    ldfld int64 Instruction::Op2
    conv.i4
    stloc.0                          // Ai reg index
    ldarg.0
    ldstr "str_marker"
    newobj instance void AtomValue::.ctor(string)
    callvirt instance int32 WamState::HeapPush(class Value)
    pop
    ldarg.0
    ldloc.0
    ldarg.0
    ldfld int32 WamState::HeapSize
    ldc.i4.1
    sub
    newobj instance void RefValue::.ctor(int32)
    callvirt instance void WamState::SetReg(int32, class Value)
    ldarg.0
    ldc.i4.2
    callvirt instance void WamState::PushWriteCtx(int32)
    ldarg.0
    callvirt instance void WamState::IncPC()
    ldc.i4.1
    ret').

wam_cil_case('L_put_list',
'    // put_list: heap marker + Ref + WriteCtx(2)
    ldarg.1
    ldfld int64 Instruction::Op1
    conv.i4
    stloc.0                          // Ai reg index
    ldarg.0
    ldstr "str(./2)"
    newobj instance void AtomValue::.ctor(string)
    callvirt instance int32 WamState::HeapPush(class Value)
    pop
    ldarg.0
    ldloc.0
    ldarg.0
    ldfld int32 WamState::HeapSize
    ldc.i4.1
    sub
    newobj instance void RefValue::.ctor(int32)
    callvirt instance void WamState::SetReg(int32, class Value)
    ldarg.0
    ldc.i4.2
    callvirt instance void WamState::PushWriteCtx(int32)
    ldarg.0
    callvirt instance void WamState::IncPC()
    ldc.i4.1
    ret').

wam_cil_case('L_set_variable',
'    // set_variable: create unbound var, push on heap, store in Xn
    ldarg.1
    ldfld int64 Instruction::Op1
    conv.i4
    stloc.0                          // Xn reg index
    ldarg.0
    ldfld int32 WamState::HeapSize
    call string [mscorlib]System.Convert::ToString(int32)
    ldstr "_H"
    call string [mscorlib]System.String::Concat(string, string)
    newobj instance void UnboundValue::.ctor(string)
    stloc.3
    ldarg.0
    ldloc.3
    callvirt instance int32 WamState::HeapPush(class Value)
    pop
    ldarg.0
    ldloc.0
    ldloc.3
    callvirt instance void WamState::SetReg(int32, class Value)
    ldarg.0
    callvirt instance void WamState::IncPC()
    ldc.i4.1
    ret').

wam_cil_case('L_set_value',
'    // set_value: push Xn value onto heap
    ldarg.0
    ldarg.1
    ldfld int64 Instruction::Op1
    conv.i4
    callvirt instance class Value WamState::GetReg(int32)
    stloc.3
    ldarg.0
    ldloc.3
    callvirt instance int32 WamState::HeapPush(class Value)
    pop
    ldarg.0
    callvirt instance void WamState::IncPC()
    ldc.i4.1
    ret').

wam_cil_case('L_set_constant',
'    // set_constant: push constant onto heap
    ldarg.0
    ldarg.1
    ldfld int64 Instruction::Op1
    newobj instance void IntegerValue::.ctor(int64)
    callvirt instance int32 WamState::HeapPush(class Value)
    pop
    ldarg.0
    callvirt instance void WamState::IncPC()
    ldc.i4.1
    ret').

% --- Control Instructions ---

wam_cil_case('L_allocate',
'    // allocate: push environment frame saving CP
    ldarg.0
    ldfld class [mscorlib]System.Collections.Generic.List`1<class StackEntry> WamState::Stack
    newobj instance void StackEntry::.ctor()
    dup
    ldc.i4.0
    stfld int32 StackEntry::Type      // type = 0 (EnvFrame)
    dup
    ldarg.0
    ldfld int32 WamState::CP
    stfld int32 StackEntry::Aux       // save CP
    callvirt instance void class [mscorlib]System.Collections.Generic.List`1<class StackEntry>::Add(!0)
    ldarg.0
    callvirt instance void WamState::IncPC()
    ldc.i4.1
    ret').

wam_cil_case('L_deallocate',
'    // deallocate: pop environment frame (top of stack, type 0), restore CP
    ldarg.0
    ldfld class [mscorlib]System.Collections.Generic.List`1<class StackEntry> WamState::Stack
    callvirt instance int32 class [mscorlib]System.Collections.Generic.List`1<class StackEntry>::get_Count()
    ldc.i4.0
    ble L_dealloc_done
    ldarg.0
    ldfld class [mscorlib]System.Collections.Generic.List`1<class StackEntry> WamState::Stack
    dup
    callvirt instance int32 class [mscorlib]System.Collections.Generic.List`1<class StackEntry>::get_Count()
    ldc.i4.1
    sub
    callvirt instance !0 class [mscorlib]System.Collections.Generic.List`1<class StackEntry>::get_Item(int32)
    stloc.s 4
    ldloc.s 4
    ldfld int32 StackEntry::Type
    brtrue L_dealloc_done             // not an EnvFrame (type != 0): skip
    // Restore CP from saved frame
    ldarg.0
    ldloc.s 4
    ldfld int32 StackEntry::Aux
    stfld int32 WamState::CP
    // Pop the frame
    ldarg.0
    ldfld class [mscorlib]System.Collections.Generic.List`1<class StackEntry> WamState::Stack
    dup
    callvirt instance int32 class [mscorlib]System.Collections.Generic.List`1<class StackEntry>::get_Count()
    ldc.i4.1
    sub
    callvirt instance void class [mscorlib]System.Collections.Generic.List`1<class StackEntry>::RemoveAt(int32)
L_dealloc_done:
    ldarg.0
    callvirt instance void WamState::IncPC()
    ldc.i4.1
    ret').

wam_cil_case('L_call',
'    // call: save continuation, jump to label
    ldarg.0
    ldarg.1
    ldfld int64 Instruction::Op1
    conv.i4
    callvirt instance int32 WamState::LabelPC(int32)
    stloc.0
    ldloc.0
    ldc.i4.m1
    beq L_call_fail
    // Save continuation
    ldarg.0
    ldarg.0
    ldfld int32 WamState::PC
    ldc.i4.1
    add
    stfld int32 WamState::CP
    // Jump: PC := LabelPC(op1)
    ldarg.0
    ldloc.0
    stfld int32 WamState::PC
    ldc.i4.1
    ret
L_call_fail:
    ldc.i4.0
    ret').

wam_cil_case('L_execute',
'    // execute: jump to label (no continuation save)
    ldarg.0
    ldarg.1
    ldfld int64 Instruction::Op1
    conv.i4
    callvirt instance int32 WamState::LabelPC(int32)
    stloc.0
    ldloc.0
    ldc.i4.m1
    beq L_exec_fail
    ldarg.0
    ldloc.0
    stfld int32 WamState::PC
    ldc.i4.1
    ret
L_exec_fail:
    ldc.i4.0
    ret').

wam_cil_case('L_proceed',
'    // proceed: return to continuation or halt
    ldarg.0
    ldfld int32 WamState::CP
    ldc.i4.0
    beq L_proc_halt
    // Return to continuation
    ldarg.0
    ldarg.0
    ldfld int32 WamState::CP
    stfld int32 WamState::PC
    ldarg.0
    ldc.i4.0
    stfld int32 WamState::CP
    ldc.i4.1
    ret
L_proc_halt:
    ldarg.0
    ldc.i4.1
    stfld bool WamState::Halted
    ldc.i4.1
    ret').

wam_cil_case('L_builtin_call',
'    // builtin_call: dispatch to builtin handler
    ldarg.0
    ldarg.1
    ldfld int64 Instruction::Op1
    conv.i4
    ldarg.1
    ldfld int64 Instruction::Op2
    conv.i4
    call bool PrologGenerated.Program::execute_builtin(class WamState, int32, int32)
    dup
    brfalse L_bi_fail
    ldarg.0
    callvirt instance void WamState::IncPC()
L_bi_fail:
    ret').

% --- Choice Point Instructions ---

wam_cil_case('L_try_me_else',
'    // try_me_else: push choice point
    ldarg.0
    ldarg.1
    ldfld int64 Instruction::Op1
    conv.i4
    callvirt instance int32 WamState::LabelPC(int32)
    stloc.0                          // nextPC
    // Create choice point (list receiver pushed first for the Add below)
    ldarg.0
    ldfld class [mscorlib]System.Collections.Generic.List`1<class ChoicePoint> WamState::ChoicePoints
    newobj instance void ChoicePoint::.ctor()
    dup
    ldloc.0
    stfld int32 ChoicePoint::NextPC
    dup
    ldarg.0
    ldfld class Value[] WamState::Regs
    callvirt instance object [mscorlib]System.Array::Clone()
    castclass class Value[]
    stfld class Value[] ChoicePoint::SavedRegs
    dup
    ldarg.0
    ldfld class [mscorlib]System.Collections.Generic.List`1<class TrailEntry> WamState::Trail
    callvirt instance int32 class [mscorlib]System.Collections.Generic.List`1<class TrailEntry>::get_Count()
    stfld int32 ChoicePoint::TrailMark
    dup
    ldarg.0
    ldfld int32 WamState::CP
    stfld int32 ChoicePoint::SavedCP
    // Push onto choice point stack (receiver already below)
    callvirt instance void class [mscorlib]System.Collections.Generic.List`1<class ChoicePoint>::Add(!0)
    ldarg.0
    callvirt instance void WamState::IncPC()
    ldc.i4.1
    ret').

wam_cil_case('L_retry_me_else',
'    // retry_me_else: update top choice point next_pc
    ldarg.0
    ldarg.1
    ldfld int64 Instruction::Op1
    conv.i4
    callvirt instance int32 WamState::LabelPC(int32)
    stloc.0
    ldarg.0
    ldfld class [mscorlib]System.Collections.Generic.List`1<class ChoicePoint> WamState::ChoicePoints
    dup
    callvirt instance int32 class [mscorlib]System.Collections.Generic.List`1<class ChoicePoint>::get_Count()
    ldc.i4.1
    sub
    callvirt instance !0 class [mscorlib]System.Collections.Generic.List`1<class ChoicePoint>::get_Item(int32)
    ldloc.0
    stfld int32 ChoicePoint::NextPC
    ldarg.0
    callvirt instance void WamState::IncPC()
    ldc.i4.1
    ret').

wam_cil_case('L_trust_me',
'    // trust_me: pop top choice point
    ldarg.0
    ldfld class [mscorlib]System.Collections.Generic.List`1<class ChoicePoint> WamState::ChoicePoints
    dup
    callvirt instance int32 class [mscorlib]System.Collections.Generic.List`1<class ChoicePoint>::get_Count()
    ldc.i4.1
    sub
    callvirt instance void class [mscorlib]System.Collections.Generic.List`1<class ChoicePoint>::RemoveAt(int32)
    ldarg.0
    callvirt instance void WamState::IncPC()
    ldc.i4.1
    ret').

wam_cil_case('L_cut_ite',
'    // cut_ite: commit to the then-branch of an if-then-else by popping
    // the guard choice point pushed by the corresponding try_me_else
    // (M144 bug class: without this, a failing then-branch backtracks
    // into the else-branch).
    ldarg.0
    ldfld class [mscorlib]System.Collections.Generic.List`1<class ChoicePoint> WamState::ChoicePoints
    callvirt instance int32 class [mscorlib]System.Collections.Generic.List`1<class ChoicePoint>::get_Count()
    ldc.i4.0
    ble L_cut_ite_done
    ldarg.0
    ldfld class [mscorlib]System.Collections.Generic.List`1<class ChoicePoint> WamState::ChoicePoints
    dup
    callvirt instance int32 class [mscorlib]System.Collections.Generic.List`1<class ChoicePoint>::get_Count()
    ldc.i4.1
    sub
    callvirt instance void class [mscorlib]System.Collections.Generic.List`1<class ChoicePoint>::RemoveAt(int32)
L_cut_ite_done:
    ldarg.0
    callvirt instance void WamState::IncPC()
    ldc.i4.1
    ret').

wam_cil_case('L_jump',
'    // jump: unconditional branch to label (op1 = label index)
    ldarg.0
    ldarg.1
    ldfld int64 Instruction::Op1
    conv.i4
    callvirt instance int32 WamState::LabelPC(int32)
    stloc.0
    ldloc.0
    ldc.i4.m1
    beq L_jump_fail
    ldarg.0
    ldloc.0
    stfld int32 WamState::PC
    ldc.i4.1
    ret
L_jump_fail:
    ldc.i4.0
    ret').

% ============================================================================
% PHASE 3: Helper predicates → CIL methods
% ============================================================================

compile_wam_helpers_to_cil(_Options, CILCode) :-
    compile_backtrack_to_cil(BacktrackCode),
    compile_unwind_trail_to_cil(UnwindCode),
    compile_execute_builtin_to_cil(BuiltinCode),
    compile_eval_arith_to_cil(ArithCode),
    compile_make_constant_to_cil(MakeConstCode),
    atomic_list_concat([
        BacktrackCode, '\n\n',
        UnwindCode, '\n\n',
        BuiltinCode, '\n\n',
        ArithCode, '\n\n',
        MakeConstCode
    ], CILCode).

compile_backtrack_to_cil(Code) :-
    Code = '.method public static bool backtrack(class WamState vm) cil managed {
    .maxstack 4
    .locals init (class ChoicePoint cp, int32 topIdx)
    // Check if any choice points exist
    ldarg.0
    ldfld class [mscorlib]System.Collections.Generic.List`1<class ChoicePoint> WamState::ChoicePoints
    callvirt instance int32 class [mscorlib]System.Collections.Generic.List`1<class ChoicePoint>::get_Count()
    ldc.i4.0
    ble L_bt_fail
    // Get top choice point
    ldarg.0
    ldfld class [mscorlib]System.Collections.Generic.List`1<class ChoicePoint> WamState::ChoicePoints
    dup
    callvirt instance int32 class [mscorlib]System.Collections.Generic.List`1<class ChoicePoint>::get_Count()
    ldc.i4.1
    sub
    callvirt instance !0 class [mscorlib]System.Collections.Generic.List`1<class ChoicePoint>::get_Item(int32)
    stloc.0
    // Unwind trail
    ldarg.0
    ldloc.0
    ldfld int32 ChoicePoint::TrailMark
    call void PrologGenerated.Program::unwind_trail(class WamState, int32)
    // Restore registers (Array.Clone)
    ldarg.0
    ldloc.0
    ldfld class Value[] ChoicePoint::SavedRegs
    callvirt instance object [mscorlib]System.Array::Clone()
    castclass class Value[]
    stfld class Value[] WamState::Regs
    // Restore PC and CP
    ldarg.0
    ldloc.0
    ldfld int32 ChoicePoint::NextPC
    stfld int32 WamState::PC
    ldarg.0
    ldloc.0
    ldfld int32 ChoicePoint::SavedCP
    stfld int32 WamState::CP
    // Clear halted
    ldarg.0
    ldc.i4.0
    stfld bool WamState::Halted
    ldc.i4.1
    ret
L_bt_fail:
    ldc.i4.0
    ret
}'.

compile_unwind_trail_to_cil(Code) :-
    Code = '.method public static void unwind_trail(class WamState vm, int32 savedMark) cil managed {
    .maxstack 4
    .locals init (int32 idx, class TrailEntry te)
L_unwind_loop:
    ldarg.0
    ldfld class [mscorlib]System.Collections.Generic.List`1<class TrailEntry> WamState::Trail
    callvirt instance int32 class [mscorlib]System.Collections.Generic.List`1<class TrailEntry>::get_Count()
    ldarg.1
    ble L_unwind_done
    // Pop last trail entry
    ldarg.0
    ldfld class [mscorlib]System.Collections.Generic.List`1<class TrailEntry> WamState::Trail
    dup
    callvirt instance int32 class [mscorlib]System.Collections.Generic.List`1<class TrailEntry>::get_Count()
    ldc.i4.1
    sub
    dup
    stloc.0
    callvirt instance !0 class [mscorlib]System.Collections.Generic.List`1<class TrailEntry>::get_Item(int32)
    stloc.1
    // Cell binding (BindVar, RegIndex = -1): dissolve the alias/binding
    // by resetting the forwarding pointer instead of touching registers.
    ldloc.1
    ldfld class Value TrailEntry::BoundVar
    brfalse L_unwind_reg
    ldloc.1
    ldfld class Value TrailEntry::BoundVar
    castclass UnboundValue
    ldnull
    stfld class Value UnboundValue::Ref
    br L_unwind_pop
L_unwind_reg:
    // Restore old register value
    ldarg.0
    ldloc.1
    ldfld int32 TrailEntry::RegIndex
    ldloc.1
    ldfld class Value TrailEntry::OldValue
    callvirt instance void WamState::SetReg(int32, class Value)
L_unwind_pop:
    // Remove trail entry
    ldarg.0
    ldfld class [mscorlib]System.Collections.Generic.List`1<class TrailEntry> WamState::Trail
    ldloc.0
    callvirt instance void class [mscorlib]System.Collections.Generic.List`1<class TrailEntry>::RemoveAt(int32)
    br L_unwind_loop
L_unwind_done:
    ret
}'.

compile_execute_builtin_to_cil(Code) :-
    Code = '// Execute builtin operations via switch dispatch
// Op ids follow the LLVM target numbering (harmonization invariant):
//   0=is 1=> 2=< 3=>= 4==< 5==:= 6==\\= 7=== 8=true 9=fail 10=!
//   13=atom 18=var 19=nonvar 24== (full unification)
// Slots 11/12/14-17/20-23 are LLVM builtins not implemented here and
// fall through to the unknown-op default (fail).
// Arithmetic comparisons evaluate both sides via eval_arith (which
// derefs alias chains); type-check builtins deref before isinst.
.method public static bool execute_builtin(class WamState vm, int32 op, int32 arity) cil managed {
    .maxstack 4
    .locals init (class Value a1, class Value a2, int64 v1, int64 v2)
    ldarg.1
    switch (L_bi_is, L_bi_gt, L_bi_lt, L_bi_ge, L_bi_le,
            L_bi_arith_eq, L_bi_arith_ne, L_bi_eq,
            L_bi_true, L_bi_fail, L_bi_cut,
            L_bi_unknown, L_bi_unknown, L_bi_atom, L_bi_unknown,
            L_bi_unknown, L_bi_unknown, L_bi_unknown, L_bi_var,
            L_bi_nonvar, L_bi_unknown, L_bi_unknown, L_bi_unknown,
            L_bi_unknown, L_bi_unify)
L_bi_unknown:
    ldc.i4.0
    ret

L_bi_is:
    // is/2: evaluate A1 expr, bind/check against A0 (alias-aware)
    ldarg.0
    ldarg.0
    ldc.i4.1
    callvirt instance class Value WamState::GetReg(int32)
    call int64 PrologGenerated.Program::eval_arith(class WamState, class Value)
    newobj instance void IntegerValue::.ctor(int64)
    stloc.0                          // result
    ldarg.0
    ldarg.0
    ldc.i4.0
    callvirt instance class Value WamState::GetReg(int32)
    callvirt instance class Value WamState::Deref(class Value)
    stloc.1                          // a0 (dereferenced)
    ldloc.1
    isinst UnboundValue
    brfalse L_bi_is_check
    // Bind the cell (trailed, alias-preserving)
    ldarg.0
    ldloc.1
    ldloc.0
    callvirt instance void WamState::BindVar(class Value, class Value)
    ldc.i4.1
    ret
L_bi_is_check:
    ldloc.1
    ldloc.0
    callvirt instance bool Value::ValueEquals(class Value)
    ret

L_bi_gt:
    ldarg.0
    ldarg.0
    ldc.i4.0
    callvirt instance class Value WamState::GetReg(int32)
    call int64 PrologGenerated.Program::eval_arith(class WamState, class Value)
    ldarg.0
    ldarg.0
    ldc.i4.1
    callvirt instance class Value WamState::GetReg(int32)
    call int64 PrologGenerated.Program::eval_arith(class WamState, class Value)
    cgt
    ret
L_bi_lt:
    ldarg.0
    ldarg.0
    ldc.i4.0
    callvirt instance class Value WamState::GetReg(int32)
    call int64 PrologGenerated.Program::eval_arith(class WamState, class Value)
    ldarg.0
    ldarg.0
    ldc.i4.1
    callvirt instance class Value WamState::GetReg(int32)
    call int64 PrologGenerated.Program::eval_arith(class WamState, class Value)
    clt
    ret
L_bi_ge:
    ldarg.0
    ldarg.0
    ldc.i4.0
    callvirt instance class Value WamState::GetReg(int32)
    call int64 PrologGenerated.Program::eval_arith(class WamState, class Value)
    ldarg.0
    ldarg.0
    ldc.i4.1
    callvirt instance class Value WamState::GetReg(int32)
    call int64 PrologGenerated.Program::eval_arith(class WamState, class Value)
    clt
    ldc.i4.0
    ceq
    ret
L_bi_le:
    ldarg.0
    ldarg.0
    ldc.i4.0
    callvirt instance class Value WamState::GetReg(int32)
    call int64 PrologGenerated.Program::eval_arith(class WamState, class Value)
    ldarg.0
    ldarg.0
    ldc.i4.1
    callvirt instance class Value WamState::GetReg(int32)
    call int64 PrologGenerated.Program::eval_arith(class WamState, class Value)
    cgt
    ldc.i4.0
    ceq
    ret
L_bi_arith_eq:
    ldarg.0
    ldarg.0
    ldc.i4.0
    callvirt instance class Value WamState::GetReg(int32)
    call int64 PrologGenerated.Program::eval_arith(class WamState, class Value)
    ldarg.0
    ldarg.0
    ldc.i4.1
    callvirt instance class Value WamState::GetReg(int32)
    call int64 PrologGenerated.Program::eval_arith(class WamState, class Value)
    ceq
    ret
L_bi_arith_ne:
    ldarg.0
    ldarg.0
    ldc.i4.0
    callvirt instance class Value WamState::GetReg(int32)
    call int64 PrologGenerated.Program::eval_arith(class WamState, class Value)
    ldarg.0
    ldarg.0
    ldc.i4.1
    callvirt instance class Value WamState::GetReg(int32)
    call int64 PrologGenerated.Program::eval_arith(class WamState, class Value)
    ceq
    ldc.i4.0
    ceq
    ret
L_bi_eq:
    // ==/2: structural equality on dereferenced values
    ldarg.0
    ldarg.0
    ldc.i4.0
    callvirt instance class Value WamState::GetReg(int32)
    callvirt instance class Value WamState::Deref(class Value)
    ldarg.0
    ldarg.0
    ldc.i4.1
    callvirt instance class Value WamState::GetReg(int32)
    callvirt instance class Value WamState::Deref(class Value)
    callvirt instance bool Value::ValueEquals(class Value)
    ret
L_bi_true:
    ldc.i4.1
    ret
L_bi_fail:
    ldc.i4.0
    ret
L_bi_cut:
    ldarg.0
    ldfld class [mscorlib]System.Collections.Generic.List`1<class ChoicePoint> WamState::ChoicePoints
    callvirt instance void class [mscorlib]System.Collections.Generic.List`1<class ChoicePoint>::Clear()
    ldc.i4.1
    ret
L_bi_var:
    // var/1: dereferenced A0 is an unbound cell
    ldarg.0
    ldarg.0
    ldc.i4.0
    callvirt instance class Value WamState::GetReg(int32)
    callvirt instance class Value WamState::Deref(class Value)
    isinst UnboundValue
    ldnull
    cgt.un
    ret
L_bi_nonvar:
    ldarg.0
    ldarg.0
    ldc.i4.0
    callvirt instance class Value WamState::GetReg(int32)
    callvirt instance class Value WamState::Deref(class Value)
    isinst UnboundValue
    ldnull
    ceq
    ret
L_bi_atom:
    ldarg.0
    ldarg.0
    ldc.i4.0
    callvirt instance class Value WamState::GetReg(int32)
    callvirt instance class Value WamState::Deref(class Value)
    isinst AtomValue
    ldnull
    cgt.un
    ret
L_bi_unify:
    // =/2: FULL unification. Var-var must ALIAS the two cells via
    // BindVar (forwarding pointer), not copy one unbound marker over
    // the other (M143 bug class).
    ldarg.0
    ldarg.0
    ldc.i4.0
    callvirt instance class Value WamState::GetReg(int32)
    callvirt instance class Value WamState::Deref(class Value)
    stloc.0
    ldarg.0
    ldarg.0
    ldc.i4.1
    callvirt instance class Value WamState::GetReg(int32)
    callvirt instance class Value WamState::Deref(class Value)
    stloc.1
    // Same cell/value object: trivially unified
    ldloc.0
    ldloc.1
    beq L_bi_true
    // a unbound: alias a -> b
    ldloc.0
    isinst UnboundValue
    brfalse L_bu_checkb
    ldarg.0
    ldloc.0
    ldloc.1
    callvirt instance void WamState::BindVar(class Value, class Value)
    ldc.i4.1
    ret
L_bu_checkb:
    // b unbound: bind b -> a
    ldloc.1
    isinst UnboundValue
    brfalse L_bu_cmp
    ldarg.0
    ldloc.1
    ldloc.0
    callvirt instance void WamState::BindVar(class Value, class Value)
    ldc.i4.1
    ret
L_bu_cmp:
    // Both bound: structural equality
    ldloc.0
    ldloc.1
    callvirt instance bool Value::ValueEquals(class Value)
    ret
}'.

compile_eval_arith_to_cil(Code) :-
    Code = '// Evaluate arithmetic expression
.method public static int64 eval_arith(class WamState vm, class Value expr) cil managed {
    .maxstack 4
    .locals init (class CompoundValue cv, int64 a, int64 b)
    // Deref alias chains first (X = Y, X = 42, Y =:= 42 must see 42)
    ldarg.0
    ldarg.1
    callvirt instance class Value WamState::Deref(class Value)
    starg.s 1
    // Integer: return value directly
    ldarg.1
    isinst IntegerValue
    brfalse L_ea_not_int
    ldarg.1
    castclass IntegerValue
    ldfld int64 IntegerValue::Val
    ret
L_ea_not_int:
    // Float: convert to int64
    ldarg.1
    isinst FloatValue
    brfalse L_ea_not_float
    ldarg.1
    castclass FloatValue
    ldfld float64 FloatValue::Val
    conv.i8
    ret
L_ea_not_float:
    // Compound: evaluate recursively
    ldarg.1
    isinst CompoundValue
    brfalse L_ea_zero
    ldarg.1
    castclass CompoundValue
    stloc.0
    ldloc.0
    ldfld int32 CompoundValue::Arity
    ldc.i4.2
    bne.un L_ea_check_unary
    // Binary: evaluate both args
    ldarg.0
    ldloc.0
    ldfld class Value[] CompoundValue::Args
    ldc.i4.0
    ldelem.ref
    call int64 PrologGenerated.Program::eval_arith(class WamState, class Value)
    stloc.1
    ldarg.0
    ldloc.0
    ldfld class Value[] CompoundValue::Args
    ldc.i4.1
    ldelem.ref
    call int64 PrologGenerated.Program::eval_arith(class WamState, class Value)
    stloc.2
    // Dispatch on functor first char
    ldloc.0
    ldfld string CompoundValue::Functor
    ldc.i4.0
    callvirt instance char [mscorlib]System.String::get_Chars(int32)
    ldc.i4 43                        // +
    beq L_ea_add
    ldloc.0
    ldfld string CompoundValue::Functor
    ldc.i4.0
    callvirt instance char [mscorlib]System.String::get_Chars(int32)
    ldc.i4 45                        // -
    beq L_ea_sub
    ldloc.0
    ldfld string CompoundValue::Functor
    ldc.i4.0
    callvirt instance char [mscorlib]System.String::get_Chars(int32)
    ldc.i4 42                        // *
    beq L_ea_mul
    ldloc.0
    ldfld string CompoundValue::Functor
    ldc.i4.0
    callvirt instance char [mscorlib]System.String::get_Chars(int32)
    ldc.i4 47                        // /
    beq L_ea_div
    br L_ea_zero
L_ea_add:
    ldloc.1
    ldloc.2
    add
    ret
L_ea_sub:
    ldloc.1
    ldloc.2
    sub
    ret
L_ea_mul:
    ldloc.1
    ldloc.2
    mul
    ret
L_ea_div:
    ldloc.2
    ldc.i8 0
    beq L_ea_zero
    ldloc.1
    ldloc.2
    div
    ret
L_ea_check_unary:
    ldloc.0
    ldfld int32 CompoundValue::Arity
    ldc.i4.1
    bne.un L_ea_zero
    ldarg.0
    ldloc.0
    ldfld class Value[] CompoundValue::Args
    ldc.i4.0
    ldelem.ref
    call int64 PrologGenerated.Program::eval_arith(class WamState, class Value)
    stloc.1
    ldloc.0
    ldfld string CompoundValue::Functor
    ldc.i4.0
    callvirt instance char [mscorlib]System.String::get_Chars(int32)
    ldc.i4 45                        // - (negation)
    bne.un L_ea_zero
    ldc.i8 0
    ldloc.1
    sub
    ret
L_ea_zero:
    ldc.i8 0
    ret
}'.

%% make_constant: build a Value from a (tag, packed) pair.
%  Tag 1 = integer (packed is the value itself); tag 0 = atom (packed is
%  the interned atom id — the IL runtime has no atom-name table, so the
%  decimal id string is the canonical atom name; equality is preserved
%  because both occurrences intern to the same id).
compile_make_constant_to_cil(Code) :-
    Code = '// Construct a constant Value from (tag, packed) — tag 1=Integer, 0=Atom
.method public static class Value make_constant(int32 tag, int64 packed) cil managed {
    .maxstack 2
    ldarg.0
    ldc.i4.1
    beq L_mc_int
    ldarg.1
    call string [mscorlib]System.Convert::ToString(int64)
    newobj instance void AtomValue::.ctor(string)
    ret
L_mc_int:
    ldarg.1
    newobj instance void IntegerValue::.ctor(int64)
    ret
}'.

% ============================================================================
% ASSEMBLY: Combine Phase 2 + Phase 3
% ============================================================================

compile_wam_runtime_to_cil(Options, CILCode) :-
    compile_step_wam_to_cil(Options, StepCode),
    compile_wam_helpers_to_cil(Options, HelpersCode),
    atomic_list_concat([StepCode, '\n\n', HelpersCode], CILCode).

% ============================================================================
% PHASE 4: WAM instructions → CIL Instruction constructor calls
% ============================================================================

% get_constant/put_constant op2 packs (tag<<16)|reg_idx — tag 0=Atom,
% 1=Integer — matching the step-case decode (op2>>16 selects the tag for
% make_constant; op2&0xFFFF is the register). Emitting the bare register
% index made every integer constant decode as an atom.
wam_instruction_to_cil_literal(get_constant(C, Ai), Lit) :-
    cil_pack_value(C, PackedVal),
    cil_const_tag(C, Tag),
    cil_reg_name_to_index(Ai, RegIdx),
    Op2 is (Tag << 16) \/ RegIdx,
    format(atom(Lit), 'new Instruction(0, ~wL, ~wL)', [PackedVal, Op2]).
wam_instruction_to_cil_literal(get_variable(Xn, Ai), Lit) :-
    cil_reg_name_to_index(Xn, XnIdx),
    cil_reg_name_to_index(Ai, AiIdx),
    format(atom(Lit), 'new Instruction(1, ~wL, ~wL)', [XnIdx, AiIdx]).
wam_instruction_to_cil_literal(get_value(Xn, Ai), Lit) :-
    cil_reg_name_to_index(Xn, XnIdx),
    cil_reg_name_to_index(Ai, AiIdx),
    format(atom(Lit), 'new Instruction(2, ~wL, ~wL)', [XnIdx, AiIdx]).
wam_instruction_to_cil_literal(put_constant(C, Ai), Lit) :-
    cil_pack_value(C, PackedVal),
    cil_const_tag(C, Tag),
    cil_reg_name_to_index(Ai, RegIdx),
    Op2 is (Tag << 16) \/ RegIdx,
    format(atom(Lit), 'new Instruction(8, ~wL, ~wL)', [PackedVal, Op2]).
wam_instruction_to_cil_literal(put_variable(Xn, Ai), Lit) :-
    cil_reg_name_to_index(Xn, XnIdx),
    cil_reg_name_to_index(Ai, AiIdx),
    format(atom(Lit), 'new Instruction(9, ~wL, ~wL)', [XnIdx, AiIdx]).
wam_instruction_to_cil_literal(put_value(Xn, Ai), Lit) :-
    cil_reg_name_to_index(Xn, XnIdx),
    cil_reg_name_to_index(Ai, AiIdx),
    format(atom(Lit), 'new Instruction(10, ~wL, ~wL)', [XnIdx, AiIdx]).
wam_instruction_to_cil_literal(allocate, 'new Instruction(16, 0L, 0L)').
wam_instruction_to_cil_literal(deallocate, 'new Instruction(17, 0L, 0L)').
wam_instruction_to_cil_literal(proceed, 'new Instruction(20, 0L, 0L)').
wam_instruction_to_cil_literal(trust_me, 'new Instruction(24, 0L, 0L)').
wam_instruction_to_cil_literal(builtin_call(Op, N), Lit) :-
    builtin_op_to_cil_id(Op, OpId),
    format(atom(Lit), 'new Instruction(21, ~wL, ~wL)', [OpId, N]).
% Label-referencing: error on /2 (use text parser with label resolution)
wam_instruction_to_cil_literal(call(P, _), _) :-
    throw(error(label_resolution_required(call, P),
          'Use text parser path with label resolution for call/execute/try_me_else/retry_me_else')).
wam_instruction_to_cil_literal(execute(P), _) :-
    throw(error(label_resolution_required(execute, P), _)).
wam_instruction_to_cil_literal(try_me_else(L), _) :-
    throw(error(label_resolution_required(try_me_else, L), _)).
wam_instruction_to_cil_literal(retry_me_else(L), _) :-
    throw(error(label_resolution_required(retry_me_else, L), _)).
% Fallback
wam_instruction_to_cil_literal(Instr, Lit) :-
    format(atom(Lit), '// TODO: ~w', [Instr]).

% --- Value packing (atoms → string hash, integers → identity) ---
% CIL uses strings for atoms, but instruction Op1/Op2 are int64,
% so we pack atom identity as a simple sequential ID.

:- dynamic cil_atom_table_entry/2.
:- dynamic cil_atom_table_next_id/1.
cil_atom_table_next_id(1).

%% cil_atom_table_reset
%  Clear the atom table between top-level compilation calls.
%  Prevents atom ID bleed-through in long-running sessions or
%  multi-module transpilation.
cil_atom_table_reset :-
    retractall(cil_atom_table_entry(_, _)),
    retractall(cil_atom_table_next_id(_)),
    assertz(cil_atom_table_next_id(1)).

cil_intern_atom(AtomName, Id) :-
    (   cil_atom_table_entry(AtomName, Id)
    ->  true
    ;   retract(cil_atom_table_next_id(Id)),
        NextId is Id + 1,
        assertz(cil_atom_table_next_id(NextId)),
        assertz(cil_atom_table_entry(AtomName, Id))
    ).

cil_pack_value(atom(A), Packed) :- !, cil_intern_atom(A, Packed).
cil_pack_value(integer(I), I) :- !.
cil_pack_value(N, N) :- integer(N), !.
cil_pack_value(N, Packed) :- float(N), !, Packed is truncate(N).
cil_pack_value(A, Packed) :- atom(A), !, cil_intern_atom(A, Packed).
cil_pack_value(_, 0).

%% cil_const_tag(+Const, -Tag): 1 = numeric (IntegerValue), 0 = atom.
cil_const_tag(integer(_), 1) :- !.
cil_const_tag(N, 1) :- number(N), !.
cil_const_tag(_, 0).

% --- Builtin op name → integer ID mapping (shared with LLVM) ---

builtin_op_to_cil_id('is/2', 0).
builtin_op_to_cil_id('>/2', 1).
builtin_op_to_cil_id('</2', 2).
builtin_op_to_cil_id('>=/2', 3).
builtin_op_to_cil_id('=</2', 4).
builtin_op_to_cil_id('=:=/2', 5).
builtin_op_to_cil_id('=\\=/2', 6).
builtin_op_to_cil_id('==/2', 7).
builtin_op_to_cil_id('true/0', 8).
builtin_op_to_cil_id('fail/0', 9).
builtin_op_to_cil_id('!/0', 10).
% Harmonization invariant (test_wam_cross_target_consistency): ids of
% builtins implemented by BOTH the LLVM and CIL targets must agree, so
% these use LLVM's numbering (11/12/14-17/20-23 are LLVM builtins the
% CIL runtime does not implement; its switch routes them to the
% unknown-op default).
builtin_op_to_cil_id('atom/1', 13).
builtin_op_to_cil_id('var/1', 18).
builtin_op_to_cil_id('nonvar/1', 19).
builtin_op_to_cil_id('=/2', 24).
builtin_op_to_cil_id(_, 99).

% ============================================================================
% WAM line parser → CIL (from WAM assembly text)
% ============================================================================

compile_wam_predicate_to_cil(Pred/Arity, WamCode, Options, CILCode) :-
    atom_string(Pred, PredStr),
    option(class_name(ClassName), Options, 'PrologGenerated.Program'),
    atom_string(WamCode, WamStr),
    split_string(WamStr, "\n", "", Lines),
    wam_lines_pass1_cil(Lines, 0, RawInstrs, LabelEntries),
    build_label_index_map_cil(LabelEntries, LabelMap),
    maplist(resolve_cil_literal(LabelMap), RawInstrs, CILLiterals),
    length(CILLiterals, InstrCount),
    length(LabelEntries, LabelCount),
    % Build CIL initializer to populate the code array.
    % NOTE: `.cctor_NAME` was both an invalid IL identifier (leading dot)
    % and never invoked; the initializer is now a valid `init_NAME` method
    % lazily called by the wrapper before first use.
    format(atom(ClassName_PredStr), '~w::~w_code', [ClassName, PredStr]),
    number_cil_instrs(CILLiterals, 0, ClassName_PredStr, NumberedInstrs),
    atomic_list_concat(NumberedInstrs, '\n', InstrStoreCode),
    % labels[LabelIdx] = PC, where LabelIdx is the sequential label index
    % used by lookup_label_cil (NOT labels[PC] = PC, which both overran
    % the array and made every LabelPC lookup wrong).
    findall(Entry,
        (   nth0(LabelIdx, LabelEntries, _Name-PC),
            format(atom(Entry),
                '    ldsfld int32[] ~w::~w_labels\n    ldc.i4 ~w\n    ldc.i4 ~w\n    stelem.i4',
                [ClassName, PredStr, LabelIdx, PC])
        ), LabelStoreCode),
    atomic_list_concat(LabelStoreCode, '\n', LabelStoreStr),
    build_cil_arg_setup(PredStr, Arity, ClassName, ArgSetup),
    build_cil_param_list(Arity, ParamList),
    format(atom(CILCode),
'// WAM-compiled predicate: ~w/~w
.field public static class Instruction[] ~w_code
.field public static int32[] ~w_labels

// Initializer for ~w (invoked lazily by the wrapper)
.method public static void init_~w() cil managed {
    .maxstack 8
    // Create instruction array
    ldc.i4 ~w
    newarr Instruction
    stsfld class Instruction[] ~w::~w_code
~w
    // Create label array
    ldc.i4 ~w
    newarr [mscorlib]System.Int32
    stsfld int32[] ~w::~w_labels
~w
    ret
}

// Wrapper method
.method public static bool ~w(~w) cil managed {
    .maxstack 4
    .locals init (class WamState vm)
    // Lazy one-time initialization of the code/label arrays
    ldsfld class Instruction[] ~w::~w_code
    brtrue L_inited
    call void ~w::init_~w()
L_inited:
    ldsfld class Instruction[] ~w::~w_code
    ldsfld int32[] ~w::~w_labels
    newobj instance void WamState::.ctor(class Instruction[], int32[])
    stloc.0
~w
    ldloc.0
    call bool ~w::run_loop(class WamState)
    ret
}
', [PredStr, Arity,
    PredStr, PredStr,
    PredStr, PredStr,
    InstrCount, ClassName, PredStr, InstrStoreCode,
    LabelCount, ClassName, PredStr, LabelStoreStr,
    PredStr, ParamList,
    ClassName, PredStr,
    ClassName, PredStr,
    ClassName, PredStr, ClassName, PredStr,
    ArgSetup, ClassName]).

%% Two-pass parsing (same approach as LLVM target)

wam_lines_pass1_cil([], _, [], []).
wam_lines_pass1_cil([Line|Rest], PC, RawInstrs, Labels) :-
    split_string(Line, " \t", " \t", Parts),
    delete(Parts, "", CleanParts),
    (   CleanParts == []
    ->  wam_lines_pass1_cil(Rest, PC, RawInstrs, Labels)
    ;   CleanParts = [First|_],
        (   sub_string(First, _, 1, 0, ":")
        ->  sub_string(First, 0, _, 1, LabelName),
            Labels = [LabelName-PC | RestLabels],
            wam_lines_pass1_cil(Rest, PC, RawInstrs, RestLabels)
        ;   RawInstrs = [CleanParts | RestInstrs],
            NPC is PC + 1,
            wam_lines_pass1_cil(Rest, NPC, RestInstrs, Labels)
        )
    ).

build_label_index_map_cil(LabelEntries, LabelMap) :-
    foldl(add_label_entry_cil, LabelEntries, 0-[], _-LabelMap).

add_label_entry_cil(Name-_PC, Idx-Map, NextIdx-[Name-Idx|Map]) :-
    NextIdx is Idx + 1.

resolve_cil_literal(LabelMap, Parts, CILLit) :-
    wam_line_to_cil_literal_resolved(Parts, LabelMap, CILLit).

% Label-referencing instructions
wam_line_to_cil_literal_resolved(["call", P, N], LabelMap, Lit) :- !,
    clean_comma_cil(P, CP), clean_comma_cil(N, CN),
    (   number_string(Arity, CN) -> true ; Arity = 0 ),
    atom_string(CP, CPAtom),
    lookup_label_cil(CPAtom, LabelMap, LabelIdx),
    format(atom(Lit), 'new Instruction(18, ~wL, ~wL)', [LabelIdx, Arity]).
wam_line_to_cil_literal_resolved(["execute", P], LabelMap, Lit) :- !,
    clean_comma_cil(P, CP),
    atom_string(CP, CPAtom),
    lookup_label_cil(CPAtom, LabelMap, LabelIdx),
    format(atom(Lit), 'new Instruction(19, ~wL, 0L)', [LabelIdx]).
wam_line_to_cil_literal_resolved(["try_me_else", L], LabelMap, Lit) :- !,
    clean_comma_cil(L, CL),
    atom_string(CL, CLAtom),
    lookup_label_cil(CLAtom, LabelMap, LabelIdx),
    format(atom(Lit), 'new Instruction(22, ~wL, 0L)', [LabelIdx]).
wam_line_to_cil_literal_resolved(["retry_me_else", L], LabelMap, Lit) :- !,
    clean_comma_cil(L, CL),
    atom_string(CL, CLAtom),
    lookup_label_cil(CLAtom, LabelMap, LabelIdx),
    format(atom(Lit), 'new Instruction(23, ~wL, 0L)', [LabelIdx]).
% jump is label-referencing too: the if-then-else lowering ends the
% then-branch with `jump L_ite_cont_N` (tag 26 = L_jump in step).
wam_line_to_cil_literal_resolved(["jump", L], LabelMap, Lit) :- !,
    clean_comma_cil(L, CL),
    atom_string(CL, CLAtom),
    lookup_label_cil(CLAtom, LabelMap, LabelIdx),
    format(atom(Lit), 'new Instruction(26, ~wL, 0L)', [LabelIdx]).
wam_line_to_cil_literal_resolved(Parts, _LabelMap, Lit) :-
    wam_line_to_cil_literal(Parts, Lit).

lookup_label_cil(LabelName, LabelMap, Index) :-
    (   member(LabelName-Index, LabelMap)
    ->  true
    ;   format(user_error,
            'Warning: unknown label "~w" in WAM CIL codegen, defaulting to index 0~n',
            [LabelName]),
        Index = 0
    ).

% Non-label instructions
wam_line_to_cil_literal(["get_constant", C, Ai], Lit) :-
    clean_comma_cil(C, CC), clean_comma_cil(Ai, CAi),
    cil_pack_value_str_tagged(CC, PackedVal, Tag),
    atom_string(CAi, CAiAtom),
    cil_reg_name_to_index(CAiAtom, RegIdx),
    Op2 is (Tag << 16) \/ RegIdx,
    format(atom(Lit), 'new Instruction(0, ~wL, ~wL)', [PackedVal, Op2]).
wam_line_to_cil_literal(["get_variable", Xn, Ai], Lit) :-
    clean_comma_cil(Xn, CXn), clean_comma_cil(Ai, CAi),
    atom_string(CXn, CXnAtom), atom_string(CAi, CAiAtom),
    cil_reg_name_to_index(CXnAtom, XnIdx),
    cil_reg_name_to_index(CAiAtom, AiIdx),
    format(atom(Lit), 'new Instruction(1, ~wL, ~wL)', [XnIdx, AiIdx]).
wam_line_to_cil_literal(["get_value", Xn, Ai], Lit) :-
    clean_comma_cil(Xn, CXn), clean_comma_cil(Ai, CAi),
    atom_string(CXn, CXnAtom), atom_string(CAi, CAiAtom),
    cil_reg_name_to_index(CXnAtom, XnIdx),
    cil_reg_name_to_index(CAiAtom, AiIdx),
    format(atom(Lit), 'new Instruction(2, ~wL, ~wL)', [XnIdx, AiIdx]).
wam_line_to_cil_literal(["put_constant", C, Ai], Lit) :-
    clean_comma_cil(C, CC), clean_comma_cil(Ai, CAi),
    cil_pack_value_str_tagged(CC, PackedVal, Tag),
    atom_string(CAi, CAiAtom),
    cil_reg_name_to_index(CAiAtom, RegIdx),
    Op2 is (Tag << 16) \/ RegIdx,
    format(atom(Lit), 'new Instruction(8, ~wL, ~wL)', [PackedVal, Op2]).
wam_line_to_cil_literal(["put_variable", Xn, Ai], Lit) :-
    clean_comma_cil(Xn, CXn), clean_comma_cil(Ai, CAi),
    atom_string(CXn, CXnAtom), atom_string(CAi, CAiAtom),
    cil_reg_name_to_index(CXnAtom, XnIdx),
    cil_reg_name_to_index(CAiAtom, AiIdx),
    format(atom(Lit), 'new Instruction(9, ~wL, ~wL)', [XnIdx, AiIdx]).
wam_line_to_cil_literal(["put_value", Xn, Ai], Lit) :-
    clean_comma_cil(Xn, CXn), clean_comma_cil(Ai, CAi),
    atom_string(CXn, CXnAtom), atom_string(CAi, CAiAtom),
    cil_reg_name_to_index(CXnAtom, XnIdx),
    cil_reg_name_to_index(CAiAtom, AiIdx),
    format(atom(Lit), 'new Instruction(10, ~wL, ~wL)', [XnIdx, AiIdx]).
wam_line_to_cil_literal(["allocate"], 'new Instruction(16, 0L, 0L)').
wam_line_to_cil_literal(["deallocate"], 'new Instruction(17, 0L, 0L)').
wam_line_to_cil_literal(["proceed"], 'new Instruction(20, 0L, 0L)').
wam_line_to_cil_literal(["trust_me"], 'new Instruction(24, 0L, 0L)').
% cut_ite commits an if-then-else by popping the guard choice point
% (tag 25 = L_cut_ite in step). Previously it fell to the '// TODO'
% comment path, leaving a null instruction slot that aborted the query.
wam_line_to_cil_literal(["cut_ite"], 'new Instruction(25, 0L, 0L)').
wam_line_to_cil_literal(["builtin_call", Op, N], Lit) :-
    clean_comma_cil(Op, COp), clean_comma_cil(N, CN),
    (   number_string(Num, CN) -> true ; Num = 0 ),
    % M147 sweep finding: atom_string/2 args were reversed, so COpAtom
    % stayed a string and never matched the atom-keyed
    % builtin_op_to_cil_id/2 table - every builtin compiled to the
    % unknown-op fallback id and unconditionally failed at runtime.
    atom_string(COpAtom, COp),
    ( builtin_op_to_cil_id(COpAtom, OpId) -> true ; OpId = 99 ),
    format(atom(Lit), 'new Instruction(21, ~wL, ~wL)', [OpId, Num]).
wam_line_to_cil_literal(Parts, Lit) :-
    atomic_list_concat(Parts, " ", Line),
    format(atom(Lit), '// TODO: ~w', [Line]).

% --- Utilities ---

clean_comma_cil(S, Clean) :-
    (   sub_string(S, Before, 1, 0, ",")
    ->  sub_string(S, 0, Before, 1, Clean)
    ;   Clean = S
    ).

cil_pack_value_str(Str, Packed) :-
    cil_pack_value_str_tagged(Str, Packed, _Tag).

%% cil_pack_value_str_tagged(+Str, -Packed, -Tag)
%  Tag 1 = numeric constant, 0 = atom (interned id).
cil_pack_value_str_tagged(Str, Packed, Tag) :-
    (   number_string(N, Str)
    ->  Packed = N, Tag = 1
    ;   atom_string(A, Str),
        cil_pack_value(atom(A), Packed),
        Tag = 0
    ).

number_cil_instrs([], _, _, []).
number_cil_instrs([Lit|Rest], Idx, ClassName_PredStr, [StoreCode|RestCodes]) :-
    % The literal is the intermediate form "new Instruction(T, XL, YL)"
    % (kept for the structural tests); lower it here to the real IL
    % construction sequence — splicing the C#-style pseudo-code into the
    % method body was a syntax error that broke every generated module.
    (   parse_cil_instr_literal(Lit, Tag, Op1, Op2)
    ->  format(atom(StoreCode),
            '    ldsfld class Instruction[] ~w\n    ldc.i4 ~w\n    ldc.i4 ~w\n    ldc.i8 ~w\n    ldc.i8 ~w\n    newobj instance void Instruction::.ctor(int32, int64, int64)\n    stelem.ref',
            [ClassName_PredStr, Idx, Tag, Op1, Op2])
    ;   % Comment or TODO — skip (leaves a null slot; should not happen
        % for any instruction the WAM compiler currently emits)
        format(atom(StoreCode), '    // [~w] ~w', [Idx, Lit])
    ),
    NextIdx is Idx + 1,
    number_cil_instrs(Rest, NextIdx, ClassName_PredStr, RestCodes).

%% parse_cil_instr_literal(+Lit, -Tag, -Op1, -Op2)
%  Decompose 'new Instruction(T, XL, YL)' into its integer components.
parse_cil_instr_literal(Lit, Tag, Op1, Op2) :-
    atom(Lit),
    atom_concat('new Instruction(', Rest, Lit),
    atom_concat(ArgsAtom, ')', Rest),
    atomic_list_concat([TagA, Op1A, Op2A], ', ', ArgsAtom),
    atom_number(TagA, Tag),
    atom_concat(Op1Num, 'L', Op1A), atom_number(Op1Num, Op1),
    atom_concat(Op2Num, 'L', Op2A), atom_number(Op2Num, Op2).

build_cil_param_list(0, "class WamState vm") :- !.
build_cil_param_list(Arity, ParamList) :-
    numlist(1, Arity, Indices),
    maplist([I, S]>>(format(atom(S), "class Value a~w", [I])), Indices, Parts),
    atomic_list_concat(['class WamState vm'|Parts], ', ', ParamList).

build_cil_arg_setup(_, 0, _, "") :- !.
build_cil_arg_setup(_PredStr, Arity, _ClassName, Setup) :-
    numlist(1, Arity, Indices),
    maplist([I, S]>>(
        RegIdx is I - 1,
        format(atom(S),
            '    ldloc.0\n    ldc.i4 ~w\n    ldarg.~w\n    callvirt instance void WamState::SetReg(int32, class Value)',
            [RegIdx, I])
    ), Indices, Parts),
    atomic_list_concat(Parts, '\n', Setup).
