:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wam_jvm_target.pl - WAM-to-JVM Assembly Transpilation Target
%
% Transpiles WAM runtime to JVM bytecode in both Jamaica and Krakatau
% assembly formats. Parameterized on OutputFmt (jamaica | krakatau).
%
% Phase 2: WAM instructions → tableswitch cases (JVM bytecode)
% Phase 3: Helper methods → JVM method bodies
%
% Reuses jvm_bytecode.pl for type descriptors and constant loading.

:- module(wam_jvm_target, [
    compile_step_wam_to_jvm/3,         % +OutputFmt, +Options, -Code
    compile_wam_helpers_to_jvm/3,       % +OutputFmt, +Options, -Code
    compile_wam_runtime_to_jvm/3,       % +OutputFmt, +Options, -Code
    compile_wam_predicate_to_jvm/5,     % +Pred/Arity, +WamCode, +OutputFmt, +Options, -Code
    write_wam_jvm_project/4,            % +Predicates, +OutputFmt, +Options, +ProjectDir
    jvm_wam_var_style/2,                % +OutputFmt, -VarStyle
    jvm_wam_comment/3,                  % +OutputFmt, +Text, -Line
    jvm_wam_class_header/3,             % +OutputFmt, +ClassName, -Code
    jvm_wam_class_footer/2,             % +OutputFmt, -Code
    jvm_wam_method_header/6,            % +OutputFmt, +Name, +Descriptor, +SL, +LL, -Code
    jvm_wam_method_footer/2,            % +OutputFmt, -Code
    jvm_wam_invoke/6,                   % +OutputFmt, +Kind, +Class, +Method, +Desc, -Instr
    wam_jvm_case/2                      % +InstrName, -BodyCode
]).

:- use_module(library(lists)).
:- use_module(library(option)).
:- use_module(library(filesex), [make_directory_path/1, directory_file_path/3]).
:- use_module('../core/template_system').
:- use_module('../core/jvm_bytecode', [
    jvm_type_descriptor/2,
    jvm_method_descriptor/3,
    jvm_load_const/2
]).
:- use_module('../bindings/jvm_wam_bindings').
:- use_module('../targets/wam_target', [compile_predicate_to_wam/3]).

:- discontiguous wam_jvm_case/2.

% ============================================================================
% FORMAT DISPATCH HELPERS
% ============================================================================

%% jvm_wam_var_style(+OutputFmt, -VarStyle)
jvm_wam_var_style(jamaica, symbolic).
jvm_wam_var_style(krakatau, numeric).

%% jvm_wam_comment(+OutputFmt, +Text, -Line)
jvm_wam_comment(jamaica, Text, Line) :-
    format(string(Line), '// ~w', [Text]).
jvm_wam_comment(krakatau, Text, Line) :-
    format(string(Line), '; ~w', [Text]).

%% jvm_wam_class_header(+OutputFmt, +ClassName, -Code)
jvm_wam_class_header(jamaica, ClassName, Code) :-
    format(string(Code),
'public class ~w {
    %default_constructor public
', [ClassName]).
jvm_wam_class_header(krakatau, ClassName, Code) :-
    format(string(Code),
'.version 52 0
.class public ~w
.super java/lang/Object

', [ClassName]).

%% jvm_wam_class_footer(+OutputFmt, -Code)
jvm_wam_class_footer(jamaica, '}').
jvm_wam_class_footer(krakatau, '').

%% jvm_wam_method_header(+OutputFmt, +Name, +Descriptor, +StackLimit, +LocalsLimit, -Code)
jvm_wam_method_header(jamaica, Name, _Descriptor, _SL, _LL, Code) :-
    format(string(Code),
'    public static int ~w(int arg0) {', [Name]).
jvm_wam_method_header(krakatau, Name, Descriptor, StackLimit, LocalsLimit, Code) :-
    format(string(Code),
'.method public static ~w : ~w
    .limit stack ~w
    .limit locals ~w', [Name, Descriptor, StackLimit, LocalsLimit]).

%% jvm_wam_method_footer(+OutputFmt, -Code)
jvm_wam_method_footer(jamaica, '    }').
jvm_wam_method_footer(krakatau, '.end method').

%% jvm_wam_field_decl(+OutputFmt, +Access, +Name, +Type, -Code)
jvm_wam_field_decl(jamaica, Access, Name, Type, Code) :-
    format(string(Code), '    ~w ~w ~w;', [Access, Type, Name]).
jvm_wam_field_decl(krakatau, _Access, Name, TypeDesc, Code) :-
    format(string(Code), '.field public ~w ~w', [Name, TypeDesc]).

%% jvm_wam_invoke(+OutputFmt, +Kind, +Class, +Method, +Descriptor, -Instr)
%  Generate a method invocation instruction.
%  Kind is one of: invokevirtual, invokestatic, invokespecial, invokeinterface.
jvm_wam_invoke(jamaica, invokevirtual, Class, Method, _Desc, Instr) :-
    format(string(Instr), '    invokevirtual ~w.~w', [Class, Method]).
jvm_wam_invoke(jamaica, invokestatic, Class, Method, _Desc, Instr) :-
    format(string(Instr), '    invokestatic ~w.~w', [Class, Method]).
jvm_wam_invoke(jamaica, invokespecial, Class, Method, _Desc, Instr) :-
    format(string(Instr), '    invokespecial ~w.~w', [Class, Method]).
jvm_wam_invoke(krakatau, invokevirtual, Class, Method, Desc, Instr) :-
    format(string(Instr), '    invokevirtual ~w ~w ~w', [Class, Method, Desc]).
jvm_wam_invoke(krakatau, invokestatic, Class, Method, Desc, Instr) :-
    format(string(Instr), '    invokestatic ~w ~w ~w', [Class, Method, Desc]).
jvm_wam_invoke(krakatau, invokespecial, Class, Method, Desc, Instr) :-
    format(string(Instr), '    invokespecial ~w ~w ~w', [Class, Method, Desc]).

%% jvm_wam_getfield(+OutputFmt, +Class, +Field, +TypeDesc, -Instr)
jvm_wam_getfield(jamaica, _Class, Field, _TypeDesc, Instr) :-
    format(string(Instr), '    getfield ~w', [Field]).
jvm_wam_getfield(krakatau, Class, Field, TypeDesc, Instr) :-
    format(string(Instr), '    getfield ~w ~w ~w', [Class, Field, TypeDesc]).

%% jvm_wam_putfield(+OutputFmt, +Class, +Field, +TypeDesc, -Instr)
jvm_wam_putfield(jamaica, _Class, Field, _TypeDesc, Instr) :-
    format(string(Instr), '    putfield ~w', [Field]).
jvm_wam_putfield(krakatau, Class, Field, TypeDesc, Instr) :-
    format(string(Instr), '    putfield ~w ~w ~w', [Class, Field, TypeDesc]).

% ============================================================================
% PHASE 2: WAM Instructions → JVM tableswitch cases
% ============================================================================

%% compile_step_wam_to_jvm(+OutputFmt, +Options, -Code)
%  Generates the step() method body with a tableswitch on instruction tag.
compile_step_wam_to_jvm(OutputFmt, _Options, Code) :-
    findall(Case, compile_jvm_step_case(OutputFmt, Case), Cases),
    atomic_list_concat(Cases, '\n', CasesCode),
    jvm_wam_comment(OutputFmt, 'step(instr) - Execute one WAM instruction', Comment),
    format(string(Code),
'~w
~w', [Comment, CasesCode]).

compile_jvm_step_case(OutputFmt, CaseCode) :-
    wam_jvm_case(InstrName, BodyCode),
    jvm_wam_comment(OutputFmt, InstrName, CmtLine),
    format(string(CaseCode), '~w\n~w', [CmtLine, BodyCode]).

% --- Head Unification Instructions ---

wam_jvm_case(get_constant,
'        aload_0
        getfield WamState/regs Ljava/util/HashMap;
        aload_1
        invokevirtual java/util/HashMap get (Ljava/lang/Object;)Ljava/lang/Object;
        dup
        ifnull L_gc_fail
        aload_2
        invokevirtual java/lang/Object equals (Ljava/lang/Object;)Z
        ifne L_gc_match
        dup
        instanceof WamUnbound
        ifeq L_gc_fail
        pop
        aload_0
        aload_1
        aload_2
        invokevirtual WamState trailAndBind (Ljava/lang/String;Ljava/lang/Object;)V
        aload_0
        dup
        getfield WamState/pc I
        iconst_1
        iadd
        putfield WamState/pc I
        iconst_1
        ireturn
    L_gc_match:
        pop
        aload_0
        dup
        getfield WamState/pc I
        iconst_1
        iadd
        putfield WamState/pc I
        iconst_1
        ireturn
    L_gc_fail:
        pop
        iconst_0
        ireturn').

wam_jvm_case(get_variable,
'        aload_0
        getfield WamState/regs Ljava/util/HashMap;
        aload_1
        invokevirtual java/util/HashMap get (Ljava/lang/Object;)Ljava/lang/Object;
        astore_3
        aload_0
        aload_1
        invokevirtual WamState trailBinding (Ljava/lang/String;)V
        aload_0
        aload_1
        aload_3
        invokevirtual WamState putReg (Ljava/lang/String;Ljava/lang/Object;)V
        aload_0
        dup
        getfield WamState/pc I
        iconst_1
        iadd
        putfield WamState/pc I
        iconst_1
        ireturn').

wam_jvm_case(get_value,
'        aload_0
        getfield WamState/regs Ljava/util/HashMap;
        aload_1
        invokevirtual java/util/HashMap get (Ljava/lang/Object;)Ljava/lang/Object;
        astore_3
        aload_0
        aload_2
        invokevirtual WamState getReg (Ljava/lang/String;)Ljava/lang/Object;
        astore 4
        aload_0
        aload_3
        aload 4
        invokevirtual WamState unify (Ljava/lang/Object;Ljava/lang/Object;)Z
        ifeq L_gv_fail
        aload_0
        dup
        getfield WamState/pc I
        iconst_1
        iadd
        putfield WamState/pc I
        iconst_1
        ireturn
    L_gv_fail:
        iconst_0
        ireturn').

wam_jvm_case(put_constant,
'        aload_0
        getfield WamState/regs Ljava/util/HashMap;
        aload_1
        aload_2
        invokevirtual java/util/HashMap put (Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;
        pop
        aload_0
        dup
        getfield WamState/pc I
        iconst_1
        iadd
        putfield WamState/pc I
        iconst_1
        ireturn').

wam_jvm_case(put_variable,
'        aload_0
        aload_0
        getfield WamState/pc I
        invokevirtual WamState freshUnbound (I)Ljava/lang/Object;
        astore_3
        aload_0
        aload_1
        invokevirtual WamState trailBinding (Ljava/lang/String;)V
        aload_0
        aload_1
        aload_3
        invokevirtual WamState putReg (Ljava/lang/String;Ljava/lang/Object;)V
        aload_0
        aload_2
        aload_3
        invokevirtual WamState putReg (Ljava/lang/String;Ljava/lang/Object;)V
        aload_0
        dup
        getfield WamState/pc I
        iconst_1
        iadd
        putfield WamState/pc I
        iconst_1
        ireturn').

wam_jvm_case(put_value,
'        aload_0
        aload_1
        invokevirtual WamState getReg (Ljava/lang/String;)Ljava/lang/Object;
        astore_3
        aload_0
        aload_2
        invokevirtual WamState trailBinding (Ljava/lang/String;)V
        aload_0
        getfield WamState/regs Ljava/util/HashMap;
        aload_2
        aload_3
        invokevirtual java/util/HashMap put (Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;
        pop
        aload_0
        dup
        getfield WamState/pc I
        iconst_1
        iadd
        putfield WamState/pc I
        iconst_1
        ireturn').

wam_jvm_case(proceed,
'        aload_0
        getfield WamState/cp I
        aload_0
        swap
        putfield WamState/pc I
        iconst_1
        ireturn').

wam_jvm_case(allocate,
'        aload_0
        invokevirtual WamState allocateEnv ()V
        aload_0
        dup
        getfield WamState/pc I
        iconst_1
        iadd
        putfield WamState/pc I
        iconst_1
        ireturn').

wam_jvm_case(deallocate,
'        aload_0
        invokevirtual WamState deallocateEnv ()V
        aload_0
        dup
        getfield WamState/pc I
        iconst_1
        iadd
        putfield WamState/pc I
        iconst_1
        ireturn').

wam_jvm_case(call,
'        aload_0
        dup
        getfield WamState/pc I
        iconst_1
        iadd
        putfield WamState/cp I
        aload_0
        aload_1
        invokevirtual WamState resolveLabel (Ljava/lang/String;)I
        putfield WamState/pc I
        iconst_1
        ireturn').

wam_jvm_case(execute,
'        aload_0
        aload_1
        invokevirtual WamState resolveLabel (Ljava/lang/String;)I
        aload_0
        swap
        putfield WamState/pc I
        iconst_1
        ireturn').

wam_jvm_case(try_me_else,
'        aload_0
        aload_1
        invokevirtual WamState pushChoicePoint (Ljava/lang/String;)V
        aload_0
        dup
        getfield WamState/pc I
        iconst_1
        iadd
        putfield WamState/pc I
        iconst_1
        ireturn').

wam_jvm_case(retry_me_else,
'        aload_0
        aload_1
        invokevirtual WamState updateChoicePoint (Ljava/lang/String;)V
        aload_0
        dup
        getfield WamState/pc I
        iconst_1
        iadd
        putfield WamState/pc I
        iconst_1
        ireturn').

wam_jvm_case(trust_me,
'        aload_0
        invokevirtual WamState popChoicePoint ()V
        aload_0
        dup
        getfield WamState/pc I
        iconst_1
        iadd
        putfield WamState/pc I
        iconst_1
        ireturn').

wam_jvm_case(set_variable,
'        aload_0
        getfield WamState/heap Ljava/util/ArrayList;
        invokevirtual java/util/ArrayList size ()I
        istore_3
        aload_0
        iload_3
        invokevirtual WamState freshHeapUnbound (I)Ljava/lang/Object;
        astore 4
        aload_0
        getfield WamState/heap Ljava/util/ArrayList;
        aload 4
        invokevirtual java/util/ArrayList add (Ljava/lang/Object;)Z
        pop
        aload_0
        aload_1
        aload 4
        invokevirtual WamState putReg (Ljava/lang/String;Ljava/lang/Object;)V
        aload_0
        dup
        getfield WamState/pc I
        iconst_1
        iadd
        putfield WamState/pc I
        iconst_1
        ireturn').

wam_jvm_case(set_value,
'        aload_0
        aload_1
        invokevirtual WamState getReg (Ljava/lang/String;)Ljava/lang/Object;
        astore_3
        aload_0
        getfield WamState/heap Ljava/util/ArrayList;
        aload_3
        invokevirtual java/util/ArrayList add (Ljava/lang/Object;)Z
        pop
        aload_0
        dup
        getfield WamState/pc I
        iconst_1
        iadd
        putfield WamState/pc I
        iconst_1
        ireturn').

wam_jvm_case(set_constant,
'        aload_0
        getfield WamState/heap Ljava/util/ArrayList;
        aload_1
        invokevirtual java/util/ArrayList add (Ljava/lang/Object;)Z
        pop
        aload_0
        dup
        getfield WamState/pc I
        iconst_1
        iadd
        putfield WamState/pc I
        iconst_1
        ireturn').

wam_jvm_case(unify_variable,
'        aload_0
        aload_1
        invokevirtual WamState stepUnifyVariable (Ljava/lang/String;)Z
        ireturn').

wam_jvm_case(unify_value,
'        aload_0
        aload_1
        invokevirtual WamState stepUnifyValue (Ljava/lang/String;)Z
        ireturn').

wam_jvm_case(unify_constant,
'        aload_0
        aload_1
        invokevirtual WamState stepUnifyConstant (Ljava/lang/Object;)Z
        ireturn').

wam_jvm_case(get_structure,
'        aload_0
        aload_1
        aload_2
        invokevirtual WamState stepGetStructure (Ljava/lang/String;Ljava/lang/String;)Z
        ireturn').

wam_jvm_case(get_list,
'        aload_0
        aload_1
        invokevirtual WamState stepGetList (Ljava/lang/String;)Z
        ireturn').

wam_jvm_case(put_structure,
'        aload_0
        aload_1
        aload_2
        invokevirtual WamState stepPutStructure (Ljava/lang/String;Ljava/lang/String;)Z
        ireturn').

wam_jvm_case(put_list,
'        aload_0
        aload_1
        invokevirtual WamState stepPutList (Ljava/lang/String;)Z
        ireturn').

wam_jvm_case(switch_on_constant,
'        aload_0
        aload_1
        invokevirtual WamState stepSwitchOnConstant (Ljava/lang/Object;)Z
        ireturn').

wam_jvm_case(builtin_call,
'        aload_0
        aload_1
        aload_2
        invokevirtual WamState executeBuiltin (Ljava/lang/String;I)Z
        ireturn').

% ============================================================================
% PHASE 3: Helper methods → JVM method bodies
% ============================================================================

%% compile_wam_helpers_to_jvm(+OutputFmt, +Options, -Code)
compile_wam_helpers_to_jvm(OutputFmt, _Options, Code) :-
    compile_run_loop_to_jvm(OutputFmt, RunCode),
    compile_backtrack_to_jvm(OutputFmt, BTCode),
    compile_unwind_trail_to_jvm(OutputFmt, UnwindCode),
    atomic_list_concat([RunCode, '\n', BTCode, '\n', UnwindCode], Code).

compile_run_loop_to_jvm(OutputFmt, Code) :-
    jvm_wam_comment(OutputFmt, 'run() - Main fetch-step-backtrack loop', Comment),
    format(string(Code),
'~w
    L_run_loop:
        aload_0
        getfield WamState/pc I
        ldc -1
        if_icmpeq L_run_halt
        aload_0
        invokevirtual WamState fetchAndStep ()Z
        ifne L_run_loop
        aload_0
        invokevirtual WamState backtrack ()Z
        ifne L_run_loop
        iconst_0
        ireturn
    L_run_halt:
        iconst_1
        ireturn', [Comment]).

compile_backtrack_to_jvm(OutputFmt, Code) :-
    jvm_wam_comment(OutputFmt, 'backtrack() - Restore from choice point', Comment),
    format(string(Code),
'~w
        aload_0
        getfield WamState/choicePoints Ljava/util/ArrayList;
        invokevirtual java/util/ArrayList size ()I
        ifle L_bt_fail
        aload_0
        invokevirtual WamState restoreChoicePoint ()V
        iconst_1
        ireturn
    L_bt_fail:
        iconst_0
        ireturn', [Comment]).

compile_unwind_trail_to_jvm(OutputFmt, Code) :-
    jvm_wam_comment(OutputFmt, 'unwindTrail(mark) - Undo register bindings', Comment),
    format(string(Code),
'~w
    L_unwind_loop:
        aload_0
        getfield WamState/trail Ljava/util/ArrayList;
        invokevirtual java/util/ArrayList size ()I
        iload_1
        if_icmple L_unwind_done
        aload_0
        invokevirtual WamState unwindOne ()V
        goto L_unwind_loop
    L_unwind_done:
        return', [Comment]).

% ============================================================================
% ASSEMBLY: Combine Phase 2 + Phase 3
% ============================================================================

%% compile_wam_runtime_to_jvm(+OutputFmt, +Options, -Code)
compile_wam_runtime_to_jvm(OutputFmt, Options, Code) :-
    compile_step_wam_to_jvm(OutputFmt, Options, StepCode),
    compile_wam_helpers_to_jvm(OutputFmt, Options, HelpersCode),
    jvm_wam_class_header(OutputFmt, 'WamState', Header),
    jvm_wam_class_footer(OutputFmt, Footer),
    format(string(Code),
'~w
~w

~w
~w', [Header, StepCode, HelpersCode, Footer]).

% ============================================================================
% PREDICATE WRAPPER: Compile a Prolog predicate via WAM to JVM assembly
% ============================================================================

%% compile_wam_predicate_to_jvm(+Pred/Arity, +WamCode, +OutputFmt, +Options, -Code)
compile_wam_predicate_to_jvm(Pred/Arity, WamCode, OutputFmt, _Options, Code) :-
    atom_string(Pred, PredStr),
    wam_code_to_jvm_instructions(WamCode, InstrLiterals, LabelLiterals),
    jvm_wam_comment(OutputFmt,
        'WAM-compiled predicate: ' + PredStr + '/' + Arity, Comment),
    format(string(Code),
'~w
~w
~w', [Comment, InstrLiterals, LabelLiterals]).

%% wam_code_to_jvm_instructions(+WamCodeStr, -InstrLiterals, -LabelLiterals)
wam_code_to_jvm_instructions(WamCode, InstrLiterals, LabelLiterals) :-
    atom_string(WamCode, WamStr),
    split_string(WamStr, "\n", "", Lines),
    wam_lines_to_jvm(Lines, 1, InstrParts, LabelParts),
    atomic_list_concat(InstrParts, '\n', InstrLiterals),
    atomic_list_concat(LabelParts, '\n', LabelLiterals).

wam_lines_to_jvm([], _, [], []).
wam_lines_to_jvm([Line|Rest], PC, Instrs, Labels) :-
    split_string(Line, " \t,", " \t,", Parts),
    delete(Parts, "", CleanParts),
    (   CleanParts == []
    ->  wam_lines_to_jvm(Rest, PC, Instrs, Labels)
    ;   CleanParts = [First|_],
        (   sub_string(First, _, 1, 0, ":")
        ->  sub_string(First, 0, _, 1, LabelName),
            format(string(LabelInsert),
                '    ldc "~w"\n    ~w\n    invokevirtual WamState registerLabel (Ljava/lang/String;I)V',
                [LabelName, PC]),
            Labels = [LabelInsert|RestLabels],
            wam_lines_to_jvm(Rest, PC, Instrs, RestLabels)
        ;   wam_line_to_jvm_instr(CleanParts, JvmInstr),
            format(string(InstrEntry), '    ~w', [JvmInstr]),
            NPC is PC + 1,
            Instrs = [InstrEntry|RestInstrs],
            wam_lines_to_jvm(Rest, NPC, RestInstrs, Labels)
        )
    ).

%% wam_line_to_jvm_instr(+Parts, -JvmInstr)
wam_line_to_jvm_instr(["get_constant", C, Ai], Instr) :-
    clean_comma(C, CC), clean_comma(Ai, CAi),
    format(string(Instr), 'ldc "get_constant" ; ~w ~w', [CC, CAi]).
wam_line_to_jvm_instr(["get_variable", Xn, Ai], Instr) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    format(string(Instr), 'ldc "get_variable" ; ~w ~w', [CXn, CAi]).
wam_line_to_jvm_instr(["put_constant", C, Ai], Instr) :-
    clean_comma(C, CC), clean_comma(Ai, CAi),
    format(string(Instr), 'ldc "put_constant" ; ~w ~w', [CC, CAi]).
wam_line_to_jvm_instr(["proceed"], 'ldc "proceed"').
wam_line_to_jvm_instr(["call", P, N], Instr) :-
    clean_comma(P, CP), clean_comma(N, CN),
    format(string(Instr), 'ldc "call" ; ~w ~w', [CP, CN]).
wam_line_to_jvm_instr(["execute", P], Instr) :-
    clean_comma(P, CP),
    format(string(Instr), 'ldc "execute" ; ~w', [CP]).
wam_line_to_jvm_instr(["allocate", N], Instr) :-
    clean_comma(N, CN),
    format(string(Instr), 'ldc "allocate" ; ~w', [CN]).
wam_line_to_jvm_instr(["deallocate"], 'ldc "deallocate"').
wam_line_to_jvm_instr(Parts, Instr) :-
    atomic_list_concat(Parts, ' ', Combined),
    format(string(Instr), 'ldc "~w"', [Combined]).

clean_comma(S, Clean) :-
    (   sub_string(S, _, 1, 0, ",")
    ->  sub_string(S, 0, _, 1, Clean)
    ;   Clean = S
    ).

% ============================================================================
% PROJECT GENERATION
% ============================================================================

%% write_wam_jvm_project(+Predicates, +OutputFmt, +Options, +ProjectDir)
write_wam_jvm_project(Predicates, OutputFmt, Options, ProjectDir) :-
    option(class_name(ClassName), Options, 'WamProgram'),
    make_directory_path(ProjectDir),
    % Generate runtime class
    compile_wam_runtime_to_jvm(OutputFmt, Options, RuntimeCode),
    (OutputFmt == jamaica -> Ext = '.ja' ; Ext = '.j'),
    atom_concat('WamState', Ext, RuntimeFile),
    directory_file_path(ProjectDir, RuntimeFile, RuntimePath),
    open(RuntimePath, write, RS),
    write(RS, RuntimeCode),
    close(RS),
    % Generate predicate wrappers
    forall(
        member(Pred/Arity-WamCode, Predicates),
        (   compile_wam_predicate_to_jvm(Pred/Arity, WamCode, OutputFmt, Options, PredCode),
            atom_string(Pred, PredStr),
            atom_concat(PredStr, Ext, PredFile),
            directory_file_path(ProjectDir, PredFile, PredPath),
            open(PredPath, write, PS),
            write(PS, PredCode),
            close(PS)
        )
    ),
    % Generate main class
    jvm_wam_class_header(OutputFmt, ClassName, MainHeader),
    jvm_wam_class_footer(OutputFmt, MainFooter),
    jvm_wam_comment(OutputFmt, 'Generated by UnifyWeaver WAM-JVM Target', GenComment),
    format(string(MainCode), '~w\n~w\n~w', [GenComment, MainHeader, MainFooter]),
    atom_concat(ClassName, Ext, MainFile),
    directory_file_path(ProjectDir, MainFile, MainPath),
    open(MainPath, write, MS),
    write(MS, MainCode),
    close(MS).
