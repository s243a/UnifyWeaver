:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% wam_kotlin_target.pl - Hybrid WAM-to-Kotlin Transpilation Target
%
% This target is the JVM-family high-level hybrid WAM backend.  It mirrors the
% Haskell/F#/Rust shape: prefer Kotlin-native lowering where available, fall
% back to WAM text for hard predicates, then package the fallback inside a
% small Kotlin WAM runtime.

:- module(wam_kotlin_target, [
    compile_predicate/3,                   % +Pred/Arity, +Options, -KotlinCode
    compile_predicate_to_kotlin_wam/3,      % +Pred/Arity, +Options, -KotlinCode
    compile_wam_predicate_to_kotlin/4,      % +Pred/Arity, +WamCode, +Options, -KotlinCode
    compile_wam_runtime_to_kotlin/2,        % +Options, -KotlinCode
    write_wam_kotlin_project/3,             % +Predicates, +Options, +ProjectDir
    wam_kotlin_resolve_emit_mode/2,         % +Options, -Mode
    wam_kotlin_partition_predicates/5,      % +Mode, +Preds, -Native, -Wam, -Failed
    wam_instruction_to_kotlin_literal/2,     % +WamInstruction, -KotlinLiteral
    kotlin_safe_identifier/2,               % +AtomOrString, -IdentifierAtom
    gradle_check_project/2                  % +ProjectDir, -Result
]).

:- use_module(library(lists)).
:- use_module(library(option)).
:- use_module(library(filesex), [make_directory_path/1, directory_file_path/3]).
:- use_module(library(process)).
:- use_module('../targets/kotlin_target').
:- use_module('../targets/wam_target', [compile_predicate_to_wam/3]).
:- use_module('../targets/wam_text_parser', [
    wam_tokenize_line/2,
    wam_recognise_label/2,
    wam_recognise_instruction/2,
    wam_classify_constant_token/2
]).
:- use_module('../core/template_system', [render_template/3]).
:- use_module('../targets/wam_kotlin_lowered_emitter', [
    wam_kotlin_lowerable/3,
    lower_predicate_to_kotlin/4,
    kotlin_lowered_func_name/2
]).

% Compatibility wrapper for the lowered emitter (rules live in wam_text_parser).
tokenize_wam_line(Line, Tokens) :-
    wam_tokenize_line(Line, Tokens).

% ============================================================================
% Public hybrid entry points
% ============================================================================

%% compile_predicate(+PredIndicator, +Options, -KotlinCode)
%  Registry-compatible alias.  Emits the fallback WAM wrapper for one
%  predicate; project-level native/WAM partitioning is handled by
%  write_wam_kotlin_project/3.
compile_predicate(PredIndicator, Options, KotlinCode) :-
    compile_predicate_to_kotlin_wam(PredIndicator, Options, KotlinCode).

%% compile_predicate_to_kotlin_wam(+PredIndicator, +Options, -KotlinCode)
%  Compile a predicate through WAM text and emit a Kotlin registrar function.
compile_predicate_to_kotlin_wam(PredIndicator, Options, KotlinCode) :-
    pi_parts(PredIndicator, _Module, Pred, Arity),
    compile_predicate_to_wam(PredIndicator, Options, WamCode),
    compile_wam_predicate_to_kotlin(Pred/Arity, WamCode, Options, KotlinCode).

%% wam_kotlin_resolve_emit_mode(+Options, -Mode)
%  Mode hierarchy follows the mature hybrid targets:
%    interpreter  — every predicate registers WAM instructions
%    functions    — try kotlin_target native lowering; WAM fallback
%    mixed(List)  — lower only listed Pred/Arity indicators; WAM fallback for rest
wam_kotlin_resolve_emit_mode(Options, Mode) :-
    (   option(emit_mode(Explicit), Options)
    ->  Mode = Explicit
    ;   current_predicate(user:wam_kotlin_emit_mode/1),
        user:wam_kotlin_emit_mode(UserMode)
    ->  Mode = UserMode
    ;   Mode = interpreter
    ).

%% wam_kotlin_partition_predicates(+Mode, +Predicates, -Native, -Wam, -Failed)
%  Classify predicates for project generation.  Native entries are
%  native(Pred/Arity, Code); WAM entries are wam(Pred/Arity, WamText).
wam_kotlin_partition_predicates(Mode, Predicates, Native, Wam, Failed) :-
    partition_predicates_(Predicates, Mode, [], NativeRev, [], WamRev, [], FailedRev),
    reverse(NativeRev, Native),
    reverse(WamRev, Wam),
    reverse(FailedRev, Failed).

partition_predicates_([], _Mode, Native, Native, Wam, Wam, Failed, Failed).
partition_predicates_([Spec|Rest], Mode, Native0, Native, Wam0, Wam, Failed0, Failed) :-
    pi_parts(Spec, Module, Pred, Arity),
    PI = Pred/Arity,
  ( catch(compile_predicate_to_wam(Module:PI, [], WamText), _, fail)
    ->  (   should_attempt_native(Mode, PI),
            catch(wam_kotlin_lowerable(Module:PI, WamText, _), _, fail),
            catch(lower_predicate_to_kotlin(Module:PI, WamText, [], Lowered), _, fail)
        ->  Native1 = [native(PI, Lowered)|Native0],
            Wam1 = [wam(PI, WamText)|Wam0],
            Failed1 = Failed0
        ;   Native1 = Native0,
            Wam1 = [wam(PI, WamText)|Wam0],
            Failed1 = Failed0
        )
    ;   Native1 = Native0,
        Wam1 = Wam0,
        Failed1 = [failed(PI)|Failed0]
    ),
    partition_predicates_(Rest, Mode, Native1, Native, Wam1, Wam, Failed1, Failed).

should_attempt_native(functions, _PI) :- !.
should_attempt_native(mixed(List), PI) :- !, member(PI, List).
should_attempt_native(_, _) :- fail.

% ============================================================================
% WAM text -> Kotlin program registration
% ============================================================================

%% compile_wam_predicate_to_kotlin(+Pred/Arity, +WamCode, +Options, -KotlinCode)
%  Converts canonical WAM text for one predicate into a Kotlin function that
%  registers an instruction list in WamProgram.
compile_wam_predicate_to_kotlin(Pred/Arity, WamCode, _Options, KotlinCode) :-
    wam_code_to_kotlin_instructions(WamCode, InstrLiterals),
    atom_string(Pred, PredStr),
    kotlin_safe_identifier(Pred, SafePred),
    format(atom(LabelKey), '~w/~w', [PredStr, Arity]),
    format(string(KotlinCode),
'fun register_~w(program: WamProgram) {
    program.register("~w", listOf(
~w
    ))
}
', [SafePred, LabelKey, InstrLiterals]).

wam_code_to_kotlin_instructions(WamCode, InstrLiterals) :-
    split_string(WamCode, "\n", "\r", Lines0),
    include(nonblank_line, Lines0, Lines),
    findall(Lit,
        (   member(Line, Lines),
            wam_line_instruction(Line, Instr),
            wam_instruction_to_kotlin_literal(Instr, Lit)
        ),
        Lits),
    indent_join(Lits, '        ', ',\n', InstrLiterals).

nonblank_line(Line) :-
    normalize_space(string(Trimmed), Line),
    Trimmed \= "".

wam_line_instruction(Line, label(Label)) :-
    normalize_space(string(Trimmed), Line),
    \+ sub_string(Trimmed, 0, 1, _, "%"),
    wam_tokenize_line(Trimmed, Tokens),
    wam_recognise_label(Tokens, Label), !.
wam_line_instruction(Line, Instr) :-
    normalize_space(string(Trimmed), Line),
    \+ sub_string(Trimmed, 0, 1, _, "%"),
    wam_tokenize_line(Trimmed, Tokens),
    wam_recognise_instruction(Tokens, Instr).

indent_join([], _Indent, _Sep, '').
indent_join([One], Indent, _Sep, Joined) :-
    format(string(Joined), '~w~w', [Indent, One]).
indent_join([One,Two|Rest], Indent, Sep, Joined) :-
    indent_join([Two|Rest], Indent, Sep, Tail),
    format(string(Joined), '~w~w~w~w', [Indent, One, Sep, Tail]).

%% wam_instruction_to_kotlin_literal(+Instruction, -Literal)
%  Deterministic public wrapper.  The catch-all unsupported clause is kept for
%  forward-compatible code generation, but must not be returned after a known
%  instruction on backtracking.
wam_instruction_to_kotlin_literal(Instr, Lit) :-
    once(wam_instruction_to_kotlin_literal_det(Instr, Lit)).

wam_instruction_to_kotlin_literal_det(label(L), Lit) :- instr_lit('label', [label(L)], Lit).
wam_instruction_to_kotlin_literal_det(get_constant(C, Ai), Lit) :- instr_lit('get_constant', [value(C), reg(Ai)], Lit).
wam_instruction_to_kotlin_literal_det(get_variable(Xn, Ai), Lit) :- instr_lit('get_variable', [reg(Xn), reg(Ai)], Lit).
wam_instruction_to_kotlin_literal_det(get_value(Xn, Ai), Lit) :- instr_lit('get_value', [reg(Xn), reg(Ai)], Lit).
wam_instruction_to_kotlin_literal_det(get_structure(F, Ai), Lit) :- instr_lit('get_structure', [string(F), reg(Ai)], Lit).
wam_instruction_to_kotlin_literal_det(get_list(Ai), Lit) :- instr_lit('get_list', [reg(Ai)], Lit).
wam_instruction_to_kotlin_literal_det(get_nil(Ai), Lit) :- instr_lit('get_nil', [value('[]'), reg(Ai)], Lit).
wam_instruction_to_kotlin_literal_det(get_integer(N, Ai), Lit) :- instr_lit('get_integer', [value(N), reg(Ai)], Lit).
wam_instruction_to_kotlin_literal_det(get_float(F, Ai), Lit) :- instr_lit('get_float', [value(F), reg(Ai)], Lit).
wam_instruction_to_kotlin_literal_det(unify_variable(Xn), Lit) :- instr_lit('unify_variable', [reg(Xn)], Lit).
wam_instruction_to_kotlin_literal_det(unify_value(Xn), Lit) :- instr_lit('unify_value', [reg(Xn)], Lit).
wam_instruction_to_kotlin_literal_det(unify_constant(C), Lit) :- instr_lit('unify_constant', [value(C)], Lit).
wam_instruction_to_kotlin_literal_det(unify_nil, Lit) :- instr_lit('unify_nil', [], Lit).
wam_instruction_to_kotlin_literal_det(unify_void(N), Lit) :- instr_lit('unify_void', [int(N)], Lit).
wam_instruction_to_kotlin_literal_det(put_variable(Xn, Ai), Lit) :- instr_lit('put_variable', [reg(Xn), reg(Ai)], Lit).
wam_instruction_to_kotlin_literal_det(put_value(Xn, Ai), Lit) :- instr_lit('put_value', [reg(Xn), reg(Ai)], Lit).
wam_instruction_to_kotlin_literal_det(put_unsafe_value(Yn, Ai), Lit) :- instr_lit('put_unsafe_value', [reg(Yn), reg(Ai)], Lit).
wam_instruction_to_kotlin_literal_det(put_constant(C, Ai), Lit) :- instr_lit('put_constant', [value(C), reg(Ai)], Lit).
wam_instruction_to_kotlin_literal_det(put_nil(Ai), Lit) :- instr_lit('put_nil', [value('[]'), reg(Ai)], Lit).
wam_instruction_to_kotlin_literal_det(put_integer(N, Ai), Lit) :- instr_lit('put_integer', [value(N), reg(Ai)], Lit).
wam_instruction_to_kotlin_literal_det(put_float(F, Ai), Lit) :- instr_lit('put_float', [value(F), reg(Ai)], Lit).
wam_instruction_to_kotlin_literal_det(put_structure(F, Ai), Lit) :- instr_lit('put_structure', [string(F), reg(Ai)], Lit).
wam_instruction_to_kotlin_literal_det(put_list(Ai), Lit) :- instr_lit('put_list', [reg(Ai)], Lit).
wam_instruction_to_kotlin_literal_det(set_variable(Xn), Lit) :- instr_lit('set_variable', [reg(Xn)], Lit).
wam_instruction_to_kotlin_literal_det(set_value(Xn), Lit) :- instr_lit('set_value', [reg(Xn)], Lit).
wam_instruction_to_kotlin_literal_det(set_local_value(Xn), Lit) :- instr_lit('set_local_value', [reg(Xn)], Lit).
wam_instruction_to_kotlin_literal_det(set_constant(C), Lit) :- instr_lit('set_constant', [value(C)], Lit).
wam_instruction_to_kotlin_literal_det(set_nil, Lit) :- instr_lit('set_nil', [], Lit).
wam_instruction_to_kotlin_literal_det(set_integer(N), Lit) :- instr_lit('set_integer', [value(N)], Lit).
wam_instruction_to_kotlin_literal_det(set_void(N), Lit) :- instr_lit('set_void', [int(N)], Lit).
wam_instruction_to_kotlin_literal_det(call(P, N), Lit) :- instr_lit('call', [pred(P), int(N)], Lit).
wam_instruction_to_kotlin_literal_det(execute(P), Lit) :- instr_lit('execute', [pred(P)], Lit).
wam_instruction_to_kotlin_literal_det(proceed, Lit) :- instr_lit('proceed', [], Lit).
wam_instruction_to_kotlin_literal_det(fail, Lit) :- instr_lit('fail', [], Lit).
wam_instruction_to_kotlin_literal_det(allocate, Lit) :- instr_lit('allocate', [], Lit).
wam_instruction_to_kotlin_literal_det(deallocate, Lit) :- instr_lit('deallocate', [], Lit).
wam_instruction_to_kotlin_literal_det(builtin_call(Op, Ar), Lit) :- instr_lit('builtin_call', [string(Op), int(Ar)], Lit).
wam_instruction_to_kotlin_literal_det(call_foreign(P, Ar), Lit) :- instr_lit('call_foreign', [pred(P), int(Ar)], Lit).
wam_instruction_to_kotlin_literal_det(arg(N, Reg, Out), Lit) :- instr_lit('arg', [int(N), reg(Reg), reg(Out)], Lit).
wam_instruction_to_kotlin_literal_det(try_me_else(L), Lit) :- instr_lit('try_me_else', [label(L)], Lit).
wam_instruction_to_kotlin_literal_det(retry_me_else(L), Lit) :- instr_lit('retry_me_else', [label(L)], Lit).
wam_instruction_to_kotlin_literal_det(trust_me, Lit) :- instr_lit('trust_me', [], Lit).
wam_instruction_to_kotlin_literal_det(try(L), Lit) :- instr_lit('try', [label(L)], Lit).
wam_instruction_to_kotlin_literal_det(retry(L), Lit) :- instr_lit('retry', [label(L)], Lit).
wam_instruction_to_kotlin_literal_det(trust(L), Lit) :- instr_lit('trust', [label(L)], Lit).
wam_instruction_to_kotlin_literal_det(jump(L), Lit) :- instr_lit('jump', [label(L)], Lit).
wam_instruction_to_kotlin_literal_det(cut_ite, Lit) :- instr_lit('cut_ite', [], Lit).
wam_instruction_to_kotlin_literal_det(get_level(Yn), Lit) :- instr_lit('get_level', [reg(Yn)], Lit).
wam_instruction_to_kotlin_literal_det(cut(Yn), Lit) :- instr_lit('cut', [reg(Yn)], Lit).
wam_instruction_to_kotlin_literal_det(begin_aggregate(K,V,R), Lit) :- instr_lit('begin_aggregate', [string(K), reg(V), reg(R)], Lit).
wam_instruction_to_kotlin_literal_det(begin_aggregate(K,V,R,W), Lit) :- instr_lit('begin_aggregate', [string(K), reg(V), reg(R), reg(W)], Lit).
wam_instruction_to_kotlin_literal_det(end_aggregate(R), Lit) :- instr_lit('end_aggregate', [reg(R)], Lit).
wam_instruction_to_kotlin_literal_det(switch_on_constant(Es), Lit) :- instr_lit('switch_on_constant', [strings(Es)], Lit).
wam_instruction_to_kotlin_literal_det(switch_on_constant_fallthrough(Es), Lit) :- instr_lit('switch_on_constant_fallthrough', [strings(Es)], Lit).
wam_instruction_to_kotlin_literal_det(switch_on_constant_a2(Es), Lit) :- instr_lit('switch_on_constant_a2', [strings(Es)], Lit).
wam_instruction_to_kotlin_literal_det(switch_on_constant_a2_fallthrough(Es), Lit) :- instr_lit('switch_on_constant_a2_fallthrough', [strings(Es)], Lit).
wam_instruction_to_kotlin_literal_det(switch_on_structure(Es), Lit) :- instr_lit('switch_on_structure', [strings(Es)], Lit).
wam_instruction_to_kotlin_literal_det(switch_on_structure_a2(Es), Lit) :- instr_lit('switch_on_structure_a2', [strings(Es)], Lit).
wam_instruction_to_kotlin_literal_det(switch_on_term(Ts), Lit) :- instr_lit('switch_on_term', [strings(Ts)], Lit).
wam_instruction_to_kotlin_literal_det(switch_on_term_a2(Ts), Lit) :- instr_lit('switch_on_term_a2', [strings(Ts)], Lit).
wam_instruction_to_kotlin_literal_det(Other, Lit) :-
    term_string(Other, S),
    instr_lit('unsupported', [string(S)], Lit).

instr_lit(Op, Args, Lit) :-
    maplist(kotlin_arg_literal, Args, ArgLits),
    atomic_list_concat(ArgLits, ', ', ArgsCode),
    escape_kotlin_string(Op, OpEsc),
    format(string(Lit), 'Instruction("~w", listOf(~w))', [OpEsc, ArgsCode]).

kotlin_arg_literal(value(Token), Lit) :- token_value_literal(Token, Lit).
kotlin_arg_literal(reg(Reg), Lit) :- atom_string_like(Reg, S), escape_kotlin_string(S, E), format(string(Lit), '"~w"', [E]).
kotlin_arg_literal(label(Label), Lit) :- atom_string_like(Label, S), escape_kotlin_string(S, E), format(string(Lit), '"~w"', [E]).
kotlin_arg_literal(pred(Pred), Lit) :- atom_string_like(Pred, S), escape_kotlin_string(S, E), format(string(Lit), '"~w"', [E]).
kotlin_arg_literal(string(S0), Lit) :- atom_string_like(S0, S), escape_kotlin_string(S, E), format(string(Lit), '"~w"', [E]).
kotlin_arg_literal(int(N0), Lit) :- atom_string_like(N0, S), (catch(number_string(N, S), _, fail) -> format(string(Lit), '~wL', [N]) ; escape_kotlin_string(S, E), format(string(Lit), '"~w"', [E])).
kotlin_arg_literal(float(F0), Lit) :- atom_string_like(F0, S), (catch(number_string(F, S), _, fail) -> format(string(Lit), '~w', [F]) ; escape_kotlin_string(S, E), format(string(Lit), '"~w"', [E])).
kotlin_arg_literal(strings(Items), Lit) :-
    maplist([I, S]>>(atom_string_like(I, Raw), escape_kotlin_string(Raw, E), format(string(S), '"~w"', [E])), Items, Parts),
    atomic_list_concat(Parts, ', ', Inner),
    format(string(Lit), 'listOf(~w)', [Inner]).

token_value_literal(Token, Lit) :-
    wam_classify_constant_token(Token, Class),
    (   Class = integer(N)
    ->  format(string(Lit), 'Value.IntVal(~wL)', [N])
    ;   Class = float(F)
    ->  format(string(Lit), 'Value.FloatVal(~w)', [F])
    ;   Class = atom(Name)
    ->  escape_kotlin_string(Name, Esc),
        format(string(Lit), 'Value.Atom("~w")', [Esc])
    ).

% ============================================================================
% Project generation
% ============================================================================

compile_wam_runtime_to_kotlin(Options, KotlinCode) :-
    option(package(Package), Options, 'generated.wam'),
    read_kotlin_wam_template('WamRuntime.kt.mustache', Template),
    render_template(Template, [package=Package], KotlinCode).

write_wam_kotlin_project(Predicates, Options, ProjectDir) :-
    option(package(Package), Options, 'generated.wam'),
    option(module_name(ModuleName), Options, 'WamKotlinGenerated'),
    wam_kotlin_resolve_emit_mode(Options, Mode),
    wam_kotlin_partition_predicates(Mode, Predicates, NativeParts, WamParts, FailedParts),
    make_directory_path(ProjectDir),

    % Gradle project files.
    render_kotlin_wam_template('settings.gradle.mustache', [module_name=ModuleName], Settings),
    write_project_file(ProjectDir, 'settings.gradle', Settings),
    render_kotlin_wam_template('build.gradle.mustache', [package=Package], Build),
    write_project_file(ProjectDir, 'build.gradle', Build),

    % Kotlin sources.
    package_to_path(Package, PackagePath),
    format(atom(SourceRel), 'src/main/kotlin/~w', [PackagePath]),
    directory_file_path(ProjectDir, SourceRel, SrcDir),
    make_directory_path(SrcDir),
    compile_wam_runtime_to_kotlin([package(Package)|Options], RuntimeCode),
    directory_file_path(SrcDir, 'WamRuntime.kt', RuntimePath),
    write_file(RuntimePath, RuntimeCode),

    compile_native_parts(NativeParts, NativeCode),
    compile_wam_parts(WamParts, Options, WamCode),
    compile_failed_parts(FailedParts, FailedCode),
    registrar_calls(NativeParts, WamParts, Registrars),
    option(conformance_main(ConfMain), Options, false),
    render_kotlin_wam_template('Main.kt.mustache', [
        package=Package,
        native_predicates=NativeCode,
        wam_predicates=WamCode,
        failed_predicates=FailedCode,
        registrar_calls=Registrars,
        conformance_main=ConfMain
    ], MainCode),
    directory_file_path(SrcDir, 'Main.kt', MainPath),
    write_file(MainPath, MainCode),
    format('WAM Kotlin project created at: ~w~n', [ProjectDir]),
    format('  Mode: ~w~n', [Mode]),
    format('  Native predicates: ~w~n', [NativeParts]),
    format('  WAM predicates: ~w~n', [WamParts]).

compile_native_parts([], '// No native Kotlin predicates selected.').
compile_native_parts(NativeParts, Code) :-
    NativeParts \= [],
    findall(Part,
        (   member(native(_PI, lowered(_PredName, _FuncName, Part)), NativeParts)
        ),
        Parts),
    atomic_list_concat(Parts, '\n\n', Code).

compile_wam_parts([], _Options, '// No WAM fallback predicates selected.').
compile_wam_parts(WamParts, Options, Code) :-
    WamParts \= [],
    findall(Part,
        (   member(wam(PI, WamText), WamParts),
            compile_wam_predicate_to_kotlin(PI, WamText, Options, Part)
        ),
        Parts),
    atomic_list_concat(Parts, '\n\n', Code).

compile_failed_parts([], '// No failed predicates.').
compile_failed_parts(FailedParts, Code) :-
    FailedParts \= [],
    findall(Line,
        (   member(failed(Pred/Arity), FailedParts),
            format(string(Line), '// Failed to compile ~w/~w', [Pred, Arity])
        ),
        Lines),
    atomic_list_concat(Lines, '\n', Code).

registrar_calls([], [], '// No predicate registrars.').
registrar_calls(NativeParts, WamParts, Calls) :-
    (NativeParts \= [] ; WamParts \= []),
    findall(Call, registrar_native_call(NativeParts, Call), NativeCalls),
    findall(Call, registrar_wam_call(WamParts, Call), WamCalls),
    append(NativeCalls, WamCalls, CallLines),
    atomic_list_concat(CallLines, '\n', Calls).

registrar_native_call(NativeParts, Call) :-
    member(native(_PI, lowered(PredName, FuncName, _)), NativeParts),
    format(string(Call), '    program.registerNative("~w", ::~w)', [PredName, FuncName]).

registrar_wam_call(WamParts, Call) :-
    member(wam(Pred/_Arity, _), WamParts),
    kotlin_safe_identifier(Pred, Safe),
    format(string(Call), '    register_~w(program)', [Safe]).

gradle_check_project(ProjectDir, Result) :-
    setup_call_cleanup(
        process_create(path(gradle), ['-q', 'compileKotlin'],
                       [cwd(ProjectDir), stdout(pipe(Out)), stderr(pipe(Err)), process(PID)]),
        ( read_string(Out, _, Stdout), read_string(Err, _, Stderr), process_wait(PID, Status) ),
        ( close(Out), close(Err) )
    ),
    Result = result(Status, Stdout, Stderr).

% ============================================================================
% File/template helpers
% ============================================================================

render_kotlin_wam_template(Name, Data, Code) :-
    read_kotlin_wam_template(Name, Template),
    render_template(Template, Data, Code).

read_kotlin_wam_template(Name, Content) :-
    source_file(wam_kotlin_target:compile_wam_runtime_to_kotlin(_, _), ThisFile),
    file_directory_name(ThisFile, TargetsDir),
    file_directory_name(TargetsDir, UnifyWeaverDir),
    file_directory_name(UnifyWeaverDir, SrcDir),
    file_directory_name(SrcDir, RepoRoot),
    directory_file_path(RepoRoot, 'templates/targets/kotlin_wam', TemplateDir),
    directory_file_path(TemplateDir, Name, Path),
    read_file_to_string(Path, Content, []).

write_project_file(ProjectDir, Rel, Content) :-
    directory_file_path(ProjectDir, Rel, Path),
    file_directory_name(Path, Dir),
    make_directory_path(Dir),
    write_file(Path, Content).

write_file(Path, Content) :-
    setup_call_cleanup(open(Path, write, S), format(S, '~w', [Content]), close(S)).

sanitize_block_comment(Input, Output) :-
    atom_string_like(Input, String),
    replace_substring_local(String, '*/', '* /', Output).

replace_substring_local(String, Find, Replace, Result) :-
    string_length(Find, FindLen),
    (   sub_string(String, Before, FindLen, After, Find)
    ->  sub_string(String, 0, Before, _, Prefix),
        Start is Before + FindLen,
        sub_string(String, Start, After, 0, Suffix),
        replace_substring_local(Suffix, Find, Replace, Rest),
        string_concat(Prefix, Replace, Head),
        string_concat(Head, Rest, Result)
    ;   Result = String
    ).

% ============================================================================
% Small utilities
% ============================================================================

pi_parts(Module:Pred/Arity, Module, Pred, Arity) :- !.
pi_parts(Pred/Arity, user, Pred, Arity).

kotlin_safe_identifier(Input, Identifier) :-
    atom_string_like(Input, Str),
    string_codes(Str, Codes),
    maplist(kotlin_safe_identifier_code, Codes, SafeCodes0),
    (   SafeCodes0 = [First|_], kotlin_identifier_start_code(First)
    ->  SafeCodes = SafeCodes0
    ;   SafeCodes = [0'_|SafeCodes0]
    ),
    string_codes(SafeStr, SafeCodes),
    atom_string(Identifier, SafeStr).

kotlin_safe_identifier_code(C, C) :-
    (   C >= 0'a, C =< 0'z
    ;   C >= 0'A, C =< 0'Z
    ;   C >= 0'0, C =< 0'9
    ;   C =:= 0'_
    ), !.
kotlin_safe_identifier_code(_, 0'_).

kotlin_identifier_start_code(C) :-
    (   C >= 0'a, C =< 0'z
    ;   C >= 0'A, C =< 0'Z
    ;   C =:= 0'_
    ).

package_to_path(Package, Path) :-
    atom_string_like(Package, PackageStr),
    split_string(PackageStr, '.', '', Parts),
    atomic_list_concat(Parts, '/', Path).

atom_string_like(Value, String) :-
    string(Value), !, String = Value.
atom_string_like(Value, String) :-
    atom(Value), !, atom_string(Value, String).
atom_string_like(Value, String) :-
    number(Value), !, number_string(Value, String).
atom_string_like(Value, String) :-
    term_string(Value, String).

escape_kotlin_string(Input, Escaped) :-
    atom_string_like(Input, S),
    string_chars(S, Chars),
    phrase(escaped_chars(Chars), OutChars),
    string_chars(Escaped, OutChars).

escaped_chars([]) --> [].
escaped_chars(['\\'|Cs]) --> ['\\','\\'], escaped_chars(Cs).
escaped_chars(['"'|Cs]) --> ['\\','"'], escaped_chars(Cs).
escaped_chars(['\n'|Cs]) --> ['\\','n'], escaped_chars(Cs).
escaped_chars(['\r'|Cs]) --> ['\\','r'], escaped_chars(Cs).
escaped_chars(['\t'|Cs]) --> ['\\','t'], escaped_chars(Cs).
escaped_chars([C|Cs]) --> [C], escaped_chars(Cs).
