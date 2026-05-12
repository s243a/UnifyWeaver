:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% wam_cpp_target.pl - WAM-to-C++ Hybrid Transpilation Target
%
% Generates a C++ project with an instruction-array WAM interpreter plus
% optional lowered per-predicate C++ functions. The architecture mirrors
% the Rust/Haskell/Lua hybrid WAM targets while exploiting C++ ergonomics
% (std::variant, std::unordered_map, std::string).
%
% Entry points:
%   compile_wam_predicate_to_cpp/4 - Translate a single predicate.
%   write_wam_cpp_project/3        - Materialise a full C++ project.
%   compile_wam_runtime_to_cpp/2   - Emit the wam_runtime.cpp source.
%   wam_instruction_to_cpp_literal/2,3 - Emit C++ struct-initializer
%       literals for raw WAM instructions (mirrors wam_c_target.pl's
%       designated-initializer style for use in the instruction array).
%
% Emit modes (resolved by cpp_wam_resolve_emit_mode/2):
%   interpreter  - All predicates dispatched via the instruction array.
%   functions    - Lowerable predicates emitted as direct C++ functions
%                  (lowered_<pred>_<arity>) on top of the array.
%   mixed(List)  - Only the listed Pred/Arity entries are lowered.

:- module(wam_cpp_target, [
    compile_wam_predicate_to_cpp/4,        % +Pred/Arity, +WamCode, +Options, -CppCode
    write_wam_cpp_project/3,               % +Predicates, +Options, +ProjectDir
    compile_wam_runtime_to_cpp/2,          % +Options, -RuntimeCppCode
    compile_wam_runtime_header_to_cpp/2,   % +Options, -RuntimeHeaderCode
    cpp_wam_resolve_emit_mode/2,           % +Options, -Mode
    wam_instruction_to_cpp_literal/2,      % +Instr, -CppCode
    wam_instruction_to_cpp_literal/3,      % +Instr, +LabelMap, -CppCode
    cpp_safe_function_name/2,              % +Pred, -CppFuncName
    escape_cpp_string/2,                   % +InStr, -OutStr
    cpp_value_literal/2                    % +Constant, -CppLiteral
]).

:- use_module(library(lists)).
:- use_module(library(option)).
:- use_module(library(filesex), [make_directory_path/1, directory_file_path/3]).
:- use_module('../targets/wam_target', [compile_predicate_to_wam/3]).
:- use_module(wam_cpp_lowered_emitter, [
    wam_cpp_lowerable/3,
    lower_predicate_to_cpp/4,
    cpp_lowered_func_name/2,
    parse_wam_text/2
]).

:- multifile user:wam_cpp_emit_mode/1.

% ============================================================================
% Emit-mode resolution
% ============================================================================

%% cpp_wam_resolve_emit_mode(+Options, -Mode)
%  Options key:    emit_mode(interpreter | functions | mixed(List))
%  User hook:      user:wam_cpp_emit_mode/1
%  Default:        interpreter
cpp_wam_resolve_emit_mode(Options, Mode) :-
    (   option(emit_mode(M0), Options)
    ->  validate_cpp_emit_mode(M0, Mode)
    ;   catch(user:wam_cpp_emit_mode(M1), _, fail)
    ->  validate_cpp_emit_mode(M1, Mode)
    ;   Mode = interpreter
    ).

validate_cpp_emit_mode(interpreter, interpreter) :- !.
validate_cpp_emit_mode(functions, functions) :- !.
validate_cpp_emit_mode(mixed(L), mixed(L)) :- is_list(L), !.
validate_cpp_emit_mode(Other, _) :-
    throw(error(domain_error(wam_cpp_emit_mode, Other),
                cpp_wam_resolve_emit_mode/2)).

should_lower(functions, _, _) :- !.
should_lower(mixed(HotPreds), P, A) :-
    member(P/A, HotPreds), !.
should_lower(_, _, _) :- fail.

% ============================================================================
% Identifier sanitisation & string helpers
% ============================================================================

cpp_safe_function_name(Pred, FuncName) :-
    atom_string(Pred, PredStr),
    string_codes(PredStr, Codes),
    maplist(cpp_safe_identifier_code, Codes, SafeCodes),
    string_codes(FuncStr, SafeCodes),
    atom_string(FuncName, FuncStr).

cpp_safe_identifier_code(C, C) :-
    (   C >= 0'a, C =< 0'z
    ;   C >= 0'A, C =< 0'Z
    ;   C >= 0'0, C =< 0'9
    ;   C =:= 0'_
    ),
    !.
cpp_safe_identifier_code(_, 0'_).

%% escape_cpp_string(+In, -Out)
%  Escape backslashes and double quotes for embedding inside a C++
%  string literal. Matches the wam_rust_target.pl helper of the same
%  shape so wam_cpp_lowered_emitter.pl can use it interchangeably.
escape_cpp_string(In, Out) :-
    atom_string(In, S),
    split_string(S, "\\", "", Parts),
    cpp_atomics_to_string(Parts, "\\\\", Escaped1),
    split_string(Escaped1, "\"", "", Parts2),
    cpp_atomics_to_string(Parts2, "\\\"", Out).

cpp_atomics_to_string([], _, "").
cpp_atomics_to_string([X], _, X).
cpp_atomics_to_string([X, Y|Rest], Sep, Result) :-
    cpp_atomics_to_string([Y|Rest], Sep, Tail),
    string_concat(X, Sep, XSep),
    string_concat(XSep, Tail, Result).

%% cpp_value_literal(+ConstantToken, -CppLiteral)
%  Convert a WAM constant (atom, integer, float, []) into a C++ Value
%  literal usable inside the generated source.
cpp_value_literal(C, Val) :-
    to_string(C, Str),
    (   number_string(N, Str), integer(N)
    ->  format(atom(Val), 'Value::Integer(~w)', [N])
    ;   number_string(F, Str), float(F)
    ->  format(atom(Val), 'Value::Float(~w)', [F])
    ;   Str == "[]"
    ->  Val = 'Value::Atom("[]")'
    ;   escape_cpp_string(Str, EscStr),
        format(atom(Val), 'Value::Atom("~w")', [EscStr])
    ).

to_string(X, S) :- string(X), !, S = X.
to_string(X, S) :- atom(X), !, atom_string(X, S).
to_string(X, S) :- number(X), !, number_string(X, S).
to_string(X, S) :- format(string(S), "~w", [X]).

% ============================================================================
% Phase 2: WAM instructions -> C++ struct-initializer literals
% ============================================================================
% These produce designated-initialiser style strings suitable for embedding
% in the interpreter's instruction array. Mirrors wam_c_target.pl's
% wam_instruction_to_c_literal/2 pattern but emits C++ aggregate init
% braces (no `.tag =` designators so the output also compiles under
% C++17 without the GCC extension).

wam_instruction_to_cpp_literal(Instr, Code) :-
    wam_instruction_to_cpp_literal(Instr, [], Code).

wam_instruction_to_cpp_literal(get_constant(C, Ai), _, Code) :-
    cpp_value_literal(C, V), to_string(Ai, R),
    format(atom(Code), 'Instruction::GetConstant(~w, "~w")', [V, R]).
wam_instruction_to_cpp_literal(get_variable(Xn, Ai), _, Code) :-
    to_string(Xn, X), to_string(Ai, R),
    format(atom(Code), 'Instruction::GetVariable("~w", "~w")', [X, R]).
wam_instruction_to_cpp_literal(get_value(Xn, Ai), _, Code) :-
    to_string(Xn, X), to_string(Ai, R),
    format(atom(Code), 'Instruction::GetValue("~w", "~w")', [X, R]).
wam_instruction_to_cpp_literal(get_structure(F, Ai), _, Code) :-
    to_string(F, FS), to_string(Ai, R),
    escape_cpp_string(FS, EF),
    format(atom(Code), 'Instruction::GetStructure("~w", "~w")', [EF, R]).
wam_instruction_to_cpp_literal(get_list(Ai), _, Code) :-
    to_string(Ai, R),
    format(atom(Code), 'Instruction::GetList("~w")', [R]).
wam_instruction_to_cpp_literal(get_nil(Ai), _, Code) :-
    to_string(Ai, R),
    format(atom(Code), 'Instruction::GetNil("~w")', [R]).
wam_instruction_to_cpp_literal(get_integer(N, Ai), _, Code) :-
    to_string(N, NS), to_string(Ai, R),
    format(atom(Code), 'Instruction::GetInteger(~w, "~w")', [NS, R]).
wam_instruction_to_cpp_literal(put_constant(C, Ai), _, Code) :-
    cpp_value_literal(C, V), to_string(Ai, R),
    format(atom(Code), 'Instruction::PutConstant(~w, "~w")', [V, R]).
wam_instruction_to_cpp_literal(put_variable(Xn, Ai), _, Code) :-
    to_string(Xn, X), to_string(Ai, R),
    format(atom(Code), 'Instruction::PutVariable("~w", "~w")', [X, R]).
wam_instruction_to_cpp_literal(put_value(Xn, Ai), _, Code) :-
    to_string(Xn, X), to_string(Ai, R),
    format(atom(Code), 'Instruction::PutValue("~w", "~w")', [X, R]).
wam_instruction_to_cpp_literal(put_structure(F, Ai), _, Code) :-
    to_string(F, FS), to_string(Ai, R),
    escape_cpp_string(FS, EF),
    format(atom(Code), 'Instruction::PutStructure("~w", "~w")', [EF, R]).
wam_instruction_to_cpp_literal(put_list(Ai), _, Code) :-
    to_string(Ai, R),
    format(atom(Code), 'Instruction::PutList("~w")', [R]).
wam_instruction_to_cpp_literal(unify_variable(Xn), _, Code) :-
    to_string(Xn, X),
    format(atom(Code), 'Instruction::UnifyVariable("~w")', [X]).
wam_instruction_to_cpp_literal(unify_value(Xn), _, Code) :-
    to_string(Xn, X),
    format(atom(Code), 'Instruction::UnifyValue("~w")', [X]).
wam_instruction_to_cpp_literal(unify_constant(C), _, Code) :-
    cpp_value_literal(C, V),
    format(atom(Code), 'Instruction::UnifyConstant(~w)', [V]).
wam_instruction_to_cpp_literal(set_variable(Xn), _, Code) :-
    to_string(Xn, X),
    format(atom(Code), 'Instruction::SetVariable("~w")', [X]).
wam_instruction_to_cpp_literal(set_value(Xn), _, Code) :-
    to_string(Xn, X),
    format(atom(Code), 'Instruction::SetValue("~w")', [X]).
wam_instruction_to_cpp_literal(set_constant(C), _, Code) :-
    cpp_value_literal(C, V),
    format(atom(Code), 'Instruction::SetConstant(~w)', [V]).
wam_instruction_to_cpp_literal(call(P, N), _, Code) :-
    to_string(P, PS), to_string(N, NS),
    escape_cpp_string(PS, EP),
    format(atom(Code), 'Instruction::Call("~w", ~w)', [EP, NS]).
wam_instruction_to_cpp_literal(execute(P), _, Code) :-
    to_string(P, PS),
    escape_cpp_string(PS, EP),
    format(atom(Code), 'Instruction::Execute("~w")', [EP]).
wam_instruction_to_cpp_literal(proceed, _, 'Instruction::Proceed()').
wam_instruction_to_cpp_literal(fail, _, 'Instruction::Fail()').
wam_instruction_to_cpp_literal(allocate, _, 'Instruction::Allocate()').
wam_instruction_to_cpp_literal(deallocate, _, 'Instruction::Deallocate()').
wam_instruction_to_cpp_literal(builtin_call(Op, Ar), _, Code) :-
    to_string(Op, OpS), to_string(Ar, ArS),
    escape_cpp_string(OpS, EOp),
    format(atom(Code), 'Instruction::BuiltinCall("~w", ~w)', [EOp, ArS]).
wam_instruction_to_cpp_literal(call_foreign(P, Ar), _, Code) :-
    to_string(P, PS), to_string(Ar, ArS),
    escape_cpp_string(PS, EP),
    format(atom(Code), 'Instruction::CallForeign("~w", ~w)', [EP, ArS]).
wam_instruction_to_cpp_literal(try_me_else(L), LabelMap, Code) :-
    label_index(L, LabelMap, Idx),
    format(atom(Code), 'Instruction::TryMeElse(~w)', [Idx]).
wam_instruction_to_cpp_literal(retry_me_else(L), LabelMap, Code) :-
    label_index(L, LabelMap, Idx),
    format(atom(Code), 'Instruction::RetryMeElse(~w)', [Idx]).
wam_instruction_to_cpp_literal(trust_me, _, 'Instruction::TrustMe()').
wam_instruction_to_cpp_literal(jump(L), LabelMap, Code) :-
    label_index(L, LabelMap, Idx),
    format(atom(Code), 'Instruction::Jump(~w)', [Idx]).
wam_instruction_to_cpp_literal(cut_ite, _, 'Instruction::CutIte()').

label_index(L, LabelMap, Idx) :-
    to_string(L, LS),
    (   member(LS-I, LabelMap)
    ->  Idx = I
    ;   format(atom(Idx), '"~w"', [LS])
    ).

% ============================================================================
% Per-predicate compilation
% ============================================================================

%% compile_wam_predicate_to_cpp(+Pred/Arity, +WamCode, +Options, -CppCode)
%  Top-level per-predicate compile. Output is one of:
%    - lowered C++ function definition (emit_mode functions / mixed match)
%    - interpreter-array wrapper (default)
compile_wam_predicate_to_cpp(PI, WamCode, Options, CppCode) :-
    ( PI = _M:Pred/Arity -> true ; PI = Pred/Arity ),
    cpp_wam_resolve_emit_mode(Options, Mode),
    (   should_lower(Mode, Pred, Arity),
        wam_cpp_lowerable(Pred/Arity, WamCode, _Reason)
    ->  lower_predicate_to_cpp(Pred/Arity, WamCode, Options, Lines),
        atomic_list_concat(Lines, '\n', CppCode)
    ;   instrs_for(WamCode, Instrs),
        compile_predicate_wrapper(Pred, Arity, Instrs, Options, CppCode)
    ).

% Accept either an instruction list (from in-memory pipelines) or raw WAM
% text (compile_predicate_to_wam/3 returns a string). The text parser is
% shared with the lowered emitter to keep a single source of truth.
instrs_for(WamCode, Instrs) :-
    is_list(WamCode), !, Instrs = WamCode.
instrs_for(WamCode, Instrs) :-
    catch(parse_wam_text(WamCode, Instrs), _, fail), !.
instrs_for(_, []).

% For per-predicate emission (used by tests inspecting individual
% predicates), produce a single-line interpreter-mode marker. Most of
% the runtime semantics live in the program-wide wam_cpp_setup function
% emitted by emit_setup_function/3 below.
compile_predicate_wrapper(Pred, Arity, _Instrs, _Options, CppCode) :-
    cpp_safe_function_name(Pred, SafeName),
    format(atom(Key), '~w/~w', [Pred, Arity]),
    format(string(CppCode),
'// Interpreter-mode predicate ~w — instructions are emitted into the
// shared wam_cpp_setup() function below; this declaration only exists
// so per-predicate emission tests have something to assert against.
extern void wam_cpp_setup_~w(WamState&);
', [Key, SafeName]).

% ============================================================================
% Program-wide assembly: parse WAM text for every predicate, collect a
% mixed (label / instruction) stream, walk it once to compute absolute
% PCs, then emit a single wam_cpp_setup(WamState&) that pushes each
% instruction with label targets resolved to those PCs.
% ============================================================================

%% parse_pred_blocks(+WamText, -Items)
%  Items is a list interleaving label(Name) and instruction terms in
%  source order. Unrecognised instructions (e.g. switch_on_constant) are
%  silently skipped — the try_me_else / trust_me path still works.
parse_pred_blocks(WamText, Items) :-
    atom_string(WamText, S),
    split_string(S, "\n", "", Lines),
    parse_block_lines(Lines, Items).

parse_block_lines([], []).
parse_block_lines([Line|Rest], Items) :-
    split_string(Line, " \t,", " \t,", Parts),
    delete(Parts, "", CleanParts),
    (   CleanParts == []
    ->  parse_block_lines(Rest, Items)
    ;   CleanParts = [First|_],
        (   sub_string(First, _, 1, 0, ":")
        ->  string_length(First, FLen),
            L1 is FLen - 1,
            sub_string(First, 0, L1, _, NameStr),
            Items = [label(NameStr)|MoreItems],
            parse_block_lines(Rest, MoreItems)
        ;   wam_cpp_lowered_emitter_instr(CleanParts, Instr)
        ->  Items = [Instr|MoreItems],
            parse_block_lines(Rest, MoreItems)
        ;   parse_block_lines(Rest, Items)
        )
    ).

% Inline copy of the lowered emitter's instr_from_parts to avoid making
% it public. Kept in lockstep with wam_cpp_lowered_emitter.pl.
wam_cpp_lowered_emitter_instr(["get_constant", C, Ai], get_constant(C, Ai)).
wam_cpp_lowered_emitter_instr(["get_variable", Xn, Ai], get_variable(Xn, Ai)).
wam_cpp_lowered_emitter_instr(["get_value", Xn, Ai], get_value(Xn, Ai)).
wam_cpp_lowered_emitter_instr(["get_structure", F, Ai], get_structure(F, Ai)).
wam_cpp_lowered_emitter_instr(["get_list", Ai], get_list(Ai)).
wam_cpp_lowered_emitter_instr(["get_nil", Ai], get_nil(Ai)).
wam_cpp_lowered_emitter_instr(["get_integer", N, Ai], get_integer(N, Ai)).
wam_cpp_lowered_emitter_instr(["unify_variable", Xn], unify_variable(Xn)).
wam_cpp_lowered_emitter_instr(["unify_value", Xn], unify_value(Xn)).
wam_cpp_lowered_emitter_instr(["unify_constant", C], unify_constant(C)).
wam_cpp_lowered_emitter_instr(["put_variable", Xn, Ai], put_variable(Xn, Ai)).
wam_cpp_lowered_emitter_instr(["put_value", Xn, Ai], put_value(Xn, Ai)).
wam_cpp_lowered_emitter_instr(["put_constant", C, Ai], put_constant(C, Ai)).
wam_cpp_lowered_emitter_instr(["put_structure", F, Ai], put_structure(F, Ai)).
wam_cpp_lowered_emitter_instr(["put_list", Ai], put_list(Ai)).
wam_cpp_lowered_emitter_instr(["set_variable", Xn], set_variable(Xn)).
wam_cpp_lowered_emitter_instr(["set_value", Xn], set_value(Xn)).
wam_cpp_lowered_emitter_instr(["set_constant", C], set_constant(C)).
wam_cpp_lowered_emitter_instr(["call", P, N], call(P, N)).
wam_cpp_lowered_emitter_instr(["execute", P], execute(P)).
wam_cpp_lowered_emitter_instr(["proceed"], proceed).
wam_cpp_lowered_emitter_instr(["fail"], fail).
wam_cpp_lowered_emitter_instr(["allocate"], allocate).
wam_cpp_lowered_emitter_instr(["deallocate"], deallocate).
wam_cpp_lowered_emitter_instr(["builtin_call", Op, Ar], builtin_call(Op, Ar)).
wam_cpp_lowered_emitter_instr(["call_foreign", Pred, Ar], call_foreign(Pred, Ar)).
wam_cpp_lowered_emitter_instr(["try_me_else", L], try_me_else(L)).
wam_cpp_lowered_emitter_instr(["retry_me_else", L], retry_me_else(L)).
wam_cpp_lowered_emitter_instr(["trust_me"], trust_me).
wam_cpp_lowered_emitter_instr(["jump", L], jump(L)).
wam_cpp_lowered_emitter_instr(["cut_ite"], cut_ite).

%% walk_blocks(+AllItems, -Labels, -FlatInstrs)
%  Single pass: every label() records its PC; every instruction goes into
%  the flat list. Labels is a list of NameStr-PC pairs.
walk_blocks(Items, Labels, FlatInstrs) :-
    walk_blocks_(Items, 0, [], LabelsRev, [], FlatRev),
    reverse(LabelsRev, Labels),
    reverse(FlatRev, FlatInstrs).

walk_blocks_([], _, LAcc, LAcc, FAcc, FAcc).
walk_blocks_([label(N)|Rest], PC, LAcc, LOut, FAcc, FOut) :- !,
    walk_blocks_(Rest, PC, [N-PC|LAcc], LOut, FAcc, FOut).
walk_blocks_([Instr|Rest], PC, LAcc, LOut, FAcc, FOut) :-
    PC1 is PC + 1,
    walk_blocks_(Rest, PC1, LAcc, LOut, [Instr|FAcc], FOut).

%% lookup_label(+NameStr, +Labels, -PC)
lookup_label(NameStr, Labels, PC) :-
    ( member(NameStr-PC, Labels) -> true ; PC = 0 ).

%% emit_setup_function(+Predicates, +Options, -SetupCpp)
%  Top-level project assembly. Compiles each predicate to WAM text,
%  parses into blocks, concatenates, resolves PCs, and emits a single
%  wam_cpp_setup() function that populates WamState::instrs / labels.
emit_setup_function(Predicates, Options, SetupCpp) :-
    foreign_pred_keys_from_options(Options, _ForeignKeys),
    findall(Items, (
        member(PI, Predicates),
        catch(
            ( compile_predicate_to_wam(PI, [], WamText),
              parse_pred_blocks(WamText, Items)
            ),
            _, fail)
    ), PerPredItems),
    flatten_blocks(PerPredItems, AllItems),
    walk_blocks(AllItems, Labels, FlatInstrs),
    findall(LabelLine, (
        member(NameStr-PC, Labels),
        format(atom(LabelLine),
               '    vm.labels["~w"] = ~w;', [NameStr, PC])
    ), LabelLines),
    atomic_list_concat(LabelLines, '\n', LabelBody),
    findall(InstrLine, (
        member(I, FlatInstrs),
        instr_to_setup_line(I, Labels, InstrLine)
    ), InstrLines),
    atomic_list_concat(InstrLines, '\n', InstrBody),
    length(FlatInstrs, Reserve),
    format(string(SetupCpp),
'void wam_cpp_setup(WamState& vm) {
    vm.instrs.clear();
    vm.labels.clear();
    vm.instrs.reserve(~w);
~w
~w
}
static const int _wam_cpp_setup_register = []() {
    Program::register_setup(&wam_cpp_setup);
    return 0;
}();
', [Reserve, LabelBody, InstrBody]).

flatten_blocks([], []).
flatten_blocks([Items|Rest], All) :-
    flatten_blocks(Rest, RestAll),
    append(Items, RestAll, All).

%% instr_to_setup_line(+Instr, +Labels, -Line)
%  Emit a single `vm.instrs.push_back(Instruction::...);` line. Label
%  references inside try_me_else / retry_me_else / jump are looked up.
instr_to_setup_line(try_me_else(L), Labels, Line) :- !,
    label_resolve(L, Labels, PC),
    format(atom(Line),
           '    vm.instrs.push_back(Instruction::TryMeElse(~w));', [PC]).
instr_to_setup_line(retry_me_else(L), Labels, Line) :- !,
    label_resolve(L, Labels, PC),
    format(atom(Line),
           '    vm.instrs.push_back(Instruction::RetryMeElse(~w));', [PC]).
instr_to_setup_line(jump(L), Labels, Line) :- !,
    label_resolve(L, Labels, PC),
    format(atom(Line),
           '    vm.instrs.push_back(Instruction::Jump(~w));', [PC]).
instr_to_setup_line(Instr, _Labels, Line) :-
    wam_instruction_to_cpp_literal(Instr, Lit),
    format(atom(Line), '    vm.instrs.push_back(~w);', [Lit]).

label_resolve(L, Labels, PC) :-
    to_string(L, LS),
    lookup_label(LS, Labels, PC).

% ============================================================================
% Project-level assembly
% ============================================================================

%% write_wam_cpp_project(+Predicates, +Options, +ProjectDir)
%  Materialise a self-contained C++ project at ProjectDir/cpp/.
%  Layout:
%    cpp/wam_runtime.h
%    cpp/wam_runtime.cpp
%    cpp/generated_program.cpp
write_wam_cpp_project(Predicates, Options, ProjectDir) :-
    make_directory_path(ProjectDir),
    directory_file_path(ProjectDir, 'cpp', CppDir),
    make_directory_path(CppDir),
    compile_wam_runtime_header_to_cpp(Options, HeaderCode),
    directory_file_path(CppDir, 'wam_runtime.h', HeaderPath),
    write_text_file(HeaderPath, HeaderCode),
    compile_wam_runtime_to_cpp(Options, RuntimeCode),
    directory_file_path(CppDir, 'wam_runtime.cpp', RuntimePath),
    write_text_file(RuntimePath, RuntimeCode),
    compile_predicates_for_project(Predicates, Options, PredicatesCode),
    emit_setup_function(Predicates, Options, SetupCpp),
    format(string(ProgramCode),
'// SPDX-License-Identifier: MIT OR Apache-2.0
// Auto-generated by wam_cpp_target.pl. Do not edit by hand.
#include "wam_runtime.h"

~w

// ----------------------------------------------------------------------
// Program assembly: wam_cpp_setup() populates the WamState with all
// predicates concatenated into one instruction vector, with absolute PCs
// resolved for try_me_else / retry_me_else / jump targets.
// ----------------------------------------------------------------------
~w
', [PredicatesCode, SetupCpp]),
    directory_file_path(CppDir, 'generated_program.cpp', ProgramPath),
    write_text_file(ProgramPath, ProgramCode),
    (   option(emit_main(true), Options, false)
    ->  emit_main_shim(MainCode),
        directory_file_path(CppDir, 'main.cpp', MainPath),
        write_text_file(MainPath, MainCode)
    ;   true
    ).

%% emit_main_shim(-MainCpp)
%  A tiny CLI driver: argv[1] is the predicate key, remaining args are
%  parsed as Prolog-style terms (atom, integer, compound foo(a,b),
%  list [a,b,c]). Prints "true" / "false" and exits 0 / 1.
emit_main_shim('// SPDX-License-Identifier: MIT OR Apache-2.0
// Auto-generated by wam_cpp_target.pl. Do not edit by hand.
#include "wam_runtime.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

struct Parser {
    const char* p;

    void skip_ws() { while (*p == \' \' || *p == \'\\t\') ++p; }

    bool eof() const { return *p == \'\\0\'; }

    bool peek(char c) { skip_ws(); return *p == c; }

    bool eat(char c) {
        skip_ws();
        if (*p == c) { ++p; return true; }
        return false;
    }

    std::string read_ident() {
        std::string r;
        while (std::isalnum(static_cast<unsigned char>(*p))
               || *p == \'_\' || *p == \'.\') {
            r.push_back(*p++);
        }
        return r;
    }

    Value parse_atom_or_int(const std::string& name) {
        // name is non-empty and starts with non-digit / non-bracket / non-paren.
        if (!name.empty()
            && (std::isdigit(static_cast<unsigned char>(name[0])) || name[0] == \'-\')) {
            try { return Value::Integer(std::stoll(name)); }
            catch (...) { return Value::Atom(name); }
        }
        return Value::Atom(name);
    }

    Value parse_term() {
        skip_ws();
        if (*p == \'[\') {
            ++p;
            return parse_list_tail();
        }
        // Read an ident / number.
        std::string name = read_ident();
        if (eat(\'(\')) {
            std::vector<wam_cpp::CellPtr> args;
            if (!eat(\')\')) {
                args.push_back(std::make_shared<Value>(parse_term()));
                while (eat(\',\')) {
                    args.push_back(std::make_shared<Value>(parse_term()));
                }
                eat(\')\');
            }
            std::string functor = name + "/" + std::to_string(args.size());
            return Value::Compound(functor, std::move(args));
        }
        return parse_atom_or_int(name);
    }

    // After "[" has been consumed: parse "]" or "elem, elem, ... ]".
    Value parse_list_tail() {
        skip_ws();
        if (eat(\']\')) return Value::Atom("[]");
        std::vector<Value> elems;
        elems.push_back(parse_term());
        while (eat(\',\')) elems.push_back(parse_term());
        eat(\']\');
        // Build right-associative cons chain: [|]/2(h, [|]/2(...,[]))
        Value tail = Value::Atom("[]");
        for (auto it = elems.rbegin(); it != elems.rend(); ++it) {
            std::vector<wam_cpp::CellPtr> args;
            args.push_back(std::make_shared<Value>(*it));
            args.push_back(std::make_shared<Value>(std::move(tail)));
            tail = Value::Compound("[|]/2", std::move(args));
        }
        return tail;
    }
};

Value parse_arg(const char* s) {
    Parser pr{s};
    return pr.parse_term();
}

} // anonymous

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s pred/arity [arg ...]\\n", argv[0]);
        return 2;
    }
    WamState vm;
    Program::apply_setup(vm);
    std::vector<Value> args;
    for (int i = 2; i < argc; ++i) args.push_back(parse_arg(argv[i]));
    bool ok = vm.query(argv[1], args);
    std::printf("%s\\n", ok ? "true" : "false");
    return ok ? 0 : 1;
}
').

compile_predicates_for_project(Predicates, Options, PredicatesCode) :-
    foreign_pred_keys_from_options(Options, ForeignKeys),
    (   member(foreign_pred_keys(_), Options)
    ->  Options1 = Options
    ;   Options1 = [foreign_pred_keys(ForeignKeys)|Options]
    ),
    findall(Code, (
        member(PI, Predicates),
        catch(
            ( compile_predicate_to_wam(PI, [], WamCode),
              compile_wam_predicate_to_cpp(PI, WamCode, Options1, Code)
            ),
            _Err,
            fail)
    ), Codes),
    atomic_list_concat(Codes, '\n\n', PredicatesCode).

foreign_pred_keys_from_options(Options, Keys) :-
    (   member(foreign_pred_keys(Keys0), Options)
    ->  Keys = Keys0
    ;   Keys = []
    ).

write_text_file(Path, Content) :-
    setup_call_cleanup(
        open(Path, write, Stream, [encoding(utf8)]),
        write(Stream, Content),
        close(Stream)
    ).

% ============================================================================
% C++ runtime header & source — bundled inline so the project is
% self-contained without a template engine. The runtime is intentionally
% compact: just enough Instruction / Value / WamState surface to satisfy
% the lowered emitter and a minimal step() interpreter loop.
% ============================================================================

compile_wam_runtime_header_to_cpp(_Options,
'// SPDX-License-Identifier: MIT OR Apache-2.0
// Auto-generated by wam_cpp_target.pl. Do not edit by hand.
//
// WAM runtime header. Provides the Value, Instruction, ChoicePoint and
// WamState types referenced by code emitted from wam_cpp_lowered_emitter.pl
// and wam_cpp_target.pl.
//
// Heap model: every "register" or compound argument is a std::shared_ptr<Value>
// (alias Cell). Binding a variable mutates *cell so all aliases see the
// update — equivalent to classic WAM heap refs but without explicit indices.
// The trail records (cell, prev_value) pairs and undoes them on backtrack.

#ifndef UNIFYWEAVER_WAM_CPP_RUNTIME_H
#define UNIFYWEAVER_WAM_CPP_RUNTIME_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace wam_cpp {

struct Value;
using Cell = Value;
using CellPtr = std::shared_ptr<Cell>;

struct Value {
    enum class Tag { Uninit, Atom, Integer, Float, Unbound, Compound };
    Tag tag = Tag::Uninit;
    std::string s;     // Atom name / Unbound name / Compound functor ("f/n")
    std::int64_t i = 0;
    double f = 0.0;
    std::vector<CellPtr> args; // Compound only

    Value() = default;
    static Value Atom(std::string a)    { Value v; v.tag = Tag::Atom;    v.s = std::move(a); return v; }
    static Value Integer(std::int64_t n){ Value v; v.tag = Tag::Integer; v.i = n;             return v; }
    static Value Float(double d)        { Value v; v.tag = Tag::Float;   v.f = d;             return v; }
    static Value Unbound(std::string n) { Value v; v.tag = Tag::Unbound; v.s = std::move(n);  return v; }
    static Value Compound(std::string fn, std::vector<CellPtr> a)
        { Value v; v.tag = Tag::Compound; v.s = std::move(fn); v.args = std::move(a); return v; }

    bool is_unbound() const { return tag == Tag::Uninit || tag == Tag::Unbound; }

    bool operator==(const Value& o) const {
        if (tag != o.tag) return false;
        switch (tag) {
            case Tag::Atom:    return s == o.s;
            case Tag::Integer: return i == o.i;
            case Tag::Float:   return f == o.f;
            case Tag::Unbound: return s == o.s;
            case Tag::Compound: {
                if (s != o.s) return false;
                if (args.size() != o.args.size()) return false;
                for (std::size_t k = 0; k < args.size(); ++k) {
                    if (!(*args[k] == *o.args[k])) return false;
                }
                return true;
            }
            default: return true;
        }
    }
};

struct Instruction {
    enum class Op {
        GetConstant, GetVariable, GetValue, GetStructure, GetList, GetNil, GetInteger,
        PutConstant, PutVariable, PutValue, PutStructure, PutList,
        UnifyVariable, UnifyValue, UnifyConstant,
        SetVariable, SetValue, SetConstant,
        Call, Execute, Proceed, Fail, Allocate, Deallocate,
        BuiltinCall, CallForeign,
        TryMeElse, RetryMeElse, TrustMe, Jump, CutIte
    };
    Op op = Op::Proceed;
    Value val;
    std::string a;
    std::string b;
    std::int64_t n = 0;
    std::size_t target = 0;

    static Instruction GetConstant(Value v, std::string ai)
        { Instruction i; i.op = Op::GetConstant; i.val = std::move(v); i.a = std::move(ai); return i; }
    static Instruction GetVariable(std::string xn, std::string ai)
        { Instruction i; i.op = Op::GetVariable; i.a = std::move(xn); i.b = std::move(ai); return i; }
    static Instruction GetValue(std::string xn, std::string ai)
        { Instruction i; i.op = Op::GetValue; i.a = std::move(xn); i.b = std::move(ai); return i; }
    static Instruction GetStructure(std::string f, std::string ai)
        { Instruction i; i.op = Op::GetStructure; i.a = std::move(f); i.b = std::move(ai); return i; }
    static Instruction GetList(std::string ai)
        { Instruction i; i.op = Op::GetList; i.a = std::move(ai); return i; }
    static Instruction GetNil(std::string ai)
        { Instruction i; i.op = Op::GetNil; i.a = std::move(ai); return i; }
    static Instruction GetInteger(std::int64_t n, std::string ai)
        { Instruction i; i.op = Op::GetInteger; i.n = n; i.a = std::move(ai); return i; }
    static Instruction PutConstant(Value v, std::string ai)
        { Instruction i; i.op = Op::PutConstant; i.val = std::move(v); i.a = std::move(ai); return i; }
    static Instruction PutVariable(std::string xn, std::string ai)
        { Instruction i; i.op = Op::PutVariable; i.a = std::move(xn); i.b = std::move(ai); return i; }
    static Instruction PutValue(std::string xn, std::string ai)
        { Instruction i; i.op = Op::PutValue; i.a = std::move(xn); i.b = std::move(ai); return i; }
    static Instruction PutStructure(std::string f, std::string ai)
        { Instruction i; i.op = Op::PutStructure; i.a = std::move(f); i.b = std::move(ai); return i; }
    static Instruction PutList(std::string ai)
        { Instruction i; i.op = Op::PutList; i.a = std::move(ai); return i; }
    static Instruction UnifyVariable(std::string xn)
        { Instruction i; i.op = Op::UnifyVariable; i.a = std::move(xn); return i; }
    static Instruction UnifyValue(std::string xn)
        { Instruction i; i.op = Op::UnifyValue; i.a = std::move(xn); return i; }
    static Instruction UnifyConstant(Value v)
        { Instruction i; i.op = Op::UnifyConstant; i.val = std::move(v); return i; }
    static Instruction SetVariable(std::string xn)
        { Instruction i; i.op = Op::SetVariable; i.a = std::move(xn); return i; }
    static Instruction SetValue(std::string xn)
        { Instruction i; i.op = Op::SetValue; i.a = std::move(xn); return i; }
    static Instruction SetConstant(Value v)
        { Instruction i; i.op = Op::SetConstant; i.val = std::move(v); return i; }
    static Instruction Call(std::string p, std::int64_t n)
        { Instruction i; i.op = Op::Call; i.a = std::move(p); i.n = n; return i; }
    static Instruction Execute(std::string p)
        { Instruction i; i.op = Op::Execute; i.a = std::move(p); return i; }
    static Instruction Proceed()    { Instruction i; i.op = Op::Proceed;    return i; }
    static Instruction Fail()       { Instruction i; i.op = Op::Fail;       return i; }
    static Instruction Allocate()   { Instruction i; i.op = Op::Allocate;   return i; }
    static Instruction Deallocate() { Instruction i; i.op = Op::Deallocate; return i; }
    static Instruction BuiltinCall(std::string op, std::int64_t n)
        { Instruction i; i.op = Op::BuiltinCall; i.a = std::move(op); i.n = n; return i; }
    static Instruction CallForeign(std::string p, std::int64_t n)
        { Instruction i; i.op = Op::CallForeign; i.a = std::move(p); i.n = n; return i; }
    static Instruction TryMeElse(std::size_t target)
        { Instruction i; i.op = Op::TryMeElse; i.target = target; return i; }
    static Instruction RetryMeElse(std::size_t target)
        { Instruction i; i.op = Op::RetryMeElse; i.target = target; return i; }
    static Instruction TrustMe()    { Instruction i; i.op = Op::TrustMe;    return i; }
    static Instruction Jump(std::size_t target)
        { Instruction i; i.op = Op::Jump; i.target = target; return i; }
    static Instruction CutIte()     { Instruction i; i.op = Op::CutIte;     return i; }
};

struct TrailEntry {
    CellPtr cell;
    Value prev;
};

// Read/write mode for the most recent Get-/Put-Structure / Get-/Put-List.
// Only one is "active" at a time — nested compounds restart the mode
// when their own Get-/Put-Structure fires.
struct ModeFrame {
    enum class Kind { None, Read, Write };
    Kind kind = Kind::None;
    CellPtr target;                // Write: the compound cell being filled
    std::vector<CellPtr> args;     // Read: the compound''s arg cells
    std::size_t idx = 0;           // Read: next arg index
    std::size_t expected_arity = 0; // Write: stop when args.size() reaches this
};

struct ChoicePoint {
    std::size_t alt_pc;
    std::size_t saved_cp;
    std::size_t trail_mark;
    std::size_t cut_barrier;
    std::unordered_map<std::string, CellPtr> saved_regs;
    ModeFrame saved_mode;
};

struct WamState {
    std::unordered_map<std::string, CellPtr> regs;
    std::unordered_map<std::string, std::size_t> labels;
    std::vector<Instruction> instrs;
    std::vector<TrailEntry> trail;
    std::vector<ChoicePoint> choice_points;
    ModeFrame mode;
    std::size_t pc = 0;
    std::size_t cp = 0;
    std::size_t cut_barrier = 0;
    std::uint64_t var_counter = 0;
    bool halt = false;

    // Cell-aware accessors. get_reg/put_reg keep their Value-shaped API
    // so existing lowered code keeps compiling; get_cell exposes the cell
    // for instructions that need sharing semantics.
    Value   get_reg(const std::string& name) const;
    void    put_reg(const std::string& name, Value v);
    CellPtr get_cell(const std::string& name);
    void    set_cell(const std::string& name, CellPtr c);

    // Deref through Unbound chains until a concrete value (or a terminal
    // unbound cell) is reached. Returns by value (snapshot).
    Value   deref(const Value& v) const;

    // bind_cell mutates *cell, recording the previous content on the trail.
    void    bind_cell(CellPtr cell, Value v);
    void    trail_binding(const std::string& name); // legacy reg-name trail

    // Unification: takes Cell pointers so binding works correctly.
    bool    unify_cells(CellPtr a, CellPtr b);
    // Convenience for the lowered emitter (no real binding — equality
    // / unbound-as-success check only).
    bool    unify(const Value& a, const Value& b);

    bool    step(const Instruction& instr);
    bool    run();
    bool    backtrack();
    bool    builtin(const std::string& op, std::int64_t arity);
    bool    query(const std::string& pred_key, const std::vector<Value>& args);
};

struct Program {
    // Setup callback: each generated_program.cpp defines a unique
    // wam_cpp_setup() function that populates instrs + labels at startup.
    using Setup = void (*)(WamState&);
    static Setup& setup_hook();
    static void register_setup(Setup s);
    static void apply_setup(WamState& vm);
};

} // namespace wam_cpp

using wam_cpp::Value;
using wam_cpp::Instruction;
using wam_cpp::WamState;
using wam_cpp::Program;

#endif // UNIFYWEAVER_WAM_CPP_RUNTIME_H
').

compile_wam_runtime_to_cpp(_Options,
'// SPDX-License-Identifier: MIT OR Apache-2.0
// Auto-generated by wam_cpp_target.pl. Do not edit by hand.
//
// WAM runtime implementation. Handles atom/integer-level unification,
// compound terms and lists via shared_ptr cells, choice points with full
// trail-based undo, and the standard control-flow opcode set.

#include "wam_runtime.h"

#include <utility>

namespace wam_cpp {

// ----------------------------------------------------------------------
// Cell helpers
// ----------------------------------------------------------------------

static CellPtr make_cell(Value v = Value{}) {
    return std::make_shared<Cell>(std::move(v));
}

CellPtr WamState::get_cell(const std::string& name) {
    auto it = regs.find(name);
    if (it != regs.end()) return it->second;
    CellPtr c = make_cell(Value::Unbound("_U_" + name));
    regs[name] = c;
    return c;
}

void WamState::set_cell(const std::string& name, CellPtr c) {
    regs[name] = std::move(c);
}

Value WamState::get_reg(const std::string& name) const {
    auto it = regs.find(name);
    if (it == regs.end()) return Value{};
    return *it->second;
}

void WamState::put_reg(const std::string& name, Value v) {
    auto it = regs.find(name);
    if (it == regs.end()) regs[name] = make_cell(std::move(v));
    else                  *it->second = std::move(v);
}

void WamState::trail_binding(const std::string& name) {
    auto it = regs.find(name);
    if (it == regs.end()) return;
    TrailEntry e;
    e.cell = it->second;
    e.prev = *it->second;
    trail.push_back(std::move(e));
}

void WamState::bind_cell(CellPtr cell, Value v) {
    TrailEntry e;
    e.cell = cell;
    e.prev = *cell;
    trail.push_back(std::move(e));
    *cell = std::move(v);
}

Value WamState::deref(const Value& v) const {
    // Followers are explicitly bound Unbound cells. With our shared_ptr
    // model the cell content is mutated in place, so deref through Value
    // alone is a no-op — we just hand back v.
    return v;
}

// Cell-aware unification. Binds variables via bind_cell so the trail
// records every change.
bool WamState::unify_cells(CellPtr a, CellPtr b) {
    if (a.get() == b.get()) return true;
    Value& va = *a;
    Value& vb = *b;
    if (va.is_unbound()) { bind_cell(a, vb); return true; }
    if (vb.is_unbound()) { bind_cell(b, va); return true; }
    if (va.tag != vb.tag) return false;
    switch (va.tag) {
        case Value::Tag::Atom:    return va.s == vb.s;
        case Value::Tag::Integer: return va.i == vb.i;
        case Value::Tag::Float:   return va.f == vb.f;
        case Value::Tag::Compound: {
            if (va.s != vb.s) return false;
            if (va.args.size() != vb.args.size()) return false;
            for (std::size_t k = 0; k < va.args.size(); ++k) {
                if (!unify_cells(va.args[k], vb.args[k])) return false;
            }
            return true;
        }
        default: return true;
    }
}

bool WamState::unify(const Value& a, const Value& b) {
    if (a.is_unbound() || b.is_unbound()) return true;
    return a == b;
}

bool WamState::builtin(const std::string& op, std::int64_t /*arity*/) {
    if (op == "true/0") { pc += 1; return true; }
    if (op == "fail/0") { return false; }
    if (op == "!/0")    {
        if (choice_points.size() > cut_barrier) choice_points.resize(cut_barrier);
        pc += 1; return true;
    }
    return false;
}

// ----------------------------------------------------------------------
// Mode helpers for read/write of Get-/Put-Structure / Get-/Put-List.
// ----------------------------------------------------------------------

namespace {

void enter_read_mode(ModeFrame& mode, const std::vector<CellPtr>& args) {
    mode.kind = ModeFrame::Kind::Read;
    mode.target.reset();
    mode.args = args;
    mode.idx = 0;
    mode.expected_arity = args.size();
}

void enter_write_mode(ModeFrame& mode, CellPtr target, std::size_t arity) {
    mode.kind = ModeFrame::Kind::Write;
    mode.target = std::move(target);
    mode.args.clear();
    mode.idx = 0;
    mode.expected_arity = arity;
    // Pre-clear the compound''s args; we will push back as Set*/Unify*
    // fire. The functor was already written by the Get-/Put- caller.
    if (mode.target) mode.target->args.clear();
}

} // anonymous

// Begin a write-mode structure / list with the given functor and arity.
// Binds (or rebinds) `target` to Compound{functor, []} and enters write mode.
static void begin_write(WamState& vm, CellPtr target, const std::string& functor,
                        std::size_t arity) {
    vm.bind_cell(target, Value::Compound(functor, {}));
    enter_write_mode(vm.mode, target, arity);
}

// ----------------------------------------------------------------------
// Step
// ----------------------------------------------------------------------

bool WamState::step(const Instruction& instr) {
    switch (instr.op) {
        // ---- Head unification --------------------------------------
        case Instruction::Op::GetConstant: {
            CellPtr a = get_cell(instr.a);
            if (a->is_unbound())          { bind_cell(a, instr.val); }
            else if (!(*a == instr.val))  { return false; }
            pc += 1; return true;
        }
        case Instruction::Op::GetInteger: {
            CellPtr a = get_cell(instr.a);
            Value want = Value::Integer(instr.n);
            if (a->is_unbound())     { bind_cell(a, want); }
            else if (!(*a == want))  { return false; }
            pc += 1; return true;
        }
        case Instruction::Op::GetNil: {
            CellPtr a = get_cell(instr.a);
            Value want = Value::Atom("[]");
            if (a->is_unbound())     { bind_cell(a, want); }
            else if (!(*a == want))  { return false; }
            pc += 1; return true;
        }
        case Instruction::Op::GetVariable: {
            // X-reg shares the same cell as A-reg.
            set_cell(instr.a, get_cell(instr.b));
            pc += 1; return true;
        }
        case Instruction::Op::GetValue: {
            if (!unify_cells(get_cell(instr.a), get_cell(instr.b))) return false;
            pc += 1; return true;
        }
        case Instruction::Op::GetStructure: {
            // instr.a = functor ("box/2"), instr.b = register ("A1" / "X1" ...).
            CellPtr a = get_cell(instr.b);
            const std::string& functor = instr.a;
            std::size_t arity = 0;
            auto p = functor.find_last_of(\'/\');
            if (p != std::string::npos) arity = std::stoull(functor.substr(p + 1));
            if (a->is_unbound()) {
                // Write mode: bind a to a fresh compound, expect `arity` Unify*.
                begin_write(*this, a, functor, arity);
                pc += 1; return true;
            }
            if (a->tag == Value::Tag::Compound && a->s == functor && a->args.size() == arity) {
                enter_read_mode(mode, a->args);
                pc += 1; return true;
            }
            return false;
        }
        case Instruction::Op::GetList: {
            CellPtr a = get_cell(instr.a);
            const std::string functor = "[|]/2";
            if (a->is_unbound()) {
                begin_write(*this, a, functor, 2);
                pc += 1; return true;
            }
            if (a->tag == Value::Tag::Compound && a->s == functor && a->args.size() == 2) {
                enter_read_mode(mode, a->args);
                pc += 1; return true;
            }
            return false;
        }

        // ---- Body construction -------------------------------------
        case Instruction::Op::PutConstant: {
            put_reg(instr.a, instr.val);
            pc += 1; return true;
        }
        case Instruction::Op::PutVariable: {
            // Fresh shared cell visible from both registers.
            CellPtr v = make_cell(Value::Unbound("_V" + std::to_string(var_counter++)));
            set_cell(instr.a, v);
            set_cell(instr.b, v);
            pc += 1; return true;
        }
        case Instruction::Op::PutValue: {
            // Target reg shares the source''s cell.
            set_cell(instr.b, get_cell(instr.a));
            pc += 1; return true;
        }
        case Instruction::Op::PutStructure: {
            const std::string& functor = instr.a;
            std::size_t arity = 0;
            auto p = functor.find_last_of(\'/\');
            if (p != std::string::npos) arity = std::stoull(functor.substr(p + 1));
            // Reuse / allocate the cell for instr.b. If it''s already an
            // Unbound shared with some parent struct, mutating *cell here
            // is what makes the chain-write pattern work.
            CellPtr target = get_cell(instr.b);
            begin_write(*this, target, functor, arity);
            pc += 1; return true;
        }
        case Instruction::Op::PutList: {
            CellPtr target = get_cell(instr.a);
            begin_write(*this, target, "[|]/2", 2);
            pc += 1; return true;
        }

        // ---- Unify (post Get-/Put-Structure / List) -----------------
        case Instruction::Op::UnifyVariable: {
            if (mode.kind == ModeFrame::Kind::Read) {
                if (mode.idx >= mode.args.size()) return false;
                set_cell(instr.a, mode.args[mode.idx]);
                mode.idx += 1;
                pc += 1; return true;
            }
            if (mode.kind == ModeFrame::Kind::Write && mode.target) {
                CellPtr fresh = make_cell(Value::Unbound("_V" + std::to_string(var_counter++)));
                set_cell(instr.a, fresh);
                mode.target->args.push_back(fresh);
                pc += 1; return true;
            }
            return false;
        }
        case Instruction::Op::UnifyValue: {
            if (mode.kind == ModeFrame::Kind::Read) {
                if (mode.idx >= mode.args.size()) return false;
                if (!unify_cells(get_cell(instr.a), mode.args[mode.idx])) return false;
                mode.idx += 1;
                pc += 1; return true;
            }
            if (mode.kind == ModeFrame::Kind::Write && mode.target) {
                mode.target->args.push_back(get_cell(instr.a));
                pc += 1; return true;
            }
            return false;
        }
        case Instruction::Op::UnifyConstant: {
            if (mode.kind == ModeFrame::Kind::Read) {
                if (mode.idx >= mode.args.size()) return false;
                CellPtr c = mode.args[mode.idx];
                if (c->is_unbound())         { bind_cell(c, instr.val); }
                else if (!(*c == instr.val)) { return false; }
                mode.idx += 1;
                pc += 1; return true;
            }
            if (mode.kind == ModeFrame::Kind::Write && mode.target) {
                mode.target->args.push_back(make_cell(instr.val));
                pc += 1; return true;
            }
            return false;
        }

        // ---- Set (always write mode, equivalent to Unify* in write) --
        case Instruction::Op::SetVariable: {
            if (mode.kind != ModeFrame::Kind::Write || !mode.target) return false;
            CellPtr fresh = make_cell(Value::Unbound("_V" + std::to_string(var_counter++)));
            set_cell(instr.a, fresh);
            mode.target->args.push_back(fresh);
            pc += 1; return true;
        }
        case Instruction::Op::SetValue: {
            if (mode.kind != ModeFrame::Kind::Write || !mode.target) return false;
            mode.target->args.push_back(get_cell(instr.a));
            pc += 1; return true;
        }
        case Instruction::Op::SetConstant: {
            if (mode.kind != ModeFrame::Kind::Write || !mode.target) return false;
            mode.target->args.push_back(make_cell(instr.val));
            pc += 1; return true;
        }

        // ---- Environment frames (no Y-reg discipline yet) ----------
        case Instruction::Op::Allocate:
        case Instruction::Op::Deallocate:
            pc += 1; return true;

        // ---- Control flow ------------------------------------------
        case Instruction::Op::Call: {
            auto it = labels.find(instr.a);
            if (it == labels.end()) return false;
            cp = pc + 1;
            pc = it->second;
            return true;
        }
        case Instruction::Op::Execute: {
            auto it = labels.find(instr.a);
            if (it == labels.end()) return false;
            pc = it->second;
            return true;
        }
        case Instruction::Op::Proceed: {
            if (cp == 0) { halt = true; return true; }
            pc = cp;
            cp = 0;
            return true;
        }
        case Instruction::Op::Fail:
            return false;
        case Instruction::Op::Jump:
            pc = instr.target;
            return true;

        // ---- Choice points -----------------------------------------
        case Instruction::Op::TryMeElse: {
            ChoicePoint cp_;
            cp_.alt_pc = instr.target;
            cp_.saved_cp = cp;
            cp_.trail_mark = trail.size();
            cp_.cut_barrier = cut_barrier;
            cp_.saved_regs = regs;
            cp_.saved_mode = mode;
            choice_points.push_back(std::move(cp_));
            pc += 1; return true;
        }
        case Instruction::Op::RetryMeElse: {
            if (choice_points.empty()) return false;
            choice_points.back().alt_pc = instr.target;
            pc += 1; return true;
        }
        case Instruction::Op::TrustMe: {
            if (!choice_points.empty()) choice_points.pop_back();
            pc += 1; return true;
        }
        case Instruction::Op::CutIte: {
            if (choice_points.size() > cut_barrier) choice_points.resize(cut_barrier);
            pc += 1; return true;
        }

        // ---- Builtins ----------------------------------------------
        case Instruction::Op::BuiltinCall:
            return builtin(instr.a, instr.n);
        case Instruction::Op::CallForeign:
            pc += 1; return true;
    }
    return false;
}

bool WamState::backtrack() {
    while (!choice_points.empty()) {
        ChoicePoint cp_ = std::move(choice_points.back());
        choice_points.pop_back();
        // Unwind cell mutations down to the trail mark.
        while (trail.size() > cp_.trail_mark) {
            TrailEntry t = std::move(trail.back());
            trail.pop_back();
            *t.cell = std::move(t.prev);
        }
        regs = std::move(cp_.saved_regs);
        mode = std::move(cp_.saved_mode);
        cp = cp_.saved_cp;
        cut_barrier = cp_.cut_barrier;
        pc = cp_.alt_pc;
        return true;
    }
    return false;
}

bool WamState::run() {
    halt = false;
    while (!halt) {
        if (pc >= instrs.size()) return false;
        if (!step(instrs[pc])) {
            if (!backtrack()) return false;
        }
    }
    return true;
}

bool WamState::query(const std::string& pred_key, const std::vector<Value>& args) {
    auto it = labels.find(pred_key);
    if (it == labels.end()) return false;
    regs.clear();
    for (std::size_t k = 0; k < args.size(); ++k) {
        regs["A" + std::to_string(k + 1)] = make_cell(args[k]);
    }
    trail.clear();
    choice_points.clear();
    mode = ModeFrame{};
    pc = it->second;
    cp = 0;
    cut_barrier = 0;
    halt = false;
    return run();
}

Program::Setup& Program::setup_hook() {
    static Setup s = nullptr;
    return s;
}

void Program::register_setup(Setup s) { setup_hook() = s; }
void Program::apply_setup(WamState& vm) { if (setup_hook()) setup_hook()(vm); }

} // namespace wam_cpp
').
