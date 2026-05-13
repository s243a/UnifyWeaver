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
    cpp_atomics_to_string(Parts2, "\\\"", Out),
    !.

cpp_atomics_to_string([], _, "") :- !.
cpp_atomics_to_string([X], _, X) :- !.
cpp_atomics_to_string([X, Y|Rest], Sep, Result) :-
    cpp_atomics_to_string([Y|Rest], Sep, Tail),
    string_concat(X, Sep, XSep),
    string_concat(XSep, Tail, Result),
    !.

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
    wam_instruction_to_cpp_literal(Instr, [], Code),
    !.

wam_instruction_to_cpp_literal(Instr, LabelMap, Code) :-
    wam_instruction_to_cpp_literal_det(Instr, LabelMap, Code),
    !.

wam_instruction_to_cpp_literal_det(get_constant(C, Ai), _, Code) :-
    cpp_value_literal(C, V), to_string(Ai, R),
    format(atom(Code), 'Instruction::GetConstant(~w, "~w")', [V, R]).
wam_instruction_to_cpp_literal_det(get_variable(Xn, Ai), _, Code) :-
    to_string(Xn, X), to_string(Ai, R),
    format(atom(Code), 'Instruction::GetVariable("~w", "~w")', [X, R]).
wam_instruction_to_cpp_literal_det(get_value(Xn, Ai), _, Code) :-
    to_string(Xn, X), to_string(Ai, R),
    format(atom(Code), 'Instruction::GetValue("~w", "~w")', [X, R]).
wam_instruction_to_cpp_literal_det(get_structure(F, Ai), _, Code) :-
    to_string(F, FS), to_string(Ai, R),
    escape_cpp_string(FS, EF),
    format(atom(Code), 'Instruction::GetStructure("~w", "~w")', [EF, R]).
wam_instruction_to_cpp_literal_det(get_list(Ai), _, Code) :-
    to_string(Ai, R),
    format(atom(Code), 'Instruction::GetList("~w")', [R]).
wam_instruction_to_cpp_literal_det(get_nil(Ai), _, Code) :-
    to_string(Ai, R),
    format(atom(Code), 'Instruction::GetNil("~w")', [R]).
wam_instruction_to_cpp_literal_det(get_integer(N, Ai), _, Code) :-
    to_string(N, NS), to_string(Ai, R),
    format(atom(Code), 'Instruction::GetInteger(~w, "~w")', [NS, R]).
wam_instruction_to_cpp_literal_det(put_constant(C, Ai), _, Code) :-
    cpp_value_literal(C, V), to_string(Ai, R),
    format(atom(Code), 'Instruction::PutConstant(~w, "~w")', [V, R]).
wam_instruction_to_cpp_literal_det(put_variable(Xn, Ai), _, Code) :-
    to_string(Xn, X), to_string(Ai, R),
    format(atom(Code), 'Instruction::PutVariable("~w", "~w")', [X, R]).
wam_instruction_to_cpp_literal_det(put_value(Xn, Ai), _, Code) :-
    to_string(Xn, X), to_string(Ai, R),
    format(atom(Code), 'Instruction::PutValue("~w", "~w")', [X, R]).
wam_instruction_to_cpp_literal_det(put_structure(F, Ai), _, Code) :-
    to_string(F, FS), to_string(Ai, R),
    escape_cpp_string(FS, EF),
    format(atom(Code), 'Instruction::PutStructure("~w", "~w")', [EF, R]).
wam_instruction_to_cpp_literal_det(put_list(Ai), _, Code) :-
    to_string(Ai, R),
    format(atom(Code), 'Instruction::PutList("~w")', [R]).
wam_instruction_to_cpp_literal_det(unify_variable(Xn), _, Code) :-
    to_string(Xn, X),
    format(atom(Code), 'Instruction::UnifyVariable("~w")', [X]).
wam_instruction_to_cpp_literal_det(unify_value(Xn), _, Code) :-
    to_string(Xn, X),
    format(atom(Code), 'Instruction::UnifyValue("~w")', [X]).
wam_instruction_to_cpp_literal_det(unify_constant(C), _, Code) :-
    cpp_value_literal(C, V),
    format(atom(Code), 'Instruction::UnifyConstant(~w)', [V]).
wam_instruction_to_cpp_literal_det(set_variable(Xn), _, Code) :-
    to_string(Xn, X),
    format(atom(Code), 'Instruction::SetVariable("~w")', [X]).
wam_instruction_to_cpp_literal_det(set_value(Xn), _, Code) :-
    to_string(Xn, X),
    format(atom(Code), 'Instruction::SetValue("~w")', [X]).
wam_instruction_to_cpp_literal_det(set_constant(C), _, Code) :-
    cpp_value_literal(C, V),
    format(atom(Code), 'Instruction::SetConstant(~w)', [V]).
wam_instruction_to_cpp_literal_det(call(P, N), _, Code) :-
    to_string(P, PS), to_string(N, NS),
    escape_cpp_string(PS, EP),
    format(atom(Code), 'Instruction::Call("~w", ~w)', [EP, NS]).
wam_instruction_to_cpp_literal_det(execute(P), _, Code) :-
    to_string(P, PS),
    escape_cpp_string(PS, EP),
    format(atom(Code), 'Instruction::Execute("~w")', [EP]).
wam_instruction_to_cpp_literal_det(proceed, _, 'Instruction::Proceed()').
wam_instruction_to_cpp_literal_det(fail, _, 'Instruction::Fail()').
wam_instruction_to_cpp_literal_det(allocate, _, 'Instruction::Allocate()').
wam_instruction_to_cpp_literal_det(deallocate, _, 'Instruction::Deallocate()').
wam_instruction_to_cpp_literal_det(builtin_call(Op, Ar), _, Code) :-
    to_string(Op, OpS), to_string(Ar, ArS),
    escape_cpp_string(OpS, EOp),
    format(atom(Code), 'Instruction::BuiltinCall("~w", ~w)', [EOp, ArS]).
wam_instruction_to_cpp_literal_det(call_foreign(P, Ar), _, Code) :-
    to_string(P, PS), to_string(Ar, ArS),
    escape_cpp_string(PS, EP),
    format(atom(Code), 'Instruction::CallForeign("~w", ~w)', [EP, ArS]).
wam_instruction_to_cpp_literal_det(try_me_else(L), LabelMap, Code) :-
    label_index(L, LabelMap, Idx),
    format(atom(Code), 'Instruction::TryMeElse(~w)', [Idx]).
wam_instruction_to_cpp_literal_det(retry_me_else(L), LabelMap, Code) :-
    label_index(L, LabelMap, Idx),
    format(atom(Code), 'Instruction::RetryMeElse(~w)', [Idx]).
wam_instruction_to_cpp_literal_det(trust_me, _, 'Instruction::TrustMe()').
wam_instruction_to_cpp_literal_det(jump(L), LabelMap, Code) :-
    label_index(L, LabelMap, Idx),
    format(atom(Code), 'Instruction::Jump(~w)', [Idx]).
wam_instruction_to_cpp_literal_det(cut_ite, _, 'Instruction::CutIte()').
wam_instruction_to_cpp_literal_det(begin_aggregate(K, V, R), _, Code) :-
    to_string(K, KS), to_string(V, VS), to_string(R, RS),
    escape_cpp_string(KS, EK), escape_cpp_string(VS, EV), escape_cpp_string(RS, ER),
    format(atom(Code),
           'Instruction::BeginAggregate("~w", "~w", "~w")', [EK, EV, ER]).
wam_instruction_to_cpp_literal_det(end_aggregate(R), _, Code) :-
    to_string(R, RS),
    escape_cpp_string(RS, ER),
    format(atom(Code), 'Instruction::EndAggregate("~w")', [ER]).

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
wam_cpp_lowered_emitter_instr(["begin_aggregate", K, V, R], begin_aggregate(K, V, R)).
wam_cpp_lowered_emitter_instr(["end_aggregate", R], end_aggregate(R)).

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
            ( compile_predicate_to_wam(PI, [inline_bagof_setof(true)], WamText),
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
            ( compile_predicate_to_wam(PI, [inline_bagof_setof(true)], WamCode),
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
        TryMeElse, RetryMeElse, TrustMe, Jump, CutIte,
        BeginAggregate, EndAggregate
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
    static Instruction BeginAggregate(std::string kind, std::string vreg, std::string rreg)
        { Instruction i; i.op = Op::BeginAggregate; i.a = std::move(kind);
          i.b = std::move(vreg); i.val = Value::Atom(std::move(rreg)); return i; }
    static Instruction EndAggregate(std::string vreg)
        { Instruction i; i.op = Op::EndAggregate; i.a = std::move(vreg); return i; }
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
    std::vector<ModeFrame> saved_mode_stack;
};

// Aggregate scope opened by BeginAggregate. Backtrack() finalises the
// frame when choice_points has been drained back to base_cp_count and
// no normal CP is available.
struct AggregateFrame {
    std::string agg_kind;       // "collect" / "count" / "sum" / "min" / "max" / "set" / "bag"
    std::string value_reg;
    std::string result_reg;
    std::size_t begin_pc = 0;   // pc of the BeginAggregate instruction
    std::size_t return_pc = 0;  // pc after end_aggregate (0 if never fired)
    bool return_pc_set = false;
    std::size_t base_cp_count = 0;
    std::size_t trail_mark = 0;
    std::size_t saved_cp = 0;
    std::size_t saved_cut_barrier = 0;
    std::unordered_map<std::string, CellPtr> saved_regs;
    std::vector<ModeFrame> saved_mode_stack;
    std::vector<Value> acc;
};

struct WamState {
    std::unordered_map<std::string, CellPtr> regs;
    std::unordered_map<std::string, std::size_t> labels;
    std::vector<Instruction> instrs;
    std::vector<TrailEntry> trail;
    std::vector<ChoicePoint> choice_points;
    std::vector<AggregateFrame> aggregate_frames;
    // Mode stack: Get-/Put-Structure / Get-/Put-List PUSH; Unify*/Set*
    // operate on top(); each frame auto-pops once its arity is filled.
    std::vector<ModeFrame> mode_stack;
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

    // ---- Builtin helpers ---------------------------------------------
    // Arithmetic: returns Integer or Float; sets ok=false on failure
    // (unbound argument, unknown operator, division by zero, ...).
    Value   eval_arith(CellPtr c, bool& ok) const;
    bool    arith_compare(const std::string& op, const Value& a, const Value& b) const;

    // Term rendering for write/1 — recursive, prints Prolog-like syntax.
    std::string render(const Value& v) const;
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

#include <algorithm>
#include <cstdio>
#include <sstream>
#include <string>
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

// ----------------------------------------------------------------------
// Arithmetic eval
// ----------------------------------------------------------------------

Value WamState::eval_arith(CellPtr c, bool& ok) const {
    if (!c) { ok = false; return Value{}; }
    const Value& v = *c;
    switch (v.tag) {
        case Value::Tag::Integer:
        case Value::Tag::Float:
            return v;
        case Value::Tag::Atom:
            // Numeric atoms ("5", "-3.14") are tolerated for robustness.
            try {
                std::size_t pos;
                long long n = std::stoll(v.s, &pos);
                if (pos == v.s.size()) return Value::Integer(n);
            } catch (...) {}
            try {
                std::size_t pos;
                double d = std::stod(v.s, &pos);
                if (pos == v.s.size()) return Value::Float(d);
            } catch (...) {}
            ok = false; return Value{};
        case Value::Tag::Compound: {
            // Unary minus
            if (v.s == "-/1" && v.args.size() == 1) {
                Value a = eval_arith(v.args[0], ok);
                if (!ok) return Value{};
                if (a.tag == Value::Tag::Integer) return Value::Integer(-a.i);
                return Value::Float(-a.f);
            }
            if (v.args.size() != 2) { ok = false; return Value{}; }
            Value a = eval_arith(v.args[0], ok);
            if (!ok) return Value{};
            Value b = eval_arith(v.args[1], ok);
            if (!ok) return Value{};
            bool either_float = (a.tag == Value::Tag::Float || b.tag == Value::Tag::Float);
            auto as_d = [](const Value& w){ return w.tag == Value::Tag::Float ? w.f : (double)w.i; };
            if (v.s == "+/2") {
                if (either_float) return Value::Float(as_d(a) + as_d(b));
                return Value::Integer(a.i + b.i);
            }
            if (v.s == "-/2") {
                if (either_float) return Value::Float(as_d(a) - as_d(b));
                return Value::Integer(a.i - b.i);
            }
            if (v.s == "*/2") {
                if (either_float) return Value::Float(as_d(a) * as_d(b));
                return Value::Integer(a.i * b.i);
            }
            if (v.s == "//2") {
                // Prolog ''/'' is float division.
                if (as_d(b) == 0.0) { ok = false; return Value{}; }
                if (either_float) return Value::Float(as_d(a) / as_d(b));
                if (a.i % b.i == 0) return Value::Integer(a.i / b.i);
                return Value::Float((double)a.i / (double)b.i);
            }
            if (v.s == "///2") {
                if (b.i == 0) { ok = false; return Value{}; }
                return Value::Integer(a.i / b.i);
            }
            if (v.s == "mod/2") {
                if (b.i == 0) { ok = false; return Value{}; }
                return Value::Integer(a.i % b.i);
            }
            ok = false; return Value{};
        }
        default:
            ok = false; return Value{};
    }
}

bool WamState::arith_compare(const std::string& op, const Value& a, const Value& b) const {
    auto as_d = [](const Value& v){ return v.tag == Value::Tag::Float ? v.f : (double)v.i; };
    double x = as_d(a), y = as_d(b);
    if (op == ">/2")  return x >  y;
    if (op == "</2")  return x <  y;
    if (op == ">=/2") return x >= y;
    if (op == "=</2") return x <= y;
    if (op == "=:=/2") return x == y;
    if (op == "=\\\\=/2") return x != y;
    return false;
}

// ----------------------------------------------------------------------
// Term rendering for write/1
// ----------------------------------------------------------------------

std::string WamState::render(const Value& v) const {
    switch (v.tag) {
        case Value::Tag::Atom:    return v.s;
        case Value::Tag::Integer: return std::to_string(v.i);
        case Value::Tag::Float: {
            std::ostringstream os; os << v.f; return os.str();
        }
        case Value::Tag::Unbound: return v.s.empty() ? std::string("_") : v.s;
        case Value::Tag::Compound: {
            // Pretty-print lists as [a, b, c] when the spine is well-formed.
            if (v.s == "[|]/2" && v.args.size() == 2) {
                std::string out = "[";
                const Value* cur = &v;
                bool first = true;
                while (cur && cur->tag == Value::Tag::Compound
                       && cur->s == "[|]/2" && cur->args.size() == 2) {
                    if (!first) out += ", ";
                    first = false;
                    out += render(*cur->args[0]);
                    cur = cur->args[1].get();
                }
                if (cur && !(cur->tag == Value::Tag::Atom && cur->s == "[]")) {
                    out += " | "; out += render(*cur);
                }
                out += "]";
                return out;
            }
            // functor(arg1, arg2, ...). Strip "/N" suffix from functor.
            std::string name = v.s;
            auto slash = name.find_last_of(\'/\');
            if (slash != std::string::npos) name.resize(slash);
            std::string out = name + "(";
            for (std::size_t k = 0; k < v.args.size(); ++k) {
                if (k) out += ", ";
                out += render(*v.args[k]);
            }
            out += ")";
            return out;
        }
        default: return "_";
    }
}

// ----------------------------------------------------------------------
// Builtin dispatch
// ----------------------------------------------------------------------

bool WamState::builtin(const std::string& op, std::int64_t /*arity*/) {
    // ---- Control ----------------------------------------------------
    if (op == "true/0") { pc += 1; return true; }
    if (op == "fail/0") { return false; }
    if (op == "!/0")    {
        if (choice_points.size() > cut_barrier) choice_points.resize(cut_barrier);
        pc += 1; return true;
    }

    // ---- =/2 (unify) -----------------------------------------------
    if (op == "=/2") {
        if (!unify_cells(get_cell("A1"), get_cell("A2"))) return false;
        pc += 1; return true;
    }
    // ---- \\=/2 (cannot unify) --------------------------------------
    if (op == "\\\\=/2") {
        // Try to unify; if it succeeds we have to undo and fail.
        std::size_t mark = trail.size();
        bool ok = unify_cells(get_cell("A1"), get_cell("A2"));
        // Roll back any bindings we just made.
        while (trail.size() > mark) {
            TrailEntry t = std::move(trail.back());
            trail.pop_back();
            *t.cell = std::move(t.prev);
        }
        if (ok) return false;
        pc += 1; return true;
    }

    // ---- is/2 -------------------------------------------------------
    if (op == "is/2") {
        bool ok = true;
        Value rhs = eval_arith(get_cell("A2"), ok);
        if (!ok) return false;
        CellPtr lhs = get_cell("A1");
        if (lhs->is_unbound()) { bind_cell(lhs, std::move(rhs)); pc += 1; return true; }
        if (!(*lhs == rhs)) return false;
        pc += 1; return true;
    }

    // ---- Arithmetic comparisons ------------------------------------
    if (op == ">/2" || op == "</2" || op == ">=/2" || op == "=</2"
        || op == "=:=/2" || op == "=\\\\=/2") {
        bool ok = true;
        Value a = eval_arith(get_cell("A1"), ok);
        if (!ok) return false;
        Value b = eval_arith(get_cell("A2"), ok);
        if (!ok) return false;
        if (!arith_compare(op, a, b)) return false;
        pc += 1; return true;
    }

    // ---- ==/2 and \\==/2 (structural equality) ----------------------
    if (op == "==/2" || op == "\\\\==/2") {
        Value a = *get_cell("A1");
        Value b = *get_cell("A2");
        bool eq = (a == b);
        if (op == "==/2"  && !eq) return false;
        if (op == "\\\\==/2" && eq) return false;
        pc += 1; return true;
    }

    // ---- Type checks -----------------------------------------------
    if (op == "atom/1") {
        Value a = *get_cell("A1");
        if (a.tag != Value::Tag::Atom) return false;
        pc += 1; return true;
    }
    if (op == "integer/1") {
        Value a = *get_cell("A1");
        if (a.tag != Value::Tag::Integer) return false;
        pc += 1; return true;
    }
    if (op == "float/1") {
        Value a = *get_cell("A1");
        if (a.tag != Value::Tag::Float) return false;
        pc += 1; return true;
    }
    if (op == "number/1") {
        Value a = *get_cell("A1");
        if (a.tag != Value::Tag::Integer && a.tag != Value::Tag::Float) return false;
        pc += 1; return true;
    }
    if (op == "atomic/1") {
        Value a = *get_cell("A1");
        if (a.tag == Value::Tag::Compound || a.is_unbound()) return false;
        pc += 1; return true;
    }
    if (op == "compound/1") {
        Value a = *get_cell("A1");
        if (a.tag != Value::Tag::Compound) return false;
        pc += 1; return true;
    }
    if (op == "var/1") {
        if (!get_cell("A1")->is_unbound()) return false;
        pc += 1; return true;
    }
    if (op == "nonvar/1") {
        if (get_cell("A1")->is_unbound()) return false;
        pc += 1; return true;
    }
    if (op == "ground/1") {
        // Recursive groundness check.
        std::function<bool(const Value&)> g = [&](const Value& v) -> bool {
            if (v.is_unbound()) return false;
            if (v.tag == Value::Tag::Compound) {
                for (auto& c : v.args) if (!g(*c)) return false;
            }
            return true;
        };
        if (!g(*get_cell("A1"))) return false;
        pc += 1; return true;
    }

    // ---- functor/3 -------------------------------------------------
    if (op == "functor/3") {
        CellPtr t = get_cell("A1");
        if (!t->is_unbound()) {
            // Decompose: t -> functor name + arity
            const Value& v = *t;
            std::string name;
            std::int64_t arity = 0;
            if (v.tag == Value::Tag::Compound) {
                name = v.s;
                auto sl = name.find_last_of(\'/\');
                if (sl != std::string::npos) name.resize(sl);
                arity = (std::int64_t)v.args.size();
            } else if (v.tag == Value::Tag::Atom) {
                name = v.s; arity = 0;
            } else if (v.tag == Value::Tag::Integer) {
                // Numbers are their own functor with arity 0.
                CellPtr fc = get_cell("A2");
                CellPtr ac = get_cell("A3");
                if (fc->is_unbound()) bind_cell(fc, v);
                else if (!(*fc == v)) return false;
                Value zero = Value::Integer(0);
                if (ac->is_unbound()) bind_cell(ac, zero);
                else if (!(*ac == zero)) return false;
                pc += 1; return true;
            } else {
                return false;
            }
            CellPtr fc = get_cell("A2");
            CellPtr ac = get_cell("A3");
            Value fv = Value::Atom(name);
            Value av = Value::Integer(arity);
            if (fc->is_unbound()) bind_cell(fc, fv); else if (!(*fc == fv)) return false;
            if (ac->is_unbound()) bind_cell(ac, av); else if (!(*ac == av)) return false;
            pc += 1; return true;
        }
        // Build mode: A2 = functor name, A3 = arity.
        Value name_v = *get_cell("A2");
        Value ar_v   = *get_cell("A3");
        if (name_v.tag != Value::Tag::Atom || ar_v.tag != Value::Tag::Integer) return false;
        if (ar_v.i == 0) { bind_cell(t, name_v); pc += 1; return true; }
        std::vector<CellPtr> args;
        for (std::int64_t k = 0; k < ar_v.i; ++k) {
            args.push_back(std::make_shared<Value>(Value::Unbound("_FA" + std::to_string(k))));
        }
        std::string functor = name_v.s + "/" + std::to_string(ar_v.i);
        bind_cell(t, Value::Compound(functor, std::move(args)));
        pc += 1; return true;
    }

    // ---- arg/3 -----------------------------------------------------
    if (op == "arg/3") {
        Value n_v = *get_cell("A1");
        Value t_v = *get_cell("A2");
        if (n_v.tag != Value::Tag::Integer) return false;
        if (t_v.tag != Value::Tag::Compound) return false;
        std::int64_t idx = n_v.i;
        if (idx < 1 || (std::size_t)idx > t_v.args.size()) return false;
        CellPtr want = t_v.args[idx - 1];
        if (!unify_cells(get_cell("A3"), want)) return false;
        pc += 1; return true;
    }

    // ---- =../2 (univ) ----------------------------------------------
    if (op == "=../2") {
        CellPtr t = get_cell("A1");
        CellPtr l = get_cell("A2");
        if (!t->is_unbound()) {
            // Decompose: t -> [Functor | Args]
            const Value& v = *t;
            std::vector<CellPtr> items;
            if (v.tag == Value::Tag::Compound) {
                std::string name = v.s;
                auto sl = name.find_last_of(\'/\');
                if (sl != std::string::npos) name.resize(sl);
                items.push_back(std::make_shared<Value>(Value::Atom(name)));
                for (auto& a : v.args) items.push_back(a);
            } else {
                // Atomic term: univ -> [v]
                items.push_back(std::make_shared<Value>(v));
            }
            // Build [|]/2 chain
            Value tail = Value::Atom("[]");
            for (auto it = items.rbegin(); it != items.rend(); ++it) {
                std::vector<CellPtr> args;
                args.push_back(*it);
                args.push_back(std::make_shared<Value>(std::move(tail)));
                tail = Value::Compound("[|]/2", std::move(args));
            }
            CellPtr built = std::make_shared<Value>(std::move(tail));
            if (!unify_cells(l, built)) return false;
            pc += 1; return true;
        }
        // Build mode: collect list elements from A2.
        std::vector<Value> items;
        const Value* cur = l.get();
        while (cur && cur->tag == Value::Tag::Compound
               && cur->s == "[|]/2" && cur->args.size() == 2) {
            items.push_back(*cur->args[0]);
            cur = cur->args[1].get();
        }
        if (!(cur && cur->tag == Value::Tag::Atom && cur->s == "[]")) return false;
        if (items.empty()) return false;
        if (items.size() == 1) {
            bind_cell(t, items[0]);
            pc += 1; return true;
        }
        if (items[0].tag != Value::Tag::Atom) return false;
        std::vector<CellPtr> args;
        for (std::size_t k = 1; k < items.size(); ++k) {
            args.push_back(std::make_shared<Value>(items[k]));
        }
        std::string functor = items[0].s + "/" + std::to_string(items.size() - 1);
        bind_cell(t, Value::Compound(functor, std::move(args)));
        pc += 1; return true;
    }

    // ---- I/O -------------------------------------------------------
    if (op == "write/1") {
        std::printf("%s", render(*get_cell("A1")).c_str());
        std::fflush(stdout);
        pc += 1; return true;
    }
    if (op == "nl/0") {
        std::printf("\\n");
        std::fflush(stdout);
        pc += 1; return true;
    }
    if (op == "write_atom/1" || op == "writeln/1") {
        std::printf("%s\\n", render(*get_cell("A1")).c_str());
        std::fflush(stdout);
        pc += 1; return true;
    }

    return false;
}

// ----------------------------------------------------------------------
// Mode helpers for read/write of Get-/Put-Structure / Get-/Put-List.
// ----------------------------------------------------------------------

namespace {

void push_read_mode(std::vector<ModeFrame>& stack, const std::vector<CellPtr>& args) {
    ModeFrame m;
    m.kind = ModeFrame::Kind::Read;
    m.args = args;
    m.idx = 0;
    m.expected_arity = args.size();
    stack.push_back(std::move(m));
}

void push_write_mode(std::vector<ModeFrame>& stack, CellPtr target, std::size_t arity) {
    if (target) target->args.clear();
    ModeFrame m;
    m.kind = ModeFrame::Kind::Write;
    m.target = std::move(target);
    m.idx = 0;
    m.expected_arity = arity;
    stack.push_back(std::move(m));
}

// Auto-pop frames whose arity has been satisfied. Cascades upward so a
// fully-filled inner struct doesn''t leave subsequent Set*/Unify* writing
// into it.
void auto_pop_modes(std::vector<ModeFrame>& stack) {
    while (!stack.empty()) {
        ModeFrame& top = stack.back();
        bool full = false;
        if (top.kind == ModeFrame::Kind::Write && top.target
            && top.target->args.size() >= top.expected_arity) full = true;
        if (top.kind == ModeFrame::Kind::Read
            && top.idx >= top.expected_arity) full = true;
        if (!full) break;
        stack.pop_back();
    }
}

// Recursive deep-copy of a Value. Atom / Integer / Float / Unbound are
// independent — Compound rebuilds its args vector with fresh cells whose
// contents are themselves deep-copied, so later mutations to the source
// tree don''t leak into a collected snapshot (used by EndAggregate).
Value deep_copy(const Value& v) {
    Value out = v;
    if (v.tag == Value::Tag::Compound) {
        out.args.clear();
        for (auto& c : v.args) {
            out.args.push_back(std::make_shared<Cell>(deep_copy(*c)));
        }
    }
    return out;
}

} // anonymous

// Begin a write-mode structure / list. When the register''s existing
// cell is Unbound, we MUTATE it via bind_cell — so chained patterns
// like set_variable + put_structure that share a cell with a parent
// struct see the new compound. When the cell already holds a concrete
// value (atom/integer/compound), we instead allocate a FRESH cell and
// rebind only the named register (classic WAM "put_structure overwrites
// Ai" semantics) — other aliases keep their old value.
static void begin_write(WamState& vm, const std::string& reg_name,
                        const std::string& functor, std::size_t arity,
                        CellPtr& chosen) {
    CellPtr existing = vm.get_cell(reg_name);
    if (existing->is_unbound()) {
        vm.bind_cell(existing, Value::Compound(functor, {}));
        chosen = existing;
    } else {
        chosen = std::make_shared<Cell>(Value::Compound(functor, {}));
        vm.set_cell(reg_name, chosen);
    }
    push_write_mode(vm.mode_stack, chosen, arity);
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
                CellPtr target;
                begin_write(*this, instr.b, functor, arity, target);
                pc += 1; return true;
            }
            if (a->tag == Value::Tag::Compound && a->s == functor && a->args.size() == arity) {
                push_read_mode(mode_stack, a->args);
                pc += 1; return true;
            }
            return false;
        }
        case Instruction::Op::GetList: {
            CellPtr a = get_cell(instr.a);
            const std::string functor = "[|]/2";
            if (a->is_unbound()) {
                CellPtr target;
                begin_write(*this, instr.a, functor, 2, target);
                pc += 1; return true;
            }
            if (a->tag == Value::Tag::Compound && a->s == functor && a->args.size() == 2) {
                push_read_mode(mode_stack, a->args);
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
            CellPtr target;
            begin_write(*this, instr.b, functor, arity, target);
            pc += 1; return true;
        }
        case Instruction::Op::PutList: {
            CellPtr target;
            begin_write(*this, instr.a, "[|]/2", 2, target);
            pc += 1; return true;
        }

        // ---- Unify (post Get-/Put-Structure / List) -----------------
        case Instruction::Op::UnifyVariable: {
            if (mode_stack.empty()) return false;
            ModeFrame& m = mode_stack.back();
            if (m.kind == ModeFrame::Kind::Read) {
                if (m.idx >= m.args.size()) return false;
                set_cell(instr.a, m.args[m.idx]);
                m.idx += 1;
            } else if (m.kind == ModeFrame::Kind::Write && m.target) {
                CellPtr fresh = make_cell(Value::Unbound("_V" + std::to_string(var_counter++)));
                set_cell(instr.a, fresh);
                m.target->args.push_back(fresh);
            } else return false;
            auto_pop_modes(mode_stack);
            pc += 1; return true;
        }
        case Instruction::Op::UnifyValue: {
            if (mode_stack.empty()) return false;
            ModeFrame& m = mode_stack.back();
            if (m.kind == ModeFrame::Kind::Read) {
                if (m.idx >= m.args.size()) return false;
                if (!unify_cells(get_cell(instr.a), m.args[m.idx])) return false;
                m.idx += 1;
            } else if (m.kind == ModeFrame::Kind::Write && m.target) {
                m.target->args.push_back(get_cell(instr.a));
            } else return false;
            auto_pop_modes(mode_stack);
            pc += 1; return true;
        }
        case Instruction::Op::UnifyConstant: {
            if (mode_stack.empty()) return false;
            ModeFrame& m = mode_stack.back();
            if (m.kind == ModeFrame::Kind::Read) {
                if (m.idx >= m.args.size()) return false;
                CellPtr c = m.args[m.idx];
                if (c->is_unbound())         { bind_cell(c, instr.val); }
                else if (!(*c == instr.val)) { return false; }
                m.idx += 1;
            } else if (m.kind == ModeFrame::Kind::Write && m.target) {
                m.target->args.push_back(make_cell(instr.val));
            } else return false;
            auto_pop_modes(mode_stack);
            pc += 1; return true;
        }

        // ---- Set (always write mode) -------------------------------
        case Instruction::Op::SetVariable: {
            if (mode_stack.empty() || mode_stack.back().kind != ModeFrame::Kind::Write
                || !mode_stack.back().target) return false;
            CellPtr fresh = make_cell(Value::Unbound("_V" + std::to_string(var_counter++)));
            set_cell(instr.a, fresh);
            mode_stack.back().target->args.push_back(fresh);
            auto_pop_modes(mode_stack);
            pc += 1; return true;
        }
        case Instruction::Op::SetValue: {
            if (mode_stack.empty() || mode_stack.back().kind != ModeFrame::Kind::Write
                || !mode_stack.back().target) return false;
            mode_stack.back().target->args.push_back(get_cell(instr.a));
            auto_pop_modes(mode_stack);
            pc += 1; return true;
        }
        case Instruction::Op::SetConstant: {
            if (mode_stack.empty() || mode_stack.back().kind != ModeFrame::Kind::Write
                || !mode_stack.back().target) return false;
            mode_stack.back().target->args.push_back(make_cell(instr.val));
            auto_pop_modes(mode_stack);
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
            cp_.saved_mode_stack = mode_stack;
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

        // ---- Aggregate / findall driver ----------------------------
        case Instruction::Op::BeginAggregate: {
            AggregateFrame frame;
            frame.agg_kind          = instr.a;
            frame.value_reg         = instr.b;
            frame.result_reg        = instr.val.s; // stuffed into val.s by factory
            frame.begin_pc          = pc;
            frame.return_pc         = 0;
            frame.return_pc_set     = false;
            frame.base_cp_count     = choice_points.size();
            frame.trail_mark        = trail.size();
            frame.saved_cp          = cp;
            frame.saved_cut_barrier = cut_barrier;
            frame.saved_regs        = regs;
            frame.saved_mode_stack  = mode_stack;
            aggregate_frames.push_back(std::move(frame));
            pc += 1;
            return true;
        }
        case Instruction::Op::EndAggregate: {
            if (aggregate_frames.empty()) return false;
            AggregateFrame& f = aggregate_frames.back();
            // Snapshot the current value of value_reg (deep copy so later
            // trail-driven mutations don''t alter what we collected).
            CellPtr v = get_cell(instr.a);
            f.acc.push_back(deep_copy(*v));
            f.return_pc     = pc + 1;
            f.return_pc_set = true;
            // Force backtrack to find the next solution of the body.
            return false;
        }
    }
    return false;
}

// (deep_copy is defined above, before step().)

// Build the finalisation result for an aggregate based on its kind.
// Falls back to the raw list when the kind isn''t recognised.
static Value finalize_aggregate(const std::string& kind, const std::vector<Value>& acc) {
    auto as_d = [](const Value& v) { return v.tag == Value::Tag::Float ? v.f : (double)v.i; };
    auto is_num = [](const Value& v) {
        return v.tag == Value::Tag::Integer || v.tag == Value::Tag::Float;
    };
    if (kind == "count") {
        return Value::Integer((std::int64_t)acc.size());
    }
    if (kind == "sum") {
        bool has_float = false;
        double sum_d = 0.0; std::int64_t sum_i = 0;
        for (auto& v : acc) {
            if (!is_num(v)) return Value{};
            if (v.tag == Value::Tag::Float) { has_float = true; sum_d += v.f; }
            else                            { sum_i += v.i; sum_d += (double)v.i; }
        }
        if (has_float) return Value::Float(sum_d);
        return Value::Integer(sum_i);
    }
    if (kind == "min" || kind == "max") {
        if (acc.empty()) return Value::Atom("[]");
        const Value* best = &acc[0];
        for (std::size_t k = 1; k < acc.size(); ++k) {
            if (!is_num(acc[k])) return Value{};
            if (kind == "min" ? (as_d(acc[k]) < as_d(*best))
                              : (as_d(acc[k]) > as_d(*best))) {
                best = &acc[k];
            }
        }
        return *best;
    }
    // "collect" (findall), bagof/setof, and aggregate_all bag/set all
    // build a list. bagof/setof fail on empty; findall and
    // aggregate_all(bag/set) keep the historical empty-list behaviour.
    if ((kind == "bagof" || kind == "setof") && acc.empty()) {
        return Value{};
    }
    std::vector<Value> items = acc;
    if (kind == "set" || kind == "setof") {
        std::vector<Value> uniq;
        for (auto& v : items) {
            bool dup = false;
            for (auto& w : uniq) { if (v == w) { dup = true; break; } }
            if (!dup) uniq.push_back(v);
        }
        items = std::move(uniq);
        std::sort(items.begin(), items.end(), [](const Value& a, const Value& b) {
            if (a.tag != b.tag) return static_cast<int>(a.tag) < static_cast<int>(b.tag);
            switch (a.tag) {
                case Value::Tag::Atom:
                case Value::Tag::Unbound:
                case Value::Tag::Compound:
                    return a.s < b.s;
                case Value::Tag::Integer:
                    return a.i < b.i;
                case Value::Tag::Float:
                    return a.f < b.f;
                default:
                    return false;
            }
        });
    }
    Value tail = Value::Atom("[]");
    for (auto it = items.rbegin(); it != items.rend(); ++it) {
        std::vector<CellPtr> args;
        args.push_back(std::make_shared<Cell>(*it));
        args.push_back(std::make_shared<Cell>(std::move(tail)));
        tail = Value::Compound("[|]/2", std::move(args));
    }
    return tail;
}

// Walk forward from begin_pc to find the matching EndAggregate, counting
// nested Begin/End pairs. Used to compute pc continuation when the body
// of an aggregate fails before any EndAggregate fires.
static std::size_t find_matching_end_aggregate(
        const std::vector<Instruction>& instrs, std::size_t begin_pc) {
    int depth = 1;
    for (std::size_t k = begin_pc + 1; k < instrs.size(); ++k) {
        if (instrs[k].op == Instruction::Op::BeginAggregate) ++depth;
        else if (instrs[k].op == Instruction::Op::EndAggregate) {
            if (--depth == 0) return k;
        }
    }
    return instrs.size();
}

bool WamState::backtrack() {
    // Pop normal choice points until we either find one to retry or run
    // into an open aggregate frame''s base — at which point the frame is
    // finalised and execution continues past its EndAggregate.
    for (;;) {
        std::size_t agg_base =
            aggregate_frames.empty() ? 0 : aggregate_frames.back().base_cp_count;
        if (!aggregate_frames.empty() && choice_points.size() == agg_base) {
            // Aggregate exhausted. Finalise.
            AggregateFrame f = std::move(aggregate_frames.back());
            aggregate_frames.pop_back();
            // Unwind any trail entries added inside the aggregate scope.
            while (trail.size() > f.trail_mark) {
                TrailEntry t = std::move(trail.back());
                trail.pop_back();
                *t.cell = std::move(t.prev);
            }
            regs        = std::move(f.saved_regs);
            mode_stack  = std::move(f.saved_mode_stack);
            cp          = f.saved_cp;
            cut_barrier = f.saved_cut_barrier;
            // Build and bind the result.
            Value result = finalize_aggregate(f.agg_kind, f.acc);
            if (result.tag == Value::Tag::Uninit) return false;
            CellPtr rcell = get_cell(f.result_reg);
            if (rcell->is_unbound()) bind_cell(rcell, result);
            else if (!unify_cells(rcell, std::make_shared<Cell>(result))) return false;
            // Jump past the matching EndAggregate.
            if (f.return_pc_set) {
                pc = f.return_pc;
            } else {
                pc = find_matching_end_aggregate(instrs, f.begin_pc) + 1;
            }
            return true;
        }
        if (choice_points.empty()) return false;
        // Classic WAM: do NOT pop the CP here. retry_me_else updates the
        // CP''s alt_pc to the next clause as it runs; trust_me pops it.
        // We restore state via COPY so the snapshot remains intact for
        // any subsequent backtracks into this same CP.
        const ChoicePoint& cp_ = choice_points.back();
        while (trail.size() > cp_.trail_mark) {
            TrailEntry t = std::move(trail.back());
            trail.pop_back();
            *t.cell = std::move(t.prev);
        }
        regs        = cp_.saved_regs;
        mode_stack  = cp_.saved_mode_stack;
        cp          = cp_.saved_cp;
        cut_barrier = cp_.cut_barrier;
        pc          = cp_.alt_pc;
        return true;
    }
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
    aggregate_frames.clear();
    mode_stack.clear();
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
