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
        catch(
            ( use_module(wam_cpp_lowered_emitter,
                         [wam_cpp_lowerable/3, lower_predicate_to_cpp/4]),
              wam_cpp_lowerable(Pred/Arity, WamCode, _Reason)
            ),
            _,
            fail)
    ->  lower_predicate_to_cpp(Pred/Arity, WamCode, Options, Lines),
        atomic_list_concat(Lines, '\n', CppCode)
    ;   instrs_for(WamCode, Instrs),
        compile_predicate_wrapper(Pred, Arity, Instrs, Options, CppCode)
    ).

% Raw-text WAM is parsed by the lowered emitter when lowering. For the
% interpreter-wrapper path we expect compile_predicate_to_wam/3 to deliver
% an instruction list, so a non-list input here is treated as empty.
instrs_for(WamCode, Instrs) :-
    is_list(WamCode), !, Instrs = WamCode.
instrs_for(_, []).

compile_predicate_wrapper(Pred, Arity, Instrs, _Options, CppCode) :-
    cpp_safe_function_name(Pred, SafeName),
    findall(Lit,
            ( member(I, Instrs),
              wam_instruction_to_cpp_literal(I, Lit)
            ),
            Literals),
    atomic_list_concat(Literals, ',\n        ', Body),
    format(atom(Key), '~w/~w', [Pred, Arity]),
    format(string(CppCode),
'// Interpreter wrapper for ~w
const std::vector<Instruction>& ~w_instrs() {
    static const std::vector<Instruction> instrs = {
        ~w
    };
    return instrs;
}
static const int _~w_register = []() {
    Program::register_predicate("~w", &~w_instrs);
    return 0;
}();
', [Key, SafeName, Body, SafeName, Key, SafeName]).

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
    format(string(ProgramCode),
'// SPDX-License-Identifier: MIT OR Apache-2.0
// Auto-generated by wam_cpp_target.pl. Do not edit by hand.
#include "wam_runtime.h"

~w
', [PredicatesCode]),
    directory_file_path(CppDir, 'generated_program.cpp', ProgramPath),
    write_text_file(ProgramPath, ProgramCode).

compile_predicates_for_project(Predicates, Options, PredicatesCode) :-
    foreign_pred_keys_from_options(Options, ForeignKeys),
    merge_options([foreign_pred_keys(ForeignKeys)], Options, Options1),
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

merge_options([], Acc, Acc).
merge_options([Opt|Rest], Acc, Out) :-
    (   member(Opt, Acc)
    ->  merge_options(Rest, Acc, Out)
    ;   merge_options(Rest, [Opt|Acc], Out)
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
// Minimal WAM runtime header. Provides the Value, Instruction and
// WamState types referenced by code emitted from
// wam_cpp_lowered_emitter.pl and wam_cpp_target.pl.

#ifndef UNIFYWEAVER_WAM_CPP_RUNTIME_H
#define UNIFYWEAVER_WAM_CPP_RUNTIME_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace wam_cpp {

struct Value {
    enum class Tag { Uninit, Atom, Integer, Float, Unbound, List, Ref };
    Tag tag = Tag::Uninit;
    std::string s;
    std::int64_t i = 0;
    double f = 0.0;

    Value() = default;
    static Value Atom(std::string a)    { Value v; v.tag = Tag::Atom;    v.s = std::move(a); return v; }
    static Value Integer(std::int64_t n){ Value v; v.tag = Tag::Integer; v.i = n;             return v; }
    static Value Float(double d)        { Value v; v.tag = Tag::Float;   v.f = d;             return v; }
    static Value Unbound(std::string n) { Value v; v.tag = Tag::Unbound; v.s = std::move(n);  return v; }

    bool is_unbound() const { return tag == Tag::Uninit || tag == Tag::Unbound; }

    bool operator==(const Value& o) const {
        if (tag != o.tag) return false;
        switch (tag) {
            case Tag::Atom:    return s == o.s;
            case Tag::Integer: return i == o.i;
            case Tag::Float:   return f == o.f;
            case Tag::Unbound: return s == o.s;
            default:           return true;
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
    std::string a;   // register / functor / pred name
    std::string b;   // optional second register
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

struct WamState {
    std::unordered_map<std::string, Value> regs;
    std::unordered_map<std::string, std::size_t> labels;
    std::vector<Instruction> instrs;
    std::vector<std::string> trail;
    std::size_t pc = 0;
    std::size_t cp = 0;
    std::uint64_t var_counter = 0;
    bool halt = false;

    Value get_reg(const std::string& name) const;
    void  put_reg(const std::string& name, Value v);
    void  trail_binding(const std::string& name);
    bool  unify(const Value& a, const Value& b);
    bool  step(const Instruction& instr);
    bool  run();
};

struct Program {
    using Provider = const std::vector<Instruction>& (*)();
    static std::unordered_map<std::string, Provider>& registry();
    static void register_predicate(const std::string& key, Provider p);
};

} // namespace wam_cpp

// Convenience aliases used by emitted lowered code (which references
// the unqualified names Value / Instruction / WamState).
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
// Minimal WAM runtime implementation. Mirrors the subset of behaviour
// needed by the lowered emitter; full builtin and choicepoint support
// is intentionally out of scope for this initial hybrid target and can
// be expanded in follow-up patches (see wam_rust_target.pl for the
// fully-featured sibling).

#include "wam_runtime.h"

namespace wam_cpp {

Value WamState::get_reg(const std::string& name) const {
    auto it = regs.find(name);
    if (it == regs.end()) return Value{};
    return it->second;
}

void WamState::put_reg(const std::string& name, Value v) {
    regs[name] = std::move(v);
}

void WamState::trail_binding(const std::string& name) {
    trail.push_back(name);
}

bool WamState::unify(const Value& a, const Value& b) {
    if (a.is_unbound() || b.is_unbound()) return true;
    return a == b;
}

bool WamState::step(const Instruction& instr) {
    // The lowered emitter inlines most ops; step() is a thin fallback
    // for instructions that cannot be inlined safely. A richer
    // implementation lives in the per-language runtime templates used
    // by wam_rust_target / wam_lua_target.
    switch (instr.op) {
        case Instruction::Op::Allocate:
        case Instruction::Op::Deallocate:
        case Instruction::Op::Proceed:
            pc += 1;
            return true;
        case Instruction::Op::Fail:
            return false;
        default:
            pc += 1;
            return true;
    }
}

bool WamState::run() {
    while (pc < instrs.size() && !halt) {
        if (!step(instrs[pc])) return false;
    }
    return true;
}

std::unordered_map<std::string, Program::Provider>& Program::registry() {
    static std::unordered_map<std::string, Provider> r;
    return r;
}

void Program::register_predicate(const std::string& key, Provider p) {
    registry()[key] = p;
}

} // namespace wam_cpp
').
