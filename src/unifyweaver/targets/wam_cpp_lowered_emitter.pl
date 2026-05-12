:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% wam_cpp_lowered_emitter.pl — WAM-lowered C++ emission
%
% Emits one C++ function per deterministic predicate (or per clause-1 of a
% multi-clause predicate). Simple register operations are inlined as direct
% C++ statements; complex instructions delegate to vm methods declared in
% wam_runtime.h.
%
% For multi-clause predicates (try_me_else), only clause 1 is lowered.
% Clause 2+ stays in the interpreter's instruction array for backtrack,
% mirroring the wam_rust_lowered_emitter / wam_haskell_lowered_emitter design.
%
% Modelled on wam_rust_lowered_emitter.pl (the closest systems-language
% sibling) and the hybrid pattern shared with wam_haskell, wam_lua, wam_r,
% wam_go, wam_clojure, wam_scala, wam_fsharp, wam_elixir.

:- module(wam_cpp_lowered_emitter, [
    wam_cpp_lowerable/3,
    lower_predicate_to_cpp/4,
    is_deterministic_pred_cpp/1,
    cpp_lowered_func_name/2,
    parse_wam_text/2
]).

:- use_module(library(lists)).
% Inlined escape helper to avoid a circular import with wam_cpp_target.
% Keeps this module standalone-loadable.

% =====================================================================
% Parsing — accept either an instruction list or a WAM-text blob.
% Mirrors parse_wam_text/2 from wam_rust_lowered_emitter.pl exactly.
% =====================================================================

parse_wam_text(WamText, Instrs) :-
    atom_string(WamText, S),
    split_string(S, "\n", "", Lines),
    parse_lines(Lines, Instrs).

parse_lines([], []).
parse_lines([Line|Rest], Instrs) :-
    split_string(Line, " \t,", " \t,", Parts),
    delete(Parts, "", CleanParts),
    (   CleanParts == []
    ->  parse_lines(Rest, Instrs)
    ;   CleanParts = [First|_],
        (   sub_string(First, _, 1, 0, ":")
        ->  parse_lines(Rest, Instrs)
        ;   instr_from_parts(CleanParts, Instr)
        ->  Instrs = [Instr|RestInstrs],
            parse_lines(Rest, RestInstrs)
        ;   parse_lines(Rest, Instrs)
        )
    ).

instr_from_parts(["get_constant", C, Ai], get_constant(C, Ai)).
instr_from_parts(["get_variable", Xn, Ai], get_variable(Xn, Ai)).
instr_from_parts(["get_value", Xn, Ai], get_value(Xn, Ai)).
instr_from_parts(["get_structure", F, Ai], get_structure(F, Ai)).
instr_from_parts(["get_list", Ai], get_list(Ai)).
instr_from_parts(["get_nil", Ai], get_nil(Ai)).
instr_from_parts(["get_integer", N, Ai], get_integer(N, Ai)).
instr_from_parts(["unify_variable", Xn], unify_variable(Xn)).
instr_from_parts(["unify_value", Xn], unify_value(Xn)).
instr_from_parts(["unify_constant", C], unify_constant(C)).
instr_from_parts(["put_variable", Xn, Ai], put_variable(Xn, Ai)).
instr_from_parts(["put_value", Xn, Ai], put_value(Xn, Ai)).
instr_from_parts(["put_constant", C, Ai], put_constant(C, Ai)).
instr_from_parts(["put_structure", F, Ai], put_structure(F, Ai)).
instr_from_parts(["put_list", Ai], put_list(Ai)).
instr_from_parts(["set_variable", Xn], set_variable(Xn)).
instr_from_parts(["set_value", Xn], set_value(Xn)).
instr_from_parts(["set_constant", C], set_constant(C)).
instr_from_parts(["call", P, N], call(P, N)).
instr_from_parts(["execute", P], execute(P)).
instr_from_parts(["proceed"], proceed).
instr_from_parts(["fail"], fail).
instr_from_parts(["allocate"], allocate).
instr_from_parts(["deallocate"], deallocate).
instr_from_parts(["builtin_call", Op, Ar], builtin_call(Op, Ar)).
instr_from_parts(["call_foreign", Pred, Ar], call_foreign(Pred, Ar)).
instr_from_parts(["try_me_else", L], try_me_else(L)).
instr_from_parts(["retry_me_else", L], retry_me_else(L)).
instr_from_parts(["trust_me"], trust_me).
instr_from_parts(["jump", L], jump(L)).
instr_from_parts(["cut_ite"], cut_ite).

% =====================================================================
% Lowerability
% =====================================================================

%% wam_cpp_lowerable(+Pred/Arity, +WamCode, -Reason)
%  True if the predicate can be lowered to a direct C++ function.
%  Reason is `deterministic` or `multi_clause_1`.
wam_cpp_lowerable(PI, WamCode, Reason) :-
    (   is_list(WamCode) -> Instrs = WamCode
    ;   atom(WamCode) -> parse_wam_text(WamCode, Instrs)
    ;   atom_string(WamCode, _), parse_wam_text(WamCode, Instrs)
    ),
    clause1_instrs(Instrs, C1),
    forall(member(I, C1), cpp_supported(I)),
    (   is_deterministic_pred_cpp(Instrs)
    ->  Reason = deterministic
    ;   Reason = multi_clause_1
    ),
    ( PI = _M:_P/_A -> true ; PI = _/_A2 -> true ; true ).

clause1_instrs([], []).
clause1_instrs([try_me_else(_)|Rest], C1) :- !,
    take_to_proceed(Rest, C1).
clause1_instrs(Instrs, Instrs).

take_to_proceed([], []).
take_to_proceed([proceed|_], [proceed]) :- !.
take_to_proceed([I|Rest], [I|More]) :- take_to_proceed(Rest, More).

%% is_deterministic_pred_cpp(+Instrs)
%  True if the instruction list has no choice point instructions.
is_deterministic_pred_cpp(Instrs) :-
    \+ member(try_me_else(_), Instrs),
    \+ member(retry_me_else(_), Instrs),
    \+ member(trust_me, Instrs).

cpp_supported(allocate).
cpp_supported(deallocate).
cpp_supported(get_constant(_, _)).
cpp_supported(get_variable(_, _)).
cpp_supported(get_value(_, _)).
cpp_supported(get_structure(_, _)).
cpp_supported(get_list(_)).
cpp_supported(get_nil(_)).
cpp_supported(get_integer(_, _)).
cpp_supported(unify_variable(_)).
cpp_supported(unify_value(_)).
cpp_supported(unify_constant(_)).
cpp_supported(put_constant(_, _)).
cpp_supported(put_variable(_, _)).
cpp_supported(put_value(_, _)).
cpp_supported(put_structure(_, _)).
cpp_supported(put_list(_)).
cpp_supported(set_variable(_)).
cpp_supported(set_value(_)).
cpp_supported(set_constant(_)).
cpp_supported(call(_, _)).
cpp_supported(execute(_)).
cpp_supported(proceed).
cpp_supported(fail).
cpp_supported(builtin_call(_, _)).
cpp_supported(call_foreign(_, _)).
cpp_supported(try_me_else(_)).
cpp_supported(trust_me).
cpp_supported(cut_ite).
cpp_supported(jump(_)).

% =====================================================================
% Function name generation
% =====================================================================

%% cpp_lowered_func_name(+Functor/Arity, -CppFuncName)
%  foo/2 -> "lowered_foo_2", my_pred/3 -> "lowered_my_pred_3"
cpp_lowered_func_name(Functor/Arity, Name) :-
    atom_string(Functor, FStr),
    sanitize_cpp_ident(FStr, SanStr),
    format(atom(Name), 'lowered_~w_~w', [SanStr, Arity]).

sanitize_cpp_ident(In, Out) :-
    string_codes(In, Codes),
    maplist(cpp_safe_code, Codes, OutCodes),
    string_codes(OutStr, OutCodes),
    atom_string(Out, OutStr).

cpp_safe_code(C, C) :-
    (   C >= 0'a, C =< 0'z
    ;   C >= 0'A, C =< 0'Z
    ;   C >= 0'0, C =< 0'9
    ;   C =:= 0'_
    ),
    !.
cpp_safe_code(_, 0'_).

% =====================================================================
% Emission
% =====================================================================

%% lower_predicate_to_cpp(+Pred/Arity, +WamCode, +Options, -CppLines)
%  Emit a C++ function for the predicate. CppLines is [Header, Body, Footer].
lower_predicate_to_cpp(PI, WamCode, Options, CppLines) :-
    ( PI = _M:Pred/Arity -> true ; PI = Pred/Arity ),
    cpp_lowered_func_name(Pred/Arity, FuncName),
    (   is_list(WamCode) -> Instrs = WamCode
    ;   parse_wam_text(WamCode, Instrs)
    ),
    clause1_instrs(Instrs, C1Instrs),
    (   member(foreign_pred_keys(ForeignPreds0), Options)
    ->  maplist(foreign_key_string, ForeignPreds0, ForeignPreds)
    ;   ForeignPreds = []
    ),
    with_output_to(string(Body), emit_instrs(C1Instrs, "    ", ForeignPreds)),
    format(string(Header),
'// ~w — lowered from ~w/~w
bool ~w(WamState* vm) {', [FuncName, Pred, Arity, FuncName]),
    format(string(Footer), '}', []),
    CppLines = [Header, Body, Footer].

%% emit_instrs(+Instrs, +Indent, +ForeignPreds)
emit_instrs([], _, _).
emit_instrs([Instr|Rest], Ind, ForeignPreds) :-
    emit_one(Instr, Ind, ForeignPreds),
    emit_instrs(Rest, Ind, ForeignPreds).

% Foreign-routed call/execute take precedence over the generic clauses.
emit_one(call(PredStr, NStr), I, ForeignPreds) :-
    member(PredStr, ForeignPreds), !,
    format("~w// call ~w via foreign kernel~n", [I, PredStr]),
    format("~wif (!vm->step(Instruction::CallForeign(\"~w\", ~w))) return false;~n",
           [I, PredStr, NStr]).
emit_one(execute(PredStr), I, ForeignPreds) :-
    member(PredStr, ForeignPreds),
    sub_atom(PredStr, _, 1, ArityLen, '/'),
    sub_atom(PredStr, _, ArityLen, 0, ArityAtom),
    atom_number(ArityAtom, Arity), !,
    format("~w// execute ~w via foreign kernel~n", [I, PredStr]),
    format("~wreturn vm->step(Instruction::CallForeign(\"~w\", ~w));~n",
           [I, PredStr, Arity]).
emit_one(Instr, I, _) :-
    emit_one(Instr, I).

foreign_key_string(Key, String) :-
    (   string(Key)
    ->  String = Key
    ;   atom_string(Key, String)
    ).

% --- Terminal instructions ---

emit_one(proceed, I) :-
    format("~wreturn true;~n", [I]).

emit_one(fail, I) :-
    format("~wreturn false;~n", [I]).

% --- Head unification (get_*) ---

emit_one(get_constant(CStr, AiStr), I) :-
    cpp_reg_name(AiStr, Ai),
    cpp_val_literal(CStr, CppVal),
    format("~w// get_constant ~w, ~w~n", [I, CStr, AiStr]),
    format("~w{~n", [I]),
    format("~w    Value _a = vm->get_reg(\"~w\");~n", [I, Ai]),
    format("~w    if (_a.is_unbound()) {~n", [I]),
    format("~w        vm->trail_binding(\"~w\");~n", [I, Ai]),
    format("~w        vm->put_reg(\"~w\", ~w);~n", [I, Ai, CppVal]),
    format("~w    } else if (!(_a == ~w)) {~n", [I, CppVal]),
    format("~w        return false;~n", [I]),
    format("~w    }~n", [I]),
    format("~w}~n", [I]).

emit_one(get_integer(NStr, AiStr), I) :-
    cpp_reg_name(AiStr, Ai),
    format("~w// get_integer ~w, ~w~n", [I, NStr, AiStr]),
    format("~w{~n", [I]),
    format("~w    Value _a = vm->get_reg(\"~w\");~n", [I, Ai]),
    format("~w    if (_a.is_unbound()) {~n", [I]),
    format("~w        vm->trail_binding(\"~w\");~n", [I, Ai]),
    format("~w        vm->put_reg(\"~w\", Value::Integer(~w));~n", [I, Ai, NStr]),
    format("~w    } else if (!(_a == Value::Integer(~w))) {~n", [I, NStr]),
    format("~w        return false;~n", [I]),
    format("~w    }~n", [I]),
    format("~w}~n", [I]).

emit_one(get_nil(AiStr), I) :-
    cpp_reg_name(AiStr, Ai),
    format("~w// get_nil ~w~n", [I, AiStr]),
    format("~w{~n", [I]),
    format("~w    Value _a = vm->get_reg(\"~w\");~n", [I, Ai]),
    format("~w    if (_a.is_unbound()) {~n", [I]),
    format("~w        vm->trail_binding(\"~w\");~n", [I, Ai]),
    format("~w        vm->put_reg(\"~w\", Value::Atom(\"[]\"));~n", [I, Ai]),
    format("~w    } else if (!(_a == Value::Atom(\"[]\"))) {~n", [I]),
    format("~w        return false;~n", [I]),
    format("~w    }~n", [I]),
    format("~w}~n", [I]).

emit_one(get_variable(XnStr, AiStr), I) :-
    cpp_reg_name(XnStr, Xn), cpp_reg_name(AiStr, Ai),
    format("~w// get_variable ~w, ~w~n", [I, XnStr, AiStr]),
    format("~wvm->put_reg(\"~w\", vm->get_reg(\"~w\"));~n", [I, Xn, Ai]).

emit_one(get_value(XnStr, AiStr), I) :-
    cpp_reg_name(XnStr, Xn), cpp_reg_name(AiStr, Ai),
    format("~w// get_value ~w, ~w~n", [I, XnStr, AiStr]),
    format("~w{~n", [I]),
    format("~w    Value va = vm->get_reg(\"~w\");~n", [I, Ai]),
    format("~w    Value vx = vm->get_reg(\"~w\");~n", [I, Xn]),
    format("~w    if (!vm->unify(va, vx)) return false;~n", [I]),
    format("~w}~n", [I]).

emit_one(get_structure(FStr, AiStr), I) :-
    cpp_reg_name(AiStr, Ai),
    format("~w// get_structure ~w, ~w (delegate to step)~n", [I, FStr, AiStr]),
    format("~wif (!vm->step(Instruction::GetStructure(\"~w\", \"~w\"))) return false;~n",
           [I, FStr, Ai]).

emit_one(get_list(AiStr), I) :-
    cpp_reg_name(AiStr, Ai),
    format("~w// get_list ~w (delegate to step)~n", [I, AiStr]),
    format("~wif (!vm->step(Instruction::GetList(\"~w\"))) return false;~n", [I, Ai]).

% --- Body construction (put_*) ---

emit_one(put_constant(CStr, AiStr), I) :-
    cpp_reg_name(AiStr, Ai),
    cpp_val_literal(CStr, CppVal),
    format("~w// put_constant ~w, ~w~n", [I, CStr, AiStr]),
    format("~wvm->put_reg(\"~w\", ~w);~n", [I, Ai, CppVal]).

emit_one(put_variable(XnStr, AiStr), I) :-
    cpp_reg_name(XnStr, Xn), cpp_reg_name(AiStr, Ai),
    format("~w// put_variable ~w, ~w~n", [I, XnStr, AiStr]),
    format("~w{~n", [I]),
    format("~w    Value v = Value::Unbound(\"_V\" + std::to_string(vm->var_counter++));~n",
           [I]),
    format("~w    vm->put_reg(\"~w\", v);~n", [I, Xn]),
    format("~w    vm->put_reg(\"~w\", v);~n", [I, Ai]),
    format("~w}~n", [I]).

emit_one(put_value(XnStr, AiStr), I) :-
    cpp_reg_name(XnStr, Xn), cpp_reg_name(AiStr, Ai),
    format("~w// put_value ~w, ~w~n", [I, XnStr, AiStr]),
    format("~wvm->put_reg(\"~w\", vm->get_reg(\"~w\"));~n", [I, Ai, Xn]).

emit_one(put_structure(FStr, AiStr), I) :-
    cpp_reg_name(AiStr, Ai),
    format("~w// put_structure ~w, ~w (delegate to step)~n", [I, FStr, AiStr]),
    format("~wif (!vm->step(Instruction::PutStructure(\"~w\", \"~w\"))) return false;~n",
           [I, FStr, Ai]).

emit_one(put_list(AiStr), I) :-
    cpp_reg_name(AiStr, Ai),
    format("~w// put_list ~w (delegate to step)~n", [I, AiStr]),
    format("~wif (!vm->step(Instruction::PutList(\"~w\"))) return false;~n", [I, Ai]).

% --- Unify instructions (delegate to step) ---

emit_one(unify_variable(XnStr), I) :-
    cpp_reg_name(XnStr, Xn),
    format("~w// unify_variable ~w (delegate to step)~n", [I, XnStr]),
    format("~wif (!vm->step(Instruction::UnifyVariable(\"~w\"))) return false;~n", [I, Xn]).

emit_one(unify_value(XnStr), I) :-
    cpp_reg_name(XnStr, Xn),
    format("~w// unify_value ~w (delegate to step)~n", [I, XnStr]),
    format("~wif (!vm->step(Instruction::UnifyValue(\"~w\"))) return false;~n", [I, Xn]).

emit_one(unify_constant(CStr), I) :-
    cpp_val_literal(CStr, CppVal),
    format("~w// unify_constant ~w (delegate to step)~n", [I, CStr]),
    format("~wif (!vm->step(Instruction::UnifyConstant(~w))) return false;~n", [I, CppVal]).

% --- Set instructions (delegate to step) ---

emit_one(set_variable(XnStr), I) :-
    cpp_reg_name(XnStr, Xn),
    format("~w// set_variable ~w (delegate to step)~n", [I, XnStr]),
    format("~wif (!vm->step(Instruction::SetVariable(\"~w\"))) return false;~n", [I, Xn]).

emit_one(set_value(XnStr), I) :-
    cpp_reg_name(XnStr, Xn),
    format("~w// set_value ~w (delegate to step)~n", [I, XnStr]),
    format("~wif (!vm->step(Instruction::SetValue(\"~w\"))) return false;~n", [I, Xn]).

emit_one(set_constant(CStr), I) :-
    cpp_val_literal(CStr, CppVal),
    format("~w// set_constant ~w (delegate to step)~n", [I, CStr]),
    format("~wif (!vm->step(Instruction::SetConstant(~w))) return false;~n", [I, CppVal]).

% --- Environment instructions ---

emit_one(allocate, I) :-
    format("~w// allocate~n", [I]),
    format("~wvm->step(Instruction::Allocate());~n", [I]).

emit_one(deallocate, I) :-
    format("~w// deallocate~n", [I]),
    format("~wvm->step(Instruction::Deallocate());~n", [I]).

% --- Control instructions ---

emit_one(call(PredStr, _NStr), I) :-
    format("~w// call ~w~n", [I, PredStr]),
    format("~w{~n", [I]),
    format("~w    std::size_t saved_cp = vm->cp;~n", [I]),
    format("~w    auto it = vm->labels.find(\"~w\");~n", [I, PredStr]),
    format("~w    if (it == vm->labels.end()) return false;~n", [I]),
    format("~w    vm->pc = it->second;~n", [I]),
    format("~w    if (!vm->run()) return false;~n", [I]),
    format("~w    vm->cp = saved_cp;~n", [I]),
    format("~w}~n", [I]).

emit_one(execute(PredStr), I) :-
    format("~w// execute ~w (tail call)~n", [I, PredStr]),
    format("~w{~n", [I]),
    format("~w    auto it = vm->labels.find(\"~w\");~n", [I, PredStr]),
    format("~w    if (it == vm->labels.end()) return false;~n", [I]),
    format("~w    vm->pc = it->second;~n", [I]),
    format("~w    return vm->run();~n", [I]),
    format("~w}~n", [I]).

emit_one(builtin_call(OpStr, NStr), I) :-
    format("~w// builtin_call ~w ~w~n", [I, OpStr, NStr]),
    local_escape_cpp_string(OpStr, EscOp),
    format("~wif (!vm->step(Instruction::BuiltinCall(\"~w\", ~w))) return false;~n",
           [I, EscOp, NStr]).

emit_one(call_foreign(PredStr, ArStr), I) :-
    format("~w// call_foreign ~w ~w~n", [I, PredStr, ArStr]),
    format("~wif (!vm->step(Instruction::CallForeign(\"~w\", ~w))) return false;~n",
           [I, PredStr, ArStr]).

% --- Choicepoint / ITE related (consumed during lowering) ---

emit_one(try_me_else(_), _) :- !.
emit_one(trust_me, _) :- !.
emit_one(cut_ite, _) :- !.
emit_one(jump(_), _) :- !.

% --- Fallback ---

emit_one(Instr, I) :-
    format("~w// TODO: lowered emission for ~w~n", [I, Instr]).

% =====================================================================
% Helpers
% =====================================================================

%% cpp_reg_name(+RegStr, -Name)
%  Pass through register name (A1, X2, Y3 etc.) — used for get_reg/put_reg.
cpp_reg_name(RegStr, Name) :-
    atom_string(RegA, RegStr),
    atom_string(RegA, Name).

%% local_escape_cpp_string(+In, -Out)
%  Inlined copy of wam_cpp_target:escape_cpp_string/2 to keep this module
%  loadable without a back-import to wam_cpp_target.
local_escape_cpp_string(In, Out) :-
    atom_string(In, S),
    split_string(S, "\\", "", Parts),
    local_join(Parts, "\\\\", Escaped1),
    split_string(Escaped1, "\"", "", Parts2),
    local_join(Parts2, "\\\"", Out).

local_join([], _, "").
local_join([X], _, X).
local_join([X, Y|Rest], Sep, Result) :-
    local_join([Y|Rest], Sep, Tail),
    string_concat(X, Sep, XSep),
    string_concat(XSep, Tail, Result).

%% cpp_val_literal(+Str, -CppLiteral)
%  Convert a WAM constant token to a C++ Value literal.
cpp_val_literal(Str, CppVal) :-
    (   number_string(N, Str), integer(N)
    ->  format(atom(CppVal), 'Value::Integer(~w)', [N])
    ;   number_string(F, Str), float(F)
    ->  format(atom(CppVal), 'Value::Float(~w)', [F])
    ;   Str == "[]"
    ->  CppVal = 'Value::Atom("[]")'
    ;   format(atom(CppVal), 'Value::Atom("~w")', [Str])
    ).
