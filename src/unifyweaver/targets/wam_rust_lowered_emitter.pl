:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wam_rust_lowered_emitter.pl — WAM-lowered Rust emission
%
% Emits one Rust function per deterministic predicate.
% Simple register operations are inlined as direct Rust code; complex
% instructions delegate to vm methods from state.rs.
%
% For multi-clause predicates (try_me_else), only clause 1 is lowered.
% Clause 2+ stays in the interpreter's instruction array for backtrack.
%
% Modelled on wam_go_lowered_emitter.pl (475 lines) and
% wam_fsharp_lowered_emitter.pl (646 lines).

:- module(wam_rust_lowered_emitter, [
    wam_rust_lowerable/3,
    lower_predicate_to_rust/4,
    is_deterministic_pred_rust/1,
    rust_lowered_func_name/2
]).

:- use_module(library(lists)).
:- use_module(wam_rust_target, [escape_rust_string/2]).

% =====================================================================
% Parsing
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

%% wam_rust_lowerable(+Pred/Arity, +WamCode, -Reason)
%  True if the predicate can be lowered to a direct Rust function.
wam_rust_lowerable(PI, WamCode, Reason) :-
    (   is_list(WamCode) -> Instrs = WamCode
    ;   atom(WamCode) -> parse_wam_text(WamCode, Instrs)
    ;   atom_string(WamCode, _), parse_wam_text(WamCode, Instrs)
    ),
    clause1_instrs(Instrs, C1),
    forall(member(I, C1), rust_supported(I)),
    (   is_deterministic_pred_rust(Instrs)
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

%% is_deterministic_pred_rust(+Instrs)
%  True if the instruction list has no choice point instructions.
is_deterministic_pred_rust(Instrs) :-
    \+ member(try_me_else(_), Instrs),
    \+ member(retry_me_else(_), Instrs),
    \+ member(trust_me, Instrs).

rust_supported(allocate).
rust_supported(deallocate).
rust_supported(get_constant(_, _)).
rust_supported(get_variable(_, _)).
rust_supported(get_value(_, _)).
rust_supported(get_structure(_, _)).
rust_supported(get_list(_)).
rust_supported(get_nil(_)).
rust_supported(get_integer(_, _)).
rust_supported(unify_variable(_)).
rust_supported(unify_value(_)).
rust_supported(unify_constant(_)).
rust_supported(put_constant(_, _)).
rust_supported(put_variable(_, _)).
rust_supported(put_value(_, _)).
rust_supported(put_structure(_, _)).
rust_supported(put_list(_)).
rust_supported(set_variable(_)).
rust_supported(set_value(_)).
rust_supported(set_constant(_)).
rust_supported(call(_, _)).
rust_supported(execute(_)).
rust_supported(proceed).
rust_supported(fail).
rust_supported(builtin_call(_, _)).
rust_supported(call_foreign(_, _)).
rust_supported(try_me_else(_)).
rust_supported(trust_me).
rust_supported(cut_ite).
rust_supported(jump(_)).

% =====================================================================
% Function name generation
% =====================================================================

%% rust_lowered_func_name(+Functor/Arity, -RustFuncName)
%  Generates a valid Rust function name.
%  foo/2 -> "lowered_foo_2", my_pred/3 -> "lowered_my_pred_3"
rust_lowered_func_name(Functor/Arity, Name) :-
    atom_string(Functor, FStr),
    sanitize_rust_ident(FStr, SanStr),
    format(atom(Name), 'lowered_~w_~w', [SanStr, Arity]).

sanitize_rust_ident(In, Out) :-
    string_codes(In, Codes),
    maplist(rust_safe_code, Codes, OutCodes),
    string_codes(OutStr, OutCodes),
    atom_string(Out, OutStr).

rust_safe_code(C, C) :-
    (   C >= 0'a, C =< 0'z -> true
    ;   C >= 0'A, C =< 0'Z -> true
    ;   C >= 0'0, C =< 0'9 -> true
    ;   C =:= 0'_ -> true
    ),
    !.
rust_safe_code(_, 0'_).

% =====================================================================
% Emission
% =====================================================================

%% lower_predicate_to_rust(+Pred/Arity, +WamCode, +Options, -RustLines)
%  Emit a Rust function for the predicate.
lower_predicate_to_rust(PI, WamCode, _Options, RustLines) :-
    ( PI = _M:Pred/Arity -> true ; PI = Pred/Arity ),
    rust_lowered_func_name(Pred/Arity, FuncName),
    (   is_list(WamCode) -> Instrs = WamCode
    ;   parse_wam_text(WamCode, Instrs)
    ),
    clause1_instrs(Instrs, C1Instrs),
    with_output_to(string(Body), emit_instrs(C1Instrs, "    ")),
    format(string(Header),
'// ~w — lowered from ~w/~w
pub fn ~w(vm: &mut WamState) -> bool {', [FuncName, Pred, Arity, FuncName]),
    format(string(Footer), '}', []),
    RustLines = [Header, Body, Footer].

%% emit_instrs(+Instrs, +Indent)
%  Emit Rust code for a list of instructions.
emit_instrs([], _).
emit_instrs([Instr|Rest], Ind) :-
    emit_one(Instr, Ind),
    emit_instrs(Rest, Ind).

% --- Terminal instructions ---

emit_one(proceed, I) :-
    format("~wreturn true;~n", [I]).

emit_one(fail, I) :-
    format("~wreturn false;~n", [I]).

% --- Head unification (get_*) ---

emit_one(get_constant(CStr, AiStr), I) :-
    rust_reg_name(AiStr, Ai),
    rust_val_literal(CStr, RustVal),
    format("~w// get_constant ~w, ~w~n", [I, CStr, AiStr]),
    format("~w{~n", [I]),
    format("~w    let _a = vm.get_reg(\"~w\").unwrap_or(Value::Uninit);~n", [I, Ai]),
    format("~w    if _a.is_unbound() {~n", [I]),
    format("~w        vm.trail_binding(\"~w\");~n", [I, Ai]),
    format("~w        vm.put_reg(\"~w\", ~w);~n", [I, Ai, RustVal]),
    format("~w    } else if _a != ~w {~n", [I, RustVal]),
    format("~w        return false;~n", [I]),
    format("~w    }~n", [I]),
    format("~w}~n", [I]).

emit_one(get_integer(NStr, AiStr), I) :-
    rust_reg_name(AiStr, Ai),
    format("~w// get_integer ~w, ~w~n", [I, NStr, AiStr]),
    format("~w{~n", [I]),
    format("~w    let _a = vm.get_reg(\"~w\").unwrap_or(Value::Uninit);~n", [I, Ai]),
    format("~w    if _a.is_unbound() {~n", [I]),
    format("~w        vm.trail_binding(\"~w\");~n", [I, Ai]),
    format("~w        vm.put_reg(\"~w\", Value::Integer(~w));~n", [I, Ai, NStr]),
    format("~w    } else if _a != Value::Integer(~w) {~n", [I, NStr]),
    format("~w        return false;~n", [I]),
    format("~w    }~n", [I]),
    format("~w}~n", [I]).

emit_one(get_nil(AiStr), I) :-
    rust_reg_name(AiStr, Ai),
    format("~w// get_nil ~w~n", [I, AiStr]),
    format("~w{~n", [I]),
    format("~w    let _a = vm.get_reg(\"~w\").unwrap_or(Value::Uninit);~n", [I, Ai]),
    format("~w    if _a.is_unbound() {~n", [I]),
    format("~w        vm.trail_binding(\"~w\");~n", [I, Ai]),
    format("~w        vm.put_reg(\"~w\", Value::Atom(\"[]\".to_string()));~n", [I, Ai]),
    format("~w    } else if _a != Value::Atom(\"[]\".to_string()) {~n", [I]),
    format("~w        return false;~n", [I]),
    format("~w    }~n", [I]),
    format("~w}~n", [I]).

emit_one(get_variable(XnStr, AiStr), I) :-
    rust_reg_name(XnStr, Xn), rust_reg_name(AiStr, Ai),
    format("~w// get_variable ~w, ~w~n", [I, XnStr, AiStr]),
    format("~wif let Some(v) = vm.get_reg(\"~w\") { vm.put_reg(\"~w\", v); }~n", [I, Ai, Xn]).

emit_one(get_value(XnStr, AiStr), I) :-
    rust_reg_name(XnStr, Xn), rust_reg_name(AiStr, Ai),
    format("~w// get_value ~w, ~w~n", [I, XnStr, AiStr]),
    format("~w{~n", [I]),
    format("~w    let va = vm.get_reg(\"~w\").unwrap_or(Value::Uninit);~n", [I, Ai]),
    format("~w    let vx = vm.get_reg(\"~w\").unwrap_or(Value::Uninit);~n", [I, Xn]),
    format("~w    if !vm.unify(&va, &vx) { return false; }~n", [I]),
    format("~w}~n", [I]).

emit_one(get_structure(FStr, AiStr), I) :-
    rust_reg_name(AiStr, Ai),
    format("~w// get_structure ~w, ~w (delegate to step)~n", [I, FStr, AiStr]),
    format("~wif !vm.step(&Instruction::GetStructure(\"~w\".to_string(), \"~w\".to_string())) { return false; }~n", [I, FStr, Ai]).

emit_one(get_list(AiStr), I) :-
    rust_reg_name(AiStr, Ai),
    format("~w// get_list ~w (delegate to step)~n", [I, AiStr]),
    format("~wif !vm.step(&Instruction::GetList(\"~w\".to_string())) { return false; }~n", [I, Ai]).

% --- Body construction (put_*) ---

emit_one(put_constant(CStr, AiStr), I) :-
    rust_reg_name(AiStr, Ai),
    rust_val_literal(CStr, RustVal),
    format("~w// put_constant ~w, ~w~n", [I, CStr, AiStr]),
    format("~wvm.put_reg(\"~w\", ~w);~n", [I, Ai, RustVal]).

emit_one(put_variable(XnStr, AiStr), I) :-
    rust_reg_name(XnStr, Xn), rust_reg_name(AiStr, Ai),
    format("~w// put_variable ~w, ~w~n", [I, XnStr, AiStr]),
    format("~w{~n", [I]),
    format("~w    let v = Value::Unbound(format!(\"_V{}\", vm.var_counter));~n", [I]),
    format("~w    vm.var_counter += 1;~n", [I]),
    format("~w    vm.put_reg(\"~w\", v.clone());~n", [I, Xn]),
    format("~w    vm.put_reg(\"~w\", v);~n", [I, Ai]),
    format("~w}~n", [I]).

emit_one(put_value(XnStr, AiStr), I) :-
    rust_reg_name(XnStr, Xn), rust_reg_name(AiStr, Ai),
    format("~w// put_value ~w, ~w~n", [I, XnStr, AiStr]),
    format("~wif let Some(v) = vm.get_reg(\"~w\") { vm.put_reg(\"~w\", v); }~n", [I, Xn, Ai]).

emit_one(put_structure(FStr, AiStr), I) :-
    rust_reg_name(AiStr, Ai),
    format("~w// put_structure ~w, ~w (delegate to step)~n", [I, FStr, AiStr]),
    format("~wif !vm.step(&Instruction::PutStructure(\"~w\".to_string(), \"~w\".to_string())) { return false; }~n", [I, FStr, Ai]).

emit_one(put_list(AiStr), I) :-
    rust_reg_name(AiStr, Ai),
    format("~w// put_list ~w (delegate to step)~n", [I, AiStr]),
    format("~wif !vm.step(&Instruction::PutList(\"~w\".to_string())) { return false; }~n", [I, Ai]).

% --- Unify instructions (delegate to step) ---

emit_one(unify_variable(XnStr), I) :-
    rust_reg_name(XnStr, Xn),
    format("~w// unify_variable ~w (delegate to step)~n", [I, XnStr]),
    format("~wif !vm.step(&Instruction::UnifyVariable(\"~w\".to_string())) { return false; }~n", [I, Xn]).

emit_one(unify_value(XnStr), I) :-
    rust_reg_name(XnStr, Xn),
    format("~w// unify_value ~w (delegate to step)~n", [I, XnStr]),
    format("~wif !vm.step(&Instruction::UnifyValue(\"~w\".to_string())) { return false; }~n", [I, Xn]).

emit_one(unify_constant(CStr), I) :-
    rust_val_literal(CStr, RustVal),
    format("~w// unify_constant ~w (delegate to step)~n", [I, CStr]),
    format("~wif !vm.step(&Instruction::UnifyConstant(~w)) { return false; }~n", [I, RustVal]).

% --- Set instructions (delegate to step) ---

emit_one(set_variable(XnStr), I) :-
    rust_reg_name(XnStr, Xn),
    format("~w// set_variable ~w (delegate to step)~n", [I, XnStr]),
    format("~wif !vm.step(&Instruction::SetVariable(\"~w\".to_string())) { return false; }~n", [I, Xn]).

emit_one(set_value(XnStr), I) :-
    rust_reg_name(XnStr, Xn),
    format("~w// set_value ~w (delegate to step)~n", [I, XnStr]),
    format("~wif !vm.step(&Instruction::SetValue(\"~w\".to_string())) { return false; }~n", [I, Xn]).

emit_one(set_constant(CStr), I) :-
    rust_val_literal(CStr, RustVal),
    format("~w// set_constant ~w (delegate to step)~n", [I, CStr]),
    format("~wif !vm.step(&Instruction::SetConstant(~w)) { return false; }~n", [I, RustVal]).

% --- Environment instructions ---

emit_one(allocate, I) :-
    format("~w// allocate~n", [I]),
    format("~wvm.step(&Instruction::Allocate);~n", [I]).

emit_one(deallocate, I) :-
    format("~w// deallocate~n", [I]),
    format("~wvm.step(&Instruction::Deallocate);~n", [I]).

% --- Control instructions ---

emit_one(call(PredStr, _NStr), I) :-
    format("~w// call ~w~n", [I, PredStr]),
    format("~w{~n", [I]),
    format("~w    let saved_cp = vm.cp;~n", [I]),
    format("~w    if let Some(&pc) = vm.labels.get(\"~w\") {~n", [I, PredStr]),
    format("~w        vm.pc = pc;~n", [I]),
    format("~w        if !vm.run() { return false; }~n", [I]),
    format("~w    } else { return false; }~n", [I]),
    format("~w    vm.cp = saved_cp;~n", [I]),
    format("~w}~n", [I]).

emit_one(execute(PredStr), I) :-
    format("~w// execute ~w (tail call)~n", [I, PredStr]),
    format("~wif let Some(&pc) = vm.labels.get(\"~w\") {~n", [I, PredStr]),
    format("~w    vm.pc = pc;~n", [I]),
    format("~w    return vm.run();~n", [I]),
    format("~w}~n", [I]),
    format("~wreturn false;~n", [I]).

emit_one(builtin_call(OpStr, NStr), I) :-
    format("~w// builtin_call ~w ~w~n", [I, OpStr, NStr]),
    escape_rust_string(OpStr, EscOp),
    format("~wif !vm.step(&Instruction::BuiltinCall(\"~w\".to_string(), ~w)) { return false; }~n", [I, EscOp, NStr]).

emit_one(call_foreign(PredStr, ArStr), I) :-
    format("~w// call_foreign ~w ~w~n", [I, PredStr, ArStr]),
    format("~wif !vm.step(&Instruction::CallForeign(\"~w\".to_string(), ~w)) { return false; }~n", [I, PredStr, ArStr]).

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

%% rust_reg_name(+RegStr, -Name)
%  Pass through register name (A1, X2, Y3 etc.) — used for get_reg/put_reg calls.
rust_reg_name(RegStr, Name) :-
    atom_string(RegA, RegStr),
    atom_string(RegA, Name).

%% rust_val_literal(+Str, -RustLiteral)
%  Convert a WAM constant to a Rust value literal.
rust_val_literal(Str, RustVal) :-
    (   number_string(N, Str), integer(N)
    ->  format(atom(RustVal), 'Value::Integer(~w)', [N])
    ;   number_string(F, Str), float(F)
    ->  format(atom(RustVal), 'Value::Float(~w)', [F])
    ;   Str == "[]"
    ->  RustVal = 'Value::Atom("[]".to_string())'
    ;   format(atom(RustVal), 'Value::Atom("~w".to_string())', [Str])
    ).
