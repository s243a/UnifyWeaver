:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% wam_go_lowered_emitter.pl — WAM-lowered Go emission
%
% Emits one Go method per deterministic predicate on *WamState.
% Simple register operations are inlined as direct Go code; complex
% instructions delegate to vm methods from state.go / runtime.go.
%
% For multi-clause predicates (try_me_else), only clause 1 is lowered.
% Clause 2+ stays in the interpreter's instruction array for backtrack.
%
% Modelled on wam_haskell_lowered_emitter.pl (464 lines) for structure
% and wam_python_lowered_emitter.pl (795 lines) for instruction detail.

:- module(wam_go_lowered_emitter, [
    wam_go_lowerable/3,
    lower_predicate_to_go/4,
    is_deterministic_pred_go/1,
    go_func_name/2,
    has_internal_ite_pattern/1   % +Instrs (true if instrs contain a complete try/cut/jump/trust ITE)
]).

:- use_module(library(lists)).
:- use_module(wam_ite_structurer, [structure_ite/2, split_commit/3, is_commit/1]).
:- use_module(wam_text_parser, [wam_classify_constant_token/2]).
:- use_module(wam_go_target, [
    escape_go_string/2,
    intern_atom_go/2
]).

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
% Label-preserving parse + if-then-else structuring
% =====================================================================
%
%  The base parser (parse_wam_text) drops label lines, which erases the
%  boundaries of an (C -> T ; E) / \+ / once block. Without those
%  boundaries the old emitter used a "peel the trailing epilogue"
%  heuristic to split the else body from the continuation — which placed a
%  *sequential* second ITE (and broke a nested one) inside the preceding
%  else branch, silently miscompiling the clause.
%
%  parse_wam_text_labeled keeps label(Name) markers (and cut_ite) so
%  structure_ite can fold each block into an ite(Cond,Then,Else) term
%  using the real else/cont labels. The continuation is then emitted as a
%  sibling of the if/else (not nested in the else), fixing sequential and
%  nested blocks. Mirrors the shared wam_ite_structurer.

parse_wam_text_labeled(WamText, Instrs) :-
    atom_string(WamText, S),
    split_string(S, "\n", "", Lines),
    parse_lines_labeled(Lines, Instrs).

parse_lines_labeled([], []).
parse_lines_labeled([Line|Rest], Instrs) :-
    split_string(Line, " \t,", " \t,", Parts),
    delete(Parts, "", CleanParts),
    (   CleanParts == []
    ->  parse_lines_labeled(Rest, Instrs)
    ;   CleanParts = [First|_],
        (   sub_string(First, _, 1, 0, ":")
        ->  sub_string(First, 0, _, 1, LabelStr),
            Instrs = [label(LabelStr)|More],   % keep as string to match try_me_else/jump args
            parse_lines_labeled(Rest, More)
        ;   instr_from_parts(CleanParts, Instr)
        ->  Instrs = [Instr|More],
            parse_lines_labeled(Rest, More)
        ;   parse_lines_labeled(Rest, Instrs)
        )
    ).

% structure_ite/2, split_commit/3 and is_commit/1 are shared across the
% lowered backends — see wam_ite_structurer.pl.

%% go_base_instrs(+WamCode, -Instrs)  — base (label-stripped) parse or list.
go_base_instrs(WamCode, Instrs) :-
    ( is_list(WamCode) -> Instrs = WamCode ; parse_wam_text(WamCode, Instrs) ).

%% go_structured_clause1(+WamCode, -Structured) is semidet.
%  Re-parse keeping labels/cut_ite, take clause 1, fold ITE blocks. Only
%  succeeds for a single-clause predicate (no predicate-level try_me_else
%  head) whose every block is consumed.
go_structured_clause1(WamCode, Structured) :-
    ( is_list(WamCode) -> LInstrs = WamCode ; parse_wam_text_labeled(WamCode, LInstrs) ),
    \+ ( LInstrs = [try_me_else(_)|_] ),   % not predicate-level multi-clause
    take_to_proceed(LInstrs, C1L),
    structure_ite(C1L, Structured),
    \+ member(try_me_else(_), Structured),  % every block consumed
    \+ member(trust_me, Structured),
    \+ member(retry_me_else(_), Structured).

%% go_supported_structured(+StructuredInstr)
go_supported_structured(ite(C, T, E)) :- !,
    forall(member(I, C), go_supported_structured(I)),
    forall(member(I, T), go_supported_structured(I)),
    forall(member(I, E), go_supported_structured(I)).
go_supported_structured(I) :- go_supported(I).

% =====================================================================
% Lowerability
% =====================================================================

%% wam_go_lowerable(+Pred/Arity, +WamCode, -Reason)
%  True if the predicate can be lowered to a direct Go function.
wam_go_lowerable(PI, WamCode, Reason) :-
    go_base_instrs(WamCode, Instrs),
    clause1_instrs(Instrs, C1),
    (   % No internal ITE: existing deterministic / multi-clause path.
        \+ member(try_me_else(_), C1),
        forall(member(I, C1), go_supported(I))
    ->  ( is_deterministic_pred_go(Instrs) -> Reason = deterministic
        ; Reason = multi_clause_1 )
    ;   % Clause-1 has an inner choice point: lower only if it is a pure
        % (C -> T ; E) / \+ / once block whose pieces are all supported.
        \+ ( Instrs = [try_me_else(_)|_] ),   % single-clause predicate
        go_structured_clause1(WamCode, Structured),
        forall(member(I, Structured), go_supported_structured(I)),
        Reason = ite_lowered
    ),
    ( PI = _M:_P/_A -> true ; PI = _/_A2 -> true ; true ).

clause1_instrs([], []).
clause1_instrs([try_me_else(_)|Rest], C1) :- !,
    take_to_proceed(Rest, C1).
clause1_instrs(Instrs, Instrs).

take_to_proceed([], []).
take_to_proceed([proceed|_], [proceed]) :- !.
take_to_proceed([I|Rest], [I|More]) :- take_to_proceed(Rest, More).

%% is_deterministic_pred_go(+Instrs)
%  True if the instruction list has no choice point instructions.
is_deterministic_pred_go(Instrs) :-
    \+ member(try_me_else(_), Instrs),
    \+ member(retry_me_else(_), Instrs),
    \+ member(trust_me, Instrs).

go_supported(allocate).
go_supported(deallocate).
go_supported(get_constant(_, _)).
go_supported(get_variable(_, _)).
go_supported(get_value(_, _)).
go_supported(get_structure(_, _)).
go_supported(get_list(_)).
go_supported(get_nil(_)).
go_supported(get_integer(_, _)).
go_supported(unify_variable(_)).
go_supported(unify_value(_)).
go_supported(unify_constant(_)).
go_supported(put_constant(_, _)).
go_supported(put_variable(_, _)).
go_supported(put_value(_, _)).
go_supported(put_structure(_, _)).
go_supported(put_list(_)).
go_supported(set_variable(_)).
go_supported(set_value(_)).
go_supported(set_constant(_)).
go_supported(call(_, _)).
go_supported(execute(_)).
go_supported(proceed).
go_supported(fail).
go_supported(builtin_call(_, _)).
go_supported(call_foreign(_, _)).
go_supported(try_me_else(_)).
go_supported(trust_me).
go_supported(cut_ite).
go_supported(jump(_)).

% =====================================================================
% Function name generation
% =====================================================================

%% go_func_name(+Functor/Arity, -GoFuncName)
%  Generates a valid exported Go function name.
%  foo/2 -> "PredFoo2", my_pred/3 -> "PredMy_pred3"
go_func_name(Functor/Arity, Name) :-
    atom_string(Functor, FStr),
    sanitize_go_ident(FStr, SanStr),
    capitalize_first(SanStr, CapStr),
    format(atom(Name), 'Pred~w~w', [CapStr, Arity]).

sanitize_go_ident(In, Out) :-
    string_codes(In, Codes),
    maplist(go_safe_code, Codes, OutCodes),
    string_codes(OutStr, OutCodes),
    atom_string(Out, OutStr).

go_safe_code(C, C) :-
    (   C >= 0'a, C =< 0'z -> true
    ;   C >= 0'A, C =< 0'Z -> true
    ;   C >= 0'0, C =< 0'9 -> true
    ;   C =:= 0'_ -> true
    ),
    !.
go_safe_code(_, 0'_).

capitalize_first("", "") :- !.
capitalize_first(Str, Cap) :-
    string_codes(Str, [First|Rest]),
    code_type(Upper, to_upper(First)),
    string_codes(Cap, [Upper|Rest]).

% =====================================================================
% Emission
% =====================================================================

%% lower_predicate_to_go(+Pred/Arity, +WamCode, +Options, -GoLines)
%  Emit a Go method on *WamState for the predicate.
lower_predicate_to_go(PI, WamCode, _Options, GoLines) :-
    ( PI = _M:Pred/Arity -> true ; PI = Pred/Arity ),
    go_func_name(Pred/Arity, FuncName),
    go_base_instrs(WamCode, Instrs),
    clause1_instrs(Instrs, C1Instrs),
    % Emit the plain clause-1 when it has no inner choice point; otherwise
    % fold its ITE block(s) into structured form (ite/3) first.
    (   \+ member(try_me_else(_), C1Instrs)
    ->  EmitInstrs = C1Instrs
    ;   go_structured_clause1(WamCode, EmitInstrs)
    ),
    with_output_to(string(Body), emit_instrs(EmitInstrs, "    ")),
    format(string(Header),
'// ~w — lowered from ~w/~w
func (vm *WamState) ~w() bool {', [FuncName, Pred, Arity, FuncName]),
    format(string(Footer), '}', []),
    GoLines = [Header, Body, Footer].

%% emit_instrs(+Instrs, +Indent)
%  Emit Go code for a list of instructions. Detects the WAM if-then-else
%  pattern (try_me_else / cut_ite / jump / trust_me) and lowers it to
%  native Go branching instead of silently consuming the choice-point
%  instructions. The condition runs inside an inline closure so existing
%  `return false` failure paths short-circuit cleanly; on failure the
%  trail is unwound to its pre-condition mark before the else branch.
emit_instrs([], _).
emit_instrs([ite(Cond, Then, Else)|Rest], Ind) :- !,
    emit_ite_block(Cond, Then, Else, Ind),
    emit_instrs(Rest, Ind).
emit_instrs([label(_)|Rest], Ind) :- !,   % safety: drop any stray label marker
    emit_instrs(Rest, Ind).
emit_instrs([Instr|Rest], Ind) :-
    emit_one(Instr, Ind),
    emit_instrs(Rest, Ind).

%% has_internal_ite_pattern(+Instrs)
%  True if the (clause-1 of the) given instruction stream contains a
%  well-formed if-then-else block, i.e. structure_ite folds out at
%  least one ite/3 term. Used by tests to assert that ITE lowering fires.
has_internal_ite_pattern(Instrs) :-
    catch(go_structured_clause1(Instrs, Structured), _, fail),
    once(member(ite(_, _, _), Structured)).

%% emit_ite_block(+CondInstrs, +ThenInstrs, +ElseInstrs, +Indent)
%  Emit native Go if/else for the WAM if-then-else pattern.
%  The condition runs in an immediately-invoked closure that returns
%  bool, so any inner `return false` short-circuits to a false outcome
%  without escaping the surrounding lowered function.
%
%  When the condition fails we must restore the pre-condition machine
%  state before running the else branch — exactly what a choice point
%  does on backtracking. Three things can change during the condition,
%  and Go's runtime tracks them in different places:
%
%    1. Variable bindings — reverted via the trail mark + unwindTrailTo.
%    2. Existing argument / Y registers that bindUnbound rewrote in place
%       (the trail does not track these) — reverted via a full snapshot of
%       the register file (copy of vm.Regs). We can't use snapshotAllRegs:
%       it sizes its Y-range from MaxYReg, but the lowered code sets
%       Y-registers with direct `vm.Regs[i] = ...` writes (not putReg), so
%       MaxYReg is stale and restoreSavedRegs would wipe live Y-regs.
%    3. Fresh variables the condition itself created with put_variable.
%       getReg reads the register array directly (not the binding table),
%       so after restoring the pre-condition registers we re-run the
%       condition's put_variable instructions to reset those registers to
%       fresh unbound vars, matching the interpreter's post-backtrack
%       state.
%
%  Without all three, a binding condition like ((Y=a,Y=b) -> ... ; ...)
%  leaves Y holding the stale `a` after the trail is unwound, so a later
%  `X = Y` in the else/continuation sees the wrong value. The then branch
%  keeps the condition's results, so only the else restores.
emit_ite_block(CondInstrs, ThenInstrs, ElseInstrs, I) :-
    atom_concat(I, "    ", InnerInd),
    cond_put_variable_instrs(CondInstrs, FreshVarInstrs),
    format("~w// if-then-else (lowered from try_me_else/cut_ite/jump/trust_me)~n", [I]),
    format("~w{~n", [I]),
    format("~w    _trailMark := vm.TrailLen~n", [I]),
    format("~w    _savedRegs := vm.Regs   // Regs is a fixed array; assignment copies it~n", [I]),
    format("~w    _condOk := func() bool {~n", [I]),
    emit_instrs(CondInstrs, InnerInd),
    format("~w        return true~n", [I]),
    format("~w    }()~n", [I]),
    format("~w    if _condOk {~n", [I]),
    emit_instrs(ThenInstrs, InnerInd),
    format("~w    } else {~n", [I]),
    format("~w        vm.Regs = _savedRegs~n", [I]),
    format("~w        vm.unwindTrailTo(_trailMark)~n", [I]),
    emit_instrs(FreshVarInstrs, InnerInd),   % reset condition-local fresh vars
    emit_instrs(ElseInstrs, InnerInd),
    format("~w    }~n", [I]),
    format("~w}~n", [I]).

%% cond_put_variable_instrs(+CondInstrs, -PutVarInstrs)
%  The put_variable instructions in a condition, in order. Re-running them
%  in the else branch resets the fresh variables the condition created.
cond_put_variable_instrs([], []).
cond_put_variable_instrs([put_variable(X, A)|T], [put_variable(X, A)|R]) :- !,
    cond_put_variable_instrs(T, R).
cond_put_variable_instrs([_|T], R) :-
    cond_put_variable_instrs(T, R).

% --- Terminal instructions ---

emit_one(proceed, I) :-
    format("~wreturn true~n", [I]).

emit_one(fail, I) :-
    format("~wreturn false~n", [I]).

% --- Head unification (get_*) ---

emit_one(get_constant(CStr, AiStr), I) :-
    go_reg_idx(AiStr, Ai),
    go_val_literal(CStr, GoVal),
    format("~w// get_constant ~w, ~w~n", [I, CStr, AiStr]),
    format("~w{~n", [I]),
    format("~w    _a := vm.deref(vm.Regs[~w])~n", [I, Ai]),
    format("~w    if _, ok := _a.(*Unbound); ok {~n", [I]),
    format("~w        u := _a.(*Unbound)~n", [I]),
    format("~w        vm.trailBinding(u.Idx)~n", [I]),
    format("~w        vm.Regs[u.Idx] = ~w~n", [I, GoVal]),
    format("~w    } else if !valueEquals(vm.deref(_a), ~w) {~n", [I, GoVal]),
    format("~w        return false~n", [I]),
    format("~w    }~n", [I]),
    format("~w}~n", [I]).

emit_one(get_integer(NStr, AiStr), I) :-
    go_reg_idx(AiStr, Ai),
    format("~w// get_integer ~w, ~w~n", [I, NStr, AiStr]),
    format("~w{~n", [I]),
    format("~w    _a := vm.deref(vm.Regs[~w])~n", [I, Ai]),
    format("~w    if _, ok := _a.(*Unbound); ok {~n", [I]),
    format("~w        u := _a.(*Unbound)~n", [I]),
    format("~w        vm.trailBinding(u.Idx)~n", [I]),
    format("~w        vm.Regs[u.Idx] = &Integer{Val: ~w}~n", [I, NStr]),
    format("~w    } else if !valueEquals(vm.deref(_a), &Integer{Val: ~w}) {~n", [I, NStr]),
    format("~w        return false~n", [I]),
    format("~w    }~n", [I]),
    format("~w}~n", [I]).

emit_one(get_nil(AiStr), I) :-
    go_reg_idx(AiStr, Ai),
    intern_atom_go("[]", NilVar),
    format("~w// get_nil ~w~n", [I, AiStr]),
    format("~w{~n", [I]),
    format("~w    _a := vm.deref(vm.Regs[~w])~n", [I, Ai]),
    format("~w    if _, ok := _a.(*Unbound); ok {~n", [I]),
    format("~w        u := _a.(*Unbound)~n", [I]),
    format("~w        vm.trailBinding(u.Idx)~n", [I]),
    format("~w        vm.Regs[u.Idx] = ~w~n", [I, NilVar]),
    format("~w    } else if !valueEquals(vm.deref(_a), ~w) {~n", [I, NilVar]),
    format("~w        return false~n", [I]),
    format("~w    }~n", [I]),
    format("~w}~n", [I]).

emit_one(get_variable(XnStr, AiStr), I) :-
    go_reg_idx(XnStr, Xn), go_reg_idx(AiStr, Ai),
    format("~w// get_variable ~w, ~w~n", [I, XnStr, AiStr]),
    format("~wvm.Regs[~w] = vm.Regs[~w]~n", [I, Xn, Ai]).

emit_one(get_value(XnStr, AiStr), I) :-
    go_reg_idx(XnStr, Xn), go_reg_idx(AiStr, Ai),
    format("~w// get_value ~w, ~w~n", [I, XnStr, AiStr]),
    format("~wif !vm.Unify(vm.Regs[~w], vm.Regs[~w]) {~n", [I, Ai, Xn]),
    format("~w    return false~n", [I]),
    format("~w}~n", [I]).

emit_one(get_structure(FStr, AiStr), I) :-
    go_reg_idx(AiStr, Ai),
    escape_go_string(FStr, EscapedFunctor),
    format("~w// get_structure ~w, ~w~n", [I, FStr, AiStr]),
    format("~wif !vm.Step(&GetStructure{Functor: \"~w\", Ai: ~w}) {~n", [I, EscapedFunctor, Ai]),
    format("~w    return false~n", [I]),
    format("~w}~n", [I]).

emit_one(get_list(AiStr), I) :-
    go_reg_idx(AiStr, Ai),
    format("~w// get_list ~w~n", [I, AiStr]),
    format("~wif !vm.Step(&GetList{Ai: ~w}) {~n", [I, Ai]),
    format("~w    return false~n", [I]),
    format("~w}~n", [I]).

% --- Body construction (put_*) ---

emit_one(put_constant(CStr, AiStr), I) :-
    go_reg_idx(AiStr, Ai),
    go_val_literal(CStr, GoVal),
    format("~w// put_constant ~w, ~w~n", [I, CStr, AiStr]),
    format("~wvm.Regs[~w] = ~w~n", [I, Ai, GoVal]).

emit_one(put_variable(XnStr, AiStr), I) :-
    go_reg_idx(XnStr, Xn), go_reg_idx(AiStr, Ai),
    format("~w// put_variable ~w, ~w~n", [I, XnStr, AiStr]),
    format("~w{~n", [I]),
    format("~w    v := &Unbound{Name: fmt.Sprintf(\"_R%d\", ~w), Idx: ~w}~n", [I, Xn, Xn]),
    format("~w    vm.putReg(~w, v)~n", [I, Xn]),
    format("~w    vm.Regs[~w] = v~n", [I, Ai]),
    format("~w}~n", [I]).

emit_one(put_value(XnStr, AiStr), I) :-
    go_reg_idx(XnStr, Xn), go_reg_idx(AiStr, Ai),
    format("~w// put_value ~w, ~w~n", [I, XnStr, AiStr]),
    format("~wvm.Regs[~w] = vm.getReg(~w)~n", [I, Ai, Xn]).

emit_one(put_structure(FStr, AiStr), I) :-
    go_reg_idx(AiStr, Ai),
    escape_go_string(FStr, EscapedFunctor),
    format("~w// put_structure ~w, ~w~n", [I, FStr, AiStr]),
    format("~wif !vm.Step(&PutStructure{Functor: \"~w\", Ai: ~w}) {~n", [I, EscapedFunctor, Ai]),
    format("~w    return false~n", [I]),
    format("~w}~n", [I]).

emit_one(put_list(AiStr), I) :-
    go_reg_idx(AiStr, Ai),
    format("~w// put_list ~w~n", [I, AiStr]),
    format("~wif !vm.Step(&PutList{Ai: ~w}) {~n", [I, Ai]),
    format("~w    return false~n", [I]),
    format("~w}~n", [I]).

% --- Unify instructions (delegate to vm.Step) ---

emit_one(unify_variable(XnStr), I) :-
    go_reg_idx(XnStr, Xn),
    format("~w// unify_variable ~w~n", [I, XnStr]),
    format("~wif !vm.Step(&UnifyVariable{Xn: ~w}) {~n", [I, Xn]),
    format("~w    return false~n", [I]),
    format("~w}~n", [I]).

emit_one(unify_value(XnStr), I) :-
    go_reg_idx(XnStr, Xn),
    format("~w// unify_value ~w~n", [I, XnStr]),
    format("~wif !vm.Step(&UnifyValue{Xn: ~w}) {~n", [I, Xn]),
    format("~w    return false~n", [I]),
    format("~w}~n", [I]).

emit_one(unify_constant(CStr), I) :-
    go_val_literal(CStr, GoVal),
    format("~w// unify_constant ~w~n", [I, CStr]),
    format("~wif !vm.Step(&UnifyConstant{C: ~w}) {~n", [I, GoVal]),
    format("~w    return false~n", [I]),
    format("~w}~n", [I]).

% --- Set instructions (delegate to vm.Step) ---

emit_one(set_variable(XnStr), I) :-
    go_reg_idx(XnStr, Xn),
    format("~w// set_variable ~w~n", [I, XnStr]),
    format("~wif !vm.Step(&SetVariable{Xn: ~w}) {~n", [I, Xn]),
    format("~w    return false~n", [I]),
    format("~w}~n", [I]).

emit_one(set_value(XnStr), I) :-
    go_reg_idx(XnStr, Xn),
    format("~w// set_value ~w~n", [I, XnStr]),
    format("~wif !vm.Step(&SetValue{Xn: ~w}) {~n", [I, Xn]),
    format("~w    return false~n", [I]),
    format("~w}~n", [I]).

emit_one(set_constant(CStr), I) :-
    go_val_literal(CStr, GoVal),
    format("~w// set_constant ~w~n", [I, CStr]),
    format("~wif !vm.Step(&SetConstant{C: ~w}) {~n", [I, GoVal]),
    format("~w    return false~n", [I]),
    format("~w}~n", [I]).

% --- Environment instructions ---

emit_one(allocate, I) :-
    format("~w// allocate~n", [I]),
    format("~wvm.Stack = append(vm.Stack, &EnvFrame{CP: vm.CP, B0: len(vm.ChoicePoints)})~n", [I]).

emit_one(deallocate, I) :-
    format("~w// deallocate~n", [I]),
    format("~wif env := vm.popEnvFrame(); env != nil {~n", [I]),
    format("~w    vm.CP = env.CP~n", [I]),
    format("~w}~n", [I]).

% --- Control instructions ---

emit_one(call(PredStr, _NStr), I) :-
    pred_to_go_call(PredStr, CallExpr),
    format("~w// call ~w~n", [I, PredStr]),
    format("~w{~n", [I]),
    format("~w    savedCP := vm.CP~n", [I]),
    format("~w    if !~w {~n", [I, CallExpr]),
    format("~w        return false~n", [I]),
    format("~w    }~n", [I]),
    format("~w    vm.CP = savedCP~n", [I]),
    format("~w}~n", [I]).

emit_one(execute(PredStr), I) :-
    pred_to_go_call(PredStr, CallExpr),
    format("~w// execute ~w (tail call)~n", [I, PredStr]),
    format("~wreturn ~w~n", [I, CallExpr]).

emit_one(builtin_call(OpStr, NStr), I) :-
    format("~w// builtin_call ~w ~w~n", [I, OpStr, NStr]),
    escape_go_string(OpStr, EscOp),
    format("~wif !vm.executeBuiltin(\"~w\", ~w) {~n", [I, EscOp, NStr]),
    format("~w    return false~n", [I]),
    format("~w}~n", [I]).

emit_one(call_foreign(PredStr, ArStr), I) :-
    format("~w// call_foreign ~w ~w~n", [I, PredStr, ArStr]),
    format("~wif !vm.executeForeignPredicate(\"~w\", ~w) {~n", [I, PredStr, ArStr]),
    format("~w    return false~n", [I]),
    format("~w}~n", [I]).

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

%% go_reg_idx(+RegStr, -Idx)
%  Parse register string to Go array index.
go_reg_idx(RegStr, Idx) :-
    atom_string(RegA, RegStr),
    (   sub_atom(RegA, 0, 1, _, 'A'),
        sub_atom(RegA, 1, _, 0, NA),
        atom_number(NA, Num)
    ->  Idx is Num - 1
    ;   sub_atom(RegA, 0, 1, _, 'X'),
        sub_atom(RegA, 1, _, 0, NA),
        atom_number(NA, Num)
    ->  Idx is Num + 99
    ;   sub_atom(RegA, 0, 1, _, 'Y'),
        sub_atom(RegA, 1, _, 0, NA),
        atom_number(NA, Num)
    ->  Idx is Num + 199
    ;   number_string(Idx, RegStr)
    ->  true
    ;   Idx = 0
    ).

%% go_val_literal(+Str, -GoLiteral)
%  Convert a WAM constant to a Go value literal. Atom literals are
%  routed through intern_atom_go/2 so identical atoms share a single
%  package-level *Atom value rather than allocating per call.
%
%  Atom-vs-number disambiguation goes through
%  wam_text_parser:wam_classify_constant_token/2: a bare token `5`
%  is the integer 5, a quoted token `'5'` is the atom whose name
%  is "5". Without the classifier, `intern_atom_go` would intern
%  the atom with the outer quotes attached (`internAtom("'5'")`).
go_val_literal(Str, GoVal) :-
    wam_classify_constant_token(Str, Class),
    (   Class = integer(N)
    ->  format(atom(GoVal), '&Integer{Val: ~w}', [N])
    ;   Class = float(F)
    ->  format(atom(GoVal), '&Float{Val: ~w}', [F])
    ;   Class = atom(Name),
        (   intern_atom_go(Name, AtomVar)
        ->  GoVal = AtomVar
        ;   % Defensive fallback if interning is somehow unavailable.
            escape_go_string(Name, Escaped),
            format(atom(GoVal), '&Atom{Name: "~w"}', [Escaped])
        )
    ).

%% pred_to_go_call(+PredStr, -CallExpr)
%  Convert "pred/arity" to a Go method call expression.
%  Generates a call via label dispatch (vm interprets from code array).
pred_to_go_call(PredStr, CallExpr) :-
    atom_string(PA, PredStr),
    (   sub_atom(PA, B, 1, _, '/')
    ->  sub_atom(PA, 0, B, _, _Functor),
        B1 is B + 1,
        sub_atom(PA, B1, _, 0, AS),
        atom_number(AS, _Arity)
    ;   true
    ),
    % Use label dispatch: set PC to label, run
    format(atom(CallExpr),
        'func() bool { if pc, ok := vm.Ctx.Labels["~w"]; ok { vm.PC = pc; return vm.Run() }; return false }()',
        [PredStr]).
