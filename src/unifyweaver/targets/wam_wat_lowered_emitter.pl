:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wam_wat_lowered_emitter.pl — WAM-lowered WAT emission
%
% Emits per-predicate WAT fast-path functions for deterministic clause-1
% prefixes.  The shape mirrors the hybrid WAM targets (Rust, Scala,
% Haskell, F#, LLVM): try the lowered clause-1 path first, and if it fails
% the generated public entry point reinitialises state and falls back to
% the complete bytecode interpreter.

:- module(wam_wat_lowered_emitter, [
    wam_wat_lowerable/3,
    lower_predicate_to_wat/4,
    wat_lowered_func_name/2
]).

:- use_module(library(lists)).

%% wam_wat_lowerable(+PI, +WamCode, -Reason) is semidet.
wam_wat_lowerable(_PI, WamCode, Reason) :-
    parse_wam_text_wat(WamCode, Instrs),
    clause1_instrs_wat(Instrs, C1, MultiClause),
    is_deterministic_pred_wat(C1),
    forall(member(I, C1), wat_supported(I)),
    ( MultiClause == true -> Reason = multi_clause_1 ; Reason = single_clause ).

parse_wam_text_wat(WamCode, Instrs) :-
    (   string(WamCode) -> S = WamCode
    ;   atom_string(WamCode, S)
    ),
    split_string(S, "\n", "", Lines),
    wam_wat_target:wam_lines_to_instrs(Lines, 0, Instrs, _Labels).

clause1_instrs_wat([try_me_else(_)|Rest], C1, true) :- !,
    take_to_proceed_wat(Rest, C1).
clause1_instrs_wat(Instrs, Instrs, false).

take_to_proceed_wat([], []).
take_to_proceed_wat([proceed|_], [proceed]) :- !.
take_to_proceed_wat([I|Rest], [I|More]) :- take_to_proceed_wat(Rest, More).

is_deterministic_pred_wat(Instrs) :-
    \+ member(try_me_else(_), Instrs),
    \+ member(retry_me_else(_), Instrs),
    \+ member(trust_me, Instrs).

wat_supported(get_constant(_, _)).
wat_supported(get_variable(_, _)).
wat_supported(get_value(_, _)).
wat_supported(get_structure(_, _)).
wat_supported(get_list(_)).
wat_supported(unify_variable(_)).
wat_supported(unify_value(_)).
wat_supported(unify_constant(_)).
wat_supported(put_constant(_, _)).
wat_supported(put_variable(_, _)).
wat_supported(put_value(_, _)).
wat_supported(put_structure(_, _)).
wat_supported(put_list(_)).
wat_supported(set_variable(_)).
wat_supported(set_value(_)).
wat_supported(set_constant(_)).
wat_supported(allocate).
wat_supported(deallocate).
wat_supported(proceed).
wat_supported(builtin_call(Op, _)) :- deterministic_builtin_wat(Op).

% Keep calls/execute in the interpreter for now: a lowered WAT function has
% no cross-predicate continuation frame, so inlining them would not be sound.
deterministic_builtin_wat('=/2').
deterministic_builtin_wat('true/0').
deterministic_builtin_wat('fail/0').
deterministic_builtin_wat('!/0').
deterministic_builtin_wat('is/2').
deterministic_builtin_wat('=:=/2').
deterministic_builtin_wat('=\\=/2').
deterministic_builtin_wat('</2').
deterministic_builtin_wat('>/2').
deterministic_builtin_wat('=</2').
deterministic_builtin_wat('>=/2').
deterministic_builtin_wat('var/1').
deterministic_builtin_wat('nonvar/1').
deterministic_builtin_wat('atom/1').
deterministic_builtin_wat('number/1').
deterministic_builtin_wat('integer/1').
deterministic_builtin_wat('float/1').
deterministic_builtin_wat('atomic/1').
deterministic_builtin_wat('is_list/1').

wat_lowered_func_name(Pred/Arity, Name) :-
    atom_string(Pred, S),
    string_codes(S, Codes),
    maplist(wat_safe_code, Codes, SafeCodes),
    string_codes(Safe, SafeCodes),
    format(atom(Name), 'lowered_~w_~w', [Safe, Arity]).

wat_safe_code(C, C) :-
    (   C >= 0'a, C =< 0'z
    ;   C >= 0'A, C =< 0'Z
    ;   C >= 0'0, C =< 0'9
    ;   C =:= 0'_
    ), !.
wat_safe_code(_, 0'_).

%% lower_predicate_to_wat(+PI, +WamCode, +Options, -lowered(PredName, FuncName, Code))
lower_predicate_to_wat(PI, WamCode, _Options, lowered(PredName, FuncName, Code)) :-
    ( PI = _M:Pred/Arity -> true ; PI = Pred/Arity ),
    format(atom(PredName), '~w/~w', [Pred, Arity]),
    wat_lowered_func_name(Pred/Arity, FuncName),
    wam_wat_lowerable(PI, WamCode, _Reason),
    parse_wam_text_wat(WamCode, Instrs),
    clause1_instrs_wat(Instrs, C1, _MultiClause),
    with_output_to(atom(Code), emit_lowered_function_wat(FuncName, C1)).

emit_lowered_function_wat(FuncName, Instrs) :-
    format('(func $~w (result i32)~n', [FuncName]),
    format('  ;; WAM-WAT lowered clause-1 fast path. On failure the public entry replays via $run_loop.~n'),
    emit_lowered_instrs_wat(Instrs),
    format('  (i32.const 1)~n'),
    format(')~n').

emit_lowered_instrs_wat([]).
emit_lowered_instrs_wat([proceed|_]) :- !,
    format('  (return (i32.const 1))~n').
emit_lowered_instrs_wat([Instr|Rest]) :-
    wam_wat_target:wam_instruction_to_wat_operands(Instr, [], DoName, Op1, Op2),
    format('  (if (i32.eqz (call $do_~w (i64.const ~w) (i64.const ~w)))~n', [DoName, Op1, Op2]),
    format('    (then (return (i32.const 0))))~n'),
    emit_lowered_instrs_wat(Rest).
