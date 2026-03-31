:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wam_runtime.pl - Enhanced Symbolic WAM Emulator
% This module provides a VM to execute symbolic WAM instructions.

:- module(wam_runtime, [
    execute_wam/3,       % +InstructionsString, +Query, -Results
    test_wam_runtime/0
]).

:- use_module(library(assoc)).
:- use_module(library(lists)).

/** <module> WAM Runtime Emulator

A symbolic emulator for the Warren Abstract Machine.
State structure: wam_state(PC, Regs, Stack, Heap, Trail, CP, ChoicePoints, Code, Labels)
*/

%% execute_wam(+Instructions, +Goal, -FinalRegs)
execute_wam(Instructions, Goal, FinalRegs) :-
    prepare_code(Instructions, Code, Labels),
    init_state(Goal, Code, Labels, S0),
    run_loop(S0, Sf),
    Sf = wam_state(_, FinalRegs, _, _, _, _, _, _, _).

prepare_code(Raw, Instructions, Labels) :-
    (   is_list(Raw) -> Lines = Raw
    ;   atomic_list_concat(Lines, '\n', Raw)
    ),
    parse_lines(Lines, 1, Instructions, Labels).

parse_lines([], _, [], Labels) :- empty_assoc(Labels).
parse_lines([Line|Rest], PC, Instrs, Labels) :-
    (   compound(Line), \+ is_label_term(Line) ->
        Instrs = [Line|IRest],
        NPC is PC + 1,
        parse_lines(Rest, NPC, IRest, Labels)
    ;   normalize_line(Line, Normalized),
        (   Normalized == "" -> parse_lines(Rest, PC, Instrs, Labels)
        ;   is_label(Normalized, Label) ->
            parse_lines(Rest, PC, Instrs, L0),
            put_assoc(Label, L0, PC, Labels)
        ;   parse_instr(Normalized, Instr) ->
            Instrs = [Instr|IRest],
            NPC is PC + 1,
            parse_lines(Rest, NPC, IRest, Labels)
        ;   parse_lines(Rest, PC, Instrs, Labels)
        )
    ).

is_label_term(Line) :-
    atom(Line),
    sub_atom(Line, _, _, 0, ':').

normalize_line(Line, Normalized) :-
    (   atom(Line) -> Str = Line
    ;   string(Line) -> atom_string(Str, Line)
    ;   term_string(Line, S), atom_string(Str, S)
    ),
    split_string(Str, " ", " \t\n\r", Parts),
    delete(Parts, "", CleanParts),
    atomic_list_concat(CleanParts, ' ', Normalized).

is_label(Line, Label) :-
    atom_concat(Label, ':', Line).

%% parse_instr(+String, -Term)
%  Parses "instr arg1, arg2" into instr(arg1, arg2)
parse_instr(Str, Term) :-
    % Handle space after opcode correctly
    (   sub_string(Str, Before, _, After, " ") ->
        sub_string(Str, 0, Before, _, OpStr),
        sub_string(Str, _, After, 0, ArgsStr),
        atom_string(Op, OpStr),
        split_string(ArgsStr, ",", " \t", ArgList),
        maplist(atom_string, ArgAtoms, ArgList),
        Term =.. [Op|ArgAtoms]
    ;   atom_string(Term, Str)
    ).

init_state(Goal, Code, Labels, wam_state(PC, Regs, [], [], [], halt, [], Code, Labels)) :-
    Goal =.. [Pred|Args],
    length(Args, Arity),
    format(atom(Target), "~w/~w", [Pred, Arity]),
    (   get_assoc(Target, Labels, PC) -> true
    ;   PC = 1 % Start at beginning if no label found
    ),
    build_regs(Args, 1, Regs).

build_regs([], _, R) :- empty_assoc(R).
build_regs([A|As], I, R) :-
    format(atom(Key), "A~w", [I]),
    NI is I + 1,
    build_regs(As, NI, R0),
    put_assoc(Key, R0, A, R).

run_loop(S, Sf) :-
    S = wam_state(PC, _, _, _, _, _, _, _, _),
    (   PC == halt -> Sf = S
    ;   (   fetch_instr(S, Instr)
        ->  (   step_wam(Instr, S, S1)
            ->  run_loop(S1, Sf)
            ;   (   backtrack(S, S_BT) -> run_loop(S_BT, Sf)
                ;   fail
                )
            )
        ;   fail
        )
    ).

fetch_instr(wam_state(PC, _, _, _, _, _, _, Code, _), Instr) :-
    nth1(PC, Code, Instr).

backtrack(wam_state(_, _, _, _, _, _, [cp(NextPC, R, S, CP, T)|CPs], Code, L), wam_state(NextPC, R, S, [], T, CP, CPs, Code, L)).

%% step_wam(+Instruction, +StateIn, -StateOut)
step_wam(get_constant(C, Ai), wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, R, S, H, T, CP, CPS, Code, L)) :-
    (   get_assoc(Ai, R, Val)
    ->  (Val == C -> NPC is PC + 1 ; fail)
    ;   fail
    ).

step_wam(get_variable(Xn, Ai), wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, NR, S, H, T, CP, CPS, Code, L)) :-
    get_assoc(Ai, R, Val),
    put_assoc(Xn, R, Val, NR),
    NPC is PC + 1.

step_wam(get_value(Xn, Ai), wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, R, S, H, T, CP, CPS, Code, L)) :-
    get_assoc(Ai, R, ValA),
    get_assoc(Xn, R, ValX),
    (ValA == ValX -> NPC is PC + 1 ; fail).

step_wam(put_constant(C, Ai), wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, NR, S, H, T, CP, CPS, Code, L)) :-
    put_assoc(Ai, R, C, NR),
    NPC is PC + 1.

step_wam(put_variable(Xn, Ai), wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, NR, S, H, T, CP, CPS, Code, L)) :-
    format(atom(NewVar), "_V~w", [PC]),
    put_assoc(Xn, R, NewVar, R1),
    put_assoc(Ai, R1, NewVar, NR),
    NPC is PC + 1.

step_wam(put_value(Xn, Ai), wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, NR, S, H, T, CP, CPS, Code, L)) :-
    get_assoc(Xn, R, Val),
    put_assoc(Ai, R, Val, NR),
    NPC is PC + 1.

step_wam(call(P, _), wam_state(PC, R, S, H, T, _, CPS, Code, L), wam_state(NPC, R, S, H, T, NCP, CPS, Code, L)) :-
    get_assoc(P, L, NPC),
    NCP is PC + 1.

step_wam(execute(P), wam_state(_, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, R, S, H, T, CP, CPS, Code, L)) :-
    get_assoc(P, L, NPC).

step_wam(proceed, wam_state(_, R, S, H, T, CP, CPS, Code, L), wam_state(CP, R, S, H, T, halt, CPS, Code, L)).

step_wam(allocate, wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, R, [env(CP)|S], H, T, CP, CPS, Code, L)) :-
    NPC is PC + 1.

step_wam(deallocate, wam_state(PC, R, [env(OldCP)|S], H, T, _, CPS, Code, L), wam_state(NPC, R, S, H, T, OldCP, CPS, Code, L)) :-
    NPC is PC + 1.

step_wam(try_me_else(NextL), wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, R, S, H, T, CP, [cp(NextPC, R, S, CP, T)|CPS], Code, L)) :-
    get_assoc(NextL, L, NextPC),
    NPC is PC + 1.

step_wam(trust_me, wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, R, S, H, T, CP, NCPS, Code, L)) :-
    (CPS = [_|NCPS] -> true ; NCPS = CPS),
    NPC is PC + 1.

step_wam(retry_me_else(NextL), wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, R, S, H, T, CP, NCPS, Code, L)) :-
    get_assoc(NextL, L, NextPC),
    (CPS = [_|Rest] -> NCPS = [cp(NextPC, R, S, CP, T)|Rest] ; NCPS = [cp(NextPC, R, S, CP, T)]),
    NPC is PC + 1.

test_wam_runtime :-
    Code = [
        'parent/2:',
        get_constant(alice, 'A1'),
        get_constant(bob, 'A2'),
        proceed
    ],
    execute_wam(Code, parent(alice, bob), _),
    format('WAM Runtime Test PASS~n').
