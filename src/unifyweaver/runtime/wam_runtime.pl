:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wam_runtime.pl - Enhanced Symbolic WAM Emulator
% This module provides a VM to execute symbolic WAM instructions.

:- module(wam_runtime, [
    execute_wam/3,       % +Instructions, +Query, -Results
    test_wam_runtime/0
]).

:- use_module(library(assoc)).
:- use_module(library(lists)).

/** <module> WAM Runtime Emulator

A symbolic emulator for the Warren Abstract Machine.
State structure: wam_state(PC, Regs, Stack, Heap, Trail, CP)
*/

%% execute_wam(+Instructions, +Goal, -FinalRegs)
execute_wam(Instructions, Goal, FinalRegs) :-
    init_state(Goal, S0),
    run_loop(Instructions, S0, Sf),
    Sf = wam_state(_, FinalRegs, _, _, _, _).

init_state(Goal, wam_state(Label, Regs, [], [], [], [])) :-
    Goal =.. [Pred|Args],
    length(Args, Arity),
    format(atom(Label), "~w/~w", [Pred, Arity]),
    build_regs(Args, 1, Regs).

build_regs([], _, R) :- empty_assoc(R).
build_regs([A|As], I, R) :-
    format(atom(Key), "A~w", [I]),
    NI is I + 1,
    build_regs(As, NI, R0),
    put_assoc(Key, R0, A, R).

run_loop(Instructions, S, Sf) :-
    S = wam_state(PC, _, _, _, _, _),
    (   PC == halt -> Sf = S
    ;   fetch_instr(Instructions, PC, Instr),
        step_wam(Instr, S, S1),
        run_loop(Instructions, S1, Sf)
    ).

fetch_instr(Instructions, PC, Instr) :-
    (   member(Line, Instructions),
        Line = (PC : Instr) -> true
    ;   member(Line, Instructions),
        atom(Line),
        sub_atom(Line, _, _, 0, ':'),
        atom_concat(PC, ':', Line) -> 
        % Found label, get next instruction
        next_line(Instructions, Line, Instr)
    ).

next_line([L, I|_], L, I) :- !.
next_line([_|Rest], L, I) :- next_line(Rest, L, I).

%% step_wam(+Instruction, +StateIn, -StateOut)
step_wam(get_constant(C, Ai), wam_state(PC, R, S, H, T, CP), wam_state(NPC, R, S, H, T, CP)) :-
    get_assoc(Ai, R, Val),
    (Val == C -> true ; fail),
    next_pc(PC, NPC).

step_wam(proceed, wam_state(_, R, S, H, T, CP), wam_state(halt, R, S, H, T, CP)).

step_wam(allocate, wam_state(PC, R, S, H, T, CP), wam_state(NPC, R, [env|S], H, T, CP)) :-
    next_pc(PC, NPC).

step_wam(deallocate, wam_state(PC, R, [_|S], H, T, CP), wam_state(NPC, R, S, H, T, CP)) :-
    next_pc(PC, NPC).

% Simplified PC increment
next_pc(PC, NPC) :-
    atom_concat(PC, '_next', NPC).

test_wam_runtime :-
    % Manual test of step_wam logic
    empty_assoc(R0),
    put_assoc('A1', R0, alice, R1),
    put_assoc('A2', R1, bob, R2),
    S0 = wam_state('parent/2', R2, [], [], [], []),
    step_wam(get_constant(alice, 'A1'), S0, S1),
    step_wam(get_constant(bob, 'A2'), S1, S2),
    step_wam(proceed, S2, wam_state(halt, _, _, _, _, _)),
    format('WAM Runtime Basic Execution: PASS~n').
