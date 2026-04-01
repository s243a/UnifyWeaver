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
    % Deterministic split for opcode/args
    (   once(sub_string(Str, Before, 1, After, " ")) ->
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

%% backtrack(+StateIn, -StateOut)
%  Restores state from choice point and unwinds the trail.
%  The trail records bindings as trail(Key, OldValue) entries made since the
%  choice point was created. On backtrack, we restore registers to their
%  saved state from the choice point, then apply trail entries to undo any
%  bindings that leaked into the saved register set.
backtrack(wam_state(_, _, _, _, Trail, _, [cp(NextPC, R, S, CP, SavedTrail)|CPs], Code, L),
          wam_state(NextPC, RestoredR, S, [], SavedTrail, CP, CPs, Code, L)) :-
    unwind_trail(Trail, SavedTrail, R, RestoredR).

%% unwind_trail(+CurrentTrail, +SavedTrail, +Regs, -RestoredRegs)
%  Undoes bindings recorded on the trail since the choice point was created.
%  Trail entries newer than SavedTrail are unwound in reverse order.
unwind_trail(Trail, SavedTrail, Regs, RestoredRegs) :-
    trail_diff(Trail, SavedTrail, NewEntries),
    undo_bindings(NewEntries, Regs, RestoredRegs).

trail_diff(Trail, Trail, []) :- !.
trail_diff([], _, []) :- !.
trail_diff([Entry|Rest], SavedTrail, [Entry|Diff]) :-
    trail_diff(Rest, SavedTrail, Diff).

undo_bindings([], Regs, Regs).
undo_bindings([trail(Key, OldValue)|Rest], Regs, RestoredRegs) :-
    (   OldValue == unbound
    ->  remove_assoc_key(Key, Regs, _, R1)
    ;   put_assoc(Key, Regs, OldValue, R1)
    ),
    undo_bindings(Rest, R1, RestoredRegs).

%% remove_assoc_key(+Key, +Assoc, -Value, -NewAssoc)
%  Remove a key from an assoc. If key not found, return assoc unchanged.
remove_assoc_key(Key, Assoc, Value, NewAssoc) :-
    (   get_assoc(Key, Assoc, Value)
    ->  put_assoc(Key, Assoc, '$deleted', NewAssoc)
    ;   NewAssoc = Assoc, Value = unbound
    ).

%% step_wam(+Instruction, +StateIn, -StateOut)
step_wam(get_constant(C, Ai), wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, R, S, H, T, CP, CPS, Code, L)) :-
    (   get_assoc(Ai, R, Val)
    ->  (Val == C -> NPC is PC + 1 ; fail)
    ;   fail
    ).

step_wam(get_variable(Xn, Ai), wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, NR, S, H, NT, CP, CPS, Code, L)) :-
    get_assoc(Ai, R, Val),
    trail_binding(Xn, R, T, NT),
    put_assoc(Xn, R, Val, NR),
    NPC is PC + 1.

step_wam(get_value(Xn, Ai), wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, R, S, H, T, CP, CPS, Code, L)) :-
    get_assoc(Ai, R, ValA),
    get_assoc(Xn, R, ValX),
    (ValA == ValX -> NPC is PC + 1 ; fail).

step_wam(put_constant(C, Ai), wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, NR, S, H, NT, CP, CPS, Code, L)) :-
    trail_binding(Ai, R, T, NT),
    put_assoc(Ai, R, C, NR),
    NPC is PC + 1.

step_wam(put_variable(Xn, Ai), wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, NR, S, H, NT, CP, CPS, Code, L)) :-
    format(atom(NewVar), "_V~w", [PC]),
    trail_binding(Xn, R, T, T1),
    trail_binding(Ai, R, T1, NT),
    put_assoc(Xn, R, NewVar, R1),
    put_assoc(Ai, R1, NewVar, NR),
    NPC is PC + 1.

step_wam(put_value(Xn, Ai), wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, NR, S, H, NT, CP, CPS, Code, L)) :-
    get_assoc(Xn, R, Val),
    trail_binding(Ai, R, T, NT),
    put_assoc(Ai, R, Val, NR),
    NPC is PC + 1.

%% put_structure F/N, Ai — begins constructing a compound term on the heap.
%  Allocates a structure cell str(F/N) on the heap, stores the heap address
%  in Ai, and enters "write mode" for subsequent set_variable/set_value.
step_wam(put_structure(FN, Ai), wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, NR, S, NH, T, CP, CPS, Code, L)) :-
    length(H, Addr),
    append(H, [str(FN)], NH),
    put_assoc(Ai, R, ref(Addr), NR),
    NPC is PC + 1.

%% set_variable Xn — pushes a new unbound variable onto the heap and binds Xn to it.
step_wam(set_variable(Xn), wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, NR, S, NH, T, CP, CPS, Code, L)) :-
    length(H, Addr),
    format(atom(Var), "_H~w", [Addr]),
    append(H, [Var], NH),
    put_assoc(Xn, R, Var, NR),
    NPC is PC + 1.

%% set_value Xn — pushes the value of Xn onto the heap.
step_wam(set_value(Xn), wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, R, S, NH, T, CP, CPS, Code, L)) :-
    get_assoc(Xn, R, Val),
    append(H, [Val], NH),
    NPC is PC + 1.

%% set_constant C — pushes a constant value onto the heap.
step_wam(set_constant(C), wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, R, S, NH, T, CP, CPS, Code, L)) :-
    append(H, [C], NH),
    NPC is PC + 1.

% NOTE: call/2 overwrites CP with the return address (PC+1). This is safe
% because the compiler emits allocate (which saves CP to the environment
% stack) before any call instruction in multi-goal bodies (N > 1 guard in
% compile_body_goals). Single-goal bodies use execute instead of call,
% so the outer CP is never lost.
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

%% trail_binding(+Key, +Regs, +TrailIn, -TrailOut)
%  Records the old value of Key (or 'unbound' if absent) on the trail.
trail_binding(Key, Regs, Trail, [trail(Key, OldValue)|Trail]) :-
    (   get_assoc(Key, Regs, OldValue) -> true
    ;   OldValue = unbound
    ).

test_wam_runtime :-
    Code = [
        'parent/2:',
        get_constant(alice, 'A1'),
        get_constant(bob, 'A2'),
        proceed
    ],
    execute_wam(Code, parent(alice, bob), _),
    format('WAM Runtime Test PASS~n').
