:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wam_runtime.pl - Enhanced Symbolic WAM Emulator
% This module provides a VM to execute symbolic WAM instructions.

:- module(wam_runtime, [
    execute_wam/3,       % +InstructionsString, +Query, -Results
    solve_wam/4,         % +InstructionsString, +Query, +VarNames, -Bindings
    findall_wam/4,       % +InstructionsString, +Query, +VarNames, -AllBindings
    test_wam_runtime/0
]).

:- use_module(library(assoc)).
:- use_module(library(lists)).
:- discontiguous step_wam/3.

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
        maplist(parse_arg, ArgList, ArgAtoms),
        Term =.. [Op|ArgAtoms]
    ;   atom_string(Term, Str)
    ).

%% parse_arg(+String, -Value)
%  Converts a string argument to an atom, or a number if it looks numeric.
parse_arg(Str, Val) :-
    (   number_string(Num, Str)
    ->  Val = Num
    ;   atom_string(Val, Str)
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
%% backtrack restores state from the top choice point but does NOT pop it.
%% The choice point is removed by trust_me, or updated by retry_me_else.
backtrack(wam_state(_, _, _, _, Trail, _, [cp(NextPC, R, S, CP, SavedTrail)|CPs], Code, L),
          wam_state(NextPC, RestoredR, S, [], SavedTrail, CP, [cp(NextPC, R, S, CP, SavedTrail)|CPs], Code, L)) :-
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
step_wam(get_constant(C, Ai), wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, NR, S, H, NT, CP, CPS, Code, L)) :-
    get_assoc(Ai, R, Val),
    (   Val == C
    ->  NR = R, NT = T, NPC is PC + 1
    ;   is_unbound_var(Val)
    ->  % Unify: bind the variable to the constant
        trail_binding(Ai, R, T, NT),
        put_assoc(Ai, R, C, NR),
        NPC is PC + 1
    ;   fail
    ).

step_wam(get_variable(Xn, Ai), wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, NR, NS, H, NT, CP, CPS, Code, L)) :-
    get_assoc(Ai, R, Val),
    trail_binding(Xn, R, T, NT),
    put_reg(Xn, Val, R, NR, S, NS),
    NPC is PC + 1.

step_wam(get_value(Xn, Ai), wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, NR, NS, H, NT, CP, CPS, Code, L)) :-
    get_assoc(Ai, R, ValA),
    get_reg(Xn, R, S, ValX),
    (   ValA == ValX
    ->  NR = R, NS = S, NT = T, NPC is PC + 1
    ;   is_unbound_var(ValA)
    ->  trail_binding(Ai, R, T, NT),
        put_assoc(Ai, R, ValX, NR), NS = S,
        NPC is PC + 1
    ;   is_unbound_var(ValX)
    ->  trail_binding(Xn, R, T, NT),
        put_reg(Xn, ValA, R, NR, S, NS),
        NPC is PC + 1
    ;   fail
    ).

%% get_structure F/N, Ai — two modes:
%  Read mode: Ai holds a compound term — verify functor/arity and push sub-args
%  as unify_ctx for subsequent unify_* instructions to match.
%  Write mode: Ai holds an unbound variable — allocate a structure on the heap,
%  bind Ai to it, and push write_ctx so unify_* instructions build sub-args.
step_wam(get_structure(FN, Ai), wam_state(PC, R, S, H, T, CP, CPS, Code, L), StateOut) :-
    get_assoc(Ai, R, Val),
    (   is_unbound_var(Val)
    ->  % Write mode: construct structure on heap
        length(H, Addr),
        append(H, [str(FN)], NH),
        trail_binding(Ai, R, T, NT),
        put_assoc(Ai, R, ref(Addr), NR),
        atom_string(FN, FNStr),
        split_string(FNStr, "/", "", [_, ArStr]),
        number_string(WriteArity, ArStr),
        NPC is PC + 1,
        StateOut = wam_state(NPC, NR, [write_ctx(WriteArity)|S], NH, NT, CP, CPS, Code, L)
    ;   Val = ref(Addr)
    ->  % Read mode on heap-constructed structure: look up str(F/N) on heap
        nth0(Addr, H, str(FN)),
        atom_string(FN, FNStr),
        split_string(FNStr, "/", "", [_, ArStr]),
        number_string(HeapArity, ArStr),
        StartIdx is Addr + 1,
        heap_subargs(H, StartIdx, HeapArity, SubArgs),
        NPC is PC + 1,
        StateOut = wam_state(NPC, R, [unify_ctx(SubArgs)|S], H, T, CP, CPS, Code, L)
    ;   % Read mode: match existing Prolog compound term
        Val =.. [F|SubArgs],
        length(SubArgs, Arity),
        format(atom(FN_check), "~w/~w", [F, Arity]),
        FN_check == FN,
        NPC is PC + 1,
        StateOut = wam_state(NPC, R, [unify_ctx(SubArgs)|S], H, T, CP, CPS, Code, L)
    ).

%% unify_variable Xn — read mode: bind next sub-arg to Xn.
%%                     write mode: create new var on heap and bind Xn to it.
step_wam(unify_variable(Xn), wam_state(PC, R, [unify_ctx([Arg|RestArgs])|S], H, T, CP, CPS, Code, L),
         wam_state(NPC, NR, NewS, H, NT, CP, CPS, Code, L)) :-
    trail_binding(Xn, R, T, NT),
    (   RestArgs == []
    ->  BaseS = S
    ;   BaseS = [unify_ctx(RestArgs)|S]
    ),
    put_reg(Xn, Arg, R, NR, BaseS, NewS),
    NPC is PC + 1.
step_wam(unify_variable(Xn), wam_state(PC, R, [write_ctx(N)|S], H, T, CP, CPS, Code, L),
         wam_state(NPC, NR, NewS, NH, T, CP, CPS, Code, L)) :-
    N > 0,
    length(H, Addr),
    format(atom(Var), "_H~w", [Addr]),
    append(H, [Var], NH),
    N1 is N - 1,
    (   N1 == 0 -> BaseS = S ; BaseS = [write_ctx(N1)|S] ),
    put_reg(Xn, Var, R, NR, BaseS, NewS),
    NPC is PC + 1.

%% unify_value Xn — read mode: check next sub-arg matches Xn.
%%                   write mode: push value of Xn onto heap.
step_wam(unify_value(Xn), wam_state(PC, R, [unify_ctx([Arg|RestArgs])|S], H, T, CP, CPS, Code, L),
         wam_state(NPC, R, NewS, H, T, CP, CPS, Code, L)) :-
    get_reg(Xn, R, S, Val),
    Val == Arg,
    (   RestArgs == []
    ->  NewS = S
    ;   NewS = [unify_ctx(RestArgs)|S]
    ),
    NPC is PC + 1.
step_wam(unify_value(Xn), wam_state(PC, R, [write_ctx(N)|S], H, T, CP, CPS, Code, L),
         wam_state(NPC, R, NewS, NH, T, CP, CPS, Code, L)) :-
    N > 0,
    get_reg(Xn, R, S, Val),
    append(H, [Val], NH),
    N1 is N - 1,
    (   N1 == 0 -> NewS = S ; NewS = [write_ctx(N1)|S] ),
    NPC is PC + 1.

%% unify_constant C — read mode: check next sub-arg equals C.
%%                     write mode: push C onto heap.
step_wam(unify_constant(C), wam_state(PC, R, [unify_ctx([Arg|RestArgs])|S], H, T, CP, CPS, Code, L),
         wam_state(NPC, R, NewS, H, T, CP, CPS, Code, L)) :-
    Arg == C,
    (   RestArgs == []
    ->  NewS = S
    ;   NewS = [unify_ctx(RestArgs)|S]
    ),
    NPC is PC + 1.
step_wam(unify_constant(C), wam_state(PC, R, [write_ctx(N)|S], H, T, CP, CPS, Code, L),
         wam_state(NPC, R, NewS, NH, T, CP, CPS, Code, L)) :-
    N > 0,
    append(H, [C], NH),
    N1 is N - 1,
    (   N1 == 0 -> NewS = S ; NewS = [write_ctx(N1)|S] ),
    NPC is PC + 1.

step_wam(put_constant(C, Ai), wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, NR, S, H, NT, CP, CPS, Code, L)) :-
    trail_binding(Ai, R, T, NT),
    put_assoc(Ai, R, C, NR),
    NPC is PC + 1.

step_wam(put_variable(Xn, Ai), wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, NR, NS, H, NT, CP, CPS, Code, L)) :-
    format(atom(NewVar), "_V~w", [PC]),
    trail_binding(Xn, R, T, T1),
    trail_binding(Ai, R, T1, NT),
    put_reg(Xn, NewVar, R, R1, S, S1),
    put_assoc(Ai, R1, NewVar, NR),
    (   is_y_reg(Xn) -> NS = S1 ; NS = S
    ),
    NPC is PC + 1.

step_wam(put_value(Xn, Ai), wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, NR, S, H, NT, CP, CPS, Code, L)) :-
    get_reg(Xn, R, S, Val),
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
step_wam(set_variable(Xn), wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, NR, NS, NH, T, CP, CPS, Code, L)) :-
    length(H, Addr),
    format(atom(Var), "_H~w", [Addr]),
    append(H, [Var], NH),
    put_reg(Xn, Var, R, NR, S, NS),
    NPC is PC + 1.

%% set_value Xn — pushes the value of Xn onto the heap.
step_wam(set_value(Xn), wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, R, S, NH, T, CP, CPS, Code, L)) :-
    get_reg(Xn, R, S, Val),
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

%% allocate — creates an environment frame with saved CP and empty Yi storage.
step_wam(allocate, wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, R, [env(CP, YRegs)|S], H, T, CP, CPS, Code, L)) :-
    empty_assoc(YRegs),
    NPC is PC + 1.

%% deallocate — restores CP from the environment frame.
step_wam(deallocate, wam_state(PC, R, [env(OldCP, _)|S], H, T, _, CPS, Code, L), wam_state(NPC, R, S, H, T, OldCP, CPS, Code, L)) :-
    NPC is PC + 1.

%% switch_on_constant — first-argument indexing.
%  The instruction is parsed as switch_on_constant('key1:label1', 'key2:label2', ...).
%  Looks up A1's value in the entries and jumps directly to the matching clause.
%  If A1 is unbound or not in the table, falls through to the next instruction.
step_wam(Instr, wam_state(PC, R, S, H, T, CP, CPS, Code, L),
         wam_state(NPC, R, S, H, T, CP, CPS, Code, L)) :-
    Instr =.. [switch_on_constant|Entries],
    Entries \= [],
    (   get_assoc('A1', R, Val),
        \+ is_unbound_var(Val),
        lookup_index(Val, Entries, L, TargetPC)
    ->  NPC = TargetPC
    ;   NPC is PC + 1
    ).

lookup_index(Val, [Entry|Rest], Labels, TargetPC) :-
    atom_string(Entry, Str),
    (   once(sub_string(Str, Before, 1, After, ":"))
    ->  sub_string(Str, 0, Before, _, KeyStr),
        sub_string(Str, _, After, 0, LabelStr),
        parse_arg(KeyStr, Key),
        atom_string(Label, LabelStr)
    ;   fail
    ),
    (   Key == Val
    ->  (   Label == default
        ->  fail  % default = first clause = fall through
        ;   get_assoc(Label, Labels, TargetPC)
        )
    ;   lookup_index(Val, Rest, Labels, TargetPC)
    ).

%% switch_on_structure — indexes on compound first argument's functor/arity.
step_wam(Instr, wam_state(PC, R, S, H, T, CP, CPS, Code, L),
         wam_state(NPC, R, S, H, T, CP, CPS, Code, L)) :-
    Instr =.. [switch_on_structure|Entries],
    Entries \= [],
    (   get_assoc('A1', R, Val),
        \+ is_unbound_var(Val),
        compound(Val),
        Val =.. [F|SubArgs],
        length(SubArgs, SArity),
        format(atom(FN), "~w/~w", [F, SArity]),
        lookup_index(FN, Entries, L, TargetPC)
    ->  NPC = TargetPC
    ;   NPC is PC + 1
    ).

%% switch_on_term — type-based dispatch for mixed first arguments.
%  Parsed as switch_on_term('constant:...', 'structure:...').
step_wam(Instr, wam_state(PC, R, S, H, T, CP, CPS, Code, L),
         wam_state(NPC, R, S, H, T, CP, CPS, Code, L)) :-
    Instr =.. [switch_on_term|Args],
    Args \= [],
    (   get_assoc('A1', R, Val),
        \+ is_unbound_var(Val)
    ->  (   atomic(Val)
        ->  find_term_index_section('constant', Args, Entries),
            (   lookup_index(Val, Entries, L, TargetPC)
            ->  NPC = TargetPC
            ;   NPC is PC + 1
            )
        ;   compound(Val)
        ->  Val =.. [F|SubArgs],
            length(SubArgs, SArity),
            format(atom(FN), "~w/~w", [F, SArity]),
            find_term_index_section('structure', Args, Entries),
            (   lookup_index(FN, Entries, L, TargetPC)
            ->  NPC = TargetPC
            ;   NPC is PC + 1
            )
        ;   NPC is PC + 1
        )
    ;   NPC is PC + 1
    ).

find_term_index_section(Type, Args, Entries) :-
    atom_string(Type, TypeStr),
    member(Arg, Args),
    atom_string(Arg, ArgStr),
    atom_concat(TypeStr, ':', Prefix),
    atom_string(Prefix, PrefixStr),
    sub_string(ArgStr, 0, _, After, PrefixStr),
    sub_string(ArgStr, _, After, 0, Rest),
    (   Rest \= "none"
    ->  split_string(Rest, ",", " ", Pairs),
        maplist([PairStr, Entry]>>(
            atom_string(PA, PairStr),
            Entry = PA
        ), Pairs, Entries)
    ;   Entries = []
    ), !.
find_term_index_section(_, _, []).

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

%% heap_subargs(+Heap, +StartIdx, +Count, -SubArgs)
%  Extracts Count elements from the heap starting at StartIdx.
heap_subargs(_, _, 0, []) :- !.
heap_subargs(Heap, Idx, N, [Val|Rest]) :-
    nth0(Idx, Heap, Val),
    NextIdx is Idx + 1,
    N1 is N - 1,
    heap_subargs(Heap, NextIdx, N1, Rest).

%% is_unbound_var(+Val)
%  Checks if a value represents a WAM unbound variable (generated by put_variable)
%  or is an actual Prolog unbound variable (from query arguments).
is_unbound_var(Val) :- var(Val), !.
is_unbound_var(Val) :-
    atom(Val),
    sub_atom(Val, 0, 2, _, '_V').

%% trail_binding(+Key, +Regs, +TrailIn, -TrailOut)
%  Records the old value of Key (or 'unbound' if absent) on the trail.
trail_binding(Key, Regs, Trail, [trail(Key, OldValue)|Trail]) :-
    (   get_assoc(Key, Regs, OldValue) -> true
    ;   OldValue = unbound
    ).

%% Yi register helpers — read/write permanent variables in the environment frame.
is_y_reg(Reg) :-
    atom(Reg),
    sub_atom(Reg, 0, 1, _, 'Y').

%% get_reg(+Reg, +Regs, +Stack, -Value)
%  Gets a register value. Yi registers are read from the top environment frame.
get_reg(Reg, _R, Stack, Val) :-
    is_y_reg(Reg), !,
    member(env(_, YRegs), Stack), !,
    get_assoc(Reg, YRegs, Val).
get_reg(Reg, R, _, Val) :-
    get_assoc(Reg, R, Val).

%% put_reg(+Reg, +Val, +Regs, -NewRegs, +Stack, -NewStack)
%  Sets a register value. Yi registers are written to the top environment frame.
put_reg(Reg, Val, R, R, Stack, NewStack) :-
    is_y_reg(Reg), !,
    update_top_env(Stack, Reg, Val, NewStack).
put_reg(Reg, Val, R, NR, Stack, Stack) :-
    put_assoc(Reg, R, Val, NR).

update_top_env([env(CP, YRegs)|Rest], Reg, Val, [env(CP, NewYRegs)|Rest]) :-
    put_assoc(Reg, YRegs, Val, NewYRegs).

%% =====================================================
%% Built-in predicate support
%% =====================================================

%% builtin_call Op, Arity — evaluates a built-in predicate using register values.
%% Arithmetic: is/2 evaluates RHS and stores in LHS register.
step_wam(builtin_call('is/2', 2), wam_state(PC, R, S, H, T, CP, CPS, Code, L),
         wam_state(NPC, NR, S, H, NT, CP, CPS, Code, L)) :-
    get_assoc('A2', R, Expr),
    eval_arith(Expr, R, S, H, Result),
    get_assoc('A1', R, LHS),
    (   is_unbound_var(LHS)
    ->  trail_binding('A1', R, T, NT),
        put_assoc('A1', R, Result, NR)
    ;   number(LHS), LHS =:= Result
    ->  NR = R, NT = T
    ;   fail
    ),
    NPC is PC + 1.

%% Comparison builtins: >/2, </2, >=/2, =</2, =:=/2, =\=/2
step_wam(builtin_call(Op, 2), wam_state(PC, R, S, H, T, CP, CPS, Code, L),
         wam_state(NPC, R, S, H, T, CP, CPS, Code, L)) :-
    is_comparison_op(Op),
    get_assoc('A1', R, V1),
    get_assoc('A2', R, V2),
    eval_arith(V1, R, S, H, N1),
    eval_arith(V2, R, S, H, N2),
    apply_comparison(Op, N1, N2),
    NPC is PC + 1.

%% Term equality: ==/2, \==/2
step_wam(builtin_call('==/2', 2), wam_state(PC, R, S, H, T, CP, CPS, Code, L),
         wam_state(NPC, R, S, H, T, CP, CPS, Code, L)) :-
    get_assoc('A1', R, V1),
    get_assoc('A2', R, V2),
    V1 == V2,
    NPC is PC + 1.

step_wam(builtin_call('\\==/2', 2), wam_state(PC, R, S, H, T, CP, CPS, Code, L),
         wam_state(NPC, R, S, H, T, CP, CPS, Code, L)) :-
    get_assoc('A1', R, V1),
    get_assoc('A2', R, V2),
    V1 \== V2,
    NPC is PC + 1.

step_wam(builtin_call('\\=/2', 2), wam_state(PC, R, S, H, T, CP, CPS, Code, L),
         wam_state(NPC, R, S, H, T, CP, CPS, Code, L)) :-
    get_assoc('A1', R, V1),
    get_assoc('A2', R, V2),
    V1 \== V2,
    NPC is PC + 1.

is_comparison_op('>/2').
is_comparison_op('</2').
is_comparison_op('>=/2').
is_comparison_op('=</2').
is_comparison_op('=:=/2').
is_comparison_op('=\\=/2').

apply_comparison('>/2', N1, N2) :- N1 > N2.
apply_comparison('</2', N1, N2) :- N1 < N2.
apply_comparison('>=/2', N1, N2) :- N1 >= N2.
apply_comparison('=</2', N1, N2) :- N1 =< N2.
apply_comparison('=:=/2', N1, N2) :- N1 =:= N2.
apply_comparison('=\\=/2', N1, N2) :- N1 =\= N2.

%% Type check builtins
step_wam(builtin_call(Op, 1), wam_state(PC, R, S, H, T, CP, CPS, Code, L),
         wam_state(NPC, R, S, H, T, CP, CPS, Code, L)) :-
    is_type_check_op(Op),
    get_assoc('A1', R, Val),
    apply_type_check(Op, Val),
    NPC is PC + 1.

is_type_check_op('integer/1').
is_type_check_op('float/1').
is_type_check_op('number/1').
is_type_check_op('atom/1').
is_type_check_op('compound/1').
is_type_check_op('var/1').
is_type_check_op('nonvar/1').
is_type_check_op('is_list/1').

apply_type_check('integer/1', V) :- integer(V).
apply_type_check('float/1', V) :- float(V).
apply_type_check('number/1', V) :- number(V).
apply_type_check('atom/1', V) :- atom(V), \+ is_unbound_var(V).
apply_type_check('compound/1', V) :- compound(V).
apply_type_check('var/1', V) :- is_unbound_var(V).
apply_type_check('nonvar/1', V) :- \+ is_unbound_var(V).
apply_type_check('is_list/1', V) :- is_list(V).

%% Control builtins
step_wam(builtin_call('true/0', 0), wam_state(PC, R, S, H, T, CP, CPS, Code, L),
         wam_state(NPC, R, S, H, T, CP, CPS, Code, L)) :-
    NPC is PC + 1.

step_wam(builtin_call('fail/0', 0), _, _) :- fail.

%% Cut — discard all choice points
step_wam(builtin_call('!/0', 0), wam_state(PC, R, S, H, T, CP, _, Code, L),
         wam_state(NPC, R, S, H, T, CP, [], Code, L)) :-
    NPC is PC + 1.

%% Negation-as-failure \+/1 — not directly callable via builtin_call
%% since it requires executing a sub-goal. For Phase 1, \+ is handled
%% as a guard that always succeeds or fails based on argument truthiness.
step_wam(builtin_call('\\+/1', 1), wam_state(PC, R, S, H, T, CP, CPS, Code, L),
         wam_state(NPC, R, S, H, T, CP, CPS, Code, L)) :-
    get_assoc('A1', R, Val),
    (   Val == fail ; Val == false ; is_unbound_var(Val)
    ->  NPC is PC + 1
    ;   fail
    ).

%% List builtins — delegate to Prolog's native list operations.
%% These are handled as runtime builtins since list operations involve
%% recursive structure manipulation better done natively.
step_wam(builtin_call('member/2', 2), wam_state(PC, R, S, H, T, CP, CPS, Code, L),
         wam_state(NPC, R, S, H, T, CP, CPS, Code, L)) :-
    get_assoc('A1', R, Elem),
    get_assoc('A2', R, List),
    member(Elem, List),
    NPC is PC + 1.

step_wam(builtin_call('append/3', 3), wam_state(PC, R, S, H, T, CP, CPS, Code, L),
         wam_state(NPC, NR, S, H, NT, CP, CPS, Code, L)) :-
    get_assoc('A1', R, L1),
    get_assoc('A2', R, L2),
    get_assoc('A3', R, L3),
    (   is_unbound_var(L3)
    ->  append(L1, L2, Result),
        trail_binding('A3', R, T, NT),
        put_assoc('A3', R, Result, NR)
    ;   append(L1, L2, L3),
        NR = R, NT = T
    ),
    NPC is PC + 1.

step_wam(builtin_call('length/2', 2), wam_state(PC, R, S, H, T, CP, CPS, Code, L),
         wam_state(NPC, NR, S, H, NT, CP, CPS, Code, L)) :-
    get_assoc('A1', R, List),
    get_assoc('A2', R, Len),
    (   is_unbound_var(Len)
    ->  length(List, Result),
        trail_binding('A2', R, T, NT),
        put_assoc('A2', R, Result, NR)
    ;   length(List, Len),
        NR = R, NT = T
    ),
    NPC is PC + 1.

%% eval_arith(+Expr, +Regs, +Stack, +Heap, -Result)
%  Evaluates an arithmetic expression. Numbers pass through. Register
%  names are dereferenced. Heap refs are reconstructed. Compounds are evaluated.
eval_arith(Expr, _, _, _, Expr) :- number(Expr), !.
eval_arith(ref(Addr), _, _, Heap, Result) :- !,
    nth0(Addr, Heap, str(FN)),
    atom_string(FN, FNStr),
    split_string(FNStr, "/", "", [_, ArStr]),
    number_string(HArity, ArStr),
    StartIdx is Addr + 1,
    heap_subargs(Heap, StartIdx, HArity, SubArgs),
    atom_string(FN, FNStr2),
    split_string(FNStr2, "/", "", [OpStr, _]),
    atom_string(Op, OpStr),
    maplist([A, V]>>eval_arith(A, t, [], Heap, V), SubArgs, Vals),
    (   Vals = [V1, V2]
    ->  apply_arith_op(Op, V1, V2, Result)
    ;   Vals = [V1]
    ->  apply_arith_unary(Op, V1, Result)
    ;   fail
    ).
eval_arith(Expr, R, S, Heap, Result) :-
    atom(Expr),
    (   is_unbound_var(Expr) -> fail
    ;   (sub_atom(Expr, 0, 1, _, 'A') ; sub_atom(Expr, 0, 1, _, 'X') ; sub_atom(Expr, 0, 1, _, 'Y'))
    ->  get_reg(Expr, R, S, Val), eval_arith(Val, R, S, Heap, Result)
    ;   fail
    ), !.
eval_arith(Expr, R, S, Heap, Result) :-
    compound(Expr),
    Expr =.. [Op|Args],
    maplist({R, S, Heap}/[A, V]>>eval_arith(A, R, S, Heap, V), Args, Vals),
    (   Vals = [V1, V2]
    ->  apply_arith_op(Op, V1, V2, Result)
    ;   Vals = [V1]
    ->  apply_arith_unary(Op, V1, Result)
    ;   fail
    ).

apply_arith_op(+, A, B, R) :- R is A + B.
apply_arith_op(-, A, B, R) :- R is A - B.
apply_arith_op(*, A, B, R) :- R is A * B.
apply_arith_op(/, A, B, R) :- B =\= 0, R is A / B.
apply_arith_op(//, A, B, R) :- B =\= 0, R is A // B.
apply_arith_op(mod, A, B, R) :- B =\= 0, R is A mod B.

apply_arith_unary(-, A, R) :- R is -A.
apply_arith_unary(abs, A, R) :- R is abs(A).

%% =====================================================
%% solve_wam/4 — extract variable bindings from a query
%% =====================================================

%% solve_wam(+Instructions, +Query, +VarNames, -Bindings)
%  VarNames is a list of Name=Var pairs (as from read_term/2 variable_names).
%  Bindings is a list of Name=Value pairs for each query variable.
solve_wam(Instructions, Query, VarNames, Bindings) :-
    prepare_code(Instructions, Code, Labels),
    init_state(Query, Code, Labels, S0),
    run_loop(S0, Sf),
    Sf = wam_state(_, FinalRegs, _, _, _, _, _, _, _),
    extract_bindings(VarNames, Query, FinalRegs, Bindings).

extract_bindings([], _, _, []).
extract_bindings([Name=Var|Rest], Query, Regs, [Name=Value|RestBindings]) :-
    Query =.. [_|Args],
    (   nth1(Idx, Args, A), A == Var
    ->  format(atom(RegKey), "A~w", [Idx]),
        (   get_assoc(RegKey, Regs, Value) -> true ; Value = Var )
    ;   Value = Var
    ),
    extract_bindings(Rest, Query, Regs, RestBindings).

%% =====================================================
%% findall_wam/4 — collect all solutions via backtracking
%% =====================================================

%% findall_wam(+Instructions, +Query, +VarNames, -AllBindings)
%  Returns a list of binding sets, one per solution. Runs the WAM
%  program and collects all successful execution paths by continuing
%  to backtrack after each solution.
findall_wam(Instructions, Query, VarNames, AllBindings) :-
    prepare_code(Instructions, Code, Labels),
    copy_term(Query-VarNames, QCopy-VNCopy),
    init_state(QCopy, Code, Labels, S0),
    run_all_solutions(S0, QCopy, VNCopy, AllBindings).

run_all_solutions(S0, Query, VarNames, AllBindings) :-
    run_all_acc(S0, Query, VarNames, [], RevBindings),
    reverse(RevBindings, AllBindings).

run_all_acc(State, Query, VarNames, Acc, AllBindings) :-
    (   run_loop(State, Sf),
        Sf = wam_state(_, FinalRegs, _, _, _, _, CPS, _, _)
    ->  extract_bindings(VarNames, Query, FinalRegs, Bindings),
        NewAcc = [Bindings|Acc],
        % Try to backtrack for more solutions
        (   CPS = [_|_],
            backtrack(Sf, SBT)
        ->  run_all_acc(SBT, Query, VarNames, NewAcc, AllBindings)
        ;   AllBindings = NewAcc
        )
    ;   AllBindings = Acc
    ).

%% =====================================================
%% Tests
%% =====================================================

test_wam_runtime :-
    Code = [
        'parent/2:',
        get_constant(alice, 'A1'),
        get_constant(bob, 'A2'),
        proceed
    ],
    execute_wam(Code, parent(alice, bob), _),
    format('WAM Runtime Test PASS~n').
