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
    empty_assoc(L0),
    parse_lines(Lines, 1, RawInstructions, L0, Labels),
    % Post-process: convert switch_on_constant to indexed form for O(log n) dispatch
    maplist(index_switch_instr, RawInstructions, IndexedInstructions),
    % Convert list to assoc for O(log n) instruction fetch
    list_to_code_assoc(IndexedInstructions, 1, Instructions).

%% list_to_code_assoc(+InstrList, +StartPC, -CodeAssoc)
%  Converts instruction list to a PC → Instruction assoc for O(log n) fetch.
list_to_code_assoc(InstrList, StartPC, CodeAssoc) :-
    empty_assoc(A0),
    foldl(add_instr_to_assoc, InstrList, StartPC-A0, _-CodeAssoc).

add_instr_to_assoc(Instr, PC-AIn, NPC-AOut) :-
    put_assoc(PC, AIn, Instr, AOut),
    NPC is PC + 1.

%% index_switch_instr(+RawInstr, -IndexedInstr)
%  Converts switch_on_constant(K1:L1, K2:L2, ...) to switch_on_constant_index(Assoc).
index_switch_instr(Instr, Indexed) :-
    (   Instr =.. [Op|Args],
        (Op == switch_on_constant ; Op == switch_on_constant_a2),
        (Args = [Entries], is_list(Entries) -> true ; Entries = Args),
        Entries \= []
    ->  entries_to_assoc(Entries, Assoc),
        Indexed = switch_on_constant_index(Assoc)
    ;   Indexed = Instr
    ).

entries_to_assoc(Entries, Assoc) :-
    empty_assoc(A0),
    foldl(add_entry, Entries, A0, Assoc).

add_entry(Entry, AIn, AOut) :-
    (   once(sub_atom(Entry, Before, 1, After, ':'))
    ->  sub_atom(Entry, 0, Before, _, KeyStr),
        sub_atom(Entry, _, After, 0, LabelStr),
        (atom_number(KeyStr, Key) -> true ; Key = KeyStr),
        put_assoc(Key, AIn, LabelStr, AOut)
    ;   AOut = AIn
    ).

parse_lines([], _, [], L, L).
parse_lines([Line|Rest], PC, Instrs, LIn, LOut) :-
    (   (compound(Line), \+ is_label_term(Line))
    ->  Instrs = [Line|IRest],
        NPC is PC + 1,
        parse_lines(Rest, NPC, IRest, LIn, LOut)
    ;   normalize_line(Line, Normalized),
        (   Normalized == "" -> parse_lines(Rest, PC, Instrs, LIn, LOut)
        ;   is_label(Normalized, Label) ->
            put_assoc(Label, LIn, PC, L1),
            parse_lines(Rest, PC, Instrs, L1, LOut)
        ;   parse_instr(Normalized, Instr) ->
            Instrs = [Instr|IRest],
            NPC is PC + 1,
            parse_lines(Rest, NPC, IRest, LIn, LOut)
        ;   parse_lines(Rest, PC, Instrs, LIn, LOut)
        )
    ).

is_label_term(Line) :-
    atom(Line),
    sub_atom(Line, _, _, 0, ':').

normalize_line(Line, Normalized) :-
    (   atom(Line) -> Str = Line
    ;   string(Line) -> atom_string(Str, Line)
    ;   is_list(Line) -> atomic_list_concat(Line, ' ', Str)
    ;   term_string(Line, S), atom_string(Str, S)
    ),
    split_string(Str, " \t\n\r", " \t\n\r", Parts),
    delete(Parts, "", CleanParts),
    atomic_list_concat(CleanParts, ' ', Normalized).

is_label(Line, Label) :-
    atom_concat(Label, ':', Line).

%% parse_instr(+String, -Term)
parse_instr(Str, Term) :-
    (   once(sub_string(Str, Before, 1, After, " ")) ->
        sub_string(Str, 0, Before, _, OpStr),
        sub_string(Str, _, After, 0, ArgsStr),
        atom_string(Op, OpStr),
        split_string(ArgsStr, ",", " \t", ArgList),
        maplist(parse_arg, ArgList, ArgAtoms),
        Term =.. [Op|ArgAtoms]
    ;   atom_string(Term, Str)
    ).

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
    ;   PC = 1
    ),
    build_regs(Args, 1, Regs),
    % Initialize the WAM variable binding table and counter (global, mutable)
    empty_assoc(EmptyBindings),
    nb_setval(wam_bindings, EmptyBindings),
    nb_setval(wam_var_counter, 0).

% ============================================================================
% Fact Table Registry
% ============================================================================

:- use_module('../core/wam_fact_table').
:- use_module('../core/wam_dict').

:- dynamic wam_fact_table_entry/2.  % Pred/Arity → FactTable

%% wam_register_fact_table(+PredKey, +FactTable)
%  Register a fact table for a predicate (e.g., "category_parent/2").
wam_register_fact_table(PredKey, FactTable) :-
    retractall(wam_fact_table_entry(PredKey, _)),
    assert(wam_fact_table_entry(PredKey, FactTable)).

%% wam_fact_table_registered(+PredKey, -FactTable)
%  Check if a predicate has a registered fact table.
wam_fact_table_registered(PredKey, FactTable) :-
    wam_fact_table_entry(PredKey, FactTable).

%% wam_auto_register_facts(+Predicates)
%  Automatically build and register fact tables for fact predicates.
wam_auto_register_facts([]).
wam_auto_register_facts([Pred/Arity|Rest]) :-
    (   is_fact_predicate(Pred, Arity)
    ->  build_fact_table(Pred, Arity, Table),
        format(atom(Key), "~w/~w", [Pred, Arity]),
        wam_register_fact_table(Key, Table),
        fact_table_size(Table, N),
        format(user_error, '  ~w/~w: ~w facts indexed~n', [Pred, Arity, N])
    ;   true
    ),
    wam_auto_register_facts(Rest).

%% call_fact_predicate(+Pred, +Arity, +FactTable, +StateIn, -StateOut)
%  Execute a fact predicate via indexed table lookup.
%  A1 is the first argument (lookup key). Remaining args are unified.
%  Creates choice points for multiple matches.
call_fact_predicate(_Pred, Arity, FactTable, StateIn, StateOut) :-
    StateIn = wam_state(_, R, S, H, T, CP, CPS, Code, L),
    get_assoc('A1', R, A1Raw),
    wam_deref(A1Raw, A1),
    (   \+ is_wam_unbound(A1)
    ->  % Bound first arg: O(log n) lookup
        wam_dict_lookup_all(A1, FactTable, Matches),
        Matches = [FirstMatch|RestMatches],
        % Save state BEFORE first match for correct backtracking
        (   RestMatches \= []
        ->  wam_save_bindings(SB),
            NCPS = [cp(_, R, S, H, CP, T, fact_retry(A1, FactTable, Arity, RestMatches, CP), SB)|CPS]
        ;   NCPS = CPS
        ),
        % Unify remaining args (A2, A3, ...) AFTER CP creation
        unify_fact_args(FirstMatch, 2, Arity, R, T, NR, NT),
        % Return to caller (proceed equivalent)
        StateOut = wam_state(CP, NR, S, H, NT, halt, NCPS, Code, L)
    ;   % Unbound first arg: iterate all facts (expensive)
        wam_dict_to_list(FactTable, AllEntries),
        all_fact_matches(AllEntries, Arity, AllMatches),
        AllMatches = [FirstKey-FirstArgs|RestAll],
        % Bind A1 to the key
        trail_binding('A1', R, T, T1),
        wam_bind(A1, FirstKey),
        put_assoc('A1', R, FirstKey, R1),
        unify_fact_args(FirstArgs, 2, Arity, R1, T1, NR, NT),
        (   RestAll \= []
        ->  wam_save_bindings(SB),
            NCPS = [cp(_, R, S, H, CP, T, fact_retry_all(FactTable, Arity, RestAll, CP), SB)|CPS]
        ;   NCPS = CPS
        ),
        StateOut = wam_state(CP, NR, S, H, NT, halt, NCPS, Code, L)
    ).

%% unify_fact_args(+ArgValues, +StartIdx, +Arity, +RegsIn, +TrailIn, -RegsOut, -TrailOut)
%  Unify A2..AN with the fact's argument values.
unify_fact_args(_, Idx, Arity, R, T, R, T) :- Idx > Arity, !.
unify_fact_args([Val|Rest], Idx, Arity, R, T, ROut, TOut) :-
    format(atom(RegKey), "A~w", [Idx]),
    (get_assoc(RegKey, R, RegVal) -> true ; RegVal = '$missing'),
    wam_deref(RegVal, DRegVal),
    (   DRegVal == Val
    ->  R1 = R, T1 = T
    ;   is_wam_unbound(DRegVal)
    ->  trail_binding(RegKey, R, T, T1),
        (atom(DRegVal), is_wam_unbound(DRegVal) -> wam_bind(DRegVal, Val) ; true),
        put_assoc(RegKey, R, Val, R1)
    ;   fail  % mismatch
    ),
    NIdx is Idx + 1,
    unify_fact_args(Rest, NIdx, Arity, R1, T1, ROut, TOut).

all_fact_matches([], _, []).
all_fact_matches([Key-ValueLists|Rest], Arity, AllMatches) :-
    maplist([Vs, Key-Vs]>>true, ValueLists, KeyedMatches),
    all_fact_matches(Rest, Arity, RestMatches),
    append(KeyedMatches, RestMatches, AllMatches).

is_wam_unbound(Val) :- var(Val), !.
is_wam_unbound(Val) :-
    atom(Val),
    (   sub_atom(Val, 0, 2, _, '_V')
    ;   sub_atom(Val, 0, 2, _, '_Q')
    ;   sub_atom(Val, 0, 2, _, '_H')
    ).

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
    integer(PC), get_assoc(PC, Code, Instr).

%% backtrack(+StateIn, -StateOut)
backtrack(wam_state(_, _, _, _, Trail, _, [cp(NextPC, R, S, H, CP, SavedTrail, BuiltinState, SavedB)|CPs], Code, L),
          StateOut) :-
    unwind_trail(Trail, SavedTrail, R, RestoredR),
    wam_restore_bindings(SavedB),
    (   nonvar(BuiltinState)
    ->  resume_builtin(BuiltinState, wam_state(NextPC, RestoredR, S, H, SavedTrail, CP, [cp(NextPC, R, S, H, CP, SavedTrail, BuiltinState, SavedB)|CPs], Code, L), StateOut)
    ;   StateOut = wam_state(NextPC, RestoredR, S, H, SavedTrail, CP, [cp(NextPC, R, S, H, CP, SavedTrail, BuiltinState, SavedB)|CPs], Code, L)
    ).

%% resume_builtin(+State, +StateIn, -StateOut)
resume_builtin(member(Elem, ListRaw, PC), StateIn, StateOut) :-
    StateIn = wam_state(_, R, S, H, T, CP, [cp(NextPC, R_orig, S_orig, H_orig, CP_orig, T_orig, _, SB)|CPs], Code, L),
    deref_wam(ListRaw, StateIn, List),
    (   List = [Next|Rest]
    ->  (   Rest == []
        ->  NCPS = CPs
        ;   NCPS = [cp(NextPC, R_orig, S_orig, H_orig, CP_orig, T_orig, member(Elem, Rest, PC), SB)|CPs]
        ),
        State1 = wam_state(PC, R, S, H, T, CP, NCPS, Code, L),
        (   unify_wam(Elem, Next, State1, State2)
        ->  State2 = wam_state(NPC_base, R2, S2, H2, T2, CP2, CPS2, Code2, L2),
            NPC is NPC_base + 1,
            StateOut = wam_state(NPC, R2, S2, H2, T2, CP2, CPS2, Code2, L2)
        ;   backtrack(State1, StateOut)
        )
    ;   fail
    ).

%% resume_builtin for fact_retry: try the next matching fact from the table.
resume_builtin(fact_retry(A1, FactTable, Arity, RestMatches, SavedCP), StateIn, StateOut) :-
    StateIn = wam_state(_, R, S, H, T, CP, [cp(NextPC, R_orig, S_orig, H_orig, CP_orig, T_orig, _, SB)|CPs], Code, L),
    RestMatches = [NextMatch|MoreMatches],
    get_assoc('A2', R, A2Val),
    % Try to unify remaining args with the next match
    (   unify_fact_args(NextMatch, 2, Arity, R, T, NR, NT)
    ->  % Success — create CP for further matches if any
        (   MoreMatches \= []
        ->  wam_save_bindings(SB2),
            NCPS = [cp(NextPC, R_orig, S_orig, H_orig, CP_orig, T_orig, fact_retry(A1, FactTable, Arity, MoreMatches, SavedCP), SB2)|CPs]
        ;   NCPS = CPs
        ),
        % Return to caller
        StateOut = wam_state(SavedCP, NR, S, H, NT, halt, NCPS, Code, L)
    ;   % This match failed — try the next one
        (   MoreMatches \= []
        ->  wam_restore_bindings(SB),
            resume_builtin(fact_retry(A1, FactTable, Arity, MoreMatches, SavedCP),
                StateIn, StateOut)
        ;   fail  % no more matches
        )
    ).

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

remove_assoc_key(Key, Assoc, Value, NewAssoc) :-
    (   del_assoc(Key, Assoc, Value, NewAssoc)
    ->  true
    ;   NewAssoc = Assoc, Value = unbound
    ).

%% step_wam(+Instruction, +StateIn, -StateOut)
step_wam(get_constant(C, Ai), wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, NR, S, H, NT, CP, CPS, Code, L)) :-
    % Read raw register value (before deref) for binding
    (   get_reg(Ai, R, S, RawVal) -> true ; RawVal = '_Vunbound' ),
    wam_deref(RawVal, Val),
    (   Val == C
    ->  NR = R, NT = T, NPC is PC + 1
    ;   is_unbound_var(Val)
    ->  trail_binding(Ai, R, T, NT),
        % Bind the RAW atom name (not the deref'd chain endpoint)
        (   atom(RawVal), is_unbound_var(RawVal)
        ->  wam_bind(RawVal, C)
        ;   true
        ),
        put_assoc(Ai, R, C, NR),
        NPC is PC + 1
    ;   fail
    ).

step_wam(get_variable(Xn, Ai), wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, NR, NS, H, NT, CP, CPS, Code, L)) :-
    get_reg_val(Ai, R, S, Val),
    trail_binding(Xn, R, T, NT),
    put_reg(Xn, Val, R, NR, S, NS),
    NPC is PC + 1.

step_wam(get_value(Xn, Ai), wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, NR, NS, H, NT, CP, CPS, Code, L)) :-
    get_reg_val(Ai, R, S, ValA),
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

step_wam(get_structure(FN, Ai), wam_state(PC, R, S, H, T, CP, CPS, Code, L), StateOut) :-
    get_reg_val(Ai, R, S, Val),
    (   is_unbound_var(Val)
    ->  length(H, Addr),
        Ref = ref(Addr),
        append(H, [str(FN)], NH),
        trail_binding(Ai, R, T, T1),
        (   deref_wam(Val, wam_state(PC, R, S, H, T, CP, CPS, Code, L), Unbound),
            is_unbound_var(Unbound)
        ->  trail_binding(Unbound, R, T1, T2),
            put_reg(Unbound, Ref, R, R1, S, S1),
            put_assoc(Ai, R1, Ref, NR), NS = S1, NT = T2
        ;   put_assoc(Ai, R, Ref, NR), NS = S, NT = T1
        ),
        get_arity(FN, Arity),
        NPC is PC + 1,
        StateOut = wam_state(NPC, NR, [write_ctx(Arity)|NS], NH, NT, CP, CPS, Code, L)
    ;   Val = ref(Addr)
    ->  nth0(Addr, H, Entry), (Entry = str(FN); Entry = FN),
        get_arity(FN, Arity),
        Arity > 0,
        StartIdx is Addr + 1,
        heap_subargs(H, StartIdx, Arity, SubArgs),
        NPC is PC + 1,
        StateOut = wam_state(NPC, R, [unify_ctx(SubArgs)|S], H, T, CP, CPS, Code, L)
    ;   Val =.. [F|SubArgs],
        length(SubArgs, Arity),
        format(atom(FN_check), "~w/~w", [F, Arity]),
        FN_check == FN,
        NPC is PC + 1,
        StateOut = wam_state(NPC, R, [unify_ctx(SubArgs)|S], H, T, CP, CPS, Code, L)
    ).

step_wam(get_list(Ai), wam_state(PC, R, S, H, T, CP, CPS, Code, L), StateOut) :-
    get_reg_val(Ai, R, S, Val),
    (   is_unbound_var(Val)
    ->  % Write mode: construct list on heap
        length(H, Addr),
        Ref = ref(Addr),
        append(H, [str('./2')], NH),
        trail_binding(Ai, R, T, T1),
        (   deref_wam(Val, wam_state(PC, R, S, H, T, CP, CPS, Code, L), Unbound),
            is_unbound_var(Unbound)
        ->  trail_binding(Unbound, R, T1, T2),
            put_reg(Unbound, Ref, R, R1, S, S1),
            put_assoc(Ai, R1, Ref, NR), NS = S1, NT = T2
        ;   put_assoc(Ai, R, Ref, NR), NS = S, NT = T1
        ),
        NPC is PC + 1,
        StateOut = wam_state(NPC, NR, [write_ctx(2)|NS], NH, NT, CP, CPS, Code, L)
    ;   Val = ref(Addr)
    ->  % Read mode on heap-constructed list
        nth0(Addr, H, str('./2')),
        StartIdx is Addr + 1,
        heap_subargs(H, StartIdx, 2, SubArgs),
        NPC is PC + 1,
        StateOut = wam_state(NPC, R, [unify_ctx(SubArgs)|S], H, T, CP, CPS, Code, L)
    ;   is_list(Val), Val = [Head|Tail]
    ->  NPC is PC + 1,
        StateOut = wam_state(NPC, R, [unify_ctx([Head, Tail])|S], H, T, CP, CPS, Code, L)
    ;   fail
    ).

step_wam(put_list(Ai), wam_state(PC, R, S, H, T, CP, CPS, Code, L),
         StateOut) :-
    get_reg_val(Ai, R, S, Val),
    length(H, Addr),
    Ref = ref(Addr),
    append(H, [str('./2')], NH),
    trail_binding(Ai, R, T, T1),
    (   is_unbound_var(Val),
        deref_wam(Val, wam_state(PC, R, S, H, T, CP, CPS, Code, L), Unbound),
        is_unbound_var(Unbound)
    ->  trail_binding(Unbound, R, T1, T2),
        put_reg(Unbound, Ref, R, R1, S, S1),
        put_assoc(Ai, R1, Ref, NR), NS = S1, NT = T2
    ;   put_assoc(Ai, R, Ref, NR), NS = S, NT = T1
    ),
    NPC is PC + 1,
    StateOut = wam_state(NPC, NR, NS, NH, NT, CP, CPS, Code, L).

step_wam(unify_variable(Xn), wam_state(PC, R, [unify_ctx([Arg|RestArgs])|S], H, T, CP, CPS, Code, L),
         wam_state(NPC, NR, NewS, H, NT, CP, CPS, Code, L)) :-
    trail_binding(Xn, R, T, NT),
    (   RestArgs == [] -> BaseS = S ; BaseS = [unify_ctx(RestArgs)|S] ),
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

%% unify_value Xn — read mode: unify next sub-arg with Xn's value.
%%                   Supports unbound variables on either side.
%%                   write mode: push value of Xn onto heap.
step_wam(unify_value(Xn), wam_state(PC, R, [unify_ctx([Arg|RestArgs])|S], H, T, CP, CPS, Code, L),
         wam_state(NPC, NR, NewS, H, NT, CP, CPS, Code, L)) :-
    get_reg(Xn, R, S, Val),
    (   Val == Arg
    ->  NR = R, NT = T
    ;   is_unbound_var(Val)
    ->  % Xn is unbound — bind it to the sub-arg
        trail_binding(Xn, R, T, NT),
        put_reg(Xn, Arg, R, NR, S, _)
    ;   is_unbound_var(Arg)
    ->  % Sub-arg is unbound — succeeds (Arg unifies with Val)
        NR = R, NT = T
    ;   fail
    ),
    (   RestArgs == []
    ->  NewS = S
    ;   NewS = [unify_ctx(RestArgs)|S]
    ),
    NPC is PC + 1.
step_wam(unify_value(Xn), wam_state(PC, R, [write_ctx(N)|S], H, T, CP, CPS, Code, L),
         wam_state(NPC, R, NewS, NH, T, CP, CPS, Code, L)) :-
    N > 0, get_reg(Xn, R, S, Val), append(H, [Val], NH),
    N1 is N - 1, ( N1 == 0 -> NewS = S ; NewS = [write_ctx(N1)|S] ),
    NPC is PC + 1.

%% unify_constant C — read mode: unify next sub-arg with constant C.
%%                     Supports unbound sub-args (bind to C).
%%                     write mode: push C onto heap.
step_wam(unify_constant(C), wam_state(PC, R, [unify_ctx([Arg|RestArgs])|S], H, T, CP, CPS, Code, L),
         wam_state(NPC, R, NewS, H, T, CP, CPS, Code, L)) :-
    (   Arg == C -> true
    ;   is_unbound_var(Arg) -> true  % unbound unifies with any constant
    ;   fail
    ),
    (   RestArgs == []
    ->  NewS = S
    ;   NewS = [unify_ctx(RestArgs)|S]
    ),
    NPC is PC + 1.
step_wam(unify_constant(C), wam_state(PC, R, [write_ctx(N)|S], H, T, CP, CPS, Code, L),
         wam_state(NPC, R, NewS, NH, T, CP, CPS, Code, L)) :-
    N > 0, append(H, [C], NH),
    N1 is N - 1, ( N1 == 0 -> NewS = S ; NewS = [write_ctx(N1)|S] ),
    NPC is PC + 1.

step_wam(put_constant(C, Ai), wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, NR, S, H, NT, CP, CPS, Code, L)) :-
    trail_binding(Ai, R, T, NT),
    put_assoc(Ai, R, C, NR),
    NPC is PC + 1.

step_wam(put_variable(Xn, Ai), wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, NR, NS, H, NT, CP, CPS, Code, L)) :-
    % Use monotonic counter for unique variable names across recursion depths.
    % PC-based names collide when the same instruction runs at different depths.
    nb_getval(wam_var_counter, VarCount),
    NVC is VarCount + 1,
    nb_setval(wam_var_counter, NVC),
    format(atom(NewVar), "_V~w", [VarCount]),
    trail_binding(Xn, R, T, T1),
    trail_binding(Ai, R, T1, NT),
    put_reg(Xn, NewVar, R, R1, S, S1),
    put_assoc(Ai, R1, NewVar, NR),
    (is_y_reg(Xn) -> NS = S1 ; NS = S),
    NPC is PC + 1.

step_wam(put_value(Xn, Ai), wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, NR, S, H, NT, CP, CPS, Code, L)) :-
    get_reg(Xn, R, S, Val),
    trail_binding(Ai, R, T, NT),
    put_assoc(Ai, R, Val, NR),
    NPC is PC + 1.

step_wam(put_structure(FN, Ai), wam_state(PC, R, S, H, T, CP, CPS, Code, L),
         StateOut) :-
    get_reg_val(Ai, R, S, Val),
    length(H, Addr),
    Ref = ref(Addr),
    append(H, [str(FN)], NH),
    trail_binding(Ai, R, T, T1),
    (   is_unbound_var(Val),
        deref_wam(Val, wam_state(PC, R, S, H, T, CP, CPS, Code, L), Unbound),
        is_unbound_var(Unbound)
    ->  trail_binding(Unbound, R, T1, T2),
        put_reg(Unbound, Ref, R, R1, S, S1),
        put_assoc(Ai, R1, Ref, NR), NS = S1, NT = T2
    ;   put_assoc(Ai, R, Ref, NR), NS = S, NT = T1
    ),
    NPC is PC + 1,
    StateOut = wam_state(NPC, NR, NS, NH, NT, CP, CPS, Code, L).

step_wam(set_variable(Xn), wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, NR, NS, NH, T, CP, CPS, Code, L)) :-
    length(H, Addr), format(atom(Var), "_H~w", [Addr]), append(H, [Var], NH),
    put_reg(Xn, Var, R, NR, S, NS),
    NPC is PC + 1.

step_wam(set_value(Xn), wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, R, S, NH, T, CP, CPS, Code, L)) :-
    get_reg(Xn, R, S, Val), append(H, [Val], NH), NPC is PC + 1.

step_wam(set_constant(C), wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, R, S, NH, T, CP, CPS, Code, L)) :-
    append(H, [C], NH), NPC is PC + 1.

step_wam(call(P, Arity), StateIn, StateOut) :-
    StateIn = wam_state(PC, R, S, H, T, _, CPS, Code, L),
    NCP is PC + 1,
    % Check fact table first (O(log n) lookup vs O(n) instruction scan)
    (   wam_fact_table_registered(P, FactTable)
    ->  call_fact_predicate(P, Arity, FactTable, wam_state(PC, R, S, H, T, NCP, CPS, Code, L), StateOut)
    ;   % Fallback to WAM code
        get_assoc(P, L, NPC),
        StateOut = wam_state(NPC, R, S, H, T, NCP, CPS, Code, L)
    ).

step_wam(execute(P), wam_state(_, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, R, S, H, T, CP, CPS, Code, L)) :-
    get_assoc(P, L, NPC).

step_wam(proceed, wam_state(_, R, S, H, T, CP, CPS, Code, L), wam_state(CP, R, S, H, T, halt, CPS, Code, L)).

step_wam(allocate, wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, R, [env(CP, YRegs, CutBarrier)|S], H, T, CP, CPS, Code, L)) :-
    empty_assoc(YRegs),
    length(CPS, CutBarrier),  % save CP stack depth as cut barrier
    NPC is PC + 1.

step_wam(deallocate, wam_state(PC, R, [env(OldCP, _, _)|S], H, T, _, CPS, Code, L), wam_state(NPC, R, S, H, T, OldCP, CPS, Code, L)) :-
    NPC is PC + 1.

step_wam(try_me_else(NextL), wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, R, S, H, T, CP, [cp(NextPC, R, S, H, CP, T, _, SavedB)|CPS], Code, L)) :-
    get_assoc(NextL, L, NextPC),
    wam_save_bindings(SavedB),
    NPC is PC + 1.

step_wam(trust_me, wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, R, S, H, T, CP, NCPS, Code, L)) :-
    (CPS = [_|NCPS] -> true ; NCPS = CPS), NPC is PC + 1.

step_wam(retry_me_else(NextL), wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, R, S, H, T, CP, NCPS, Code, L)) :-
    get_assoc(NextL, L, NextPC),
    % Only update next_pc, preserve saved state from try_me_else
    (CPS = [cp(_, SR, SS, SH, SCP, ST, SBuiltin, SB)|Rest]
    ->  NCPS = [cp(NextPC, SR, SS, SH, SCP, ST, SBuiltin, SB)|Rest]
    ;   wam_save_bindings(SB),
        NCPS = [cp(NextPC, R, S, H, CP, T, _, SB)]
    ),
    NPC is PC + 1.

%% Indexing instructions — O(log n) dispatch via assoc lookup.
%  switch_on_constant entries are pre-parsed into assoc during prepare_code.
%  Format: switch_on_constant_index(Assoc) where Assoc maps Key → Label.
step_wam(switch_on_constant_index(IndexAssoc), wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, R, S, H, T, CP, CPS, Code, L)) :-
    get_assoc('A1', R, Val0),
    wam_deref(Val0, Val),
    (   \+ is_unbound_var(Val),
        get_assoc(Val, IndexAssoc, LabelStr),
        LabelStr \== default,
        get_assoc(LabelStr, L, TargetPC)
    ->  NPC = TargetPC
    ;   NPC is PC + 1
    ).

%% Legacy switch_on_constant with flat list (fallback for unprocessed code)
step_wam(Instr, wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, R, S, H, T, CP, CPS, Code, L)) :-
    Instr =.. [Op|Args],
    (Op == switch_on_constant ; Op == switch_on_structure ; Op == switch_on_constant_a2),
    (Args = [Entries], is_list(Entries) -> true ; Entries = Args),
    (get_assoc('A1', R, Val0), wam_deref(Val0, Val), \+ is_unbound_var(Val), lookup_index(Val, Entries, L, TargetPC) -> NPC = TargetPC ; NPC is PC + 1).

lookup_index(Val, [Entry|Rest], Labels, TargetPC) :-
    (once(sub_atom(Entry, Before, 1, After, ':')) ->
        sub_atom(Entry, 0, Before, _, KeyStr), sub_atom(Entry, _, After, 0, LabelStr),
        (atom_number(KeyStr, Key) -> true ; Key = KeyStr),
        (Key == Val -> (LabelStr == 'default' -> fail ; get_assoc(LabelStr, Labels, TargetPC)) ; lookup_index(Val, Rest, Labels, TargetPC))
    ; fail).

%% !/0 — cut: truncate choice points to the cut barrier.
%  The cut barrier is saved at Allocate time in the env frame.
step_wam(builtin_call('!/0', 0), wam_state(PC, R, S, H, T, CP, CPS, Code, L),
         wam_state(NPC, R, S, H, T, CP, NCPS, Code, L)) :-
    NPC is PC + 1,
    % Find the cut barrier from the topmost env frame.
    % CutBarrier is the CP stack length at Allocate time.
    % Cut removes CPs added AFTER the barrier (newer CPs at list head).
    % Keep only the last CutBarrier CPs (the tail of the list).
    (   member(env(_, _, CutBarrier), S)
    ->  length(CPS, CPLen),
        DropN is max(0, CPLen - CutBarrier),
        length(Prefix, DropN),
        append(Prefix, NCPS, CPS)
    ;   NCPS = []  % no env frame — clear all (fallback)
    ).

%% length/2 — measure list length, bind to A2.
step_wam(builtin_call('length/2', 2), wam_state(PC, R, S, H, T, CP, CPS, Code, L),
         wam_state(NPC, NR, S, H, NT, CP, CPS, Code, L)) :-
    get_assoc('A1', R, ListRaw),
    wam_deref(ListRaw, DList0),
    deref_wam(DList0, wam_state(PC, R, S, H, T, CP, CPS, Code, L), DList),
    is_list(DList),
    length(DList, Len),
    get_assoc('A2', R, LHS0), wam_deref(LHS0, LHS),
    (   is_unbound_var(LHS)
    ->  trail_binding('A2', R, T, NT),
        wam_bind(LHS, Len),
        put_assoc('A2', R, Len, NR)
    ;   number(LHS), LHS =:= Len
    ->  NR = R, NT = T
    ;   fail
    ),
    NPC is PC + 1.

step_wam(builtin_call(Op, 2), StateIn, StateOut) :-
    is_comparison_op(Op), !, StateIn = wam_state(PC, R, S, H, T, CP, CPS, Code, L),
    get_assoc('A1', R, V1), get_assoc('A2', R, V2),
    eval_arith(V1, R, S, H, N1), eval_arith(V2, R, S, H, N2),
    apply_comparison(Op, N1, N2), NPC is PC + 1,
    StateOut = wam_state(NPC, R, S, H, T, CP, CPS, Code, L).

step_wam(builtin_call('is/2', 2), wam_state(PC, R, S, H, T, CP, CPS, Code, L), wam_state(NPC, NR, S, H, NT, CP, CPS, Code, L)) :-
    get_assoc('A2', R, Expr0), wam_deref(Expr0, Expr),
    eval_arith(Expr, R, S, H, Result),
    get_assoc('A1', R, RawLHS), wam_deref(RawLHS, LHS),
    (is_unbound_var(LHS) ->
        trail_binding('A1', R, T, NT),
        (atom(RawLHS), is_unbound_var(RawLHS) -> wam_bind(RawLHS, Result) ; true),
        put_assoc('A1', R, Result, NR)
    ; number(LHS), LHS =:= Result -> NR = R, NT = T ; fail),
    NPC is PC + 1.

step_wam(builtin_call('member/2', 2), StateIn, StateOut) :-
    StateIn = wam_state(PC, R, S, H, T, CP, CPS, Code, L),
    get_assoc('A1', R, ElemRaw), get_assoc('A2', R, ListRaw),
    deref_wam(ElemRaw, StateIn, Elem), deref_wam(ListRaw, StateIn, List),
    (List = [Next|Rest] ->
        wam_save_bindings(MemberSB),
        (Rest == [] -> NCPS = CPS ; NCPS = [cp(PC, R, S, H, CP, T, member(Elem, Rest, PC), MemberSB)|CPS]),
        State1 = wam_state(PC, R, S, H, T, CP, NCPS, Code, L),
        (unify_wam(Elem, Next, State1, State2) ->
            State2 = wam_state(NPC_base, R2, S2, H2, T2, CP2, CPS2, Code2, L2),
            NPC is NPC_base + 1, StateOut = wam_state(NPC, R2, S2, H2, T2, CP2, CPS2, Code2, L2)
        ; backtrack(State1, StateOut))
    ; fail).

%% \+/1 (negation-as-failure): the goal is in A1 as a structure.
%  Fast path for member/2: directly check list membership.
%  General path: save state, try goal, negate result.
step_wam(builtin_call('\\+/1', 1),
         wam_state(PC, R, S, H, T, CP, CPS, Code, L),
         wam_state(NPC, R, S, H, T, CP, CPS, Code, L)) :-
    get_assoc('A1', R, GoalTerm),
    State = wam_state(PC, R, S, H, T, CP, CPS, Code, L),
    deref_wam(GoalTerm, State, DGoal0),
    % If DGoal0 is a heap ref to a str, reconstruct with args from heap
    reconstruct_goal(DGoal0, H, DGoal),
    (   naf_try_goal(DGoal, State)
    ->  fail    % goal succeeded → \+ fails
    ;   NPC is PC + 1   % goal failed → \+ succeeds
    ).

%% reconstruct_goal(+HeapVal, +Heap, -Goal)
%  If the value is str(F/N), extract N args from the heap after it.
reconstruct_goal(str(FN), Heap, str(FN, Args)) :-
    !,
    % Parse arity from functor (e.g., "member/2" → 2)
    (   sub_atom(FN, B, 1, _, '/'),
        sub_atom(FN, _, _, 0, ArityAtom),
        atom_number(ArityAtom, Arity),
        B > 0
    ->  true
    ;   Arity = 0
    ),
    % Find the str(FN) on the heap and get the next Arity values
    (   nth0(Idx, Heap, str(FN))
    ->  StartIdx is Idx + 1,
        heap_subargs(Heap, StartIdx, Arity, Args)
    ;   Args = []
    ).
reconstruct_goal(Goal, _, Goal).

%% heap_subargs(+Heap, +Start, +Count, -Args)
heap_subargs(_, _, 0, []) :- !.
heap_subargs(Heap, Idx, N, [Arg|Rest]) :-
    N > 0,
    nth0(Idx, Heap, Arg),
    Idx1 is Idx + 1,
    N1 is N - 1,
    heap_subargs(Heap, Idx1, N1, Rest).

%% naf_try_goal(+Goal, +State) — succeeds if Goal can be proved
naf_try_goal(str('member/2', [Elem, List]), State) :-
    !,
    % Fast path: check membership directly
    deref_wam(Elem, State, DElem),
    deref_wam(List, State, DList),
    is_list(DList),
    member(DElem, DList).
naf_try_goal(Goal, wam_state(_, R, S, H, T, CP, CPS, Code, L)) :-
    % General path: extract functor/args, set up registers, run
    Goal = str(Functor, Args),
    length(Args, Arity),
    set_args(Args, 1, R, R1),
    format(atom(PredKey), "~w", [Functor]),
    init_state_for_call(PredKey, R1, S, H, T, CPS, Code, L, CallState),
    run_loop(CallState, _).

set_args([], _, R, R).
set_args([Arg|Rest], N, R, ROut) :-
    format(atom(RegName), "A~w", [N]),
    put_assoc(RegName, R, Arg, R1),
    N1 is N + 1,
    set_args(Rest, N1, R1, ROut).

init_state_for_call(PredKey, R, S, H, T, CPS, Code, L, wam_state(PC, R, S, H, T, halt, CPS, Code, L)) :-
    (   get_assoc(PredKey, L, PC) -> true
    ;   PC = 0  % not found → will fail immediately
    ).

%% unify_wam(+V1, +V2, +StateIn, -StateOut)
unify_wam(V1, V2, StateIn, StateOut) :-
    deref_wam(V1, StateIn, DV1), deref_wam(V2, StateIn, DV2),
    (DV1 == DV2 -> StateOut = StateIn
    ; var(DV1) -> DV1 = DV2, StateOut = StateIn
    ; var(DV2) -> DV2 = DV1, StateOut = StateIn
    ; is_unbound_var(DV1) ->
        StateIn = wam_state(PC, R, S, H, T, CP, CPS, Code, L),
        trail_binding(DV1, R, T, NT), put_reg(DV1, DV2, R, NR, S, NS),
        StateOut = wam_state(PC, NR, NS, H, NT, CP, CPS, Code, L)
    ; is_unbound_var(DV2) ->
        StateIn = wam_state(PC, R, S, H, T, CP, CPS, Code, L),
        trail_binding(DV2, R, T, NT), put_reg(DV2, DV1, R, NR, S, NS),
        StateOut = wam_state(PC, NR, NS, H, NT, CP, CPS, Code, L)
    ; compound(DV1), compound(DV2) ->
        DV1 =.. [F|Args1], DV2 =.. [F|Args2], length(Args1, Len), length(Args2, Len),
        unify_args(Args1, Args2, StateIn, StateOut)
    ; fail).

unify_args([], [], S, S).
unify_args([A1|R1], [A2|R2], S0, Sf) :- unify_wam(A1, A2, S0, S1), unify_args(R1, R2, S1, Sf).

deref_wam(Val, State, Derefed) :-
    State = wam_state(_, R, S, _, _, _, _, _, _),
    (   is_unbound_var(Val), atom(Val)
    ->  % First check the binding table
        (   nb_getval(wam_bindings, B), get_assoc(Val, B, BoundVal)
        ->  deref_wam(BoundVal, State, Derefed)
        ;   % Then check registers
            (   get_reg(Val, R, S, Next)
            ->  (Next == Val -> Derefed = Val ; deref_wam(Next, State, Derefed))
            ;   Derefed = Val
            )
        )
    ;   Val = ref(Addr) -> deref_heap(ref(Addr), State, Derefed)
    ;   Derefed = Val
    ).

deref_heap(ref(Addr), State, Term) :- !,
    State = wam_state(_, _, _, Heap, _, _, _, _, _),
    nth0(Addr, Heap, Entry),
    (   (Entry = str(FN) ; Entry = FN)
    ->  (   (atom(Entry) ; (Entry = str(AN), atom(AN)))
        ->  atom_string(FN, FNStr),
            (   once(sub_atom(FNStr, _, 1, After, '/'))
            ->  sub_atom(FNStr, _, After, 0, ArStr),
                atom_number(ArStr, Arity),
                (   once(sub_atom(FNStr, Before, 1, _, '/'))
                ->  sub_atom(FNStr, 0, Before, _, F)
                ;   F = FN
                ),
                (   Arity > 0
                ->  StartIdx is Addr + 1,
                    heap_subargs(Heap, StartIdx, Arity, SubArgs),
                    maplist({State}/[A, D]>>deref_wam(A, State, D), SubArgs, DerefArgs)
                ;   DerefArgs = []
                ),
                (   (F == '.'; F == '[|]')
                ->  (   DerefArgs = [Head, Tail]
                    ->  (   Tail == [] -> Term = [Head]
                        ;   is_list(Tail) -> Term = [Head|Tail]
                        ;   Term = [Head|Tail]
                        )
                    ;   Term =.. [F|DerefArgs]
                    )
                ;   Term =.. [F|DerefArgs]
                )
            ;   Term = FN
            )
        ;   Term = Entry
        )
    ;   Term = Entry
    ).
deref_heap(Val, _, Val).

heap_subargs(_, _, 0, []) :- !.
heap_subargs(Heap, Idx, N, [Val|Rest]) :- nth0(Idx, Heap, Val), NextIdx is Idx + 1, N1 is N - 1, heap_subargs(Heap, NextIdx, N1, Rest).

is_unbound_var(Val) :- var(Val), !.
is_unbound_var(Val) :- atom(Val), (sub_atom(Val, 0, 2, _, '_V') ; sub_atom(Val, 0, 2, _, '_H') ; sub_atom(Val, 0, 2, _, '_Q')).

trail_binding(Key, Regs, Trail, [trail(Key, OldValue)|Trail]) :- (get_assoc(Key, Regs, OldValue) -> true ; OldValue = unbound).

is_y_reg(Reg) :- atom(Reg), sub_atom(Reg, 0, 1, _, 'Y').

%% wam_bind(+VarName, +Value)
%  Bind a WAM variable name to a value in the global binding table.
wam_bind(VarName, Value) :-
    nb_getval(wam_bindings, B),
    put_assoc(VarName, B, Value, B1),
    nb_setval(wam_bindings, B1).

%% wam_deref(+Val, -Derefed)
%  Dereference a value through the binding table.
%  If Val is an unbound WAM variable (_V*, _Q*, _H*), look up its binding.
%  Follows chains: _V1 → _V2 → 10 → returns 10.
wam_deref(Val, Derefed) :-
    (   atom(Val), is_unbound_var(Val)
    ->  nb_getval(wam_bindings, B),
        (   get_assoc(Val, B, Bound)
        ->  wam_deref(Bound, Derefed)
        ;   Derefed = Val
        )
    ;   Derefed = Val
    ).

%% wam_save_bindings(-Saved)
%  Save the current binding table (for choice points).
wam_save_bindings(Saved) :-
    nb_getval(wam_bindings, Saved).

%% wam_restore_bindings(+Saved)
%  Restore binding table from a saved snapshot (on backtrack).
wam_restore_bindings(Saved) :-
    nb_setval(wam_bindings, Saved).

get_reg_val(Reg, R, S, Val) :-
    (   get_reg(Reg, R, S, V)
    ->  wam_deref(V, Val)
    ;   Val = '_Vunbound'
    ).

get_reg(Reg, _R, Stack, Val) :-
    is_y_reg(Reg), !,
    member(env(_, YRegs, _), Stack), !,
    get_assoc(Reg, YRegs, Val).
get_reg(Reg, R, _, Val) :-
    get_assoc(Reg, R, Val).

put_reg(Reg, Val, R, R, Stack, NewStack) :- is_y_reg(Reg), !, update_top_env(Stack, Reg, Val, NewStack).
put_reg(Reg, Val, R, NR, Stack, Stack) :- put_assoc(Reg, R, Val, NR).
update_top_env([env(CP, YRegs, CB)|Rest], Reg, Val, [env(CP, NewYRegs, CB)|Rest]) :- put_assoc(Reg, YRegs, Val, NewYRegs).

get_arity(FN, Arity) :-
    (   once(sub_atom(FN, _, 1, After, '/'))
    ->  sub_atom(FN, _, After, 0, ArStr),
        atom_number(ArStr, Arity)
    ;   Arity = 0
    ).

is_comparison_op('>/2'). is_comparison_op('</2'). is_comparison_op('>=/2'). is_comparison_op('=</2'). is_comparison_op('=:=/2'). is_comparison_op('=\\=/2').
apply_comparison('>/2', N1, N2) :- N1 > N2. apply_comparison('</2', N1, N2) :- N1 < N2. apply_comparison('>=/2', N1, N2) :- N1 >= N2.
apply_comparison('=</2', N1, N2) :- N1 =< N2. apply_comparison('=:=/2', N1, N2) :- N1 =:= N2. apply_comparison('=\\=/2', N1, N2) :- N1 =\= N2.

is_type_check_op('integer/1'). is_type_check_op('float/1'). is_type_check_op('number/1'). is_type_check_op('atom/1').
is_type_check_op('compound/1'). is_type_check_op('var/1'). is_type_check_op('nonvar/1'). is_type_check_op('is_list/1').
apply_type_check('integer/1', V) :- integer(V). apply_type_check('float/1', V) :- float(V). apply_type_check('number/1', V) :- number(V).
apply_type_check('atom/1', V) :- atom(V), \+ is_unbound_var(V). apply_type_check('compound/1', V) :- compound(V).
apply_type_check('var/1', V) :- is_unbound_var(V). apply_type_check('nonvar/1', V) :- \+ is_unbound_var(V). apply_type_check('is_list/1', V) :- is_list(V).

eval_arith(Expr, R, S, Heap, Result) :- deref_wam(Expr, wam_state(0, R, S, Heap, [], 0, [], [], _), D), eval_arith_derefed(D, R, S, Heap, Result).
eval_arith_derefed(Expr, _, _, _, Expr) :- number(Expr), !.
eval_arith_derefed(Expr, R, S, Heap, Result) :- atom(Expr), (is_unbound_var(Expr) -> fail ; (sub_atom(Expr, 0, 1, _, 'A') ; sub_atom(Expr, 0, 1, _, 'X') ; sub_atom(Expr, 0, 1, _, 'Y')) -> get_reg(Expr, R, S, Val), eval_arith(Val, R, S, Heap, Result) ; fail), !.
eval_arith_derefed(Expr, R, S, Heap, Result) :- compound(Expr), Expr =.. [Op|Args], maplist({R, S, Heap}/[A, V]>>eval_arith(A, R, S, Heap, V), Args, Vals), (Vals = [V1, V2] -> apply_arith_op(Op, V1, V2, Result) ; Vals = [V1] -> apply_arith_unary(Op, V1, Result) ; fail).
apply_arith_op(+, A, B, R) :- R is A + B. apply_arith_op(-, A, B, R) :- R is A - B. apply_arith_op(*, A, B, R) :- R is A * B.
apply_arith_op(/, A, B, R) :- B =\= 0, R is A / B. apply_arith_op(//, A, B, R) :- B =\= 0, R is A // B. apply_arith_op(mod, A, B, R) :- B =\= 0, R is A mod B. apply_arith_op(div, A, B, R) :- B =\= 0, R is A div B.
apply_arith_unary(-, A, R) :- R is -A. apply_arith_unary(abs, A, R) :- R is abs(A).

solve_wam(Instructions, Query, VarNames, Bindings) :-
    prepare_code(Instructions, Code, Labels),
    prepare_query(Query, VarNames, QSymbolic, VNSymbolic),
    init_state(QSymbolic, Code, Labels, S0),
    run_loop(S0, Sf),
    extract_bindings(VNSymbolic, QSymbolic, Sf, Bindings).

prepare_query(Query, VarNames, QSymbolic, VNSymbolic) :-
    copy_term(Query-VarNames, QSymbolic-VNSymbolic),
    term_variables(QSymbolic, Vars),
    map_query_vars(Vars, 1).

map_query_vars([], _).
map_query_vars([V|Vs], I) :-
    format(atom(V), "_Q~w", [I]),
    NI is I + 1,
    map_query_vars(Vs, NI).

findall_wam(Instructions, Query, VarNames, AllBindings) :-
    prepare_code(Instructions, Code, Labels),
    prepare_query(Query, VarNames, QSymbolic, VNSymbolic),
    init_state(QSymbolic, Code, Labels, S0),
    run_all_solutions(S0, QSymbolic, VNSymbolic, AllBindings).

run_all_solutions(S0, Query, VarNames, AllBindings) :-
    run_all_acc(S0, Query, VarNames, [], RevBindings),
    reverse(RevBindings, AllBindings).

run_all_acc(State, Query, VarNames, Acc, AllBindings) :- 
    (   run_loop(State, Sf)
    ->  extract_bindings(VarNames, Query, Sf, Bindings),
        Sf = wam_state(_, _, _, _, _, _, CPS, _, _),
        NewAcc = [Bindings|Acc],
        (   CPS = [_|_]
        ->  (   backtrack(Sf, SBT) -> run_all_acc(SBT, Query, VarNames, NewAcc, AllBindings)
            ;   AllBindings = NewAcc
            )
        ;   AllBindings = NewAcc
        )
    ;   AllBindings = Acc
    ).

extract_bindings(VarNames, Query, wam_state(_, Regs, _, Heap, _, _, _, _, _), Bindings) :-
    nb_getval(wam_bindings, BTable),
    extract_bindings_iter(VarNames, Query, Regs, Heap, BTable, Bindings).

extract_bindings_iter([], _, _, _, _, []).
extract_bindings_iter([Name=Var|Rest], Query, Regs, Heap, BTable, [Name=Value|RestBindings]) :-
    Query =.. [_|Args],
    (   nth1(Idx, Args, A), A == Var
    ->  % First check the binding table for the query variable name
        (   atom(Var), get_assoc(Var, BTable, BoundVal)
        ->  Value = BoundVal
        ;   % Fallback to register value
            format(atom(RegKey), "A~w", [Idx]),
            (   get_assoc(RegKey, Regs, RawValue)
            ->  deref_wam(RawValue, wam_state(0, Regs, [], Heap, [], 0, [], [], _), Value)
            ;   Value = Var
            )
        )
    ;   Value = Var
    ),
    extract_bindings_iter(Rest, Query, Regs, Heap, BTable, RestBindings).

test_wam_runtime :-
    Test1 = 'WAM Runtime: basic parent/2',
    Code1 = ['parent/2:', get_constant(alice, 'A1'), get_constant(bob, 'A2'), proceed],
    (execute_wam(Code1, parent(alice, bob), _) -> format('[PASS] ~w~n', [Test1]) ; format('[FAIL] ~w~n', [Test1])),
    Test2 = 'WAM Runtime: relational member/2',
    Code2 = [
        'test_member/1:',
        put_list('A2'),
        set_constant(1),
        set_variable('X3'),
        put_structure('[|]/2', 'X3'),
        set_constant(2),
        set_constant([]),
        put_variable('X1', 'A1'),
        builtin_call('member/2', 2),
        proceed
    ],
    (findall_wam(Code2, test_member(X), ['X'=X], Sols) -> (Sols == [['X'=1], ['X'=2]] -> format('[PASS] ~w~n', [Test2]) ; format('[FAIL] ~w: expected [[X=1], [X=2]], got ~w~n', [Test2, Sols])) ; format('[FAIL] ~w: findall failed~n', [Test2])),
    format('WAM Runtime Tests Complete~n').
