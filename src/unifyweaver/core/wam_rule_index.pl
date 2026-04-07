:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wam_rule_index.pl — Rule head separation for the WAM runtime.
%
% Splits WAM code into:
%   - Head patterns: constant matches on arguments (for dispatch)
%   - Head bindings: get_variable mappings (register copies)
%   - Body code: everything after the head (WAM instructions)
%
% The dispatcher uses native pattern matching on head constants
% instead of WAM try_me_else/trust_me instruction chains.

:- module(wam_rule_index, [
    build_rule_index/3,       % +WamCodeString, -RuleIndex, -Labels
    rule_index_dispatch/4,    % +PredKey, +RuleIndex, +State, -BodyCode+Bindings
    apply_head_bindings/5     % +Bindings, +Regs, +Stack, -NewRegs, -NewStack
]).

:- use_module(wam_dict).
:- use_module(library(lists)).

% ============================================================================
% Rule Index Construction
% ============================================================================

%% build_rule_index(+WamCodeString, -RuleIndex, -Labels)
%  Parse WAM code for a predicate and build a rule index.
%  RuleIndex is a list of rule(HeadPattern, HeadBindings, BodyCode).
%  Labels is the label map (for body code PC references).
build_rule_index(WamCodeString, RuleIndex, Labels) :-
    atom_string(WamCodeString, WamStr),
    split_string(WamStr, "\n", "", Lines),
    parse_wam_lines(Lines, 1, Instructions, Labels),
    split_into_clauses(Instructions, Clauses),
    maplist(analyze_clause, Clauses, RuleIndex).

%% parse_wam_lines(+Lines, +PC, -Instructions, -Labels)
%  Parse WAM assembly into instruction terms and a label map.
parse_wam_lines([], _, [], Labels) :- wam_dict_new(Labels).
parse_wam_lines([Line|Rest], PC, Instrs, Labels) :-
    split_string(Line, " \t,", " \t,", Parts),
    delete(Parts, "", CleanParts),
    (   CleanParts == []
    ->  parse_wam_lines(Rest, PC, Instrs, Labels)
    ;   CleanParts = [First|_],
        (   sub_string(First, _, 1, 0, ":")
        ->  sub_string(First, 0, _, 1, LabelName),
            atom_string(LabelAtom, LabelName),
            parse_wam_lines(Rest, PC, Instrs, Labels0),
            wam_dict_insert(LabelAtom, PC, Labels0, Labels)
        ;   parse_wam_instr(CleanParts, Instr),
            NPC is PC + 1,
            Instrs = [pc(PC, Instr)|RestInstrs],
            parse_wam_lines(Rest, NPC, RestInstrs, Labels)
        )
    ).

parse_wam_instr(["get_constant", C, Ai], get_constant(CA, AiA)) :-
    clean_comma(C, CC), clean_comma(Ai, CAi),
    parse_val(CC, CA), atom_string(AiA, CAi).
parse_wam_instr(["get_variable", Xn, Ai], get_variable(XnA, AiA)) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    atom_string(XnA, CXn), atom_string(AiA, CAi).
parse_wam_instr(["try_me_else", L], try_me_else(LA)) :- atom_string(LA, L).
parse_wam_instr(["trust_me"], trust_me).
parse_wam_instr(["retry_me_else", L], retry_me_else(LA)) :- atom_string(LA, L).
parse_wam_instr(["allocate"], allocate).
parse_wam_instr(["deallocate"], deallocate).
parse_wam_instr(["proceed"], proceed).
parse_wam_instr(Parts, generic(Parts)).  % fallback for body instructions

clean_comma(Str, Clean) :-
    (sub_string(Str, _, 1, 0, ",") -> sub_string(Str, 0, _, 1, Clean) ; Clean = Str).

parse_val(Str, Val) :-
    (number_string(N, Str) -> Val = N ; atom_string(Val, Str)).

% ============================================================================
% Clause Splitting
% ============================================================================

%% split_into_clauses(+Instructions, -Clauses)
%  Split a list of pc(PC, Instr) into clauses, delimited by
%  try_me_else/retry_me_else/trust_me.
split_into_clauses([], []).
split_into_clauses(Instrs, [Clause|Rest]) :-
    take_clause(Instrs, Clause, Remaining),
    split_into_clauses(Remaining, Rest).

take_clause([], [], []).
take_clause([pc(_, try_me_else(_))|Rest], Clause, Remaining) :-
    !, take_until_next_clause(Rest, Clause, Remaining).
take_clause([pc(_, trust_me)|Rest], Clause, Remaining) :-
    !, take_until_next_clause(Rest, Clause, Remaining).
take_clause([pc(_, retry_me_else(_))|Rest], Clause, Remaining) :-
    !, take_until_next_clause(Rest, Clause, Remaining).
take_clause(Instrs, Clause, Remaining) :-
    % No clause selector — single clause predicate
    take_until_next_clause(Instrs, Clause, Remaining).

take_until_next_clause([], [], []).
take_until_next_clause([pc(_, trust_me)|Rest], [], [pc(_, trust_me)|Rest]) :- !.
take_until_next_clause([pc(_, retry_me_else(_))|Rest], [], [pc(_, retry_me_else(_))|Rest]) :- !.
take_until_next_clause([pc(PC, Instr)|Rest], [pc(PC, Instr)|Clause], Remaining) :-
    take_until_next_clause(Rest, Clause, Remaining).

% ============================================================================
% Clause Analysis
% ============================================================================

%% analyze_clause(+InstrList, -Rule)
%  Analyze a clause's instructions to extract head pattern, bindings, and body.
%  Rule = rule(HeadPattern, HeadBindings, BodyInstructions)
%  HeadPattern = list of match(ArgReg, Value) or var(ArgReg)
%  HeadBindings = list of bind(TargetReg, SourceReg)
analyze_clause(InstrList, rule(HeadPattern, HeadBindings, BodyInstrs)) :-
    % Skip allocate at the start
    (   InstrList = [pc(_, allocate)|AfterAlloc]
    ->  true
    ;   AfterAlloc = InstrList
    ),
    % Extract head instructions (get_constant, get_variable)
    extract_head(AfterAlloc, HeadPattern, HeadBindings, BodyInstrs).

extract_head([], [], [], []).
extract_head([pc(_, get_constant(Val, Ai))|Rest], [match(Ai, Val)|HP], HB, Body) :-
    !, extract_head(Rest, HP, HB, Body).
extract_head([pc(_, get_variable(Xn, Ai))|Rest], HP, [bind(Xn, Ai)|HB], Body) :-
    !, extract_head(Rest, HP, HB, Body).
extract_head(Instrs, [], [], Instrs).
    % First non-head instruction → everything else is body

% ============================================================================
% Rule Dispatch
% ============================================================================

%% rule_index_dispatch(+RuleIndex, +Regs, -MatchedRule, -RemainingRules)
%  Find the first rule whose head pattern matches the current registers.
%  Returns the matched rule and remaining rules for backtracking.
rule_index_dispatch([Rule|Rest], Regs, Rule, Rest) :-
    Rule = rule(HeadPattern, _, _),
    head_matches(HeadPattern, Regs).
rule_index_dispatch([_|Rest], Regs, MatchedRule, Remaining) :-
    rule_index_dispatch(Rest, Regs, MatchedRule, Remaining).

head_matches([], _).
head_matches([match(Ai, ExpectedVal)|Rest], Regs) :-
    wam_dict_lookup(Ai, Regs, ActualVal),
    ActualVal == ExpectedVal,
    head_matches(Rest, Regs).
head_matches([var(_)|Rest], Regs) :-
    head_matches(Rest, Regs).

%% apply_head_bindings(+Bindings, +RegsIn, +StackIn, -RegsOut, -StackOut)
%  Apply head bindings: copy argument registers to target registers.
apply_head_bindings([], R, S, R, S).
apply_head_bindings([bind(Target, Source)|Rest], R, S, ROut, SOut) :-
    wam_dict_lookup(Source, R, Val),
    (   is_y_reg(Target)
    ->  update_top_env_dict(S, Target, Val, S1), R1 = R
    ;   wam_dict_insert(Target, Val, R, R1), S1 = S
    ),
    apply_head_bindings(Rest, R1, S1, ROut, SOut).

is_y_reg(Reg) :-
    atom(Reg),
    sub_atom(Reg, 0, 1, _, 'Y'),
    sub_atom(Reg, 1, _, 0, Rest),
    atom_number(Rest, _).

update_top_env_dict([env(CP, YRegs, CB)|Rest], Reg, Val, [env(CP, NewYRegs, CB)|Rest]) :-
    wam_dict_insert(Reg, Val, YRegs, NewYRegs).
