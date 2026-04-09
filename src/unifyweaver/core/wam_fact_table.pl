:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wam_fact_table.pl — Indexed fact tables for the WAM runtime.
%
% Facts are stored in hash-indexed dictionaries instead of WAM
% instruction chains. A query like category_parent(a, X) becomes
% a single O(log n) dictionary lookup instead of scanning through
% thousands of try_me_else/get_constant/proceed instruction triples.
%
% Architecture:
%   - Facts are grouped by predicate name and indexed by first argument
%   - call_fact/N dispatches to the indexed table
%   - Multi-value results create choice points for backtracking
%   - Unbound first argument falls back to full scan (rare case)

:- module(wam_fact_table, [
    build_fact_table/3,       % +PredicateName, +Arity, -FactTable
    load_fact_table/4,        % +PredicateName, +Arity, +Clauses, -FactTable
    fact_table_lookup/4,      % +A1, +FactTable, -RestArgs, -ChoicePointData
    fact_table_size/2,        % +FactTable, -Count
    is_fact_predicate/2       % +PredicateName, +Arity
]).

:- use_module(wam_dict).

% ============================================================================
% Fact Table Construction
% ============================================================================

%% build_fact_table(+Pred, +Arity, -FactTable)
%  Build an indexed fact table from asserted clauses of Pred/Arity.
%  Groups facts by first argument for O(log n) lookup.
build_fact_table(Pred, Arity, FactTable) :-
    functor(Head, Pred, Arity),
    findall(Head, clause(Head, true), Clauses),
    load_fact_table(Pred, Arity, Clauses, FactTable).

%% load_fact_table(+Pred, +Arity, +Clauses, -FactTable)
%  Build an indexed fact table from a list of clause heads.
%  FactTable is a dict: FirstArg → [RestArgsTuple]
load_fact_table(_Pred, Arity, Clauses, FactTable) :-
    Arity >= 1,
    clauses_to_pairs(Clauses, Pairs),
    wam_dict_from_grouped(Pairs, FactTable).

clauses_to_pairs([], []).
clauses_to_pairs([Head|Rest], [A1-RestArgs|Pairs]) :-
    Head =.. [_Functor, A1 | RestArgsList],
    RestArgs = RestArgsList,
    clauses_to_pairs(Rest, Pairs).

% ============================================================================
% Fact Table Lookup
% ============================================================================

%% fact_table_lookup(+A1, +FactTable, -RestArgs, -Remaining)
%  Look up facts matching first argument A1.
%  Returns the rest-args of the first match and a list of remaining matches
%  for backtracking.
%
%  Usage pattern:
%    fact_table_lookup(A1, Table, RestArgs, Remaining)
%    → RestArgs is the first match
%    → Remaining is the list of additional matches (for choice points)
%
%  If A1 is unbound (variable or _V* atom), returns ALL facts.
fact_table_lookup(A1, FactTable, RestArgs, Remaining) :-
    (   is_wam_unbound(A1)
    ->  % Unbound: collect all facts (rare, expensive)
        wam_dict_to_list(FactTable, AllEntries),
        all_facts_from_entries(AllEntries, AllFacts),
        AllFacts = [First|Rest],
        First = K1-RestArgs0,
        RestArgs = [K1|RestArgs0],  % include first arg in output
        Remaining = Rest
    ;   % Bound: O(log n) lookup on first argument
        wam_dict_lookup_all(A1, FactTable, Matches),
        Matches = [RestArgs|Remaining]
    ).

all_facts_from_entries([], []).
all_facts_from_entries([Key-Values|Rest], AllFacts) :-
    maplist([V, Key-V]>>true, Values, KeyedValues),
    all_facts_from_entries(Rest, RestFacts),
    append(KeyedValues, RestFacts, AllFacts).

%% is_wam_unbound(+Val)
%  True if Val represents an unbound WAM variable.
is_wam_unbound(Val) :- var(Val), !.
is_wam_unbound(Val) :-
    atom(Val),
    (   sub_atom(Val, 0, 2, _, '_V')
    ;   sub_atom(Val, 0, 2, _, '_Q')
    ;   sub_atom(Val, 0, 2, _, '_H')
    ).

% ============================================================================
% Fact Table Metadata
% ============================================================================

%% fact_table_size(+FactTable, -Count)
%  Total number of facts in the table.
fact_table_size(FactTable, Count) :-
    wam_dict_fold([_K, Vs, Acc, Out]>>(length(Vs, N), Out is Acc + N),
                  FactTable, 0, Count).

%% is_fact_predicate(+Pred, +Arity)
%  True if Pred/Arity has only fact clauses (no rule bodies).
is_fact_predicate(Pred, Arity) :-
    functor(Head, Pred, Arity),
    \+ (clause(Head, Body), Body \== true).
