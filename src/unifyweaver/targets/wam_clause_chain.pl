:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% wam_clause_chain.pl — shared front-end for "multi-clause as an
% if-then-else chain" lowering (lowering type T5 in
% docs/proposals/WAM_LOWERING_TAXONOMY_AND_MATRIX.md).
%
% A multi-clause predicate whose clauses discriminate on a *distinct*
% first-argument constant —
%
%     p(a) :- B1.   p(b) :- B2.   p(c) :- B3.
%
% compiles to a try_me_else / retry_me_else / trust_me chain where each
% clause body opens with `get_constant V, A1`. When the first argument is
% *bound* at call time this is deterministic first-argument dispatch, and a
% target can lower it to a native `->`-style cascade:
%
%     deref A1;  if unbound -> defer to the interpreter (enumeration);
%     else if A1 == a then B1  else if A1 == b then B2  else …  else fail
%
% This module is the *target-agnostic* front-end: it recognises the shape
% and returns the discriminator + per-clause remainder. Each target keeps
% its own back-end (deref / equality / clause-body emit). It deliberately
% mirrors how wam_ite_structurer.pl is shared across the T2 (if-then-else)
% back-ends.
%
% Soundness contract. The cascade is sound only because:
%   1. The discriminators are DISTINCT, so at most one clause matches a
%      bound first argument — the predicate is deterministic in that mode,
%      so taking the single matching clause is correct for boolean AND
%      enumeration queries (no solution is dropped).
%   2. The UNBOUND first-argument case is NOT handled by the cascade (it is
%      genuinely nondeterministic — every clause would match by binding
%      A1) and must be deferred to the full interpreter by the caller.
% Callers therefore emit the cascade behind a bound-check and rely on their
% existing interpreter fallback for the unbound case.

:- module(wam_clause_chain, [
    clause_chain/2          % +ParsedInstrs, -chain(Guards)
]).

:- use_module(library(lists)).

%% clause_chain(+ParsedInstrs, -chain(Guards)) is semidet.
%
%  ParsedInstrs is a predicate's instruction list with the choice-point
%  separators present (try_me_else/retry_me_else/trust_me) and labels /
%  switch_on_* already dropped — exactly what the per-target flat parsers
%  produce. Succeeds iff the predicate is a clean distinct-first-argument
%  constant discrimination of two or more clauses.
%
%  Guards is a list of `guard(Discriminator, Remainder)`:
%    - Discriminator : the first-argument constant token (the V in
%      `get_constant V, A1`), as produced by the parser (a string/atom).
%    - Remainder     : the clause body with that leading discriminator
%      removed (still ending in its `proceed`/terminal).
clause_chain(ParsedInstrs, chain(Guards)) :-
    split_clauses(ParsedInstrs, Clauses),
    Clauses = [_, _ | _],                       % at least two clauses
    maplist(clause_guard, Clauses, Guards),
    findall(V, member(guard(V, _), Guards), Discriminators),
    all_distinct_chain(Discriminators).

%% split_clauses(+Instrs, -Clauses) is semidet.
%  Split at the choice-point separators into per-clause instruction lists,
%  dropping the separators. Requires the list to open with try_me_else
%  (i.e. a genuine multi-clause predicate).
split_clauses([try_me_else(_) | Rest], Clauses) :- !,
    split_clauses_(Rest, [], Clauses).
split_clauses(_, _) :- fail.

split_clauses_([], Acc, [Clause]) :-
    reverse(Acc, Clause).
split_clauses_([retry_me_else(_) | Rest], Acc, [Clause | More]) :- !,
    reverse(Acc, Clause),
    split_clauses_(Rest, [], More).
split_clauses_([trust_me | Rest], Acc, [Clause | More]) :- !,
    reverse(Acc, Clause),
    split_clauses_(Rest, [], More).
split_clauses_([I | Rest], Acc, Clauses) :-
    split_clauses_(Rest, [I | Acc], Clauses).

%% clause_guard(+ClauseInstrs, -guard(V, Remainder)) is semidet.
%  A lowerable clause opens with `get_constant V, A1`. The shared compiler
%  emits get_constant for both atom and integer first arguments (e.g.
%  `get_constant red, A1` and `get_constant 1, A2`), so this single shape
%  covers flat constant discrimination. get_structure / get_list first
%  arguments are out of scope — they discriminate on a functor, not a flat
%  constant. Anything else fails the whole chain, so the predicate falls
%  back to the caller's default path.
clause_guard([get_constant(V, A1) | Remainder], guard(V, Remainder)) :-
    is_first_arg_reg(A1).

is_first_arg_reg(A1) :- ( A1 == "A1" ; A1 == 'A1' ; A1 == a1 ), !.

%% all_distinct_chain(+List) is semidet.  No two elements are == equal.
all_distinct_chain(List) :-
    \+ ( append(_, [X | Rest], List), member(Y, Rest), X == Y ).
