:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% pe_canonical.pl - canonical labelling of a pattern state's variables:
% colour refinement to stability, then INDIVIDUALIZATION.
%
% Implements ruling 1 of DESIGN_pattern_state_identity.md as amended
% (PR #4098): refinement alone is sound but INCOMPLETE — it can
% stabilize with a non-singleton colour class on symmetric structures,
% and an input-order tie-break at that point would rebuild the §2
% ordering defect one layer deeper ("accidentally stable when
% refinement separates").  Individualization is what makes this a
% canonical FORM rather than merely an invariant.
%
% ============================================================
% WHAT THIS IS AND IS NOT
% ============================================================
% This computes a canonical FORM — a normal form under which two
% presentations of one pattern state produce identical output.  A
% canonical form is NOT an identity: no digest is computed here, none
% may be derived from this output, and no cache or namespace may be
% keyed by it.  Ruling 4 stands (no early sealing of a candidate
% view); pattern-state digests remain fenced behind peid-v1.  If
% having a canonical form makes a digest look easy, that is precisely
% when not to add one.
%
% ============================================================
% THE ALGORITHM
% ============================================================
%   1. Colour every variable identically to start.
%   2. REFINE: recolour each variable by (its old colour, the sorted
%      multiset of its occurrence contexts, where each context is the
%      enclosing term or goal with every OTHER variable replaced by
%      its colour and this variable's own positions marked `self`).
%      Re-rank the resulting keys to integers.  Repeat until the
%      colour assignment stops changing.  Refinement only ever splits
%      classes, and the distinct-colour count is bounded by the
%      variable count, so this terminates.
%   3. If every colour class is a singleton, the colouring IS the
%      canonical variable order: done.
%   4. Otherwise INDIVIDUALIZE: take the SMALLEST non-singleton class
%      (ties broken by lowest colour, which is itself canonical), and
%      for each member branch — give that member a colour strictly
%      below its classmates, re-refine, recurse.  Compare the
%      completed forms and take the LEXICOGRAPHIC MINIMUM.
%      Every member is tried, so the result does not depend on which
%      member the enumeration happened to reach first: that is the
%      difference between this and a tie-break.
%
% Correctness property (ruled): for any store, EVERY input permutation
% must yield =@= output.  See permutation_stable/2 in
% test_pstate_views.pl — digest-free, so it sits left of the identity
% fence.
%
% ============================================================
% THE BRANCH CAP: REFUSE, NEVER DEGRADE
% ============================================================
% Individualization-refinement is exponential in the worst case.  At
% pattern-state sizes (a handful of goals) that is irrelevant, but a
% pathological store must REFUSE rather than hang or silently fall
% back to a weaker labelling — this lane's standing discipline.
%
% The cap counts individualization BRANCHES EXPLORED (not depth, not
% time): every recursive descent into one member of a non-singleton
% class consumes one unit.  On exhaustion the labelling throws
%     error(pe_canonical(branch_cap_exceeded(Cap, Vars, Goals)), _)
% and computes nothing.  There is no timeout, no partial result, and
% no fallback to the old ordering device: a state that cannot be
% canonically labelled within the cap has NO canonical form, and the
% caller is told so rather than handed an arbitrary one.

:- module(pe_canonical, [
    canonical_number/2,          % +Term, +Goals   (destructive on a copy)
    canonical_branch_cap/1,      % -Cap
    canonical_colours/3          % +Term, +Goals, -ColourList  (introspection)
]).

:- use_module(library(lists)).
:- use_module(library(apply)).
:- use_module(library(pairs)).

%% canonical_branch_cap(-Cap)
%  Individualization branches explored before refusal.  1000 is far
%  above anything a real pattern state reaches (the largest store the
%  lane has produced is 3 goals, which needs zero individualization),
%  and far below anything that would hang a test run.
canonical_branch_cap(1000).

%% canonical_number(+Term, +Goals)
%  Bind every variable in Term-Goals to '$VAR'(N) in canonical order.
%  DESTRUCTIVE: the caller must pass a private copy.  Goals is the
%  store as a list; its list order is irrelevant to the result, which
%  is the entire point.
canonical_number(Term, Goals) :-
    term_variables(Term-Goals, Vars),
    (   Vars == []
    ->  true
    ;   length(Vars, NV),
        maplist([_, 0]>>true, Vars, Colours0),
        canonical_branch_cap(Cap),
        canonicalize(Term, Goals, Vars, Colours0, Cap, _Left, BestColours),
        (   BestColours == none
        ->  throw(error(pe_canonical(branch_cap_exceeded(Cap, NV, Goals)), _))
        ;   true
        ),
        bind_colours(Vars, BestColours)
    ).

%% canonical_colours(+Term, +Goals, -Colours)
%  The canonical colour of each variable of Term-Goals, aligned with
%  term_variables/2 order.  Introspection for tests; computes on a
%  copy so the caller's term is untouched.
canonical_colours(Term, Goals, Colours) :-
    copy_term(Term-Goals, T2-G2),
    term_variables(T2-G2, Vars),
    (   Vars == []
    ->  Colours = []
    ;   maplist([_, 0]>>true, Vars, C0),
        canonical_branch_cap(Cap),
        canonicalize(T2, G2, Vars, C0, Cap, _, Colours),
        Colours \== none
    ).

bind_colours([], []).
bind_colours([V|Vs], [C|Cs]) :-
    V = '$VAR'(C),
    bind_colours(Vs, Cs).

%% ============================================
%% REFINE, THEN INDIVIDUALIZE
%% ============================================

%% canonicalize(+Term, +Goals, +Vars, +Colours0, +Budget0, -Budget, -Best)
%  Best is the winning colour list, or `none` when the budget ran out
%  anywhere in the search (propagated so the caller refuses rather
%  than accepting a partial exploration).
canonicalize(Term, Goals, Vars, Colours0, Budget0, Budget, Best) :-
    refine_to_fixpoint(Term, Goals, Vars, Colours0, Colours),
    (   all_singletons(Colours)
    ->  Best = Colours,
        Budget = Budget0
    ;   smallest_nonsingleton_class(Vars, Colours, Members),
        branch_members(Members, Term, Goals, Vars, Colours,
                       Budget0, Budget, none, Best)
    ).

%% branch_members(+Members, ..., +BestSoFar, -Best)
%  Try EVERY member of the chosen class; keep the lexicographically
%  least completed form.  Trying every member is what makes this
%  independent of enumeration order.
branch_members([], _, _, _, _, Budget, Budget, Best, Best).
branch_members([V|Vs], Term, Goals, Vars, Colours, Budget0, Budget, Best0, Best) :-
    (   Budget0 =< 0
    ->  Budget = 0,
        Best = none                    % refuse: propagate exhaustion
    ;   Budget1 is Budget0 - 1,
        individualize(V, Vars, Colours, Colours1),
        canonicalize(Term, Goals, Vars, Colours1, Budget1, Budget2, Candidate),
        (   Candidate == none
        ->  Budget = 0,
            Best = none
        ;   better_of(Best0, Candidate, Term, Goals, Vars, Best1),
            branch_members(Vs, Term, Goals, Vars, Colours,
                           Budget2, Budget, Best1, Best)
        )
    ).

%% better_of(+BestSoFar, +Candidate, ..., -Winner)
%  Compare completed colourings by the FORM they induce — the numbered
%  term paired with the numbered, sorted store — not by the colour
%  list itself, because the form is the thing that must be canonical.
better_of(none, Candidate, _, _, _, Candidate) :- !.
better_of(Best, Candidate, Term, Goals, Vars, Winner) :-
    induced_form(Term, Goals, Vars, Best, FormB),
    induced_form(Term, Goals, Vars, Candidate, FormC),
    (   FormC @< FormB
    ->  Winner = Candidate
    ;   Winner = Best
    ).

%% induced_form(+Term, +Goals, +Vars, +Colours, -Form)
%  The ground form a colouring induces: variables replaced by their
%  colours, store sorted (it is a set, so its list order carries no
%  information and must not survive into the form).
induced_form(Term, Goals, Vars, Colours, Term1-Sorted) :-
    copy_term(Vars-Term-Goals, Vars1-Term1-Goals1),
    bind_colours(Vars1, Colours),
    msort(Goals1, Sorted).

%% individualize(+V, +Vars, +Colours, -Colours1)
%  Give V a colour strictly below its classmates, then re-rank.  The
%  caller re-refines, which propagates the new distinction outward.
individualize(V, Vars, Colours, Colours1) :-
    maplist(ind_key(V), Vars, Colours, Keys),
    rank_keys(Keys, Colours1).

ind_key(V, W, C, Key) :-
    (   W == V
    ->  Key = C-0
    ;   Key = C-1
    ).

%% ============================================
%% COLOUR REFINEMENT
%% ============================================

refine_to_fixpoint(Term, Goals, Vars, Colours0, Colours) :-
    refine_once(Term, Goals, Vars, Colours0, Colours1),
    (   Colours1 == Colours0
    ->  Colours = Colours0
    ;   refine_to_fixpoint(Term, Goals, Vars, Colours1, Colours)
    ).

%% refine_once(+Term, +Goals, +Vars, +Colours, -Colours1)
%  New key per variable: its old colour paired with the sorted
%  multiset of its occurrence contexts.  Re-ranked to integers, so
%  equal partitions always produce equal colour lists and the
%  fixpoint test is a plain ==.
refine_once(Term, Goals, Vars, Colours, Colours1) :-
    maplist(var_key(Term, Goals, Vars, Colours), Vars, Keys),
    rank_keys(Keys, Colours1).

var_key(Term, Goals, Vars, Colours, V, C-Sig) :-
    colour_of(V, Vars, Colours, C),
    findall(S,
            ( context(Term, Goals, Ctx),
              occurs_in(V, Ctx),
              skeleton(Ctx, V, Vars, Colours, S)
            ),
            Sigs0),
    msort(Sigs0, Sig).

% The term and each goal are distinct context kinds: a variable
% occurring in the term is not interchangeable with one occurring only
% in the store, and the wrapper keeps that visible to refinement.
context(Term, _, term_ctx(Term)).
context(_, Goals, goal_ctx(G)) :- member(G, Goals).

occurs_in(V, T) :-
    term_variables(T, Vs),
    memberchk_eq(V, Vs).

memberchk_eq(V, [X|Xs]) :-
    (   V == X
    ->  true
    ;   memberchk_eq(V, Xs)
    ).

%% skeleton(+Ctx, +V, +Vars, +Colours, -Skel)
%  Ctx with every variable replaced by its colour, except occurrences
%  of V, which become `self`.  Marking V's own positions is what makes
%  two variables in the same goal distinguishable by position.
skeleton(T, V, _, _, self) :-
    var(T),
    T == V,
    !.
skeleton(T, _, Vars, Colours, c(C)) :-
    var(T),
    !,
    colour_of(T, Vars, Colours, C).
skeleton(T, _, _, _, T) :-
    atomic(T),
    !.
skeleton(T, V, Vars, Colours, S) :-
    compound(T),
    compound_name_arguments(T, F, Args),
    skeleton_args(Args, V, Vars, Colours, SArgs),
    compound_name_arguments(S, F, SArgs).

% NOT a yall lambda, deliberately.  `maplist([A,SA]>>skeleton(A,V,...))`
% is wrong here and silently so: yall's >> COPIES the lambda's free
% variables on every call, so V would be renamed to a fresh variable,
% `T == V` would never hold, and colour_of/4 would fail on a copied
% Vars list — leaving refinement inert while individualization quietly
% produced correct-but-exponential answers.  This is the same
% copy-severs-variable-identity hazard recorded in desugaring §12 for
% findall/3, in a third venue.  A plain helper predicate shares its
% arguments and cannot exhibit it.
skeleton_args([], _, _, _, []).
skeleton_args([A|As], V, Vars, Colours, [S|Ss]) :-
    skeleton(A, V, Vars, Colours, S),
    skeleton_args(As, V, Vars, Colours, Ss).

colour_of(V, [W|Ws], [C|Cs], Colour) :-
    (   V == W
    ->  Colour = C
    ;   colour_of(V, Ws, Cs, Colour)
    ).

%% rank_keys(+Keys, -Ranks)
%  Replace each key by its index in the sorted set of distinct keys.
%  This is what makes colours canonical given the partition, so the
%  fixpoint test can be a plain term comparison.
rank_keys(Keys, Ranks) :-
    sort(Keys, Distinct),
    maplist(rank_of(Distinct), Keys, Ranks).

rank_of(Distinct, Key, Rank) :-
    nth0(Rank, Distinct, K),
    K == Key,
    !.

%% ============================================
%% PARTITION HELPERS
%% ============================================

all_singletons(Colours) :-
    msort(Colours, Sorted),
    no_adjacent_duplicates(Sorted).

no_adjacent_duplicates([]).
no_adjacent_duplicates([_]) :- !.
no_adjacent_duplicates([A, B|T]) :-
    A \== B,
    no_adjacent_duplicates([B|T]).

%% smallest_nonsingleton_class(+Vars, +Colours, -Members)
%  The smallest non-singleton class; ties broken by lowest colour.
%  Both criteria are functions of the colouring, which is itself
%  canonical at this point, so the choice of class is canonical too —
%  only the choice WITHIN the class needs branching.
smallest_nonsingleton_class(Vars, Colours, Members) :-
    msort(Colours, Sorted),
    setof(C, member(C, Sorted), Distinct),
    findall(Size-C,
            ( member(C, Distinct),
              class_members(Vars, Colours, C, Ms),
              length(Ms, Size),
              Size > 1
            ),
            Classes),
    msort(Classes, [_-Chosen|_]),
    class_members(Vars, Colours, Chosen, Members).

class_members([], [], _, []).
class_members([V|Vs], [C|Cs], Target, Members) :-
    (   C == Target
    ->  Members = [V|Rest]
    ;   Members = Rest
    ),
    class_members(Vs, Cs, Target, Rest).
