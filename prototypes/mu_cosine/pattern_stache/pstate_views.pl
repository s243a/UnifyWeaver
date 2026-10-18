:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% pstate_views.pl - EXPERIMENT HARNESS: two candidate views of a
% pattern state, emitted from the same real elaboration so the encoder
% lane can ablate them against each other.
%
% ============================================================
% THIS IS NOT AN IDENTITY ENCODING.  READ THIS PARAGRAPH.
% ============================================================
% Ruling 6 (DESIGN_prolog_elaborator.md §5) asks whether residual goals
% reach the encoder as STRUCTURE (tokens in the stream) or as FEATURES
% (a side channel).  It is deliberately undecided, and deciding it by
% shipping one view would freeze a canonical form by accident — the
% exact hazard this lane keeps recording.  So this module emits BOTH
% and commits to NEITHER.
%
% Consequently, and without exception:
%   - NO digest is computed here, and none may be computed from these
%     views;
%   - NO persistent name is assigned to any pattern state;
%   - NO cache key, no seal, no golden bundle derives from this output;
%   - the version stamp is `pstate-probe-v0` — an EXPERIMENT version,
%     never a contract version.  It shares no namespace with pec-v3,
%     peid-v1, tok-v1, or any registry version, and a later reader must
%     not mistake it for one.
%   - whatever `peid-v1` rules for pattern-state canonical form
%     SUPERSEDES this file entirely.  These views are scaffolding to be
%     discarded, not a default to be inherited.
%
% Fixtures that name a state use file-local names under the mandatory
% no_identity/1 marker (ruling 7(b)); see test_pstate_views.pl.
%
% ============================================================
% KNOWN DEFECT INHERITED FROM THE STORE ORDERING
% ============================================================
% The STRUCTURE view serializes the residual store in the order
% pe_elaborate's canonical_store produces.  That ordering has a
% measured input-order dependence when two or more residual goals
% share a projection AND are projection-least (see
% DESIGN_pattern_state_identity.md "The ordering defect", and the
% characterization test in test_pstate_views.pl).  The defect is NOT
% patched around here — the harness emits what the elaborator gives it,
% so the instability stays visible to whoever ablates these views.
%
% The FEATURES view is immune by construction (it is a sorted bag of
% counts, so goal order cannot reach it).  That asymmetry is itself an
% input to ruling 6 and is why emitting both matters.

:- module(pstate_views, [
    pstate_probe_version/1,       % -Version
    pstate_of/3,                  % +Term, +Goals, -PatternState
    pstate_structure_view/2,      % +PatternState, -Tokens
    pstate_features_view/2,       % +PatternState, -Features
    pstate_both_views/4,          % +Term, +Goals, -Tokens, -Features
    print_views/2                 % +Term, +Goals   (runnable demo)
]).

:- use_module(pe_elaborate, [elaborate/3]).
:- use_module(library(lists)).
:- use_module(library(apply)).
:- use_module(library(pairs)).

pstate_probe_version('pstate-probe-v0').

%% pstate_of(+Term, +Goals, -PatternState)
%  Generate a pattern state from a REAL elaboration (never a
%  hand-written state).  A store that fully discharges yields
%  ground/1 — the views refuse it, because a ground term is not a
%  pattern state and already has a sealed canonical form.
pstate_of(Term, Goals, State) :-
    elaborate(Term, Goals, State0),
    (   State0 = pattern(_, _)
    ->  State = State0
    ;   throw(error(pstate_views(not_a_pattern_state(State0)), _))
    ).

%% pstate_both_views(+Term, +Goals, -Tokens, -Features)
pstate_both_views(Term, Goals, Tokens, Features) :-
    pstate_of(Term, Goals, State),
    pstate_structure_view(State, Tokens),
    pstate_features_view(State, Features).

%% ============================================
%% VIEW 1: STRUCTURE  (ruling 6 answer: "as structure")
%% ============================================
%
% A token stream with role paths, shaped like the sealed bundle's
% token/role-path pairs so an encoder that already consumes those can
% consume these — but in a DELIBERATELY DISTINCT namespace
% (`<PSTATE:...>`, `PSTATE_ROOT`) so no reader and no tool can mistake
% probe tokens for pec-v3 tokens.
%
% Variables appear as <VAR:N> slots.  N comes from the elaborator's
% numbering, which is an ORDERING DEVICE, not a variable identity
% scheme (DESIGN_pattern_state_identity.md, "The VarId question").
% Anything downstream that treats <VAR:N> as an identity is reading
% more into it than exists.

pstate_structure_view(pattern(Term, Store), Tokens) :-
    copy_term(Term-Store, TermC-StoreC),
    numbervars(TermC-StoreC, 0, _),
    pstate_probe_version(V),
    format(atom(Header), '<PSTATE:~w>', [V]),
    term_tokens(TermC, 'PSTATE_ROOT/TERM', TermToks),
    store_tokens(StoreC, 0, StoreToks),
    append([[tok(Header, 'PSTATE_ROOT'), tok('<TERM>', 'PSTATE_ROOT')],
            TermToks,
            [tok('<STORE>', 'PSTATE_ROOT')],
            StoreToks,
            [tok('<PSTATE_EOS>', 'PSTATE_ROOT')]],
           Tokens).

store_tokens([], _, []).
store_tokens([G|Gs], I, Tokens) :-
    format(atom(Path), 'PSTATE_ROOT/STORE/GOAL(~w)', [I]),
    format(atom(GoalTok), '<GOAL:~w>', [I]),
    term_tokens(G, Path, GTs),
    I1 is I + 1,
    store_tokens(Gs, I1, Rest),
    append([[tok(GoalTok, Path)], GTs, Rest], Tokens).

%% term_tokens(+NumberedTerm, +Path, -Tokens)
term_tokens('$VAR'(N), Path, [tok(Tok, Path)]) :-
    !,
    format(atom(Tok), '<VAR:~w>', [N]).
term_tokens(A, Path, [tok(Tok, Path)]) :-
    atom(A),
    !,
    format(atom(Tok), '<ATOM:~w>', [A]).
term_tokens(N, Path, [tok(Tok, Path)]) :-
    number(N),
    !,
    format(atom(Tok), '<NUM:~w>', [N]).
term_tokens(S, Path, [tok(Tok, Path)]) :-
    string(S),
    !,
    format(atom(Tok), '<STR:~w>', [S]).
term_tokens(T, Path, Tokens) :-
    compound(T),
    compound_name_arity(T, Name, Arity),
    format(atom(FTok), '<FUNCTOR:~w/~w>', [Name, Arity]),
    T =.. [_|Args],
    arg_tokens(Args, 0, Path, ArgToks),
    Tokens = [tok(FTok, Path)|ArgToks].

arg_tokens([], _, _, []).
arg_tokens([A|As], I, Path, Tokens) :-
    format(atom(SubPath), '~w/ARG(~w)', [Path, I]),
    term_tokens(A, SubPath, T1),
    I1 is I + 1,
    arg_tokens(As, I1, Path, T2),
    append(T1, T2, Tokens).

%% ============================================
%% VIEW 2: FEATURES  (ruling 6 answer: "as features")
%% ============================================
%
% A sorted bag of Key-Value counts: a fixed-shape side channel that
% says WHAT constrains the term without saying in what order.  Order
% cannot reach this view — every feature is a count or a sorted set
% membership — which is precisely why it is immune to the store
% ordering defect the structure view inherits.
%
% Features are deliberately shallow: they describe the residual store's
% shape, not its content-addressed identity.  Adding a feature here is
% cheap; nothing downstream may treat this list as complete or stable
% across probe versions.

pstate_features_view(pattern(Term, Store), Features) :-
    copy_term(Term-Store, TermC-StoreC),
    numbervars(TermC-StoreC, 0, NVars),
    length(StoreC, GoalCount),
    term_variables(Term, TermVars0),
    length(TermVars0, TermVarCount),
    StoreOnly is NVars - TermVarCount,
    % per-functor goal counts, sorted
    maplist(goal_functor, StoreC, Functors),
    msort(Functors, SortedFunctors),
    count_runs(SortedFunctors, FunctorCounts),
    % per-constraint-kind counts (the type-side functor of has_type/2)
    convlist(constraint_kind, StoreC, Kinds),
    msort(Kinds, SortedKinds),
    count_runs(SortedKinds, KindCounts),
    % variable degree: how many goals mention each variable
    var_degrees(StoreC, NVars, Degrees),
    max_or_zero(Degrees, MaxDegree),
    sum_list(Degrees, DegreeSum),
    maplist([F-C, Key-C]>>format(atom(Key), 'goal_functor(~w)', [F]),
            FunctorCounts, FunctorFeatures),
    maplist([F-C, Key-C]>>format(atom(Key), 'constraint_kind(~w)', [F]),
            KindCounts, KindFeatures),
    Base = [ 'residual_goal_count'-GoalCount,
             'distinct_var_count'-NVars,
             'term_var_count'-TermVarCount,
             'store_only_var_count'-StoreOnly,
             'var_degree_max'-MaxDegree,
             'var_occurrence_total'-DegreeSum,
             'term_is_ground'-0 ],
    append([Base, FunctorFeatures, KindFeatures], All),
    msort(All, Features).

goal_functor(G, Name/Arity) :-
    (   compound(G)
    ->  compound_name_arity(G, Name, Arity)
    ;   Name = G, Arity = 0
    ).

% the type-side constructor of a has_type/2 goal: substrate, judge, ...
constraint_kind(has_type(_, T), Kind) :-
    compound(T),
    !,
    compound_name_arity(T, Kind, _).
constraint_kind(has_type(_, T), T) :-
    atom(T).

count_runs([], []).
count_runs([X|Xs], [X-N|Rest]) :-
    take_run(X, Xs, 1, N, Tail),
    count_runs(Tail, Rest).

take_run(X, [Y|Ys], N0, N, Tail) :-
    Y == X,
    !,
    N1 is N0 + 1,
    take_run(X, Ys, N1, N, Tail).
take_run(_, Rest, N, N, Rest).

%% var_degrees(+NumberedStore, +NVars, -Degrees)
%  Degrees[i] = number of goals mentioning '$VAR'(i).
var_degrees(Store, NVars, Degrees) :-
    Last is NVars - 1,
    (   NVars =:= 0
    ->  Degrees = []
    ;   numlist(0, Last, Is),
        maplist(degree_of(Store), Is, Degrees)
    ).

degree_of(Store, I, D) :-
    include(mentions_var(I), Store, Ms),
    length(Ms, D).

mentions_var(I, Goal) :-
    term_contains_var(Goal, I),
    !.

term_contains_var('$VAR'(I), I) :- !.
term_contains_var(T, I) :-
    compound(T),
    T =.. [_|Args],
    member(A, Args),
    term_contains_var(A, I),
    !.

max_or_zero([], 0) :- !.
max_or_zero(L, M) :- max_list(L, M).

%% ============================================
%% RUNNABLE DEMO
%% ============================================

%% print_views(+Term, +Goals)
%  Given a goal term and a goal store, print both views.  Run e.g.:
%    swipl -g "use_module(pstate_views), print_views(
%        product(hop_decay(C,gamma(0.6)), lca_frac(C)),
%        [has_type(_X, substrate(C))])" -t halt
print_views(Term, Goals) :-
    pstate_of(Term, Goals, State),
    pstate_probe_version(V),
    format("=== pattern state (~w) - NOT an identity encoding ===~n", [V]),
    copy_term(State, Shown),
    numbervars(Shown, 0, _),
    format("~q~n~n", [Shown]),
    pstate_structure_view(State, Tokens),
    length(Tokens, NT),
    format("-- view 1: STRUCTURE (~w tokens) --~n", [NT]),
    forall(member(tok(T, P), Tokens), format("  ~w~t~28|~w~n", [T, P])),
    pstate_features_view(State, Features),
    length(Features, NF),
    format("~n-- view 2: FEATURES (~w keys) --~n", [NF]),
    forall(member(K-Val, Features), format("  ~w~t~34|~w~n", [K, Val])).
