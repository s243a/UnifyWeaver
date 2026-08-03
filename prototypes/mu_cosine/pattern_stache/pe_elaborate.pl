:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% pe_elaborate.pl - PROTOTYPE Prolog-side elaborator: the fixpoint
% discharge loop over the goal store.
%
% Implements DESIGN_prolog_elaborator.md to its ruled design (all
% seven §5 rulings decided in PR #4093).  Ground goals DISCHARGE at
% elaboration; under-instantiated goals RESIDUATE — they travel with
% the term as part of what it is (desugaring doc §§3-6).
%
% Desugar scope is ruling 4(a), nothing else:
%   - BINDING GOALS   V = Value      discharged by substitution (§12);
%   - has_type/2      has_type(X, T) discharged by the closed registry
%                     check when ground;
%   - anything else is refused fail-closed (unknown_constraint) — its
%     functor can never become known, so it errors at first sight.
%
% States (type-per-state, constructor-checked):
%   ground(Term)            every goal discharged, Term ground.  This
%                           is exactly what pe_emit renders; its
%                           canonical bytes are the sealed golden
%                           surface.
%   pattern(Term, Store)    Term may contain variables; Store is the
%                           canonicalized residual store (ruling 1:
%                           numbervars-by-traversal projection order,
%                           ==-dedup).  NO digest, NO canonical bytes,
%                           NO persistent name — peid-v1's fence.
%
% Origin metadata (ruling 3): carried ALONGSIDE the store, never
% inside it.  elaborate/4 takes Goal-Origin pairs and returns residual
% origins as pairs aligned with the canonical store; the store inside
% pattern/2 holds pure goals.  Errors name the origin when one exists.
%
% Reuse (the note's §1 architecture map): pe_where's validation
% machinery is imported — binding shapes, duplicate refusal, the
% occurrence walker (walking, never copying: the findall/3 hazard is
% recorded in desugaring §12), and per-position legality including the
% ratified both-directions pin refusals.  Registry facts come from the
% generated, hash-checked mirror (ruling 5(b)).
%
% What replaced pe_where's "residuation is the elaborator's future"
% clause: exactly one thing — where pe_where throws
% unbound_after_elaboration, this module returns pattern(Term, Store).
% pe_where itself is unchanged and remains the ground-only fast path;
% pe_emit's nonground_dispatch backstop also survives untouched.
%
% Termination: each pass either discharges at least one goal — the
% store strictly shrinks, and nothing ever adds a goal — or the loop
% stops.  Substitution only instantiates variables (monotone), never
% unbinds, so a discharged goal cannot become undischarged.  With a
% finite store of N goals the loop runs at most N passes.

:- module(pe_elaborate, [
    elaborate/3,              % +Term, +Goals, -State
    elaborate/4,              % +Term, +GoalOriginPairs, -State, -ResidualOrigins
    canonical_store/3         % +Term, +Goals, -CanonGoals
]).

:- use_module(pe_where,
              [check_binding_shapes/1, check_no_duplicates/1,
               occurrences/3, check_value_at/2]).
:- use_module(pe_registry_mirror,
              [pe_output/2, pe_modifier/2]).
:- use_module(library(lists)).
:- use_module(library(apply)).
:- use_module(library(pairs)).

%% elaborate(+Term, +Goals, -State)
%  Plain-goal entry: every goal gets origin `none`.
elaborate(Term, Goals, State) :-
    maplist([G, G-none]>>true, Goals, Pairs),
    elaborate(Term, Pairs, State, _ResidualOrigins).

%% elaborate(+Term0, +Pairs0, -State, -ResidualOrigins)
%  Pairs0 is a list of Goal-Origin.  State is ground(Term) or
%  pattern(Term, Store); ResidualOrigins is a list of Goal-Origin
%  aligned with the canonical store order ([] on the ground path).
elaborate(Term0, Pairs0, State, ResidualOrigins) :-
    must_be_pairs(Pairs0),
    pairs_keys(Pairs0, Goals0),
    % one copy, sharing preserved, caller's term never mutated;
    % origins are ground metadata and re-attach by position
    copy_term(Term0-Goals0, Term-Goals),
    pairs_values(Pairs0, Origins),
    pairs_keys_values(Pairs, Goals, Origins),
    % binding-goal validation up front, pe_where semantics: shapes and
    % duplicates are spelling accidents, refused before any work
    include(is_binding_pair, Pairs, BindingPairs),
    pairs_keys(BindingPairs, Bindings),
    check_binding_shapes(Bindings),
    check_no_duplicates(Bindings),
    check_no_dead_bindings(Pairs, Term),
    fixpoint(Pairs, Term, Residual),
    (   Residual == [],
        ground(Term)
    ->  State = ground(Term),
        ResidualOrigins = []
    ;   canonical_pairs(Term, Residual, CanonPairs),
        pairs_keys(CanonPairs, Store),
        State = pattern(Term, Store),
        ResidualOrigins = CanonPairs
    ).

must_be_pairs(Pairs) :-
    (   is_list(Pairs),
        forall(member(P, Pairs), (nonvar(P), P = _-_))
    ->  true
    ;   throw(error(pe_elaborate(bad_goal_pairs(Pairs)), _))
    ).

% Any =/2 goal is claimed by the binding channel, so a malformed one
% (nonvar left-hand side) is refused by the reused shape check rather
% than misread as a constraint.
is_binding_pair(G-_) :-
    nonvar(G),
    G = (_ = _).

%% check_no_dead_bindings(+Pairs, +Term)
%  A binding's variable must occur in the Term or in some OTHER goal —
%  its own left-hand side does not count, or every binding would be
%  trivially live.  Dead bindings hide typos (pe_where semantics,
%  domain widened to the store per the note's architecture).
check_no_dead_bindings(Pairs, Term) :-
    pairs_keys(Pairs, Goals),
    forall(( member((V = Val)-_, Pairs), var(V) ),
           (   (   term_var_member(V, Term)
               ;   member(Other, Goals),
                   Other \== (V = Val),
                   \+ same_binding(Other, V),
                   term_var_member(V, Other)
               )
           ->  true
           ;   throw(error(pe_elaborate(dead_binding(V = Val)), _))
           )).

same_binding(G, V) :- nonvar(G), G = (W = _), W == V.

term_var_member(V, T) :-
    term_variables(T, Vs),
    member(X, Vs),
    X == V,
    !.

%% ============================================
%% THE FIXPOINT LOOP
%% ============================================

%% fixpoint(+Pairs, +Term, -ResidualPairs)
%  One pass attempts every goal; discharging a binding instantiates
%  variables, which can make another residual checkable, so the loop
%  re-runs until a pass discharges nothing (the note's own
%  observation).  See the module header for the termination argument.
fixpoint(Pairs, Term, Residual) :-
    pass(Pairs, Term, Kept, Progress),
    (   Progress == true
    ->  fixpoint(Kept, Term, Residual)
    ;   Residual = Kept
    ).

%% pass(+Pairs, +Term, -KeptPairs, -Progress)
pass([], _, [], false).
pass([Pair|Rest], Term, Kept, Progress) :-
    attempt(Pair, Term, Outcome),
    (   Outcome == discharged
    ->  Kept = KeptRest,
        Progress = true,
        pass(Rest, Term, KeptRest, _)
    ;   Kept = [Pair|KeptRest],
        pass(Rest, Term, KeptRest, Progress)
    ).

%% attempt(+Goal-Origin, +Term, -Outcome)   Outcome: discharged | residual
%
% BINDING GOAL: discharges when its value is ground — first checking
% structural legality at every position the variable reaches (in the
% term AND the other goals' arguments), including the ratified pin
% refusals, then substituting by unification so every occurrence is
% reached at once.  A binding whose value is still nonground waits:
% another discharge may ground it.
attempt((V = Val)-Origin, Term, Outcome) :-
    var(V),
    !,
    (   ground(Val)
    ->  check_binding_positions(V, Val, Term, Origin),
        V = Val,
        Outcome = discharged
    ;   Outcome = residual
    ).
% A binding whose variable got instantiated by an earlier substitution
% chain cannot happen — duplicates are refused up front and nothing
% else binds a store variable except its own discharge — but fail
% closed rather than assume.
attempt((V = Val)-Origin, _Term, _) :-
    nonvar(V),
    !,
    throw(error(pe_elaborate(binding_variable_already_bound(V = Val, origin(Origin))), _)).
% has_type/2: discharges when GROUND, by the closed registry check.
% A failed check is an ERROR in the author's vocabulary (§7), not a
% silent residual; an under-instantiated has_type residuates.
%
% v1.1 (ruled on PR #4095): EAGER VALIDATION WITHOUT GOAL REWRITING.
% All-or-nothing discharge stays, but a residual whose ground portion
% is checkable gets checked eagerly — throw on illegal, change nothing
% on legal.  The deciding distinction: under-instantiated residuates,
% but ground-and-false is an error — has_type(_X, substrate(frobnicate))
% must fail at elaboration, not sit dormant in a pattern state as a
% constraint that can never ground.  No goal splitting, no narrowed
% residuals: the residual goal stays textually whole, so the store's
% content (and any future peid-v1 canonical form over it) is untouched
% by the check.
attempt(has_type(X, T)-Origin, _Term, Outcome) :-
    !,
    (   ground(has_type(X, T))
    ->  (   has_type_holds(T)
        ->  Outcome = discharged
        ;   throw(error(pe_elaborate(constraint_failed(has_type(X, T), origin(Origin))), _))
        )
    ;   eager_check_has_type(has_type(X, T), Origin),
        Outcome = residual
    ).
% Anything else: outside ruling 4(a)'s scope.  Its functor can never
% become known, so this is an error now, not a residual (fail closed).
attempt(Goal-Origin, _Term, _) :-
    throw(error(pe_elaborate(unknown_constraint(Goal, origin(Origin))), _)).

%% eager_check_has_type(+Goal, +Origin)
%  The v1.1 check over a RESIDUATING has_type: validate exactly the
%  ground portion, mutate nothing.
%    - type side fully ground -> the full registry check must hold
%      (the ruling's own example: substrate(frobnicate) with a free
%      subject throws now, not never);
%    - type side partially ground -> its ground SPINE must still be
%      satisfiable: the constructor must be an arity-1 kind with at
%      least one registered inhabitant (has_type(_X, frobnicate(_C))
%      can never ground to a registered check, so it throws), and a
%      partially-ground mod/2 subject must keep its ground half
%      registered;
%    - type side unbound -> nothing ground to check; residuate.
%  Failure throws constraint_unsatisfiable/2 — named distinctly from
%  constraint_failed/2 (a GROUND goal failing at discharge) so the
%  diagnostic says which situation the author is in.
eager_check_has_type(has_type(_X, T), _Origin) :-
    var(T),
    !.
eager_check_has_type(has_type(X, T), Origin) :-
    ground(T),
    !,
    (   has_type_holds(T)
    ->  true
    ;   throw(error(pe_elaborate(constraint_unsatisfiable(has_type(X, T), origin(Origin))), _))
    ).
eager_check_has_type(has_type(X, T), Origin) :-
    (   ground_spine_satisfiable(T)
    ->  true
    ;   throw(error(pe_elaborate(constraint_unsatisfiable(has_type(X, T), origin(Origin))), _))
    ).

% ground_spine_satisfiable(+T): T is a nonground type term; its ground
% parts must still admit SOME registered grounding.
ground_spine_satisfiable(T) :-
    compound(T),
    compound_name_arity(T, Kind, 1),
    arg(1, T, C),
    kind_inhabited(Kind),
    subject_spine_satisfiable(C, Kind).

kind_inhabited(Kind) :-
    pe_output(_, Kind),
    !.

subject_spine_satisfiable(C, _Kind) :-
    var(C),
    !.
subject_spine_satisfiable(mod(B, M), Kind) :-
    !,
    (   var(B)
    ->  true
    ;   atom(B),
        pe_output(B, Kind)
    ),
    (   var(M)
    ->  (   atom(B)
        ->  pe_modifier(B, _)          % some modifier must exist on B
        ;   true
        )
    ;   atom(B)
    ->  pe_modifier(B, M)
    ;   true
    ).
% a nonvar, non-mod subject inside a NONGROUND type term: the only
% remaining nonground position was the kind side, which is impossible
% here (the kind is the functor) — so a ground atom subject with a
% ground kind was handled by the ground(T) clause; anything else is a
% shape that can never satisfy the registry check.
subject_spine_satisfiable(C, Kind) :-
    atom(C),
    pe_output(C, Kind).

%% has_type_holds(+TypeTerm)
%  The closed check the generated mirror supports: has_type(_X, K(C))
%  holds when C is registered with output kind K — for every output
%  kind (substrate, judge, score, target-set, pick) — and a modified
%  base mod(B, M) requires the modifier registered on B.  This is
%  structural registry lookup, not μ-typing; the type system proper
%  remains vNext's.
has_type_holds(TypeTerm) :-
    compound(TypeTerm),
    compound_name_arguments(TypeTerm, Kind, [C]),
    subject_output(C, Kind).

subject_output(C, Kind) :-
    atom(C),
    !,
    pe_output(C, Kind).
subject_output(mod(B, M), Kind) :-
    atom(B),
    atom(M),
    pe_output(B, Kind),
    pe_modifier(B, M).

%% check_binding_positions(+V, +Val, +Term, +Origin)
%  pe_where's per-position legality over the TERM's occurrences of V —
%  including the ratified pin refusals, which only the term can
%  trigger (pins never occur inside store goals).  Occurrences of V
%  inside OTHER goals are validated by those goals' own discharge:
%  a has_type/2 that V's substitution grounds is checked when it
%  discharges, and another binding's right-hand side is checked at
%  that binding's discharge.  occurrences/3 walks, never copies.
check_binding_positions(V, Val, Term, Origin) :-
    occurrences(Term, root, TermOccs),
    forall(( member(occ(X, Ctx), TermOccs), X == V ),
           check_at(Ctx, Val, Origin)).

check_at(Ctx, Val, Origin) :-
    catch(
        check_value_at(Ctx, Val),
        error(pe_where(E), _),
        throw(error(pe_elaborate(binding_rejected(E, origin(Origin))), _))
    ).

%% ============================================
%% CANONICAL STORE (ruling 1)
%% ============================================
%
% Order derives from the elaborated TERM's own traversal, never from
% variable allocation age (the measured accidentally-stable hazard,
% probe_residual_order.pl): copy Term-Store once (sharing preserved),
% number the term's variables first, then number store-only variables
% goal-by-goal — each step numbering the projection-least remaining
% goal, so numbering is a function of logical content, not input
% order.  Sort by the numbered copies, dedup ==-identical goals
% (identical goal over identical store variables; goals differing only
% in WHICH variable they constrain are distinct and both kept).
% This is an ORDERING DEVICE, superseded by peid-v1's numbering when
% that freezes — no identity claim attaches to it.

%% canonical_store(+Term, +Goals, -CanonGoals)
canonical_store(Term, Goals, CanonGoals) :-
    maplist([G, G-none]>>true, Goals, Pairs),
    canonical_pairs(Term, Pairs, CanonPairs),
    pairs_keys(CanonPairs, CanonGoals).

%% canonical_pairs(+Term, +Pairs, -CanonPairs)
%  Pairs are Goal-Origin; the canonical order and dedup are computed
%  on numbered COPIES (keys) while the original goals — with their
%  live, shared variables — ride as values.  Deduplication keeps the
%  first origin of a merged pair.
canonical_pairs(Term, Pairs, CanonPairs) :-
    pairs_keys(Pairs, Goals),
    copy_term(Term-Goals, TermC-GoalsC),
    numbervars(TermC, 0, N0),
    number_store_goals(GoalsC, N0),
    pairs_keys_values(KeyedPairs, GoalsC, Pairs),
    msort(KeyedPairs, Sorted),
    dedup_by_key(Sorted, CanonPairs).

%% number_store_goals(+GoalsC, +N0)
%  Number store-only variables: repeatedly take the projection-least
%  goal still containing variables and number it (numbervars reaches
%  shared variables across goals, so later goals inherit numbering).
number_store_goals(GoalsC, N0) :-
    include(goal_has_var, GoalsC, WithVars),
    (   WithVars == []
    ->  true
    ;   least_by_projection(WithVars, Least),
        numbervars(Least, N0, N1),
        number_store_goals(GoalsC, N1)
    ).

goal_has_var(G) :-
    term_variables(G, [_|_]).

%% least_by_projection(+Goals, -Least)
%  Projection: a COPY (safe — used only for comparison, never kept)
%  with every remaining variable bound to a fixed sentinel, so
%  comparison is over logical shape, not variable age.
least_by_projection([G|Gs], Least) :-
    projection(G, P),
    least_by_projection_(Gs, G, P, Least).

least_by_projection_([], Least, _, Least).
least_by_projection_([G|Gs], Best, BestP, Least) :-
    projection(G, P),
    (   P @< BestP
    ->  least_by_projection_(Gs, G, P, Least)
    ;   least_by_projection_(Gs, Best, BestP, Least)
    ).

projection(G, P) :-
    copy_term(G, P),
    term_variables(P, Vs),
    maplist(=('$unnumbered'), Vs).

dedup_by_key([], []).
dedup_by_key([K-V, K2-_|Rest], Out) :-
    K == K2,
    !,
    dedup_by_key([K-V|Rest], Out).
dedup_by_key([_K-V|Rest], [V|Out]) :-
    dedup_by_key(Rest, Out).
