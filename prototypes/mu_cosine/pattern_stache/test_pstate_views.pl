:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_pstate_views.pl - tests for the pattern-state experiment
% harness (pstate_views.pl) and the CHARACTERIZATION of the store
% ordering defect it inherits.
%
% Run from this directory:
%   swipl -g run_tests -t halt test_pstate_views.pl
%
% Fixtures that name a pattern state carry the mandatory no_identity/1
% marker (ruling 7(b)); nothing here computes a digest, assigns a
% persistent name, or derives a cache key.

:- use_module(pstate_views).
:- use_module(pe_elaborate, [elaborate/3, canonical_store/3]).
:- use_module(library(plunit)).
:- use_module(library(lists)).

%% Fixtures: file-local names, mandatory no-identity marker.
%% pstate_fixture(no_identity(LocalName), Term, Goals)

pstate_fixture(no_identity(pv_one_residual),
    product(hop_decay(C, gamma(0.6)), lca_frac(C)),
    [has_type(_X, substrate(C))]).
pstate_fixture(no_identity(pv_two_kinds),
    lca_frac(C),
    [has_type(_X, substrate(C)), has_type(_Y, judge(_J))]).
pstate_fixture(no_identity(pv_ground_term_residual_store),
    fs,
    [has_type(_X, substrate(_C))]).
pstate_fixture(no_identity(pv_open_term_no_goals),
    lca_frac(_C),
    []).

:- begin_tests(pstate_harness).

test(fixtures_carry_no_identity_marker) :-
    forall(clause(pstate_fixture(Marker, _, _), _),
           Marker = no_identity(_)).

test(version_is_experiment_scoped) :-
    pstate_probe_version(V),
    V == 'pstate-probe-v0'.

% Both views come out of one real elaboration, for every fixture.
test(both_views_from_real_elaboration, [forall(pstate_fixture(_, T, G))]) :-
    pstate_both_views(T, G, Tokens, Features),
    Tokens = [tok('<PSTATE:pstate-probe-v0>', 'PSTATE_ROOT')|_],
    last(Tokens, tok('<PSTATE_EOS>', 'PSTATE_ROOT')),
    memberchk('residual_goal_count'-_, Features).

% A ground elaboration is NOT a pattern state: refused, because a
% ground term already has a sealed canonical form and these views must
% never compete with it.
test(ground_state_refused,
     error(pstate_views(not_a_pattern_state(ground(_))))) :-
    pstate_both_views(lca_frac(C), [C = simplemind], _, _).

% Structure view: variables appear as <VAR:N> slots and a variable
% shared between term and store carries the same slot in both.
test(shared_variable_same_slot_in_term_and_store) :-
    pstate_fixture(no_identity(pv_one_residual), T, G),
    pstate_both_views(T, G, Tokens, _),
    % the term's variable and the store goal's substrate argument are
    % the same variable, so the same <VAR:N> token appears under both
    % a TERM path and a STORE path
    memberchk(tok('<VAR:0>', TermPath), Tokens),
    sub_atom(TermPath, 0, _, _, 'PSTATE_ROOT/TERM'),
    once(( member(tok('<VAR:0>', StorePath), Tokens),
           sub_atom(StorePath, 0, _, _, 'PSTATE_ROOT/STORE') )).

% Features view: counts describe the store's shape.
test(features_describe_store_shape) :-
    pstate_fixture(no_identity(pv_two_kinds), T, G),
    pstate_features_view_of(T, G, F),
    memberchk('residual_goal_count'-2, F),
    memberchk('goal_functor(has_type/2)'-2, F),
    memberchk('constraint_kind(substrate)'-1, F),
    memberchk('constraint_kind(judge)'-1, F),
    memberchk('term_var_count'-1, F).

% An open term with an empty store still yields a pattern state with
% an empty store: zero goals, one variable.
test(open_term_empty_store_features) :-
    pstate_fixture(no_identity(pv_open_term_no_goals), T, G),
    pstate_features_view_of(T, G, F),
    memberchk('residual_goal_count'-0, F),
    memberchk('term_var_count'-1, F),
    memberchk('var_degree_max'-0, F).

pstate_features_view_of(T, G, F) :-
    pstate_of(T, G, S),
    pstate_features_view(S, F).

:- end_tests(pstate_harness).

%% ============================================
%% The ordering defect: CHARACTERIZED, not patched
%% ============================================
%
% DESIGN_pattern_state_identity.md "The ordering defect" reports a
% measured input-order dependence in pe_elaborate's canonical store:
% when two or more residual goals share a projection AND are
% projection-least, least_by_projection_/4 breaks the tie with a
% strict @<, so the earlier goal in the INPUT list wins and the
% canonical output follows presentation order.
%
% These tests pin the CURRENT (defective) behaviour, in the same
% recorded-divergence style the mode-classification suite uses: they
% assert what the code does today, so the day the ordering is fixed
% they FAIL and force the fix, this file, and the design note to be
% reconciled in one review.  The defect is deliberately NOT patched
% around in pstate_views.pl — the structure view emits what the
% elaborator gives it.

:- begin_tests(ordering_defect_characterization).

% The stable case, for contrast: goals with DISTINCT projections
% canonicalize order-independently (this is what the merged suites
% cover, and it still holds).
test(distinct_projections_are_order_independent) :-
    G1 = has_type(_A, substrate(pearltrees)),
    G2 = has_type(_B, judge(haiku)),
    canonical_store(fs, [G1, G2], C1),
    canonical_store(fs, [G2, G1], C2),
    numbered(C1, N1),
    numbered(C2, N2),
    N1 == N2.

% The fully symmetric case, also stable — and stable for a REASON:
% swapping the two goals is an automorphism, so both presentations
% denote the same state up to renaming.
test(symmetric_tie_is_stable_because_swap_is_a_variant) :-
    G1 = has_type(_A, substrate(pearltrees)),
    G2 = has_type(_B, substrate(pearltrees)),
    canonical_store(fs, [G1, G2], C1),
    canonical_store(fs, [G2, G1], C2),
    numbered(C1, N1),
    numbered(C2, N2),
    N1 == N2,
    [G1, G2] =@= [G2, G1].

% THE DEFECT.  One store, three goals over two shared variables:
% variable A is constrained twice (judge + substrate), B once (judge).
% The two judge goals share a projection and are projection-least
% (judge @< substrate), so the tie is reached BEFORE the substrate
% goal can distinguish A from B — and the tie falls to input order.
%
% Asserted as the defect it is: the two presentations of ONE store
% produce canonical forms that are not even variants of each other.
test(near_symmetric_tie_is_input_order_dependent_KNOWN_DEFECT) :-
    GJa = has_type(A, judge(_J1)),
    GJb = has_type(_B, judge(_J2)),
    GSa = has_type(A, substrate(_C)),
    elaborate(fs, [GJa, GJb, GSa], P1),
    elaborate(fs, [GJb, GJa, GSa], P2),
    numbered(P1, N1),
    numbered(P2, N2),
    N1 \== N2,                 % <- the defect; should be == once fixed
    \+ P1 =@= P2.              % <- not even alpha-equivalent

% The features view is IMMUNE to the same defect: counts cannot carry
% order.  This asymmetry is a ruling-6 input (see the design note's
% constraint mapping), which is why the harness emits both views.
test(features_view_immune_to_the_ordering_defect) :-
    GJa = has_type(A, judge(_J1)),
    GJb = has_type(_B, judge(_J2)),
    GSa = has_type(A, substrate(_C)),
    pstate_features_view_of2(fs, [GJa, GJb, GSa], F1),
    pstate_features_view_of2(fs, [GJb, GJa, GSa], F2),
    F1 == F2.

% ...and the structure view is NOT immune, precisely because it
% serializes the store in canonical order.
test(structure_view_inherits_the_defect_KNOWN) :-
    GJa = has_type(A, judge(_J1)),
    GJb = has_type(_B, judge(_J2)),
    GSa = has_type(A, substrate(_C)),
    pstate_structure_view_of2(fs, [GJa, GJb, GSa], T1),
    pstate_structure_view_of2(fs, [GJb, GJa, GSa], T2),
    T1 \== T2.                 % <- should become == once fixed

pstate_features_view_of2(T, G, F) :-
    pstate_of(T, G, S),
    pstate_features_view(S, F).

pstate_structure_view_of2(T, G, Toks) :-
    pstate_of(T, G, S),
    pstate_structure_view(S, Toks).

numbered(X, N) :-
    copy_term(X, N),
    numbervars(N, 0, _).

:- end_tests(ordering_defect_characterization).

%% ============================================
%% The ruled acceptance criterion for canonical labelling
%% ============================================
%
% Ruling 1 (AST-lane, on this note): adopt canonical labelling by
% refinement PLUS INDIVIDUALIZATION — refinement alone is sound but
% incomplete (it stabilizes with non-singleton colour classes on
% symmetric structures), and if the tie-break there were input order
% we would rebuild the defect one layer deeper.  The ruling's stated
% correctness property is transcribed here as a reusable predicate, so
% whoever builds the scheme has the acceptance test ready and does not
% have to re-derive it:
%
%   for ANY store, EVERY input permutation must yield =@= outputs.
%
% It is checkable without a digest, so it stays left of the identity
% fence (DESIGN_pattern_state_identity.md §7).

%% permutation_stable(+Term, +Goals) is semidet.
%  True when elaborating Term against every permutation of Goals
%  yields pairwise alpha-equivalent (=@=) states.  This is the
%  property the future canonical labelling must satisfy for all
%  inputs; today it holds for some and fails for the §2 shape.
permutation_stable(Term, Goals) :-
    findall(S,
            ( permutation(Goals, P),
              elaborate(Term, P, S)
            ),
            States),
    States = [First|Rest],
    forall(member(S, Rest), S =@= First).

:- begin_tests(permutation_property).

% Holds today: distinct projections.
test(distinct_projections_permutation_stable) :-
    permutation_stable(fs, [has_type(_A, substrate(pearltrees)),
                            has_type(_B, judge(haiku))]).

% Holds today, and for the right reason: a fully symmetric pair is an
% automorphism, so every permutation is a variant of every other.
test(symmetric_pair_permutation_stable) :-
    permutation_stable(fs, [has_type(_A, substrate(pearltrees)),
                            has_type(_B, substrate(pearltrees))]).

% Holds today: goals sharing a term variable, distinct kinds.
test(shared_term_var_permutation_stable) :-
    permutation_stable(lca_frac(C), [has_type(_X, substrate(C)),
                                     has_type(_Y, judge(_J))]).

% DOES NOT hold today — the §2 near-symmetric shape.  Asserted as the
% known failure it is; when canonical labelling lands this test fails
% and forces the reconciliation.
test(near_symmetric_NOT_permutation_stable_KNOWN_DEFECT) :-
    GJa = has_type(A, judge(_J1)),
    GJb = has_type(_B, judge(_J2)),
    GSa = has_type(A, substrate(_C)),
    \+ permutation_stable(fs, [GJa, GJb, GSa]).

:- end_tests(permutation_property).
