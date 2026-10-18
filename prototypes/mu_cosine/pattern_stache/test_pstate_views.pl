:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_pstate_views.pl - tests for the pattern-state experiment
% harness (pstate_views.pl) and for the store ordering property that
% canonical labelling now provides.
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
:- use_module(library(random)).

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
%% Store ordering: the defect, and its repair
%% ============================================
%
% HISTORY, kept because the reconciliation was the point.  These tests
% were written as recorded-divergence characterizations of a measured
% defect: pe_elaborate's canonical store used a projection-least
% ordering device whose tie-break was a strict @<, so when two
% residual goals shared a projection AND were projection-least, the
% earlier goal in the INPUT list won and the canonical output followed
% presentation order (DESIGN_pattern_state_identity.md §2).  Their
% headers said that when the ordering was fixed they would FAIL and
% force the fix, this file, and the note to be reconciled in one pass.
%
% That is what happened.  pe_canonical.pl now provides canonical
% labelling (refinement + individualization, ruling 1 as amended), the
% three assertions flipped, and they are rewritten here as the
% positive properties they were always the negative of.  The lane's
% recorded-divergence discipline worked exactly as designed, so the
% mechanism is worth leaving visible rather than tidying away.

:- begin_tests(store_ordering).

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

% THE REPAIRED CASE (was: near_symmetric_tie_is_input_order_dependent).
% One store, three goals over two shared variables: A is constrained
% twice (judge + substrate), B once (judge).  The two judge goals are
% locally indistinguishable, so refinement's first round must carry
% the substrate goal's evidence into A's colour before any tie is
% considered — which is exactly what the old projection-least device
% failed to do.
test(near_symmetric_tie_is_now_order_independent) :-
    GJa = has_type(A, judge(_J1)),
    GJb = has_type(_B, judge(_J2)),
    GSa = has_type(A, substrate(_C)),
    elaborate(fs, [GJa, GJb, GSa], P1),
    elaborate(fs, [GJb, GJa, GSa], P2),
    numbered(P1, N1),
    numbered(P2, N2),
    N1 == N2,
    P1 =@= P2.

% The features view was immune to the defect by construction (counts
% cannot carry order), and remains stable now that the structure view
% is stable too.  The asymmetry it exposed is what made ruling 6's
% cost mapping measurable, so the test stays.
test(features_view_immune_to_the_ordering_defect) :-
    GJa = has_type(A, judge(_J1)),
    GJb = has_type(_B, judge(_J2)),
    GSa = has_type(A, substrate(_C)),
    pstate_features_view_of2(fs, [GJa, GJb, GSa], F1),
    pstate_features_view_of2(fs, [GJb, GJa, GSa], F2),
    F1 == F2.

% ...and the structure view, which serializes the store in canonical
% order, is now stable too — it was the view that inherited the defect,
% so it is the view that demonstrates the repair.
test(structure_view_now_stable_under_presentation_order) :-
    GJa = has_type(A, judge(_J1)),
    GJb = has_type(_B, judge(_J2)),
    GSa = has_type(A, substrate(_C)),
    pstate_structure_view_of2(fs, [GJa, GJb, GSa], T1),
    pstate_structure_view_of2(fs, [GJb, GJa, GSa], T2),
    T1 == T2.

pstate_features_view_of2(T, G, F) :-
    pstate_of(T, G, S),
    pstate_features_view(S, F).

pstate_structure_view_of2(T, G, Toks) :-
    pstate_of(T, G, S),
    pstate_structure_view(S, Toks).

numbered(X, N) :-
    copy_term(X, N),
    numbervars(N, 0, _).

:- end_tests(store_ordering).

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
%  yields pairwise alpha-equivalent (=@=) states.  This is the ruled
%  correctness property for canonical labelling, and it is digest-free,
%  so it sits left of the identity fence.
%
%  SCALING BOUND, same refuse-don't-degrade discipline as the algorithm
%  it validates: this enumerates n!, so it REFUSES above
%  permutation_exhaustive_limit/1 rather than hanging on a realistic
%  store.  Refusal is an error, never a silent pass — a property test
%  that quietly stops testing is worse than one that stops.
%  For larger stores use permutation_stable_sampled/4, which is
%  explicitly probabilistic at the call site.
permutation_stable(Term, Goals) :-
    length(Goals, N),
    permutation_exhaustive_limit(Limit),
    (   N =< Limit
    ->  true
    ;   throw(error(permutation_bound_exceeded(N, Limit,
                    use(permutation_stable_sampled/4)), _))
    ),
    findall(S,
            ( permutation(Goals, P),
              elaborate(Term, P, S)
            ),
            States),
    all_variants(States).

%% permutation_exhaustive_limit(-N)
%  8 goals = 40320 permutations, a few seconds; 9 would be ~10x that.
%  No pattern state the lane has produced exceeds 3 goals.
permutation_exhaustive_limit(8).

%% permutation_stable_sampled(+Term, +Goals, +K, +Seed) is semidet.
%  PROBABILISTIC: checks K seeded random permutations rather than all
%  n!.  Passing means "no counterexample among K samples", NOT
%  "stable" — the weaker claim is the caller's to make knowingly,
%  which is why the seed and K are explicit arguments and there is no
%  default.  Seeded, so a failure reproduces exactly.
permutation_stable_sampled(Term, Goals, K, Seed) :-
    set_random(seed(Seed)),
    findall(S,
            ( between(1, K, _),
              random_permutation(Goals, P),
              elaborate(Term, P, S)
            ),
            States),
    all_variants(States).

all_variants([]) :- !.
all_variants([First|Rest]) :-
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

% THE FLIP.  This was asserted as a known failure — the §2
% near-symmetric shape — with a header saying canonical labelling
% landing would fail it and force the reconciliation.  It did, and
% this is the reconciled form: all 6 permutations now yield one
% canonical form.
test(near_symmetric_now_permutation_stable) :-
    GJa = has_type(A, judge(_J1)),
    GJb = has_type(_B, judge(_J2)),
    GSa = has_type(A, substrate(_C)),
    permutation_stable(fs, [GJa, GJb, GSa]).

% The bound refuses rather than hanging: 9 goals is past the
% exhaustive limit, so the property REFUSES and names the sampled
% alternative instead of quietly enumerating 362880 permutations.
test(permutation_bound_refuses_above_limit,
     error(permutation_bound_exceeded(9, 8, use(permutation_stable_sampled/4)))) :-
    length(Goals, 9),
    maplist([has_type(_, substrate(pearltrees))]>>true, Goals),
    permutation_stable(fs, Goals).

% ...and the sampled variant handles the same store, probabilistically
% and with an explicit seed.  Nine DISTINGUISHABLE goals (distinct
% registered judges), so refinement separates them outright and no
% individualization is needed.
test(sampled_variant_handles_large_stores) :-
    Judges = [graph, human, luna, sonnet, haiku, gemini, opus, llm, 'gpt-5.5-low'],
    maplist([J, has_type(_, judge(J))]>>true, Judges, Goals),
    permutation_stable_sampled(fs, Goals, 20, 42).

%% ============================================
%% The branch cap: refuse, never degrade
%% ============================================
%
% MEASURED: individualization is exponential on FULLY SYMMETRIC stores
% (n goals identical up to renaming, nothing to distinguish them), and
% at the ruled cap of 1000 branches the refusal threshold is 6 such
% goals.  Recorded as a stated property rather than left to be
% discovered: below it the labelling succeeds, at and above it the
% labelling REFUSES with a named error and computes nothing.  There is
% no timeout, no partial result, and no fallback to the old ordering
% device.
%
% Realistic stores are unaffected: distinguishable goals are separated
% by refinement alone and consume zero branches.

test(five_symmetric_goals_still_labelled) :-
    length(Goals, 5),
    maplist([has_type(_, substrate(pearltrees))]>>true, Goals),
    elaborate(fs, Goals, pattern(fs, Store)),
    length(Store, 5).

test(six_symmetric_goals_refused_not_degraded,
     error(pe_canonical(branch_cap_exceeded(1000, _, _)))) :-
    length(Goals, 6),
    maplist([has_type(_, substrate(pearltrees))]>>true, Goals),
    elaborate(fs, Goals, _).

% A store of the same size but DISTINGUISHABLE is labelled without
% touching the cap — the cost is symmetry, not size.
test(six_distinguishable_goals_unaffected_by_cap) :-
    Judges = [graph, human, luna, sonnet, haiku, gemini],
    maplist([J, has_type(_, judge(J))]>>true, Judges, Goals),
    elaborate(fs, Goals, pattern(fs, Store)),
    length(Store, 6).

:- end_tests(permutation_property).
