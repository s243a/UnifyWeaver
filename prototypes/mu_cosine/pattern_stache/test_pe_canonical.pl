:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_pe_canonical.pl - tests for the canonical labelling itself
% (pe_canonical.pl): refinement, individualization, the branch cap,
% and the ruled correctness property.
%
% Run from this directory:
%   swipl -g run_tests -t halt test_pe_canonical.pl
%
% Nothing here computes a digest or names a state: a canonical FORM is
% not an identity (ruling 4 stands, peid-v1 still fences digests).

:- use_module(pe_canonical).
:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module(library(apply)).

% label(+Term, +Goals, -NumberedTermGoals): canonical form of a state,
% computed on a private copy so the caller's term is untouched.
label(Term, Goals, Numbered) :-
    copy_term(Term-Goals, T-G),
    canonical_number(T, G),
    msort(G, SortedG),
    Numbered = T-SortedG.

%% ============================================
%% Refinement does the work when it can
%% ============================================

:- begin_tests(refinement).

% Distinguishable variables are separated by refinement alone — one
% round, no individualization, so the branch cap is never touched.
test(distinguishable_vars_separated_by_refinement) :-
    canonical_colours(fs, [has_type(_, judge(graph)),
                           has_type(_, judge(human)),
                           has_type(_, judge(luna))], Cs),
    msort(Cs, Sorted),
    Sorted == [0, 1, 2].

% Refinement is GLOBAL, not local: the two judge goals are locally
% identical, and only the substrate goal distinguishes their subjects.
% Refinement must carry that evidence into the colour — this is
% exactly what the superseded projection-least device failed to do.
test(refinement_uses_global_evidence) :-
    GJa = has_type(A, judge(_)),
    GJb = has_type(_B, judge(_)),
    GSa = has_type(A, substrate(_)),
    canonical_colours(fs, [GJa, GJb, GSa], Cs),
    % five variables, all distinguished: A and B differ because A is
    % reached by a substrate goal and B is not
    length(Cs, 5),
    sort(Cs, Distinct),
    length(Distinct, 5).

% A variable occurring in the term is distinguishable from one that
% occurs only in the store, because the context wrapper differs.
test(term_and_store_contexts_are_distinct) :-
    canonical_colours(lca_frac(C), [has_type(_X, substrate(C))], Cs),
    sort(Cs, Distinct),
    length(Distinct, 2).

% Position within one goal is distinguishing: the two arguments of a
% single goal must not collapse to one colour.
test(argument_position_distinguishes) :-
    canonical_colours(fs, [pair_probe(_P, _Q)], Cs),
    sort(Cs, D),
    length(D, 2).

:- end_tests(refinement).

%% ============================================
%% Individualization handles what refinement cannot
%% ============================================

:- begin_tests(individualization).

% A fully symmetric pair: refinement stabilizes with a non-singleton
% class, individualization branches over both members, and the
% lexicographic minimum is taken.  Both presentations agree.
test(symmetric_pair_still_canonical) :-
    G1 = has_type(_A, substrate(pearltrees)),
    G2 = has_type(_B, substrate(pearltrees)),
    label(fs, [G1, G2], L1),
    label(fs, [G2, G1], L2),
    L1 == L2.

% The classic automorphism g(A,B)/g(B,A): swapping is an automorphism,
% so every presentation must yield one form.
test(automorphism_yields_one_form) :-
    label(fs, [g_probe(X, Y), g_probe(Y, X)], L1),
    label(fs, [g_probe(P, Q), g_probe(Q, P)], L2),
    L1 == L2.

% Individualization takes the MINIMUM over branches, not the first
% branch reached — so the answer cannot depend on enumeration order.
% Checked by permuting the store: every permutation agrees.
test(minimum_over_branches_not_first_branch) :-
    Goals = [has_type(_A, substrate(pearltrees)),
             has_type(_B, substrate(pearltrees)),
             has_type(_C, judge(haiku))],
    findall(L, ( permutation(Goals, P), label(fs, P, L) ), Ls),
    sort(Ls, [_]).

:- end_tests(individualization).

%% ============================================
%% The ruled correctness property
%% ============================================
%
% For any store, every input permutation must yield the same canonical
% form.  Exhaustive over a corpus of shapes chosen to include the ones
% that broke the superseded device.

:- begin_tests(permutation_property).

corpus(distinct_kinds,
       fs, [has_type(_, substrate(pearltrees)), has_type(_, judge(haiku))]).
corpus(symmetric_pair,
       fs, [has_type(_, substrate(pearltrees)), has_type(_, substrate(pearltrees))]).
corpus(near_symmetric,
       fs, [has_type(A, judge(_)), has_type(_, judge(_)), has_type(A, substrate(_))]).
corpus(shared_term_var,
       lca_frac(C), [has_type(_, substrate(C)), has_type(_, judge(_))]).
corpus(chain,
       fs, [has_type(A, judge(_)), has_type(A, substrate(_)),
            has_type(B, judge(_)), has_type(B, substrate(_)),
            has_type(B, judge(haiku))]).
corpus(term_and_store_sharing,
       product(hop_decay(C, gamma(0.6)), lca_frac(C)),
       [has_type(_, substrate(C)), has_type(_, judge(_))]).
corpus(three_way_symmetric,
       fs, [has_type(_, judge(haiku)), has_type(_, judge(haiku)),
            has_type(_, judge(haiku))]).

test(every_permutation_agrees, [forall(corpus(_Name, Term, Goals))]) :-
    findall(L, ( permutation(Goals, P), label(Term, P, L) ), Ls),
    sort(Ls, Distinct),
    Distinct = [_].

:- end_tests(permutation_property).

%% ============================================
%% The branch cap: refuse, never degrade
%% ============================================

:- begin_tests(branch_cap).

test(cap_is_stated) :-
    canonical_branch_cap(Cap),
    integer(Cap),
    Cap > 0.

% Refusal is an error naming the cap — not a timeout, not a partial
% result, and not a fallback to a weaker ordering.
test(pathological_symmetry_refused,
     error(pe_canonical(branch_cap_exceeded(_, _, _)))) :-
    length(Goals, 6),
    maplist([has_type(_, substrate(pearltrees))]>>true, Goals),
    label(fs, Goals, _).

% Size alone does not trigger it: the cost is symmetry.  Nine
% distinguishable goals are separated by refinement and never branch.
test(size_alone_does_not_trigger_the_cap) :-
    Judges = [graph, human, luna, sonnet, haiku, gemini, opus, llm, 'gpt-5.5-low'],
    maplist([J, has_type(_, judge(J))]>>true, Judges, Goals),
    label(fs, Goals, _-Store),
    length(Store, 9).

:- end_tests(branch_cap).

%% ============================================
%% Non-destructiveness and edge cases
%% ============================================

:- begin_tests(canonical_edges).

% canonical_colours/3 computes on a copy: the caller's variables stay
% unbound (the copy-don't-mutate discipline the lane applies
% everywhere).
test(colours_do_not_bind_the_caller) :-
    T = lca_frac(C),
    canonical_colours(T, [has_type(_X, substrate(C))], _),
    var(C).

% A ground state has no variables to label.
test(ground_state_has_no_colours) :-
    canonical_colours(lca_frac(simplemind), [], []).

% An empty store over an open term still labels the term's variables.
test(open_term_empty_store) :-
    canonical_colours(lca_frac(_C), [], Cs),
    Cs == [0].

:- end_tests(canonical_edges).
