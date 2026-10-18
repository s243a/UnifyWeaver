:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% probe_residual_order.pl - measurement probe for
% DESIGN_prolog_elaborator.md §"Order-dependence".
%
% Claim measured: sorting a residual goal store by the RAW standard
% order of terms (@<, msort/2) is construction-order dependent,
% because two fresh variables compare by AGE — so the same logical
% store, built with its variables created in a different order, sorts
% differently.  A canonical store order must therefore derive variable
% order from the elaborated TERM's own traversal (numbervars on a
% copy), not from variable identity.
%
% Run:
%   swipl -g run_tests -t halt probe_residual_order.pl
%
% This is a probe, not production code and not a consumer.

:- use_module(library(plunit)).

% fresh_var(-V): allocate a variable at RUN TIME, at this exact call,
% so creation order is under the probe's control.  (Clause-local
% variables are allocated by head unification and term construction in
% ways the compiler may reorder; reading a term allocates its
% variables at the moment of the call, in call order.)
fresh_var(V) :-
    term_string(T, "v(_)"),
    T = v(V).

% The stable key: copy the term-plus-store, number variables by first
% occurrence in a FIXED traversal (term first, then store), then sort
% by standard order of the numbered copies.
stable_store_order(Term, Store, Ordered) :-
    copy_term(Term-Store, TermC-StoreC),
    numbervars(TermC-StoreC, 0, _),
    pairs_keys_values(Pairs, StoreC, Store),
    msort(Pairs, SortedPairs),
    pairs_values(SortedPairs, Ordered).

:- begin_tests(residual_order_probe).

% MEASUREMENT 1a: for goals identical up to their variable, raw
% msort/2 order follows variable CREATION order.  The two runs build
% the textually identical store [g(V2,m2), g(V1,m1)] — marker AFTER
% the variable, so variable age is compared first — differing only in
% which variable was allocated first.  The marker that sorts first
% flips with creation order.  (If this test ever fails, SWI changed
% variable ordering semantics and the design note's premise must be
% re-measured.)
test(raw_standard_order_is_creation_order_dependent) :-
    % run A: the m1 goal's variable is created first
    fresh_var(VA1), fresh_var(VA2),
    msort([g(VA2, m2), g(VA1, m1)], [g(_, MA)|_]),
    % run B: the m2 goal's variable is created first
    fresh_var(VB2), fresh_var(VB1),
    msort([g(VB2, m2), g(VB1, m1)], [g(_, MB)|_]),
    MA == m1,
    MB == m2,
    MA \== MB.

% MEASUREMENT 1b: the hazard is CONDITIONAL — when functors differ,
% the functor comparison dominates before variable age is consulted,
% and the order is accidentally stable.  This is why the bug would
% survive most test suites: a mixed store looks stable in every test
% that doesn't hit the same-shape case.
test(raw_order_accidentally_stable_across_functors) :-
    fresh_var(A), fresh_var(B),
    msort([in_support(decay, B), has_type(A, substrate(B))], M1),
    fresh_var(D), fresh_var(C),
    msort([in_support(decay, D), has_type(C, substrate(D))], M2),
    M1 =@= M2,
    M1 = [has_type(_, _)|_].

% Contrast: with ground goals, raw msort/2 is stable — the hazard is
% variables only.
test(raw_order_stable_when_ground) :-
    msort([in_support(decay, d1), has_type(x, substrate(c))], M1),
    msort([in_support(decay, d1), has_type(x, substrate(c))], M2),
    M1 == M2.

% MEASUREMENT 2: the numbervars-by-term-traversal key is stable across
% variable creation order — the same two runs that flip under raw
% msort/2 agree once variable order derives from the elaborated TERM's
% own traversal instead of allocation age.
test(numbervars_key_is_creation_order_independent) :-
    fresh_var(VA1), fresh_var(VA2),
    stable_store_order(f(VA1, VA2), [g(VA2, m2), g(VA1, m1)], O1),
    fresh_var(VB2), fresh_var(VB1),
    stable_store_order(f(VB1, VB2), [g(VB2, m2), g(VB1, m1)], O2),
    O1 =@= O2.

% MEASUREMENT 3: sort/2 deduplicates by ==, which is the right dedup
% for a store — two occurrences of the SAME goal over the SAME store
% variables collapse; goals over distinct variables (distinct
% constraints) do not.
test(sort_dedups_identical_goals_only) :-
    G1 = has_type(X, substrate(C)),
    G2 = has_type(_Y, substrate(C)),
    sort([G1, G1, G2], Sorted),
    length(Sorted, 2),
    X = X.   % keep X shared, silence singleton

:- end_tests(residual_order_probe).
