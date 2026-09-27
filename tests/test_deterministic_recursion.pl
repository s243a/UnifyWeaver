% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% test_deterministic_recursion.pl — tests for the general deterministic-
% recursion classifier src/unifyweaver/core/deterministic_recursion.pl
% (docs/proposals/DETERMINISTIC_RECURSION_TAXONOMY.md §11).
%
% The fixtures are HERMETIC (defined here in `user`, not read from resolver.pl)
% and use deliberately unrelated predicate names, so the tests prove the
% classifier recognizes each taxonomy SHAPE structurally — not a frozen literal
% or a hard-coded predicate name. This is the property that lets one classifier
% replace the per-predicate =@= region recognizers.
%
%   swipl -q -g run_tests -t halt tests/test_deterministic_recursion.pl

:- use_module('../src/unifyweaver/core/deterministic_recursion').

% ---- fixtures (asserted into `user`) ---------------------------------------

% Pattern 1 / list_filter (filter_satisfies kin) — bare-element keep/drop with
% a satisfies/2 committed guard, tail self-call, difference-list build.
drt_filter([], _C, []).
drt_filter([V|Vs], C, Out) :-
    ( satisfies(V, C) -> Out = [V|Os] ; Out = Os ),
    drt_filter(Vs, C, Os).

% Pattern 1 / list_map_index, pkg row (key_pkg_rows kin) — package(N,V) -> N-I-V.
drt_pkgmap([], _I, []).
drt_pkgmap([package(N, V)|Rest], I, [N-I-V|Ks]) :-
    I1 is I + 1,
    drt_pkgmap(Rest, I1, Ks).

% Pattern 1 / list_map_index, dep row (key_dep_rows kin) — depends(N,V,D,C) ->
% (N-V)-I-Req via dep_to_req/3.
drt_depmap([], _I, []).
drt_depmap([depends(N, V, D, C)|Rest], I, [(N-V)-I-Req|Ks]) :-
    dep_to_req(D, C, Req),
    I1 is I + 1,
    drt_depmap(Rest, I1, Ks).

% Pattern 1 / bst_descent (tree_lookup kin) — compare/3 trichotomy, one of two
% tail self-calls, semidet.
drt_bst(t(L, K, V, R), Key, Val) :-
    compare(Ord, Key, K),
    ( Ord = (=) -> Val = V
    ; Ord = (<) -> drt_bst(L, Key, Val)
    ;             drt_bst(R, Key, Val) ).

% Pattern 6 / committed-choice tail (close_moving / dep_breaks kin) — a
% committing `->` whose condition contains a nondet-looking call, sole tail
% self-call after the commit.
drt_committed([], Acc, Acc).
drt_committed([X|Xs], Acc, Out) :-
    ( pick_something(X, Acc, New)
    -> drt_committed(Xs, [New|Acc], Out)   % sole self-call, after the commit
    ;  Out = Acc ).                          % else-branch terminates (fixpoint)

% Pattern 5 / mutual recursion, tail (segs_lt / segs_lt_1 kin).
drt_a([], _).
drt_a([X|Xs], K) :- ( X == K -> true ; drt_b(Xs, K) ).
drt_b([], _).
drt_b([X|Xs], K) :- ( X == K -> true ; drt_a(Xs, K) ).

% Pattern 5 / mutual recursion, NON-tail post-order (topo_all / topo_one kin).
drt_p([], Acc, Acc).
drt_p([N|Ns], Acc0, Acc) :- drt_q(N, Acc0, Acc1), drt_p(Ns, Acc1, Acc).
drt_q(node(Name, Kids), Acc0, [Name|Acc1]) :- drt_p(Kids, Acc0, Acc1).

% Decline cases.
drt_plain(X, Y) :- Y = f(X).                       % not recursive
drt_meta(G, X) :- call(G, X), drt_meta(G, X).      % meta-call unresolved

% guards referenced above need no definition for classification, but define
% trivial ones so current_predicate enumeration is clean.
satisfies(_, _).
dep_to_req(_, _, req).
pick_something(_, _, new).

% ---- tests ------------------------------------------------------------------

:- begin_tests(deterministic_recursion).

test(list_filter_shape) :-
    deterministic_recursion_class(user, drt_filter/3, C),
    assertion(C = tail_loop(user:drt_filter/3, list_filter(satisfies/2, 1, 2, 3))).

test(list_map_index_pkg_row) :-
    deterministic_recursion_class(user, drt_pkgmap/3, C),
    assertion(C = tail_loop(user:drt_pkgmap/3, list_map_index(package/2, pkg_row, 1, 2, 3))).

test(list_map_index_dep_row) :-
    deterministic_recursion_class(user, drt_depmap/3, C),
    assertion(C = tail_loop(user:drt_depmap/3, list_map_index(depends/4, dep_row, 1, 2, 3))).

test(bst_descent_shape) :-
    deterministic_recursion_class(user, drt_bst/3, C),
    assertion(C = tail_loop(user:drt_bst/3, bst_descent('t'/4, 2, 3, 1, 2, 3))).

test(committed_choice_tail) :-
    deterministic_recursion_class(user, drt_committed/3, C),
    assertion(C = committed(user:drt_committed/3, committed_tail(1))).

test(mutual_tail_scc) :-
    deterministic_recursion_class(user, drt_a/2, C),
    assertion(C = mutual([drt_a/2, drt_b/2], tail_shaped(true), _)).

test(mutual_tail_scc_symmetric) :-
    deterministic_recursion_class(user, drt_b/2, C),
    assertion(C = mutual([drt_a/2, drt_b/2], tail_shaped(true), _)).

test(mutual_nontail_scc) :-
    deterministic_recursion_class(user, drt_p/3, C),
    assertion(C = mutual([drt_p/3, drt_q/3], tail_shaped(false), _)).

test(decline_non_recursive) :-
    deterministic_recursion_class(user, drt_plain/2, C),
    assertion(C = decline(not_recursive)).

test(decline_meta_call) :-
    deterministic_recursion_class(user, drt_meta/2, C),
    assertion(C = decline(meta_call_unresolved)).

test(scc_membership) :-
    dr_scc(user, drt_a/2, SCC),
    assertion(SCC == [drt_a/2, drt_b/2]).

test(self_recursive_detected) :-
    assertion(dr_self_recursive(user, drt_filter/3)),
    assertion(\+ dr_self_recursive(user, drt_plain/2)).

:- end_tests(deterministic_recursion).
